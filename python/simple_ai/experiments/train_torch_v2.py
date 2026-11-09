"""Train the larger Simple AI v2 Indonesian decoder-only Transformer.

This entry point is separate from train_torch.py so v1 experiments and
checkpoints remain reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from ..byte_bpe import ByteBPETokenizer
from ..tokenizer import CharTokenizer, BPETokenizer, load_tokenizer
from ..torch_backend.checkpoint import load_checkpoint, save_checkpoint
from ..torch_backend.dataset import PackedTokenDataset, load_documents
from ..torch_backend.model import GPTConfig, GPTLanguageModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def build_tokenizer(kind: str, documents: list[str], vocab_size: int):
    if kind == "byte-bpe":
        return ByteBPETokenizer.train(documents, vocab_size=vocab_size)
    if kind == "bpe":
        return BPETokenizer.train(
            "\n".join(documents), vocab_size=vocab_size
        )
    return CharTokenizer.from_text("\n".join(documents))


def evaluate(
    model: GPTLanguageModel,
    dataset: PackedTokenDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> float | None:
    if not len(dataset):
        return None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    try:
        with torch.inference_mode():
            for batch_index, (inputs, targets) in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                _, loss = model(inputs, targets)
                count = targets.numel()
                total_loss += float(loss.item()) * count
                total_tokens += count
    finally:
        model.train(was_training)
    return total_loss / total_tokens if total_tokens else None


def learning_rate_at_step(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay to a non-zero LR floor."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Simple AI v2 with byte-level BPE and a larger GPT."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--tokenizer", choices=("byte-bpe", "bpe", "char"), default="byte-bpe"
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument(
        "--window-stride", type=int, default=None,
        help="Token-stream window stride; defaults to context size (non-overlapping input blocks).",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument(
        "--eval-batches", type=int, default=100,
        help="Validation batches per epoch; 0 means full evaluation.",
    )
    parser.add_argument(
        "--full-final-eval", action="store_true",
        help="Evaluate all validation and test windows at the end.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="Limit this invocation for a smoke test; 0 means no limit.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    positive = (
        args.vocab_size, args.context_size, args.embedding_dim,
        args.num_heads, args.num_layers, args.epochs, args.batch_size,
        args.checkpoint_steps,
    )
    if any(value < 1 for value in positive):
        parser.error("Vocabulary, model dimensions, epochs, batch, and checkpoint interval must be positive.")
    if args.embedding_dim % args.num_heads:
        parser.error("--embedding-dim must be divisible by --num-heads.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay non-negative.")
    if not 0.0 < args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be in (0, 1].")
    if args.warmup_steps < 0 or args.max_steps < 0 or args.num_workers < 0:
        parser.error("Warmup, max steps, and workers cannot be negative.")
    if args.window_stride is not None and args.window_stride < 1:
        parser.error("--window-stride must be positive.")
    if args.eval_batches < 0 or args.gradient_clip <= 0:
        parser.error("Evaluation batches cannot be negative and gradient clip must be positive.")

    train_path = Path(args.train)
    validation_path = Path(args.validation)
    test_path = Path(args.test)
    for path in (train_path, validation_path, test_path):
        if not path.is_file():
            raise SystemExit(f"Corpus file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    if not args.resume:
        # A fresh run must not leave an older checkpoint/report looking current.
        for artifact_name in (
            "checkpoint_latest.pt",
            "checkpoint_best.pt",
            "training_report.json",
        ):
            artifact = output_dir / artifact_name
            if artifact.exists():
                artifact.unlink()

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.perf_counter()
    train_documents = load_documents(train_path)
    validation_documents = load_documents(validation_path)
    test_documents = load_documents(test_path)

    if args.resume:
        if not tokenizer_path.is_file():
            raise SystemExit(f"Tokenizer is missing; cannot resume: {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = build_tokenizer(
            args.tokenizer, train_documents, args.vocab_size
        )
        tokenizer.save(tokenizer_path)

    train_data = PackedTokenDataset(
        train_documents, tokenizer, context_size=args.context_size,
        stride=args.window_stride,
    )
    validation_data = PackedTokenDataset(
        validation_documents, tokenizer, context_size=args.context_size,
        stride=train_data.stride,
    )
    test_data = PackedTokenDataset(
        test_documents, tokenizer, context_size=args.context_size,
        stride=train_data.stride,
    )
    if not len(train_data) or not len(validation_data) or not len(test_data):
        raise SystemExit("At least one split produced zero training windows.")

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    run_config = {
        "version": 2,
        "model": model.get_config(),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_steps": args.warmup_steps,
        "total_steps": total_steps,
        "weight_decay": args.weight_decay,
        "tokenizer_format": tokenizer.format_name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "dataset_format": "packed-token-stream-v1",
        "window_stride": train_data.stride,
        "context_size": args.context_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "training_file": str(train_path),
    }

    start_epoch = 0
    resume_step = 0
    global_step = 0
    best_validation_loss = None
    if args.resume:
        state = load_checkpoint(
            args.resume, model=model, optimizer=optimizer,
            map_location=device,
        )
        saved = state.get("config", {})
        if saved != run_config:
            raise SystemExit(
                "Resume configuration differs from checkpoint. Keep model, "
                "tokenizer, batch size, epochs, and schedule settings unchanged."
            )
        start_epoch = state["epoch"]
        resume_step = state["step_in_epoch"]
        global_step = state["global_step"]
        best_validation_loss = state["best_validation_loss"]
        print(
            f"Resumed at epoch={start_epoch + 1}, "
            f"step_in_epoch={resume_step}, global_step={global_step}",
            flush=True,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"tokenizer: {tokenizer.format_name}")
    print(f"vocabulary_size: {tokenizer.vocab_size:,}")
    print(f"documents: train={len(train_documents):,} validation={len(validation_documents):,} test={len(test_documents):,}")
    print(f"windows: train={len(train_data):,} validation={len(validation_data):,} test={len(test_data):,}")
    print(f"packed_tokens: train={train_data.total_tokens:,} validation={validation_data.total_tokens:,} test={test_data.total_tokens:,}")
    print(f"window_stride: {train_data.stride}")
    print(f"dropped_tail_tokens: train={train_data.dropped_tokens:,} validation={validation_data.dropped_tokens:,} test={test_data.dropped_tokens:,}")
    print(f"empty_documents_skipped: train={train_data.skipped_documents} validation={validation_data.skipped_documents} test={test_data.skipped_documents}")
    print(f"parameters: {parameter_count:,}")
    print(f"steps_per_epoch: {steps_per_epoch:,}")
    print(f"planned_total_steps: {total_steps:,}")
    print(f"preparation_seconds: {time.perf_counter() - started:.2f}")

    latest_path = output_dir / "checkpoint_latest.pt"
    best_path = output_dir / "checkpoint_best.pt"
    steps_this_run = 0
    stop_early = False
    history: list[dict[str, float | int | None]] = []

    for epoch in range(start_epoch, args.epochs):
        sampler = RandomSampler(
            train_data,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        model.train()
        loss_sum = 0.0
        token_sum = 0
        processed_batches = 0

        for batch_index, (inputs, targets) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_step:
                continue

            next_global_step = global_step + 1
            current_lr = learning_rate_at_step(
                next_global_step,
                base_lr=args.learning_rate,
                warmup_steps=args.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Model did not return a training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            optimizer.step()

            global_step = next_global_step
            steps_this_run += 1
            processed_batches += 1
            count = targets.numel()
            loss_sum += float(loss.detach().item()) * count
            token_sum += count

            next_step_in_epoch = batch_index + 1
            if global_step % args.checkpoint_steps == 0:
                save_checkpoint(
                    latest_path, model=model, optimizer=optimizer,
                    epoch=epoch, step_in_epoch=next_step_in_epoch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    config=run_config,
                )
                print(
                    f"epoch={epoch + 1} step={global_step:,} "
                    f"loss={loss.item():.4f} lr={current_lr:.2e} "
                    f"checkpoint={latest_path.name}",
                    flush=True,
                )

            if args.max_steps and steps_this_run >= args.max_steps:
                stop_early = True
                break

        train_loss = loss_sum / token_sum if token_sum else None
        validation_loss = evaluate(
            model, validation_data,
            batch_size=args.batch_size, device=device,
            max_batches=args.eval_batches,
        )
        history.append({
            "epoch": epoch + 1,
            "training_loss_this_invocation": train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"epoch_complete={epoch + 1} "
            f"train_loss={train_loss} validation_loss={validation_loss}",
            flush=True,
        )

        if validation_loss is not None and (
            best_validation_loss is None or validation_loss < best_validation_loss
        ):
            best_validation_loss = validation_loss
            save_checkpoint(
                best_path, model=model, optimizer=optimizer,
                epoch=epoch + 1, step_in_epoch=0,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                config=run_config,
            )

        if stop_early:
            save_checkpoint(
                latest_path, model=model, optimizer=optimizer,
                epoch=epoch, step_in_epoch=resume_step + processed_batches,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                config=run_config,
            )
            break

        save_checkpoint(
            latest_path, model=model, optimizer=optimizer,
            epoch=epoch + 1, step_in_epoch=0,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            config=run_config,
        )
        start_epoch = epoch + 1
        resume_step = 0

    final_eval_batches = 0 if args.full_final_eval else args.eval_batches
    final_validation_loss = evaluate(
        model, validation_data,
        batch_size=args.batch_size, device=device,
        max_batches=final_eval_batches,
    )
    final_test_loss = evaluate(
        model, test_data,
        batch_size=args.batch_size, device=device,
        max_batches=final_eval_batches,
    )

    report = {
        "version": 2,
        "device": str(device),
        "tokenizer": tokenizer.format_name,
        "vocabulary_size": tokenizer.vocab_size,
        "model_config": model.get_config(),
        "parameters": parameter_count,
        "dataset": {
            "format": "packed-token-stream-v1",
            "window_stride": train_data.stride,
            "train_tokens": train_data.total_tokens,
            "validation_tokens": validation_data.total_tokens,
            "test_tokens": test_data.total_tokens,
            "train_windows": len(train_data),
            "validation_windows": len(validation_data),
            "test_windows": len(test_data),
            "dropped_tail_tokens": {
                "train": train_data.dropped_tokens,
                "validation": validation_data.dropped_tokens,
                "test": test_data.dropped_tokens,
            },
        },
        "global_step": global_step,
        "planned_total_steps": total_steps,
        "best_validation_loss": best_validation_loss,
        "final_validation_loss": final_validation_loss,
        "final_test_loss": final_test_loss,
        "final_validation_perplexity": (
            math.exp(min(final_validation_loss, 700.0))
            if final_validation_loss is not None else None
        ),
        "final_test_perplexity": (
            math.exp(min(final_test_loss, 700.0))
            if final_test_loss is not None else None
        ),
        "epoch_history_this_run": history,
        "checkpoint_latest": str(latest_path),
        "checkpoint_best": str(best_path) if best_path.exists() else None,
        "corpus_license_note": (
            "The corpus is CC BY-SA 4.0. Review attribution, share-alike, "
            "and model-weight implications before redistribution."
        ),
    }
    write_json(output_dir / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
