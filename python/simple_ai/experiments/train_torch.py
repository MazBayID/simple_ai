"""Train the PyTorch GPT backend on the prepared Indonesian corpus."""

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

from ..tokenizer import BPETokenizer, CharTokenizer, load_tokenizer
from ..torch_backend.checkpoint import load_checkpoint, save_checkpoint
from ..torch_backend.dataset import DocumentTokenDataset, load_documents
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


def make_tokenizer(
    kind: str,
    train_documents: list[str],
    vocab_size: int,
):
    if kind == "char":
        return CharTokenizer.from_text("\n".join(train_documents))
    # The project's educational BPE trainer is CPU-intensive on large corpora.
    # Start with the character tokenizer for the first cloud smoke test.
    return BPETokenizer.train(
        "\n".join(train_documents), vocab_size=vocab_size
    )


def evaluate(
    model: GPTLanguageModel,
    dataset: DocumentTokenDataset,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> float | None:
    if len(dataset) == 0:
        return None

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
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
        description="Train Simple AI's PyTorch decoder-only Transformer."
    )
    parser.add_argument("--train", default=None, help="Training .jsonl or .txt corpus.")
    parser.add_argument("--validation", default=None, help="Optional validation corpus.")
    parser.add_argument("--test", default=None, help="Optional untouched test corpus.")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=2048,
                        help="BPE vocabulary size; ignored for char tokenizer.")
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=100,
                        help="Maximum validation/test batches; 0 means all.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optional limit for this invocation; 0 means no limit.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/torch_indonesian")
    parser.add_argument("--resume", default=None, help="Path to checkpoint_latest.pt.")
    args = parser.parse_args()

    positive = {
        "context-size": args.context_size,
        "embedding-dim": args.embedding_dim,
        "num-heads": args.num_heads,
        "num-layers": args.num_layers,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "checkpoint-steps": args.checkpoint_steps,
    }
    if any(value < 1 for value in positive.values()):
        parser.error("Context, model dimensions, epochs, batch size, and checkpoint steps must be positive.")
    if args.embedding_dim % args.num_heads:
        parser.error("--embedding-dim must be divisible by --num-heads.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay non-negative.")
    if args.gradient_clip <= 0 or args.max_steps < 0 or args.num_workers < 0:
        parser.error("Gradient clip must be positive; max steps and workers cannot be negative.")
    if args.eval_batches < 0:
        parser.error("--eval-batches cannot be negative.")

    train_path = Path(args.train) if args.train else Path(
        "data/processed/indonesian_wikipedia/train.jsonl"
    )
    if args.train is None and not train_path.exists():
        train_path = Path("data/processed/indonesian_wikipedia/train.txt")
    validation_path = Path(args.validation) if args.validation else None
    if args.validation is None:
        candidate = train_path.with_name("validation.jsonl")
        if not candidate.exists():
            candidate = train_path.with_name("validation.txt")
        if candidate.exists():
            validation_path = candidate
    test_path = Path(args.test) if args.test else None
    if args.test is None:
        candidate = train_path.with_name("test.jsonl")
        if not candidate.exists():
            candidate = train_path.with_name("test.txt")
        if candidate.exists():
            test_path = candidate

    if not train_path.is_file():
        raise SystemExit(f"Training corpus not found: {train_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"

    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    started = time.perf_counter()
    train_documents = load_documents(train_path)
    validation_documents = (
        load_documents(validation_path)
        if validation_path and validation_path.is_file() else []
    )
    test_documents = (
        load_documents(test_path)
        if test_path and test_path.is_file() else []
    )

    if args.resume:
        if not tokenizer_path.is_file():
            raise SystemExit(
                f"Cannot resume: tokenizer artifact is missing at {tokenizer_path}"
            )
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = make_tokenizer(
            args.tokenizer, train_documents, args.vocab_size
        )
        tokenizer.save(tokenizer_path)

    train_data = DocumentTokenDataset(
        train_documents, tokenizer, context_size=args.context_size
    )
    validation_data = DocumentTokenDataset(
        validation_documents, tokenizer, context_size=args.context_size
    )
    test_data = DocumentTokenDataset(
        test_documents, tokenizer, context_size=args.context_size
    )
    if not len(train_data):
        raise SystemExit(
            "No training windows were produced. Reduce --context-size or inspect the corpus."
        )
    if validation_documents and not len(validation_data):
        raise SystemExit(
            "Validation documents are too short for this context size."
        )

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
    )

    run_config = {
        "model": model.get_config(),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tokenizer_format": tokenizer.format_name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
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
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        saved_config = state.get("config", {})
        if saved_config.get("model") != run_config["model"]:
            raise SystemExit(
                "Checkpoint model configuration differs from this run. "
                "Use the original model dimensions and context size."
            )
        if saved_config.get("tokenizer_vocab_size") != tokenizer.vocab_size:
            raise SystemExit("Checkpoint and tokenizer vocabulary sizes differ.")
        start_epoch = state["epoch"]
        resume_step = state["step_in_epoch"]
        global_step = state["global_step"]
        best_validation_loss = state["best_validation_loss"]
        print(
            f"Resumed at epoch={start_epoch + 1}, "
            f"step_in_epoch={resume_step}, global_step={global_step}"
        )

    print(f"device: {device}")
    print(f"tokenizer: {tokenizer.format_name}")
    print(f"vocabulary_size: {tokenizer.vocab_size}")
    print(f"train_documents: {len(train_documents)}")
    print(f"train_windows: {len(train_data):,}")
    print(f"validation_windows: {len(validation_data):,}")
    print(f"test_windows: {len(test_data):,}")
    print(f"skipped_short_train_documents: {train_data.skipped_documents}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"preparation_seconds: {time.perf_counter() - started:.2f}")

    steps_this_run = 0
    stop_after_step_limit = False
    latest_path = output_dir / "checkpoint_latest.pt"
    best_path = output_dir / "checkpoint_best.pt"
    epoch_history: list[dict[str, float | int | None]] = []

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
        total_loss = 0.0
        total_tokens = 0
        processed_batches = 0

        for batch_index, (inputs, targets) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_step:
                continue

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

            token_count = targets.numel()
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
            processed_batches += 1
            global_step += 1
            steps_this_run += 1

            next_step_in_epoch = batch_index + 1
            if global_step % args.checkpoint_steps == 0:
                save_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step_in_epoch=next_step_in_epoch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    config=run_config,
                )
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f} checkpoint={latest_path.name}",
                    flush=True,
                )

            if args.max_steps and steps_this_run >= args.max_steps:
                save_checkpoint(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step_in_epoch=next_step_in_epoch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    config=run_config,
                )
                stop_after_step_limit = True
                break

        if total_tokens:
            epoch_train_loss = total_loss / total_tokens
        else:
            epoch_train_loss = None

        validation_loss = evaluate(
            model,
            validation_data,
            batch_size=args.batch_size,
            device=device,
            max_batches=args.eval_batches,
        ) if len(validation_data) else None

        epoch_history.append({
            "epoch": epoch + 1,
            "training_loss": epoch_train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"epoch_complete={epoch + 1} train_loss={epoch_train_loss} "
            f"validation_loss={validation_loss}",
            flush=True,
        )

        improved = (
            validation_loss is not None
            and (
                best_validation_loss is None
                or validation_loss < best_validation_loss
            )
        )
        if improved:
            best_validation_loss = validation_loss
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                step_in_epoch=0,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                config=run_config,
            )

        if stop_after_step_limit:
            # Resume this same epoch at the next unprocessed batch.
            pass
        else:
            start_epoch = epoch + 1
            resume_step = 0

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch if stop_after_step_limit else epoch + 1,
            step_in_epoch=(
                min(resume_step + processed_batches, len(loader))
                if stop_after_step_limit else 0
            ),
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            config=run_config,
        )

        if stop_after_step_limit:
            break

    final_validation_loss = evaluate(
        model,
        validation_data,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.eval_batches,
    ) if len(validation_data) else None
    final_test_loss = evaluate(
        model,
        test_data,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.eval_batches,
    ) if len(test_data) else None

    report = {
        "device": str(device),
        "tokenizer": tokenizer.format_name,
        "vocabulary_size": tokenizer.vocab_size,
        "model_config": model.get_config(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "global_step": global_step,
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
        "epoch_history_this_run": epoch_history,
        "checkpoint_latest": str(latest_path),
        "checkpoint_best": str(best_path) if best_path.exists() else None,
        "corpus_license_note": (
            "The prepared corpus is CC BY-SA 4.0. Review attribution, "
            "share-alike, and model-weight implications before redistribution."
        ),
    }
    write_json(output_dir / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
