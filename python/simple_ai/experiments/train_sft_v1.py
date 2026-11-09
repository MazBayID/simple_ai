"""Supervised fine-tuning for a pretrained Simple AI v2 GPT checkpoint.

The prompt tokens are masked from the loss; only assistant-answer tokens
(including EOS) contribute to the SFT objective.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..byte_bpe import ByteBPETokenizer
from ..torch_backend.checkpoint import load_checkpoint, save_checkpoint
from ..torch_backend.model import GPTConfig, GPTLanguageModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InstructionDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: ByteBPETokenizer,
        *,
        context_size: int,
    ) -> None:
        self.path = Path(path)
        self.examples: list[tuple[list[int], list[int]]] = []
        self.skipped = 0
        max_total = context_size + 1

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc

                instruction = str(record.get("instruction") or "").strip()
                output = str(record.get("output") or "").strip()
                if not instruction or not output:
                    self.skipped += 1
                    continue

                prompt = f"### Instruksi:\n{instruction}\n\n### Jawaban:\n"
                prefix_ids = [tokenizer.bos_id] + tokenizer.encode(
                    prompt, add_bos=False, add_eos=False
                )
                answer_ids = tokenizer.encode(
                    output, add_bos=False, add_eos=False
                ) + [tokenizer.eos_id]

                # Preserve an EOS target and as much of the answer as possible.
                if len(answer_ids) > max_total - 1:
                    answer_ids = answer_ids[: max_total - 2] + [tokenizer.eos_id]
                prefix_budget = max_total - len(answer_ids)
                if prefix_budget < 1:
                    raise RuntimeError("Unable to reserve prompt and answer tokens.")
                prefix_ids = prefix_ids[-prefix_budget:]

                token_ids = prefix_ids + answer_ids
                if len(token_ids) < 2:
                    self.skipped += 1
                    continue

                response_start = len(prefix_ids)
                input_ids = token_ids[:-1]
                labels = [
                    token if target_position >= response_start else -100
                    for target_position, token in enumerate(token_ids[1:], start=1)
                ]
                if all(label == -100 for label in labels):
                    self.skipped += 1
                    continue
                self.examples.append((input_ids, labels))

        if not self.examples:
            raise ValueError(f"No usable SFT examples in {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def make_collator(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]):
        max_length = max(len(item[0]) for item in batch)
        inputs, labels = [], []
        for input_ids, target_ids in batch:
            padding = max_length - len(input_ids)
            inputs.append(input_ids + [pad_id] * padding)
            labels.append(target_ids + [-100] * padding)
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
    return collate


def evaluate(
    model: GPTLanguageModel,
    dataset: InstructionDataset,
    *,
    batch_size: int,
    device: torch.device,
    pad_id: int,
) -> float:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=make_collator(pad_id), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    try:
        with torch.inference_mode():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                _, loss = model(inputs, labels)
                active = int((labels != -100).sum().item())
                if loss is not None and active:
                    loss_sum += float(loss.item()) * active
                    token_count += active
    finally:
        model.train(was_training)
    return loss_sum / max(1, token_count)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT fine-tune a Simple AI v2 checkpoint.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test")
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.checkpoint_steps) < 1:
        parser.error("epochs, batch size, and checkpoint interval must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0:
        parser.error("Learning rate and gradient clip must be positive; weight decay non-negative.")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative.")

    train_path = Path(args.train)
    validation_path = Path(args.validation)
    pretrained_path = Path(args.pretrained_checkpoint)
    tokenizer_path = Path(args.tokenizer)
    for path in (train_path, validation_path, pretrained_path, tokenizer_path):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    seed_everything(args.seed)

    # Checkpoints must come from this project or another trusted source.
    payload = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise SystemExit("Unsupported pretrained checkpoint.")
    model_values = payload["config"].get("model")
    if not isinstance(model_values, dict):
        raise SystemExit("Checkpoint does not contain a model configuration.")

    model_config = GPTConfig(**model_values)
    model = GPTLanguageModel(model_config)
    load_checkpoint(pretrained_path, model=model, map_location="cpu", restore_rng=False)
    model.to(device)

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match "
            f"checkpoint vocab ({model_config.vocab_size})."
        )

    train_data = InstructionDataset(
        train_path, tokenizer, context_size=model_config.context_size
    )
    validation_data = InstructionDataset(
        validation_path, tokenizer, context_size=model_config.context_size
    )
    test_data = (
        InstructionDataset(args.test, tokenizer, context_size=model_config.context_size)
        if args.test else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "checkpoint_latest.pt"
    best_path = output_dir / "checkpoint_best.pt"

    run_config: dict[str, Any] = {
        "task": "supervised-fine-tuning-v1",
        "model": model.get_config(),
        "pretrained_checkpoint_sha256": sha256_file(pretrained_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_file_sha256": sha256_file(train_path),
        "validation_file_sha256": sha256_file(validation_path),
        "test_file_sha256": sha256_file(Path(args.test)) if args.test else None,
        "context_size": model_config.context_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "seed": args.seed,
    }

    start_epoch = 0
    resume_step = 0
    global_step = 0
    best_validation_loss: float | None = None
    if args.resume:
        state = load_checkpoint(
            args.resume, model=model, optimizer=optimizer,
            map_location=device, restore_rng=True,
        )
        if state["config"] != run_config:
            raise SystemExit("SFT resume configuration differs from this run.")
        start_epoch = state["epoch"]
        resume_step = state["step_in_epoch"]
        global_step = state["global_step"]
        best_validation_loss = state["best_validation_loss"]
        print(
            f"Resumed epoch={start_epoch + 1}, batch={resume_step}, "
            f"global_step={global_step}"
        )

    collate = make_collator(tokenizer.pad_id)
    steps_this_run = 0
    stop_early = False
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"examples: train={len(train_data):,} validation={len(validation_data):,}"
        + (f" test={len(test_data):,}" if test_data else "")
    )
    print(f"skipped: train={train_data.skipped} validation={validation_data.skipped}")

    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            generator=generator, collate_fn=collate, num_workers=0,
            pin_memory=device.type == "cuda",
        )
        model.train()
        loss_sum = 0.0
        active_tokens = 0
        processed_batches = 0

        for batch_index, (inputs, labels) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_step:
                continue
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, labels)
            if loss is None:
                raise RuntimeError("Model returned no SFT loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            active = int((labels != -100).sum().item())
            loss_sum += float(loss.detach().item()) * active
            active_tokens += active
            global_step += 1
            steps_this_run += 1
            processed_batches += 1

            if global_step % args.checkpoint_steps == 0:
                save_checkpoint(
                    latest_path, model=model, optimizer=optimizer,
                    epoch=epoch, step_in_epoch=batch_index + 1,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    config=run_config,
                )
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

            if args.max_steps and steps_this_run >= args.max_steps:
                stop_early = True
                break

        validation_loss = evaluate(
            model, validation_data, batch_size=args.batch_size,
            device=device, pad_id=tokenizer.pad_id,
        )
        train_loss = loss_sum / max(1, active_tokens)
        history.append({
            "epoch": epoch + 1,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"epoch_complete={epoch + 1} train_loss={train_loss:.5f} "
            f"validation_loss={validation_loss:.5f}",
            flush=True,
        )

        if best_validation_loss is None or validation_loss < best_validation_loss:
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
        resume_step = 0

    report = {
        "task": "supervised-fine-tuning-v1",
        "device": str(device),
        "examples": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data) if test_data else 0,
        },
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "test_loss": (
            evaluate(
                model, test_data, batch_size=args.batch_size,
                device=device, pad_id=tokenizer.pad_id,
            ) if test_data else None
        ),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_latest": str(latest_path),
        "checkpoint_best": str(best_path),
        "note": "SFT loss is computed only on answer tokens. It does not prove factual accuracy or instruction-following quality.",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
