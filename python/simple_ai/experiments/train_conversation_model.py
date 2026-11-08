"""Train Simple AI on Indonesian text or JSONL conversation pairs."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from ..conversation import load_conversations, render_conversations
from ..dataset import NextTokenDataset
from ..language_model import TinyLanguageModel, train_language_model
from ..transformer import TinyTransformerLanguageModel
from ..losses import CrossEntropyLoss
from ..optim import Adam, SGD
from ..tokenizer import BPETokenizer, CharTokenizer


def evaluate(model: TinyLanguageModel, dataset: NextTokenDataset, batch_size: int) -> float | None:
    if not len(dataset):
        return None
    criterion = CrossEntropyLoss()
    total = 0.0
    seen = 0
    for inputs, targets in dataset.batches(batch_size):
        value = criterion(model(inputs), targets).item()
        total += value * len(targets)
        seen += len(targets)
    return total / seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Indonesian Simple AI language-model baseline.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", default="data/indonesian_seed.txt")
    source.add_argument("--conversations")
    parser.add_argument("--validation-input", help="Optional independent UTF-8 validation corpus.")
    parser.add_argument("--tokenizer", choices=("bpe", "char"), default="bpe")
    parser.add_argument("--architecture", choices=("mlp", "transformer"), default="mlp")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/indonesian_model")
    args = parser.parse_args()

    if args.context_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("context size, epochs, and batch size must be positive.")
    if not 0.0 <= args.validation_fraction <= 0.5:
        parser.error("--validation-fraction must be in [0, 0.5].")

    validation_source = args.validation_input or "contiguous_holdout"
    if args.conversations:
        records = load_conversations(args.conversations)
        source_description = f"{len(records)} conversation pairs"
        if args.validation_input:
            train_text = render_conversations(records)
            validation_text = Path(args.validation_input).read_text(encoding="utf-8")
            validation_source = args.validation_input
        else:
            split_at = max(1, min(
                int(len(records) * (1.0 - args.validation_fraction)), len(records)
            ))
            train_text = render_conversations(records[:split_at])
            validation_records = records[split_at:]
            validation_text = render_conversations(validation_records) if validation_records else ""
            validation_source = "held_out_conversations"
        validation_ids_are_independent = True
        text = train_text
    else:
        text = Path(args.input).read_text(encoding="utf-8")
        source_description = args.input
        if args.validation_input:
            train_text = text
            validation_text = Path(args.validation_input).read_text(encoding="utf-8")
            validation_source = args.validation_input
            validation_ids_are_independent = True
        else:
            split_at = int(len(text) * (1.0 - args.validation_fraction))
            split_at = max(1, min(split_at, len(text)))
            train_text = text[:split_at]
            validation_text = text[split_at:]
            validation_ids_are_independent = False
    if not train_text.strip():
        raise SystemExit("Training corpus is empty or whitespace-only.")
    if not validation_text and not args.validation_input:
        validation_source = "none"
    if args.validation_input and not validation_text.strip():
        raise SystemExit("Validation corpus is empty or whitespace-only.")
    tokenization_started = time.perf_counter()
    if args.tokenizer == "bpe":
        tokenizer = BPETokenizer.train(train_text, vocab_size=args.vocab_size)
    else:
        tokenizer = CharTokenizer.from_text(train_text)

    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=False)
    tokenization_seconds = time.perf_counter() - tokenization_started
    if validation_ids_are_independent:
        validation_ids = tokenizer.encode(validation_text, add_bos=True, add_eos=True)
    else:
        validation_body = tokenizer.encode(validation_text, add_bos=False, add_eos=True)
        validation_ids = train_ids[-args.context_size:] + validation_body
    if len(train_ids) <= args.context_size:
        raise SystemExit("Training corpus is too short for the selected context size.")
    train_data = NextTokenDataset(train_ids, context_size=args.context_size)
    validation_enabled = bool(args.validation_input) or args.validation_fraction > 0
    validation_data = (NextTokenDataset(validation_ids, context_size=args.context_size)
                       if validation_enabled and validation_text else None)

    if args.architecture == "transformer":
        model = TinyTransformerLanguageModel(
            tokenizer.vocab_size, args.context_size,
            embedding_dim=max(2, args.embedding_dim), seed=args.seed,
        )
    else:
        model = TinyLanguageModel(
            tokenizer.vocab_size, args.context_size,
            embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, seed=args.seed,
        )
    before_validation = evaluate(model, validation_data, args.batch_size) if validation_data else None
    optimizer_name = args.optimizer or ("adam" if args.architecture == "transformer" else "sgd")
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.001 if optimizer_name == "adam" else 0.03
    if learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    optimizer = (
        Adam(model.parameters(), learning_rate=learning_rate)
        if optimizer_name == "adam"
        else SGD(model.parameters(), learning_rate=learning_rate)
    )
    training_started = time.perf_counter()
    history = train_language_model(
        model, train_data, optimizer, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_started
    after_validation = evaluate(model, validation_data, args.batch_size) if validation_data else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.json")
    tokenizer.save(output_dir / "tokenizer.json")
    def perplexity(loss: float | None) -> float | None:
        if loss is None:
            return None
        return math.exp(min(loss, 700.0))

    report = {
        "source": source_description, "validation_source": validation_source,
        "training_characters": len(train_text), "validation_characters": len(validation_text),
        "tokenization_tokens_per_character": len(train_ids) / max(1, len(train_text)),
        "characters": len(train_text) + len(validation_text),
        "tokenizer": args.tokenizer, "architecture": args.architecture,
        "optimizer": optimizer_name, "learning_rate": learning_rate,
        "vocabulary_size": tokenizer.vocab_size,
        "learned_merges": len(tokenizer.merges) if isinstance(tokenizer, BPETokenizer) else 0,
        "context_size": args.context_size, "training_tokens": len(train_ids),
        "validation_tokens": len(validation_ids) if validation_data else 0,
        "training_examples": len(train_data),
        "validation_examples": len(validation_data) if validation_data else 0,
        "epochs": args.epochs, "tokenization_seconds": tokenization_seconds,
        "training_seconds": training_seconds,
        "training_examples_per_second": (
            len(train_data) * args.epochs / training_seconds if training_seconds > 0 else None
        ),
        "initial_training_loss": history[0],
        "final_training_loss": history[-1], "initial_validation_loss": before_validation,
        "final_validation_loss": after_validation,
        "initial_validation_perplexity": perplexity(before_validation),
        "final_validation_perplexity": perplexity(after_validation),
        "parameters": model.parameter_count(),
        "warning": "Lower loss is not proof of semantic understanding or useful chat ability.",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"checkpoint: {output_dir / 'model.json'}")
    print(f"tokenizer: {output_dir / 'tokenizer.json'}")
    print(f"report: {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
