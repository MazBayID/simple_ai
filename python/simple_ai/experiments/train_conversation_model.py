"""Train Simple AI on Indonesian text or JSONL conversation pairs.

Examples:
  PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
  PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --conversations data/conversations.example.jsonl
  PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --input data/indonesian_seed.txt --context-size 48 --epochs 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..conversation import load_conversations, render_conversations
from ..dataset import NextTokenDataset
from ..language_model import TinyLanguageModel, train_language_model
from ..losses import CrossEntropyLoss
from ..optim import SGD
from ..tokenizer import CharTokenizer


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
    source.add_argument("--input", default="data/indonesian_seed.txt", help="UTF-8 Indonesian plain-text corpus.")
    source.add_argument("--conversations", help="JSONL file with one {user, assistant} object per line.")
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/indonesian_model")
    args = parser.parse_args()

    if args.context_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("context size, epochs, and batch size must be positive.")
    if not 0.0 <= args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in [0, 0.5).")

    if args.conversations:
        records = load_conversations(args.conversations)
        text = render_conversations(records)
        source_description = f"{len(records)} conversation pairs"
    else:
        text = Path(args.input).read_text(encoding="utf-8")
        source_description = args.input
    if not text.strip():
        raise SystemExit("Training corpus is empty or whitespace-only.")

    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= args.context_size + 1:
        raise SystemExit("Corpus is too short for the selected context size.")

    split = int(len(token_ids) * (1.0 - args.validation_fraction))
    split = max(args.context_size + 1, min(split, len(token_ids)))
    train_ids = token_ids[:split]
    validation_ids = token_ids[max(0, split - args.context_size):]
    train_data = NextTokenDataset(train_ids, context_size=args.context_size)
    validation_data = NextTokenDataset(validation_ids, context_size=args.context_size) if args.validation_fraction else None

    model = TinyLanguageModel(
        tokenizer.vocab_size, args.context_size,
        embedding_dim=args.embedding_dim, hidden_size=args.hidden_size, seed=args.seed,
    )
    before_validation = evaluate(model, validation_data, args.batch_size) if validation_data else None
    optimizer = SGD(model.parameters(), learning_rate=args.learning_rate)
    history = train_language_model(
        model, train_data, optimizer, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )
    after_validation = evaluate(model, validation_data, args.batch_size) if validation_data else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.json")
    tokenizer.save(output_dir / "tokenizer.json")
    report = {
        "source": source_description,
        "characters": len(text),
        "vocabulary_size": tokenizer.vocab_size,
        "context_size": args.context_size,
        "training_examples": len(train_data),
        "validation_examples": len(validation_data) if validation_data else 0,
        "epochs": args.epochs,
        "initial_training_loss": history[0],
        "final_training_loss": history[-1],
        "initial_validation_loss": before_validation,
        "final_validation_loss": after_validation,
        "parameters": model.parameter_count(),
        "warning": "A lower loss is not proof of semantic understanding or useful chat ability.",
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
