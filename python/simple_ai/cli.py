"""Command line interface for the first Simple AI prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import TinyCharModel

DEFAULT_TEXT = """Simple AI is an offline artificial intelligence project.
We learn language patterns from text.
This tiny prototype is a first step toward a model built from scratch.
Python is readable and useful for experiments.
The future model should be efficient, understandable, and modifiable.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simple-ai",
        description="Train and run the first tiny offline character model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from a text file.")
    train_parser.add_argument("--data", type=Path, help="UTF-8 training text.")
    train_parser.add_argument(
        "--output", type=Path, default=Path("tiny_model.json"),
        help="Checkpoint output path.",
    )

    chat_parser = subparsers.add_parser("generate", help="Generate text.")
    chat_parser.add_argument("--model", type=Path, default=Path("tiny_model.json"))
    chat_parser.add_argument("--prompt", default="")
    chat_parser.add_argument("--tokens", type=int, default=120)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--seed", type=int)

    demo_parser = subparsers.add_parser("demo", help="Train a tiny built-in demo.")
    demo_parser.add_argument("--prompt", default="Simple AI")
    demo_parser.add_argument("--tokens", type=int, default=120)
    demo_parser.add_argument("--seed", type=int, default=7)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        text = args.data.read_text(encoding="utf-8") if args.data else DEFAULT_TEXT
        model = TinyCharModel()
        stats = model.train(text)
        model.save(args.output)
        print("Training complete.")
        print(f"Stats: {stats}")
        print(f"Checkpoint: {args.output}")
        return

    if args.command == "generate":
        model = TinyCharModel.load(args.model)
        print(model.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            seed=args.seed,
        ))
        return

    if args.command == "demo":
        model = TinyCharModel()
        stats = model.train(DEFAULT_TEXT)
        print(f"Demo training stats: {stats}")
        print(model.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            seed=args.seed,
        ))


if __name__ == "__main__":
    main()
