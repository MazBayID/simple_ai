"""Prepare local UTF-8 text for next-token language-model experiments.

Run:
    PYTHONPATH=python python -m simple_ai.pipeline --input sample.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_ai.dataset import NextTokenDataset
from simple_ai.tokenizer import CharTokenizer


def prepare_corpus(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    context_size: int = 16,
) -> dict[str, int | str]:
    source = Path(input_path)
    text = source.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"Input corpus is empty: {source}")
    if not text.strip():
        raise ValueError(f"Input corpus contains only whitespace: {source}")

    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(token_ids, context_size=context_size)
    if len(dataset) == 0:
        raise ValueError(
            f"Corpus is too short for context_size={context_size}; "
            f"need at least {context_size + 1} encoded tokens."
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer.save(destination / "tokenizer.json")
    (destination / "token_ids.json").write_text(
        json.dumps(token_ids, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    stats: dict[str, int | str] = {
        "source": source.name,
        "characters": len(text),
        "token_count": len(token_ids),
        "vocabulary_size": tokenizer.vocab_size,
        "context_size": context_size,
        "training_examples": len(dataset),
    }
    (destination / "dataset.json").write_text(
        json.dumps(
            {
                "format": "simple-ai-next-token-dataset",
                "version": 1,
                **stats,
                "tokenizer_file": "tokenizer.json",
                "token_ids_file": "token_ids.json",
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 text corpus.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/text_pipeline"),
        help="Directory for tokenizer and prepared token data.",
    )
    parser.add_argument("--context-size", type=int, default=16)
    args = parser.parse_args()

    stats = prepare_corpus(
        args.input,
        args.output_dir,
        context_size=args.context_size,
    )
    print("Text pipeline preparation complete.")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
