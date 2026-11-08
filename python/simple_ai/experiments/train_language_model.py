"""Train and sample a tiny character-level language model.

Run:
    PYTHONPATH=python python -m simple_ai.experiments.train_language_model \
        --input sample.txt --context-size 8 --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..dataset import NextTokenDataset
from ..language_model import TinyLanguageModel, train_language_model
from ..optim import SGD
from ..tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Simple AI's tiny language model.")
    parser.add_argument("--input", default="sample.txt")
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/language_model")
    args = parser.parse_args()

    source = Path(args.input)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit("Input corpus is empty or whitespace-only.")

    tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)
    dataset = NextTokenDataset(token_ids, context_size=args.context_size)
    if not len(dataset):
        raise SystemExit("Corpus is too short for the selected context size.")

    model = TinyLanguageModel(
        tokenizer.vocab_size,
        args.context_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        seed=args.seed,
    )
    optimizer = SGD(model.parameters(), learning_rate=args.learning_rate)
    history = train_language_model(
        model,
        dataset,
        optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.json")
    tokenizer.save(output_dir / "tokenizer.json")

    prompt = text[:args.context_size]
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=5,
        eos_id=tokenizer.eos_id,
        seed=args.seed,
    )
    print(f"Corpus characters: {len(text)}")
    print(f"Vocabulary size:   {tokenizer.vocab_size}")
    print(f"Training examples: {len(dataset)}")
    print(f"Parameters:        {model.parameter_count()}")
    print(f"Initial loss:      {history[0]:.6f}")
    print(f"Final loss:        {history[-1]:.6f}")
    print(f"Checkpoint:        {output_dir / 'model.json'}")
    print(f"Tokenizer:         {output_dir / 'tokenizer.json'}")
    print(f"Prompt:            {prompt!r}")
    print(f"Generated sample:  {tokenizer.decode(generated_ids)!r}")
    print("Note: this is a tiny corpus memorization experiment, not a useful chatbot yet.")


if __name__ == "__main__":
    main()
