"""Interactive terminal chat for a trained experimental language model."""
from __future__ import annotations

import argparse
from pathlib import Path

from .conversation import build_chat_prompt
from .language_model import load_language_model
from .tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a local Simple AI checkpoint.")
    parser.add_argument("--model", default="artifacts/indonesian_model/model.json")
    parser.add_argument("--tokenizer", default="artifacts/indonesian_model/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--history-turns", type=int, default=2)
    args = parser.parse_args()

    if args.history_turns < 0:
        parser.error("--history-turns must not be negative.")
    model = load_language_model(Path(args.model))
    tokenizer = load_tokenizer(Path(args.tokenizer))
    if model.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Model and tokenizer vocabulary sizes do not match.")

    print("Simple AI experimental chat — ketik /exit untuk keluar.")
    print("Catatan: model ini masih eksperimental dan belum menjamin jawaban benar.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            message = input("\nKamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa.")
            break
        if message.lower() in {"/exit", "/quit"}:
            print("Sampai jumpa.")
            break
        if not message:
            continue

        prompt = build_chat_prompt(history[-args.history_turns:] if args.history_turns else [], message)
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        generated = model.generate(
            prompt_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, eos_id=tokenizer.eos_id, seed=args.seed + len(history),
        )
        answer = tokenizer.decode(generated).split("\n\n", 1)[0].strip()
        print(f"Simple AI: {answer or '[tidak menghasilkan teks]'}")
        history.append((message, answer))


if __name__ == "__main__":
    main()
