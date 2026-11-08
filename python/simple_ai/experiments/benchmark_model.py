"""Measure local checkpoint loading and autoregressive generation."""
from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from ..language_model import load_language_model
from ..core import backend_name
from ..tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local Simple AI checkpoint.")
    parser.add_argument("--model", default="artifacts/indonesian_model/model.json")
    parser.add_argument("--tokenizer", default="artifacts/indonesian_model/tokenizer.json")
    parser.add_argument("--prompt", default="Halo, siapa kamu?")
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.generation_tokens < 1:
        parser.error("--generation-tokens must be positive.")
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    tracemalloc.start()
    load_started = time.perf_counter()
    model = load_language_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    load_seconds = time.perf_counter() - load_started
    if model.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Model and tokenizer vocabulary sizes do not match.")

    prompt_ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    generation_started = time.perf_counter()
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.generation_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
        seed=args.seed,
    )
    generation_seconds = time.perf_counter() - generation_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rate = len(generated) / generation_seconds if generation_seconds > 0 else 0.0
    print(f"architecture: {model.__class__.__name__}")
    print(f"compute_backend: {backend_name()}")
    print(f"parameters: {model.parameter_count()}")
    print(f"vocabulary_size: {tokenizer.vocab_size}")
    print(f"context_size: {model.context_size}")
    print(f"checkpoint_bytes: {model_path.stat().st_size}")
    print(f"load_seconds: {load_seconds:.6f}")
    print(f"generated_tokens: {len(generated)}")
    print(f"generation_seconds: {generation_seconds:.6f}")
    print(f"tokens_per_second: {rate:.3f}")
    print(f"python_traced_peak_mib: {peak_bytes / (1024 * 1024):.3f}")
    print("Note: tracemalloc measures Python allocations, not total process RSS.")


if __name__ == "__main__":
    main()
