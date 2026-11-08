"""Run reproducible prompt-based generation diagnostics on a local checkpoint.

This tool measures generation behavior and speed. It does not assign a semantic
quality score; generated samples still require human review.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..core import backend_name
from ..language_model import load_language_model
from ..tokenizer import load_tokenizer


DEFAULT_PROMPTS = (
    "Jelaskan apa itu kecerdasan buatan.",
    "Mengapa air penting bagi kehidupan?",
    "Apa perbedaan Python dan C++?",
    "Bagaimana cara kerja sebuah komputer?",
    "Buat ringkasan tentang bahasa Indonesia.",
)


def evaluate_prompts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> dict:
    """Generate deterministic samples and collect basic diagnostic metrics."""
    samples = []
    total_tokens = 0
    total_seconds = 0.0

    for index, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        started = time.perf_counter()
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            seed=seed + index,
        )
        elapsed = time.perf_counter() - started
        decoded = tokenizer.decode(generated).strip()
        unique_ratio = len(set(generated)) / len(generated) if generated else 0.0
        samples.append({
            "prompt": prompt,
            "generated_text": decoded,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "generation_seconds": elapsed,
            "tokens_per_second": len(generated) / elapsed if elapsed > 0 else 0.0,
            "unique_token_ratio": unique_ratio,
            "eos_reached": tokenizer.eos_id in generated,
        })
        total_tokens += len(generated)
        total_seconds += elapsed

    return {
        "architecture": model.__class__.__name__,
        "compute_backend": backend_name(),
        "parameters": model.parameter_count(),
        "vocabulary_size": tokenizer.vocab_size,
        "context_size": model.context_size,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "prompt_count": len(prompts),
        "generated_tokens": total_tokens,
        "generation_seconds": total_seconds,
        "tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else 0.0,
        "samples": samples,
        "limitations": (
            "Speed and token-diversity diagnostics are not semantic quality scores. "
            "Review samples for relevance, fluency, factuality, and repetition."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt generation from a local Simple AI checkpoint."
    )
    parser.add_argument("--model", default="artifacts/indonesian_model/model.json")
    parser.add_argument("--tokenizer", default="artifacts/indonesian_model/tokenizer.json")
    parser.add_argument(
        "--prompt", action="append", dest="prompts",
        help="Prompt to evaluate; repeat this option for multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        help="Optional UTF-8 file containing one prompt per non-empty line.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", default="artifacts/evaluation/generation_report.json"
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if args.temperature <= 0:
        parser.error("--temperature must be positive.")
    if args.top_k < 1:
        parser.error("--top-k must be positive.")

    prompts = list(args.prompts or [])
    if args.prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    model = load_language_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    if model.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Model and tokenizer vocabulary sizes do not match.")

    report = evaluate_prompts(
        model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    report["model_path"] = str(model_path)
    report["tokenizer_path"] = str(tokenizer_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for key in (
        "architecture", "compute_backend", "parameters", "prompt_count",
        "generated_tokens", "generation_seconds", "tokens_per_second",
    ):
        print(f"{key}: {report[key]}")
    for index, sample in enumerate(report["samples"], 1):
        print(f"\n[{index}] Prompt: {sample['prompt']}")
        print(f"Generated: {sample['generated_text'] or '[empty]'}")
        print(f"tokens_per_second: {sample['tokens_per_second']:.3f}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
