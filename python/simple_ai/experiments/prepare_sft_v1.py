"""Normalize approved instruction datasets into deterministic SFT JSONL splits.

Input files are JSONL records with instruction/output fields. Optional metadata
(source, category, original_index) is preserved. Exact duplicate pairs are
removed, and all records sharing a normalized instruction stay in one split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def read_jsonl(paths: list[Path], source_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc
                if not isinstance(item, dict):
                    continue
                instruction = str(item.get("instruction") or "").strip()
                output = str(item.get("output") or "").strip()
                if not instruction or not output:
                    continue
                context = str(item.get("input") or "").strip()
                if context:
                    instruction = f"{instruction}\n\nKonteks:\n{context}"
                records.append({
                    "instruction": instruction,
                    "output": output,
                    "source": str(item.get("source") or source_name),
                    "category": str(item.get("category") or item.get("task_type") or ""),
                    "original_index": item.get("original_index", item.get("id", "")),
                })
    return records


def split_for_instruction(instruction: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{normalize(instruction)}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Simple AI SFT JSONL splits.")
    parser.add_argument("--garudacoder", nargs="*", default=[])
    parser.add_argument("--aksara", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    sources: list[dict[str, Any]] = []
    if args.garudacoder:
        sources.extend(read_jsonl([Path(p) for p in args.garudacoder], "garudacoder-id-coding"))
    if args.aksara:
        sources.extend(read_jsonl([Path(p) for p in args.aksara], "aksara-sft-clean-v1"))
    if not sources:
        parser.error("Provide at least one --garudacoder or --aksara JSONL file.")

    seen_pairs: set[tuple[str, str]] = set()
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    dropped_duplicates = 0
    source_counts: Counter[str] = Counter()

    for item in sources:
        pair = (normalize(item["instruction"]), normalize(item["output"]))
        if pair in seen_pairs:
            dropped_duplicates += 1
            continue
        seen_pairs.add(pair)
        split = split_for_instruction(item["instruction"], args.seed)
        item["split"] = split
        splits[split].append(item)
        source_counts[item["source"]] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        destination = output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "format": "simple-ai-sft-jsonl-v1",
        "seed": args.seed,
        "split_policy": "stable SHA-256 of normalized instruction; 90/5/5 buckets",
        "records_input": len(sources),
        "exact_duplicate_pairs_removed": dropped_duplicates,
        "records_written": {name: len(items) for name, items in splits.items()},
        "records_by_source": dict(source_counts),
        "files": {name: f"{name}.jsonl" for name in splits},
        "note": "SFT candidate data only; source licensing and quality decisions remain the dataset owner's responsibility.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SIMPLE AI SFT DATASET")
    print("=" * 40)
    print("Input:", len(sources))
    print("Exact duplicate pairs removed:", dropped_duplicates)
    for name, items in splits.items():
        print(f"{name}: {len(items)}")
    print("By source:", dict(source_counts))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
