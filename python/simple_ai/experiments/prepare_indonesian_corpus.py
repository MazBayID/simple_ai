"""Prepare a provenance-tracked Indonesian Wikipedia text corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

DATASET_URL = "https://huggingface.co/datasets/apricitea/bahasa-indonesia-corpus/resolve/main/data/bahasa_wikipedia_ccbysa.jsonl"
DATASET_CARD = "https://huggingface.co/datasets/apricitea/bahasa-indonesia-corpus"
LICENSE = "cc-by-sa-4.0"


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = "".join(ch for ch in value if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    value = re.sub(r"[ \t\u00a0]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def split_name(record: dict) -> str:
    key = f"{record['source_url']}#{record['source_revision_id']}"
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def download_dataset(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "SimpleAI-corpus-preparer/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        destination.write_bytes(response.read())


def load_eligible_records(source: Path) -> tuple[list[dict], int, int]:
    records, seen = [], set()
    rejected = duplicates = 0
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                source_url = str(raw["source_url"])
                revision_id = int(raw["source_revision_id"])
                text = normalize_text(str(raw["text"]))
                title = normalize_text(str(raw.get("title", "")))
                parsed = urlparse(source_url)
                if (raw.get("license") != LICENSE or raw.get("rights_reviewed") is not True
                    or parsed.scheme != "https" or parsed.hostname != "id.wikipedia.org"
                    or revision_id <= 0 or len(text) < 80):
                    rejected += 1
                    continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                rejected += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            records.append({
                "title": title, "text": text, "source_url": source_url,
                "source_revision_id": revision_id, "retrieved_at": raw.get("retrieved_at"),
                "text_sha256": digest,
            })
    return records, rejected, duplicates


def write_corpus(source: Path, output_dir: Path, minimum_training_characters: int) -> dict:
    records, rejected, duplicates = load_eligible_records(source)
    splits = {name: [] for name in ("train", "validation", "test")}
    for record in records:
        splits[split_name(record)].append(record)
    rendered = {
        name: "\n\n".join(
            f"{item['title']}\n{item['text']}" if item["title"] else item["text"]
            for item in items
        ).strip() + "\n"
        for name, items in splits.items()
    }
    counts = {name: len(value.strip()) for name, value in rendered.items()}
    if counts["train"] < minimum_training_characters:
        raise SystemExit(
            f"Only {counts['train']:,} training characters; "
            f"minimum is {minimum_training_characters:,}. No output written."
        )
    if not splits["validation"] or not splits["test"]:
        raise SystemExit("Validation or test split is empty; no output written.")

    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for name, value in rendered.items():
        path = output_dir / f"{name}.txt"
        path.write_text(value, encoding="utf-8")
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    # Preserve article boundaries for document-aware language-model training.
    for name, items in splits.items():
        path = output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(
                    {"title": item["title"], "text": item["text"]},
                    ensure_ascii=False,
                ) + "\\n")
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = [{
        "title": item["title"], "source_url": item["source_url"],
        "source_revision_id": item["source_revision_id"],
        "retrieved_at": item["retrieved_at"], "text_sha256": item["text_sha256"],
        "license": LICENSE,
    } for item in records]
    manifest = {
        "dataset": "Bahasa Indonesia Wikipedia extracts (rights-cleared subset)",
        "dataset_url": DATASET_CARD, "source_file": str(source),
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "license": "CC BY-SA 4.0",
        "attribution": "Indonesian Wikipedia contributors; Nataputra, A. (2026), Bahasa Indonesia Wikipedia extracts (rights-cleared subset), Hugging Face.",
        "split_method": "Stable SHA-256 of source URL + revision ID; 90/5/5 document buckets.",
        "minimum_training_characters": minimum_training_characters,
        "source_records_kept": len(records), "records_rejected": rejected,
        "duplicate_texts_removed": duplicates,
        "records_per_split": {name: len(items) for name, items in splits.items()},
        "characters_per_split": counts, "file_sha256": file_hashes,
        "provenance": provenance,
        "warning": "CC BY-SA 4.0 obligations apply. Review attribution, share-alike, and model-weight implications before redistribution or commercial use.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "ATTRIBUTION.md").write_text(
        "# Corpus attribution\n\n"
        "Normalized introductory extracts from Indonesian Wikipedia.\n\n"
        f"Dataset: {DATASET_CARD}\nLicense: CC BY-SA 4.0\n\n"
        "Attribute Indonesian Wikipedia contributors and Nataputra, A. (2026), "
        "Bahasa Indonesia Wikipedia extracts (rights-cleared subset). "
        "Per-article URLs, revision IDs, retrieval timestamps, and hashes are in manifest.json.\n\n"
        "Review attribution and share-alike obligations before redistributing the corpus "
        "or trained artifacts. This is not legal advice.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a licensed Indonesian Wikipedia corpus.")
    parser.add_argument("--source", default="data/local/bahasa_wikipedia_ccbysa.jsonl")
    parser.add_argument("--output-dir", default="data/processed/indonesian_wikipedia")
    parser.add_argument("--minimum-training-characters", type=int, default=1_000_000)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.minimum_training_characters < 1:
        parser.error("--minimum-training-characters must be positive.")
    source = Path(args.source)
    if args.force_download or not source.exists():
        print(f"Downloading: {DATASET_URL}")
        download_dataset(source)
    started = time.perf_counter()
    manifest = write_corpus(source, Path(args.output_dir), args.minimum_training_characters)
    print(f"records kept: {manifest['source_records_kept']}")
    print(f"records rejected: {manifest['records_rejected']}")
    print(f"duplicate texts removed: {manifest['duplicate_texts_removed']}")
    for name in ("train", "validation", "test"):
        print(f"{name}: {manifest['records_per_split'][name]} documents, {manifest['characters_per_split'][name]:,} characters")
    print(f"license: {manifest['license']}")
    print(f"output: {args.output_dir}")
    print(f"elapsed_seconds: {time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
