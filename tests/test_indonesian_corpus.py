import json

from simple_ai.experiments.prepare_indonesian_corpus import (
    LICENSE, load_eligible_records, normalize_text, split_name,
)


def test_normalize_text_preserves_words_and_cleans_spacing():
    assert normalize_text("  Bahasa  Indonesia\u00a0\n\n\n bagus!  ") == "Bahasa Indonesia\n\nbagus!"


def test_loader_requires_reviewed_cc_by_sa_wikipedia_provenance(tmp_path):
    source = tmp_path / "source.jsonl"
    text = "Bahasa Indonesia adalah bahasa resmi Republik Indonesia. " * 3
    rows = [
        {"license": LICENSE, "rights_reviewed": True, "source_url": "https://id.wikipedia.org/wiki/Bahasa_Indonesia", "source_revision_id": 123, "title": "Bahasa Indonesia", "text": text},
        {"license": LICENSE, "rights_reviewed": False, "source_url": "https://id.wikipedia.org/wiki/Invalid", "source_revision_id": 124, "title": "Invalid", "text": text + " extra"},
        {"license": "unknown", "rights_reviewed": True, "source_url": "https://id.wikipedia.org/wiki/Unknown", "source_revision_id": 125, "title": "Unknown", "text": text + " other"},
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    records, rejected, duplicates = load_eligible_records(source)
    assert len(records) == 1
    assert rejected == 2
    assert duplicates == 0
    assert records[0]["source_revision_id"] == 123


def test_split_is_deterministic():
    record = {"source_url": "https://id.wikipedia.org/wiki/Komputer", "source_revision_id": 987654}
    assert split_name(record) == split_name(record)
    assert split_name(record) in {"train", "validation", "test"}
