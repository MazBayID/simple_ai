import json

import pytest

from simple_ai.pipeline import prepare_corpus


def test_prepare_corpus_writes_tokenizer_tokens_and_manifest(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("Halo dunia. Ini data uji.", encoding="utf-8")
    output = tmp_path / "prepared"

    stats = prepare_corpus(source, output, context_size=4)

    assert stats["characters"] == len("Halo dunia. Ini data uji.")
    assert stats["training_examples"] > 0
    assert (output / "tokenizer.json").is_file()
    assert (output / "token_ids.json").is_file()
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["context_size"] == 4
    assert manifest["source"] == "corpus.txt"


def test_prepare_corpus_rejects_empty_file(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prepare_corpus(source, tmp_path / "out", context_size=2)


def test_prepare_corpus_rejects_whitespace_only_file(tmp_path):
    source = tmp_path / "spaces.txt"
    source.write_text("  \n\t", encoding="utf-8")
    with pytest.raises(ValueError, match="whitespace"):
        prepare_corpus(source, tmp_path / "out", context_size=2)


def test_prepare_corpus_rejects_too_short_corpus(tmp_path):
    source = tmp_path / "short.txt"
    source.write_text("abc", encoding="utf-8")
    with pytest.raises(ValueError, match="too short"):
        prepare_corpus(source, tmp_path / "out", context_size=8)
