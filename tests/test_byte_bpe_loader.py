import pytest

pytest.importorskip("tokenizers")

from simple_ai.byte_bpe import ByteBPETokenizer
from simple_ai.tokenizer import load_tokenizer


def test_loader_recognizes_byte_bpe(tmp_path):
    tokenizer = ByteBPETokenizer.train(
        ["Indonesia adalah negara kepulauan.", "Bahasa Indonesia."],
        vocab_size=512,
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    restored = load_tokenizer(path)
    assert restored.format_name == ByteBPETokenizer.format_name
    assert restored.decode(restored.encode("Indonesia")) == "Indonesia"
