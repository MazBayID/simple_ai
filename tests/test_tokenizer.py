import pytest

from simple_ai.tokenizer import CharTokenizer


def test_tokenizer_round_trip_unicode():
    text = "Halo, dunia! 🌍\n"
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocabulary_is_deterministic():
    first = CharTokenizer.from_text("cab")
    second = CharTokenizer.from_text("abc")
    assert first.to_dict() == second.to_dict()


def test_unknown_character_uses_unknown_id():
    tokenizer = CharTokenizer.from_text("abc")
    assert tokenizer.encode("z", add_bos=False, add_eos=False) == [tokenizer.unk_id]


def test_special_tokens_can_be_disabled():
    tokenizer = CharTokenizer.from_text("abc")
    assert tokenizer.encode("ab", add_bos=False, add_eos=False) == [
        tokenizer.token_to_id["a"],
        tokenizer.token_to_id["b"],
    ]


def test_tokenizer_save_and_load(tmp_path):
    tokenizer = CharTokenizer.from_text("Halo dunia")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = CharTokenizer.load(path)
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.encode("Halo dunia") == tokenizer.encode("Halo dunia")


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_invalid_token_id_rejected():
    tokenizer = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tokenizer.decode([999])
