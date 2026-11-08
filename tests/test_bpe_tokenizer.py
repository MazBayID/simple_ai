import pytest

from simple_ai.tokenizer import BPETokenizer, load_tokenizer


def test_bpe_round_trip_repeated_indonesian_text():
    text = "makan makanan. makan bersama."
    tokenizer = BPETokenizer.train(text, vocab_size=40)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert len(tokenizer.merges) > 0


def test_bpe_is_deterministic():
    first = BPETokenizer.train("halo dunia halo dunia", vocab_size=24)
    second = BPETokenizer.train("halo dunia halo dunia", vocab_size=24)
    assert first.to_dict() == second.to_dict()


def test_bpe_can_encode_unseen_character_as_unknown():
    tokenizer = BPETokenizer.train("abcabc", vocab_size=12)
    ids = tokenizer.encode("z", add_bos=False, add_eos=False)
    assert ids == [tokenizer.unk_id]


def test_bpe_save_load_and_auto_loader(tmp_path):
    tokenizer = BPETokenizer.train("bahasa indonesia bahasa", vocab_size=32)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.to_dict() == tokenizer.to_dict()
    assert load_tokenizer(path).encode("bahasa") == tokenizer.encode("bahasa")


def test_bpe_rejects_vocab_smaller_than_base_vocabulary():
    with pytest.raises(ValueError, match="smaller"):
        BPETokenizer.train("abcxyz", vocab_size=5)


def test_bpe_merge_validation():
    with pytest.raises(ValueError, match="merges"):
        BPETokenizer(["a", "b"], [["a", "z"]])
