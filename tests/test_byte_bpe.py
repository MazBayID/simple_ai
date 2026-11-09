import pytest

pytest.importorskip("tokenizers")

from simple_ai.byte_bpe import ByteBPETokenizer


def tiny_tokenizer():
    return ByteBPETokenizer.train(
        [
            "Bahasa Indonesia adalah bahasa persatuan.",
            "Kecerdasan buatan mempelajari pola dari data.",
            "Internet of Things menghubungkan perangkat.",
            "Bahasa Indonesia digunakan di berbagai wilayah.",
        ],
        vocab_size=512,
    )


def test_byte_bpe_round_trip_unicode():
    tokenizer = tiny_tokenizer()
    sample = "Halo, dunia! Indonesia 🇮🇩 — teknologi."
    ids = tokenizer.encode(sample, add_bos=False, add_eos=False)
    assert tokenizer.decode(ids) == sample


def test_prompt_can_exclude_eos_and_document_can_include_it():
    tokenizer = tiny_tokenizer()
    prompt = "A Wizard of Earthsea"
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    document_ids = tokenizer.encode(prompt, add_bos=True, add_eos=True)

    assert prompt_ids[0] == tokenizer.bos_id
    assert prompt_ids[-1] != tokenizer.eos_id
    assert document_ids[:-1] == prompt_ids
    assert document_ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(prompt_ids) == prompt


def test_byte_bpe_save_and_load(tmp_path):
    tokenizer = tiny_tokenizer()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    restored = ByteBPETokenizer.load(path)
    sample = "Pemrosesan bahasa alami."
    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.encode(sample) == tokenizer.encode(sample)
    assert restored.decode(restored.encode(sample)) == sample


def test_invalid_vocab_size_is_rejected():
    with pytest.raises(ValueError):
        ByteBPETokenizer.train(["teks"], vocab_size=100)
