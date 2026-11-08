from pathlib import Path

import pytest

from simple_ai.model import TinyCharModel


def test_train_reports_stats():
    model = TinyCharModel()
    stats = model.train("ababa")
    assert stats["characters"] == 5
    assert stats["vocabulary_size"] == 2
    assert stats["transition_contexts"] == 3


def test_generate_is_reproducible():
    model = TinyCharModel()
    model.train("hello world. hello simple ai.")
    first = model.generate("h", max_new_tokens=30, seed=123)
    second = model.generate("h", max_new_tokens=30, seed=123)
    assert first == second
    assert first.startswith("h")


def test_generate_requires_training():
    model = TinyCharModel()
    with pytest.raises(RuntimeError):
        model.generate()


def test_save_and_load(tmp_path: Path):
    model = TinyCharModel()
    model.train("abcabc")
    checkpoint = tmp_path / "model.json"
    model.save(checkpoint)

    loaded = TinyCharModel.load(checkpoint)
    assert loaded.generate("a", max_new_tokens=10, seed=4) == model.generate(
        "a", max_new_tokens=10, seed=4
    )


def test_empty_training_data_rejected():
    model = TinyCharModel()
    with pytest.raises(ValueError):
        model.train("")
