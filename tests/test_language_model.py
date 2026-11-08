import pytest

from simple_ai.dataset import NextTokenDataset
from simple_ai.language_model import TinyLanguageModel, train_language_model
from simple_ai.optim import SGD
from simple_ai.tokenizer import CharTokenizer


def test_model_returns_one_logit_per_vocabulary_token():
    model = TinyLanguageModel(7, 3, embedding_dim=4, hidden_size=6, seed=2)
    output = model([[1, 2, 3], [3, 2, 1]])
    assert output.shape == (2, 7)


def test_model_parameter_count_and_shapes():
    model = TinyLanguageModel(5, 2, embedding_dim=3, hidden_size=4)
    assert model.parameter_count() == (5 * 3) + (6 * 4) + (4 * 5)
    assert len(model.parameters()) == 3


def test_model_rejects_wrong_context_size():
    model = TinyLanguageModel(5, 2)
    with pytest.raises(ValueError, match="context size"):
        model([[1]])


def test_training_reduces_loss_on_repeating_pattern():
    token_ids = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    dataset = NextTokenDataset(token_ids, context_size=2)
    model = TinyLanguageModel(3, 2, embedding_dim=4, hidden_size=8, seed=4)
    optimizer = SGD(model.parameters(), learning_rate=0.08)
    history = train_language_model(
        model, dataset, optimizer, epochs=35, batch_size=4, seed=3
    )
    assert history[-1] < history[0]


def test_checkpoint_round_trip_preserves_logits(tmp_path):
    model = TinyLanguageModel(6, 2, embedding_dim=3, hidden_size=5, seed=9)
    expected = model([[1, 2]]).data
    checkpoint = tmp_path / "model.json"
    model.save(checkpoint)
    restored = TinyLanguageModel.load(checkpoint)
    assert restored([[1, 2]]).data == expected


def test_generation_is_reproducible_and_returns_only_new_tokens():
    model = TinyLanguageModel(6, 2, embedding_dim=3, hidden_size=5, seed=9)
    first = model.generate([1, 2], max_new_tokens=5, seed=11)
    second = model.generate([1, 2], max_new_tokens=5, seed=11)
    assert first == second
    assert len(first) == 5
    assert all(0 <= token_id < 6 for token_id in first)


def test_generation_stops_at_eos():
    model = TinyLanguageModel(5, 2, embedding_dim=3, hidden_size=4, seed=2)
    generated = model.generate(
        [1, 2], max_new_tokens=10, eos_id=0, temperature=0.01, seed=0
    )
    assert len(generated) <= 10


def test_training_rejects_optimizer_from_another_model():
    dataset = NextTokenDataset([1, 2, 1, 2], context_size=1)
    model = TinyLanguageModel(3, 1)
    other = TinyLanguageModel(3, 1)
    optimizer = SGD(other.parameters())
    with pytest.raises(ValueError, match="this model"):
        train_language_model(model, dataset, optimizer, epochs=1)
