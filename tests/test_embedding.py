import pytest

from simple_ai.embedding import Embedding


def test_embedding_flattens_context_and_preserves_position_order():
    layer = Embedding(5, 3, seed=7)
    output = layer([[1, 2], [3, 4]])
    assert output.shape == (2, 6)
    assert output.data[0][:3] == layer.weights.data[1]
    assert output.data[0][3:] == layer.weights.data[2]


def test_embedding_is_deterministic_for_same_seed():
    left = Embedding(4, 2, seed=11)
    right = Embedding(4, 2, seed=11)
    assert left.weights.data == right.weights.data


def test_embedding_backpropagates_and_accumulates_repeated_tokens():
    layer = Embedding(4, 2, seed=3)
    output = layer([[1, 1]])
    output.sum().backward()
    assert layer.weights.grad[1] == [2.0, 2.0]
    assert layer.weights.grad[0] == [0.0, 0.0]


def test_embedding_parameters_exposes_weight_table():
    layer = Embedding(4, 2)
    assert layer.parameters() == [layer.weights]


@pytest.mark.parametrize("token_ids", [[], [[]], [[1], [1, 2]]])
def test_embedding_rejects_empty_or_ragged_batches(token_ids):
    layer = Embedding(4, 2)
    with pytest.raises(ValueError):
        layer(token_ids)


def test_embedding_rejects_out_of_range_token():
    layer = Embedding(4, 2)
    with pytest.raises(ValueError, match="outside vocabulary"):
        layer([[4]])


def test_embedding_rejects_non_integer_token():
    layer = Embedding(4, 2)
    with pytest.raises(TypeError, match="integers"):
        layer([[1.5]])
