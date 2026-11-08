import pytest

from simple_ai.dataset import NextTokenDataset


def test_dataset_creates_context_target_pairs():
    dataset = NextTokenDataset([10, 11, 12, 13, 14], context_size=2)
    assert len(dataset) == 3
    assert dataset[0] == ([10, 11], 12)
    assert dataset[2] == ([12, 13], 14)


def test_dataset_supports_negative_indices():
    dataset = NextTokenDataset([1, 2, 3, 4], context_size=2)
    assert dataset[-1] == ([2, 3], 4)


def test_dataset_rejects_out_of_range_index():
    dataset = NextTokenDataset([1, 2, 3], context_size=2)
    with pytest.raises(IndexError):
        _ = dataset[1]


def test_dataset_length_zero_when_sequence_too_short():
    assert len(NextTokenDataset([1, 2], context_size=2)) == 0


def test_batches_keep_partial_final_batch():
    dataset = NextTokenDataset([1, 2, 3, 4, 5, 6], context_size=2)
    batches = list(dataset.batches(2))
    assert batches == [
        ([[1, 2], [2, 3]], [3, 4]),
        ([[3, 4], [4, 5]], [5, 6]),
    ]


def test_shuffled_batches_are_reproducible_with_seed():
    dataset = NextTokenDataset(list(range(20)), context_size=2)
    first = list(dataset.batches(3, shuffle=True, seed=7))
    second = list(dataset.batches(3, shuffle=True, seed=7))
    assert first == second


@pytest.mark.parametrize("context_size", [0, -1])
def test_invalid_context_size_rejected(context_size):
    with pytest.raises(ValueError):
        NextTokenDataset([1, 2, 3], context_size=context_size)


def test_invalid_batch_size_rejected():
    dataset = NextTokenDataset([1, 2, 3], context_size=1)
    with pytest.raises(ValueError):
        list(dataset.batches(0))


def test_negative_token_id_rejected():
    with pytest.raises(ValueError):
        NextTokenDataset([1, -2, 3], context_size=1)
