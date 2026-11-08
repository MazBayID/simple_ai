import math

import pytest

from simple_ai.core import Tensor
from simple_ai.losses import CrossEntropyLoss


def test_cross_entropy_uniform_two_class_logits():
    logits = Tensor([[0.0, 0.0]])
    loss = CrossEntropyLoss()(logits, [0])
    assert loss.item() == pytest.approx(math.log(2.0))


def test_cross_entropy_gradient_is_softmax_minus_one_hot():
    logits = Tensor([[0.0, 0.0]])
    CrossEntropyLoss()(logits, [0]).backward()
    assert logits.grad[0] == pytest.approx([-0.5, 0.5])


def test_cross_entropy_averages_gradient_across_batch():
    logits = Tensor([[0.0, 0.0], [0.0, 0.0]])
    CrossEntropyLoss()(logits, [0, 1]).backward()
    flattened = [value for row in logits.grad for value in row]
    assert flattened == pytest.approx([-0.25, 0.25, 0.25, -0.25])


def test_cross_entropy_is_stable_for_large_logits():
    logits = Tensor([[10000.0, -10000.0]])
    loss = CrossEntropyLoss()(logits, [0])
    assert loss.item() == pytest.approx(0.0)
    assert math.isfinite(loss.item())


def test_cross_entropy_penalizes_wrong_extreme_class():
    logits = Tensor([[10000.0, -10000.0]])
    loss = CrossEntropyLoss()(logits, [1])
    assert loss.item() == pytest.approx(20000.0)


def test_cross_entropy_validates_target_count_and_class_range():
    logits = Tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="targets"):
        CrossEntropyLoss()(logits, [])
    with pytest.raises(ValueError, match="outside class count"):
        CrossEntropyLoss()(logits, [2])


def test_cross_entropy_rejects_non_integer_targets():
    with pytest.raises(TypeError, match="integers"):
        CrossEntropyLoss()(Tensor([[1.0, 2.0]]), [0.5])
