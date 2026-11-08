import pytest

from simple_ai.core import Tensor
from simple_ai.optim import Adam


def test_adam_first_step_uses_bias_corrected_moments():
    parameter = Tensor([[1.0]])
    optimizer = Adam([parameter], learning_rate=0.1)
    parameter.grad = [[2.0]]
    optimizer.step()
    assert parameter.data[0][0] == pytest.approx(0.9)
    assert optimizer.step_count == 1


def test_adam_zero_grad_and_second_step():
    parameter = Tensor([[1.0, -1.0]])
    optimizer = Adam([parameter], learning_rate=0.01)
    parameter.grad = [[1.0, -1.0]]
    optimizer.step()
    optimizer.zero_grad()
    assert parameter.grad == [[0.0, 0.0]]
    parameter.grad = [[0.5, -0.5]]
    optimizer.step()
    assert optimizer.step_count == 2
    assert parameter.data[0][0] < 1.0
    assert parameter.data[0][1] > -1.0


def test_adam_rejects_invalid_hyperparameters():
    parameter = Tensor([[1.0]])
    with pytest.raises(ValueError):
        Adam([parameter], learning_rate=0)
    with pytest.raises(ValueError):
        Adam([parameter], beta1=1.0)
    with pytest.raises(ValueError):
        Adam([], learning_rate=0.1)
