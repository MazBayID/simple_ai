import pytest

from simple_ai.core import Tensor


def test_matrix_multiplication_and_gradients():
    left = Tensor([[1, 2], [3, 4]])
    right = Tensor([[5, 6], [7, 8]])

    loss = (left @ right).sum()
    assert loss.item() == 134.0
    loss.backward()

    assert left.grad == [[11.0, 15.0], [11.0, 15.0]]
    assert right.grad == [[4.0, 4.0], [6.0, 6.0]]


def test_elementwise_multiply_and_relu_gradients():
    values = Tensor([[-1, 2], [3, -4]])
    loss = values.relu().mean()

    assert loss.item() == 1.25
    loss.backward()
    assert values.grad == [[0.0, 0.25], [0.25, 0.0]]


def test_scalar_arithmetic_broadcasts_and_backpropagates():
    values = Tensor([[1, 2], [3, 4]])
    result = ((values * 2) + 1).mean()

    assert result.item() == 6.0
    result.backward()
    assert values.grad == [[0.5, 0.5], [0.5, 0.5]]


def test_transpose_gradient():
    values = Tensor([[1, 2, 3], [4, 5, 6]])
    values.T.sum().backward()
    assert values.grad == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_shape_validation():
    with pytest.raises(ValueError):
        Tensor([[1, 2], [3]])
    with pytest.raises(ValueError):
        Tensor([[1, 2]]) @ Tensor([[1, 2]])
    with pytest.raises(ValueError):
        Tensor([[1, 2]]) + Tensor([[1], [2]])


def test_backward_requires_scalar_output():
    with pytest.raises(ValueError):
        Tensor([[1, 2]]).backward()


def test_constants_do_not_receive_gradients():
    values = Tensor([[2, 3]])
    constant = Tensor([[4, 5]], requires_grad=False)
    (values * constant).sum().backward()

    assert values.grad == [[4.0, 5.0]]
    assert constant.grad == [[0.0, 0.0]]
