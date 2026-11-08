import math

import pytest

from simple_ai.core import Tensor


def test_layer_norm_normalizes_each_row_and_backpropagates():
    values = Tensor([[1, 2, 4], [2, 0, 1]])
    normalized = values.layer_norm()
    for row in normalized.data:
        assert abs(sum(row) / len(row)) < 1e-12
        variance = sum(value * value for value in row) / len(row)
        assert abs(variance - 1.0) < 1e-4
    weights = Tensor([[1.0, -2.0, 0.5], [0.3, 1.0, -0.5]], requires_grad=False)
    (normalized * weights).sum().backward()
    assert all(math.isfinite(value) for row in values.grad for value in row)
    assert any(abs(value) > 1e-8 for row in values.grad for value in row)


def test_layer_norm_gradient_matches_finite_difference():
    source = [[0.5, 1.5, 3.0]]
    coefficients = [[0.3, -0.2, 1.1]]
    values = Tensor(source)
    loss = (values.layer_norm() * Tensor(coefficients, requires_grad=False)).sum()
    loss.backward()
    analytical = values.grad[0][1]

    delta = 1e-6
    plus = [list(source[0])]
    minus = [list(source[0])]
    plus[0][1] += delta
    minus[0][1] -= delta
    plus_loss = (Tensor(plus, requires_grad=False).layer_norm()
                 * Tensor(coefficients, requires_grad=False)).sum().item()
    minus_loss = (Tensor(minus, requires_grad=False).layer_norm()
                  * Tensor(coefficients, requires_grad=False)).sum().item()
    numerical = (plus_loss - minus_loss) / (2 * delta)
    assert abs(analytical - numerical) < 1e-5


def test_repeat_rows_accumulates_parameter_gradients():
    parameter = Tensor([[2.0, -1.0]])
    repeated = parameter.repeat_rows(3)
    assert repeated.data == [[2.0, -1.0]] * 3
    (repeated * Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=False)).sum().backward()
    assert parameter.grad == [[9.0, 12.0]]


def test_concatenate_columns_routes_gradients_to_each_input():
    left = Tensor([[1, 2], [3, 4]])
    right = Tensor([[5], [6]])
    combined = Tensor.concatenate_columns([left, right])
    assert combined.data == [[1, 2, 5], [3, 4, 6]]
    (combined * Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=False)).sum().backward()
    assert left.grad == [[1.0, 2.0], [4.0, 5.0]]
    assert right.grad == [[3.0], [6.0]]


def test_layer_norm_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        Tensor([[1, 2]]).layer_norm(0)
