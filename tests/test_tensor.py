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



def test_scalar_subtraction():
    values = Tensor([[3, 5]])
    (values - 2).sum().backward()
    assert values.grad == [[1.0, 1.0]]



def test_zero_grad_resets_accumulated_gradient():
    values = Tensor([[2, 3]])
    values.sum().backward()
    values.sum().backward()
    assert values.grad == [[2.0, 2.0]]

    values.zero_grad()
    assert values.grad == [[0.0, 0.0]]


from simple_ai.core import Tensor


def test_reshape_preserves_values_and_backpropagates():
    source = Tensor([[1, 2, 3, 4]])
    result = source.reshape(2, 2)
    assert result.data == [[1, 2], [3, 4]]
    result.sum().backward()
    assert source.grad == [[1, 1, 1, 1]]


def test_row_selection_routes_gradient_to_selected_row():
    source = Tensor([[1, 2], [3, 4], [5, 6]])
    source.row(-1).sum().backward()
    assert source.grad == [[0, 0], [0, 0], [1, 1]]


def test_stack_rows_routes_gradient_to_each_input():
    first = Tensor([[1, 2]])
    second = Tensor([[3, 4]])
    Tensor.stack_rows([first, second]).sum().backward()
    assert first.grad == [[1, 1]]
    assert second.grad == [[1, 1]]


def test_masked_fill_blocks_gradient_at_masked_entries():
    source = Tensor([[1, 2], [3, 4]])
    result = source.masked_fill([[False, True], [True, False]], -1.0)
    assert result.data == [[1, -1], [-1, 4]]
    result.sum().backward()
    assert source.grad == [[1, 0], [0, 1]]


def test_row_softmax_is_stable_and_backpropagates():
    source = Tensor([[1000, 1001], [1, 1]])
    result = source.softmax()
    assert abs(sum(result.data[0]) - 1.0) < 1e-12
    assert abs(sum(result.data[1]) - 1.0) < 1e-12
    result.data[0][0] = result.data[0][0]
    (result * Tensor([[1, 0], [0, 1]])).sum().backward()
    assert any(abs(value) > 0 for row in source.grad for value in row)
