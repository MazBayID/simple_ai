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
