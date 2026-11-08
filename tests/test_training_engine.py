from simple_ai.core import Tensor
from simple_ai.losses import MeanSquaredError
from simple_ai.nn import Linear, ReLU
from simple_ai.optim import SGD
from simple_ai.training import fit, train_step


class TinyRegression:
    def __init__(self):
        self.layer = Linear(1, 1, seed=3)

    def __call__(self, inputs):
        return self.layer(inputs)

    def parameters(self):
        return self.layer.parameters()


def test_linear_forward_shape_and_values():
    layer = Linear(2, 1, seed=1)
    output = layer(Tensor([[1, 2], [3, 4]], requires_grad=False))
    assert output.shape == (2, 1)


def test_relu_layer():
    result = ReLU()(Tensor([[-2, 3]], requires_grad=False))
    assert result.data == [[0.0, 3.0]]


def test_mse_loss_and_gradient():
    prediction = Tensor([[1, 3]])
    target = Tensor([[0, 1]], requires_grad=False)
    loss = MeanSquaredError()(prediction, target)
    assert loss.item() == 2.5
    loss.backward()
    assert prediction.grad == [[1.0, 2.0]]


def test_sgd_updates_parameter():
    parameter = Tensor([[2.0]])
    parameter.grad = [[4.0]]
    SGD([parameter], learning_rate=0.1).step()
    assert parameter.data == [[1.6]]


def test_sgd_zero_grad():
    parameter = Tensor([[2.0]])
    parameter.grad = [[4.0]]
    optimizer = SGD([parameter])
    optimizer.zero_grad()
    assert parameter.grad == [[0.0]]


def test_train_step_reduces_simple_loss():
    model = TinyRegression()
    optimizer = SGD(model.parameters(), learning_rate=0.05)
    x = Tensor([[-2], [-1], [1], [2]], requires_grad=False)
    y = Tensor([[-4], [-2], [2], [4]], requires_grad=False)

    before = train_step(model, x, y, optimizer)
    after = train_step(model, x, y, optimizer)
    assert after < before


def test_fit_history_and_callback():
    model = TinyRegression()
    optimizer = SGD(model.parameters(), learning_rate=0.05)
    x = Tensor([[1], [2]], requires_grad=False)
    y = Tensor([[2], [4]], requires_grad=False)
    seen = []
    history = fit(model, x, y, optimizer, epochs=5, callback=lambda epoch, loss: seen.append(epoch))
    assert len(history) == 5
    assert seen == [1, 2, 3, 4, 5]
    assert history[-1] < history[0]
