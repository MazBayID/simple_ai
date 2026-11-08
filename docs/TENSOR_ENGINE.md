# Tensor Engine: first implementation

Simple AI now includes a small 2-D tensor engine written with Python's standard
library. It does not import NumPy, PyTorch, TensorFlow, or another numerical
framework.

## Location

- Implementation: `python/simple_ai/core/tensor.py`
- Public import: `from simple_ai.core import Tensor`
- Tests: `tests/test_tensor.py`

## Supported operations

- Rectangular 2-D tensor construction and shape validation
- Addition and subtraction
- Element-wise multiplication
- Scalar addition, subtraction, and multiplication
- Matrix multiplication
- Transpose
- ReLU activation
- Sum and mean reductions
- Reverse-mode automatic differentiation for these operations

This is an educational prototype. It is not yet optimized and does not support
1-D tensors, broadcasting between arbitrary shapes, GPU execution, batching,
serialization, or a neural-network layer API.

## Quick example

```python
from simple_ai.core import Tensor

weights = Tensor([[0.2, -0.1], [0.4, 0.3]])
inputs = Tensor([[1.0, 2.0]])
bias = Tensor([[0.1, 0.1]])

prediction = (inputs @ weights + bias).relu()
loss = prediction.mean()
loss.backward()

print("loss:", loss.item())
print("weight gradients:", weights.grad)
```

## How backward propagation works

Each operation creates a result tensor and remembers the tensors that produced
it. Calling `backward()` on a scalar result walks that computation graph in
reverse order. Each operation applies its local derivative and adds its
contribution to its parents' gradients using the chain rule.

For example, for element-wise multiplication (z = x * y):

- gradient for (x): (partial L/partial x = (partial L/partial z) * y)
- gradient for (y): (partial L/partial y = (partial L/partial z) * x)

The current implementation accumulates gradients. Call `zero_grad()` before
a new training step when reusing parameters. An optimizer will be added in a
later milestone.

## Test

From the repository root:

```bash
PYTHONPATH=python python -m pytest -q
```

The engine must pass numerical correctness tests before it becomes the base of
our neural-network layers. Passing unit tests does not establish useful model
quality or compliance with any resource ceiling.
