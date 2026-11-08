"""Minimal neural-network layers built on Simple AI's Tensor engine."""

from __future__ import annotations

import random

from .core import Tensor


class Linear:
    """A trainable fully connected layer without bias.

    Inputs have shape (batch_size, in_features). Weights have shape
    (in_features, out_features). Bias is intentionally omitted in this
    first correctness-first milestone.
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int = 0) -> None:
        if in_features < 1 or out_features < 1:
            raise ValueError("Layer dimensions must be positive.")
        rng = random.Random(seed)
        scale = (2.0 / in_features) ** 0.5
        self.weights = Tensor(
            [[rng.uniform(-scale, scale) for _ in range(out_features)]
             for _ in range(in_features)]
        )

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Expected {self.weights.shape[0]} input features, got {inputs.shape[1]}."
            )
        return inputs @ self.weights

    def parameters(self) -> list[Tensor]:
        return [self.weights]


class ReLU:
    """Rectified linear activation."""

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.relu()

    def parameters(self) -> list[Tensor]:
        return []



class LayerNorm:
    """Learnable row-wise layer normalization (scale and shift)."""

    def __init__(self, dimension: int, *, epsilon: float = 1e-5) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.dimension = dimension
        self.epsilon = float(epsilon)
        self.weight = Tensor([[1.0 for _ in range(dimension)]])
        self.bias = Tensor([[0.0 for _ in range(dimension)]])

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} features, got {inputs.shape[1]}."
            )
        normalized = inputs.layer_norm(self.epsilon)
        return (
            normalized * self.weight.repeat_rows(inputs.shape[0])
            + self.bias.repeat_rows(inputs.shape[0])
        )

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
