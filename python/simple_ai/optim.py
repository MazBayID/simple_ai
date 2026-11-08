"""Simple optimizers for the educational Tensor engine."""

from __future__ import annotations

from .core import Tensor


class SGD:
    """Stochastic gradient descent for a collection of leaf tensors."""

    def __init__(self, parameters: list[Tensor], learning_rate: float = 0.01) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not parameters:
            raise ValueError("SGD requires at least one parameter.")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            if not parameter.requires_grad:
                continue
            for row in range(parameter.shape[0]):
                for column in range(parameter.shape[1]):
                    parameter.data[row][column] -= (
                        self.learning_rate * parameter.grad[row][column]
                    )
