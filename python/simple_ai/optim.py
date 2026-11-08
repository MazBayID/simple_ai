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


class Adam:
    """Adam optimizer with bias-corrected first and second moments."""

    def __init__(
        self,
        parameters: list[Tensor],
        learning_rate: float = 0.001,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1).")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not parameters:
            raise ValueError("Adam requires at least one parameter.")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.step_count = 0
        self.first_moment = [
            [[0.0 for _ in range(parameter.shape[1])]
             for _ in range(parameter.shape[0])]
            for parameter in self.parameters
        ]
        self.second_moment = [
            [[0.0 for _ in range(parameter.shape[1])]
             for _ in range(parameter.shape[0])]
            for parameter in self.parameters
        ]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        self.step_count += 1
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count
        for parameter_index, parameter in enumerate(self.parameters):
            if not parameter.requires_grad:
                continue
            first = self.first_moment[parameter_index]
            second = self.second_moment[parameter_index]
            for row in range(parameter.shape[0]):
                for column in range(parameter.shape[1]):
                    gradient = parameter.grad[row][column]
                    first[row][column] = (
                        self.beta1 * first[row][column]
                        + (1.0 - self.beta1) * gradient
                    )
                    second[row][column] = (
                        self.beta2 * second[row][column]
                        + (1.0 - self.beta2) * gradient * gradient
                    )
                    mean = first[row][column] / correction1
                    variance = second[row][column] / correction2
                    parameter.data[row][column] -= (
                        self.learning_rate * mean
                        / (variance ** 0.5 + self.epsilon)
                    )
