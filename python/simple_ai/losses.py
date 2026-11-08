"""Loss functions for small neural-network experiments."""

from __future__ import annotations

from .core import Tensor


class MeanSquaredError:
    """Mean squared error over every element in two equally shaped tensors."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}."
            )
        difference = prediction - target
        return (difference * difference).mean()
