"""Loss functions for small neural-network experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

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


class CrossEntropyLoss:
    """Stable mean cross-entropy computed directly from unnormalized logits.

    logits has shape (batch_size, vocabulary_size); targets contains one
    expected class ID per row. The implementation uses log-sum-exp for
    numerical stability and supplies the softmax-minus-one-hot gradient.
    """

    def __call__(self, logits: Tensor, targets: Sequence[int]) -> Tensor:
        batch_size, class_count = logits.shape
        if batch_size < 1 or class_count < 2:
            raise ValueError("logits must have at least one row and two classes.")
        if len(targets) != batch_size:
            raise ValueError(
                f"Expected {batch_size} targets, received {len(targets)}."
            )

        probabilities: list[list[float]] = []
        losses: list[float] = []
        for row_index, row in enumerate(logits.data):
            target = targets[row_index]
            if not isinstance(target, int) or isinstance(target, bool):
                raise TypeError("Class targets must be integers.")
            if target < 0 or target >= class_count:
                raise ValueError(
                    f"Target class {target} is outside class count {class_count}."
                )

            maximum = max(row)
            exponentials = [math.exp(value - maximum) for value in row]
            denominator = sum(exponentials)
            probabilities.append([value / denominator for value in exponentials])
            log_partition = maximum + math.log(denominator)
            losses.append(log_partition - row[target])

        result = Tensor(
            sum(losses) / batch_size,
            requires_grad=logits.requires_grad,
            _parents=(logits,),
        )

        def backward() -> None:
            if not logits.requires_grad:
                return
            scale = result.grad[0][0] / batch_size
            gradients = []
            for row_index, probability_row in enumerate(probabilities):
                target = targets[row_index]
                gradients.append([
                    (probability - (1.0 if class_index == target else 0.0)) * scale
                    for class_index, probability in enumerate(probability_row)
                ])
            logits._add_grad(gradients)

        result._backward = backward
        return result
