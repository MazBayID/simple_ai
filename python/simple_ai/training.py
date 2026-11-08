"""Small, explicit training loop utilities."""

from __future__ import annotations

from collections.abc import Callable

from .core import Tensor
from .losses import MeanSquaredError
from .optim import SGD


def train_step(
    model: Callable[[Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    optimizer: SGD,
    loss_function: MeanSquaredError | None = None,
) -> float:
    """Run one forward/loss/backward/update step and return its pre-update loss."""
    criterion = loss_function or MeanSquaredError()
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    value = loss.item()
    loss.backward()
    optimizer.step()
    return value


def fit(
    model: Callable[[Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    optimizer: SGD,
    *,
    epochs: int,
    loss_function: MeanSquaredError | None = None,
    callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train on one full batch repeatedly; returns the loss from each epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        loss = train_step(model, inputs, targets, optimizer, loss_function)
        history.append(loss)
        if callback is not None:
            callback(epoch, loss)
    return history
