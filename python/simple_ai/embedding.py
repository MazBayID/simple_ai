"""Trainable token embeddings for fixed-context language experiments.

The current 2-D Tensor engine has no native 3-D tensors. This layer therefore
flattens each context's position-specific embeddings into one feature row:
(batch, context) token IDs -> (batch, context * embedding_dim) Tensor.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .core import Tensor


class Embedding:
    """Look up trainable vectors and flatten them by token position.

    Repeated token IDs accumulate gradients into the same vocabulary row.
    Position order is preserved, so the downstream projection can distinguish
    the same token appearing at different positions in a context.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        seed: int = 0,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        rng = random.Random(seed)
        scale = 1.0 / (embedding_dim ** 0.5)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = Tensor([
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ])

    def __call__(self, token_ids: Sequence[Sequence[int]]) -> Tensor:
        if not token_ids:
            raise ValueError("token_ids batch must not be empty.")
        context_size = len(token_ids[0])
        if context_size < 1:
            raise ValueError("Each token context must not be empty.")
        if any(len(row) != context_size for row in token_ids):
            raise ValueError("All token contexts must have equal length.")

        batch_size = len(token_ids)
        output: list[list[float]] = []
        for row in token_ids:
            features: list[float] = []
            for token_id in row:
                if not isinstance(token_id, int) or isinstance(token_id, bool):
                    raise TypeError("Token IDs must be integers.")
                if token_id < 0 or token_id >= self.vocab_size:
                    raise ValueError(
                        f"Token ID {token_id} is outside vocabulary size {self.vocab_size}."
                    )
                features.extend(self.weights.data[token_id])
            output.append(features)

        result = Tensor(
            output,
            requires_grad=self.weights.requires_grad,
            _parents=(self.weights,),
        )

        def backward() -> None:
            if not self.weights.requires_grad:
                return
            for batch_index, row in enumerate(token_ids):
                for position, token_id in enumerate(row):
                    offset = position * self.embedding_dim
                    for dimension in range(self.embedding_dim):
                        self.weights.grad[token_id][dimension] += (
                            result.grad[batch_index][offset + dimension]
                        )

        result._backward = backward
        return result

    def parameters(self) -> list[Tensor]:
        return [self.weights]
