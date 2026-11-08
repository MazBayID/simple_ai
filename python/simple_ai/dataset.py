"""Small next-token dataset and deterministic mini-batch iterator."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence


class NextTokenDataset:
    """Create fixed-context next-token examples from a token-ID sequence.

    For each valid start position, the input is a context_size window and the
    target is the immediately following token. No padding is needed because
    incomplete windows are not emitted.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        *,
        context_size: int,
    ) -> None:
        if context_size < 1:
            raise ValueError("context_size must be >= 1.")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers.")
        self.token_ids = tuple(token_ids)
        self.context_size = context_size
        self._length = max(0, len(self.token_ids) - context_size)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("NextTokenDataset index out of range.")
        start = index
        end = start + self.context_size
        return list(self.token_ids[start:end]), self.token_ids[end]

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Iterator[tuple[list[list[int]], list[int]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        indices = list(range(len(self)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            inputs: list[list[int]] = []
            targets: list[int] = []
            for index in selected:
                context, target = self[index]
                inputs.append(context)
                targets.append(target)
            yield inputs, targets
