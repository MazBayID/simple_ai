"""A tiny decoder-only causal Transformer built on the Simple AI Tensor engine.

This is a correctness-first, single-head educational implementation. It uses
Python lists and 2-D tensors, so it is not intended for large-scale training.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from collections.abc import Sequence

from .core import Tensor
from .embedding import Embedding
from .language_model import train_language_model
from .nn import Linear, ReLU


class TinyTransformerLanguageModel:
    """One-block, single-head causal self-attention language model."""

    format_name = "simple-ai-tiny-transformer"
    format_version = 1

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        *,
        embedding_dim: int = 8,
        seed: int = 0,
    ) -> None:
        if vocab_size < 2 or context_size < 1 or embedding_dim < 2:
            raise ValueError("vocab_size/context_size must be positive and embedding_dim >= 2.")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        rng = random.Random(seed + 10)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.position = Tensor([
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(context_size)
        ])
        self.query = Linear(embedding_dim, embedding_dim, seed=seed + 1)
        self.key = Linear(embedding_dim, embedding_dim, seed=seed + 2)
        self.value = Linear(embedding_dim, embedding_dim, seed=seed + 3)
        self.feed_forward_in = Linear(embedding_dim, embedding_dim * 2, seed=seed + 4)
        self.activation = ReLU()
        self.feed_forward_out = Linear(embedding_dim * 2, embedding_dim, seed=seed + 5)
        self.output = Linear(embedding_dim, vocab_size, seed=seed + 6)
        self.causal_mask = [
            [column > row for column in range(context_size)]
            for row in range(context_size)
        ]

    def _forward_one(self, token_ids: Sequence[int]) -> Tensor:
        if len(token_ids) != self.context_size:
            raise ValueError(f"Expected context size {self.context_size}.")
        embedded = self.embedding([token_ids]).reshape(self.context_size, self.embedding_dim)
        hidden = embedded + self.position

        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = (query @ key.T) * (1.0 / math.sqrt(self.embedding_dim))
        attention = scores.masked_fill(self.causal_mask, -1.0e9).softmax()
        attended = attention @ value
        hidden = hidden + attended

        feed_forward = self.feed_forward_out(
            self.activation(self.feed_forward_in(hidden))
        )
        hidden = hidden + feed_forward
        return self.output(hidden.row(-1))

    def __call__(self, token_ids: Sequence[Sequence[int]]) -> Tensor:
        if not token_ids:
            raise ValueError("Expected a non-empty batch.")
        logits = [self._forward_one(row) for row in token_ids]
        return Tensor.stack_rows(logits)

    def parameters(self) -> list[Tensor]:
        return (
            self.embedding.parameters()
            + [self.position]
            + self.query.parameters()
            + self.key.parameters()
            + self.value.parameters()
            + self.feed_forward_in.parameters()
            + self.feed_forward_out.parameters()
            + self.output.parameters()
        )

    def parameter_count(self) -> int:
        return sum(parameter.shape[0] * parameter.shape[1] for parameter in self.parameters())

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        seed: int = 0,
    ) -> list[int]:
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be non-negative and temperature positive.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when provided.")
        if any(not isinstance(value, int) or value < 0 or value >= self.vocab_size
               for value in prompt_ids):
            raise ValueError("Prompt contains a token ID outside the vocabulary.")
        generated = list(prompt_ids)
        rng = random.Random(seed)
        for _ in range(max_new_tokens):
            context = generated[-self.context_size:]
            if len(context) < self.context_size:
                context = [0] * (self.context_size - len(context)) + context
            logits = self([context]).data[0]
            scaled = [value / temperature for value in logits]
            if top_k is not None and top_k < self.vocab_size:
                allowed = set(sorted(range(self.vocab_size),
                                     key=lambda index: scaled[index], reverse=True)[:top_k])
                scaled = [value if index in allowed else float("-inf")
                          for index, value in enumerate(scaled)]
            maximum = max(scaled)
            probabilities = [math.exp(value - maximum) for value in scaled]
            total = sum(probabilities)
            threshold = rng.random()
            cumulative = 0.0
            next_id = len(probabilities) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability / total
                if threshold <= cumulative:
                    next_id = index
                    break
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
        return generated[len(prompt_ids):]

    def save(self, path: str | Path) -> None:
        payload = {
            "format": self.format_name,
            "version": self.format_version,
            "config": {
                "vocab_size": self.vocab_size,
                "context_size": self.context_size,
                "embedding_dim": self.embedding_dim,
            },
            "parameters": [parameter.data for parameter in self.parameters()],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyTransformerLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported Transformer checkpoint format or version.")
        model = cls(**payload.get("config", {}))
        saved = payload.get("parameters")
        parameters = model.parameters()
        if not isinstance(saved, list) or len(saved) != len(parameters):
            raise ValueError("Transformer checkpoint parameter list is invalid.")
        for parameter, values in zip(parameters, saved):
            restored = Tensor(values, requires_grad=parameter.requires_grad)
            if restored.shape != parameter.shape:
                raise ValueError("Transformer checkpoint parameter shape mismatch.")
            parameter.data = restored.data
        return model
