"""Small decoder-only Transformer with multi-head causal self-attention.

This version is kept separate from the original single-head implementation so
old checkpoints and baseline experiments remain reproducible.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from collections.abc import Sequence

from .core import Tensor
from .embedding import Embedding
from .nn import LayerNorm, Linear, ReLU


class _CausalAttentionHead:
    """One causal attention head with independent query, key, and value maps."""

    def __init__(self, embedding_dim: int, head_dim: int, *, seed: int) -> None:
        self.query = Linear(embedding_dim, head_dim, seed=seed)
        self.key = Linear(embedding_dim, head_dim, seed=seed + 1)
        self.value = Linear(embedding_dim, head_dim, seed=seed + 2)
        self.scale = 1.0 / math.sqrt(head_dim)

    def __call__(self, hidden: Tensor, causal_mask: list[list[bool]]) -> Tensor:
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        scores = (query @ key.T) * self.scale
        attention = scores.masked_fill(causal_mask, -1.0e9).softmax()
        return attention @ value

    def parameters(self) -> list[Tensor]:
        return self.query.parameters() + self.key.parameters() + self.value.parameters()


class MultiHeadSelfAttention:
    """Parallel causal heads concatenated and projected back to model width."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        context_size: int,
        seed: int,
    ) -> None:
        if num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by a positive num_heads.")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.heads = [
            _CausalAttentionHead(embedding_dim, self.head_dim, seed=seed + index * 3)
            for index in range(num_heads)
        ]
        self.output = Linear(embedding_dim, embedding_dim, seed=seed + num_heads * 3 + 1)
        self.causal_mask = [
            [column > row for column in range(context_size)]
            for row in range(context_size)
        ]

    def __call__(self, hidden: Tensor) -> Tensor:
        if hidden.shape[0] != len(self.causal_mask):
            raise ValueError("Attention input sequence length does not match context size.")
        head_outputs = [head(hidden, self.causal_mask) for head in self.heads]
        combined = Tensor.concatenate_columns(head_outputs)
        return self.output(combined)

    def parameters(self) -> list[Tensor]:
        result: list[Tensor] = []
        for head in self.heads:
            result.extend(head.parameters())
        result.extend(self.output.parameters())
        return result


class MultiHeadTransformerLanguageModel:
    """One-block, pre-normalized decoder with multi-head causal attention."""

    format_name = "simple-ai-multihead-transformer"
    format_version = 1

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        *,
        embedding_dim: int = 8,
        num_heads: int = 2,
        seed: int = 0,
    ) -> None:
        if vocab_size < 2 or context_size < 1 or embedding_dim < 2:
            raise ValueError("vocab_size/context_size must be positive and embedding_dim >= 2.")
        if num_heads < 1 or embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by a positive num_heads.")
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        rng = random.Random(seed + 10)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.position = Tensor([
            [rng.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(context_size)
        ])
        self.attention_norm = LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(
            embedding_dim, num_heads, context_size=context_size, seed=seed + 20
        )
        self.feed_forward_norm = LayerNorm(embedding_dim)
        self.feed_forward_in = Linear(embedding_dim, embedding_dim * 2, seed=seed + 40)
        self.activation = ReLU()
        self.feed_forward_out = Linear(embedding_dim * 2, embedding_dim, seed=seed + 41)
        self.final_norm = LayerNorm(embedding_dim)
        self.output = Linear(embedding_dim, vocab_size, seed=seed + 50)

    def _forward_one(self, token_ids: Sequence[int]) -> Tensor:
        if len(token_ids) != self.context_size:
            raise ValueError(f"Expected context size {self.context_size}.")
        embedded = self.embedding([token_ids]).reshape(self.context_size, self.embedding_dim)
        hidden = embedded + self.position
        hidden = hidden + self.attention(self.attention_norm(hidden))
        feed_forward = self.feed_forward_out(
            self.activation(self.feed_forward_in(self.feed_forward_norm(hidden)))
        )
        hidden = self.final_norm(hidden + feed_forward)
        return self.output(hidden.row(-1))

    def __call__(self, token_ids: Sequence[Sequence[int]]) -> Tensor:
        if not token_ids:
            raise ValueError("Expected a non-empty batch.")
        return Tensor.stack_rows([self._forward_one(row) for row in token_ids])

    def parameters(self) -> list[Tensor]:
        return (
            self.embedding.parameters()
            + [self.position]
            + self.attention_norm.parameters()
            + self.attention.parameters()
            + self.feed_forward_norm.parameters()
            + self.feed_forward_in.parameters()
            + self.feed_forward_out.parameters()
            + self.final_norm.parameters()
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
                allowed = set(sorted(
                    range(self.vocab_size), key=lambda index: scaled[index], reverse=True
                )[:top_k])
                scaled = [
                    value if index in allowed else float("-inf")
                    for index, value in enumerate(scaled)
                ]
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
                "num_heads": self.num_heads,
            },
            "parameters": [parameter.data for parameter in self.parameters()],
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadTransformerLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported multi-head Transformer checkpoint format or version.")
        model = cls(**payload.get("config", {}))
        saved = payload.get("parameters")
        parameters = model.parameters()
        if not isinstance(saved, list) or len(saved) != len(parameters):
            raise ValueError("Multi-head Transformer checkpoint parameter list is invalid.")
        for parameter, values in zip(parameters, saved):
            restored = Tensor(values, requires_grad=parameter.requires_grad)
            if restored.shape != parameter.shape:
                raise ValueError("Multi-head Transformer checkpoint parameter shape mismatch.")
            parameter.data = restored.data
        return model
