"""A tiny fixed-context neural language model built from Simple AI layers."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from collections.abc import Sequence

from .core import Tensor
from .dataset import NextTokenDataset
from .embedding import Embedding
from .losses import CrossEntropyLoss
from .nn import Linear, ReLU
from .optim import SGD


class TinyLanguageModel:
    """Position-aware MLP language model for next-character prediction.

    This is a deliberately small baseline, not a Transformer. It embeds each
    token in a fixed context, flattens those vectors, applies one hidden
    projection and ReLU, then projects to vocabulary logits.
    """

    format_name = "simple-ai-tiny-language-model"
    format_version = 1

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        *,
        embedding_dim: int = 8,
        hidden_size: int = 32,
        seed: int = 0,
    ) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")
        if context_size < 1:
            raise ValueError("context_size must be positive.")
        if embedding_dim < 1 or hidden_size < 1:
            raise ValueError("embedding_dim and hidden_size must be positive.")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        self.hidden = Linear(context_size * embedding_dim, hidden_size, seed=seed + 1)
        self.activation = ReLU()
        self.output = Linear(hidden_size, vocab_size, seed=seed + 2)

    def __call__(self, token_ids: Sequence[Sequence[int]]) -> Tensor:
        if not token_ids or any(len(row) != self.context_size for row in token_ids):
            raise ValueError(
                f"Expected a non-empty batch with context size {self.context_size}."
            )
        features = self.embedding(token_ids)
        hidden = self.activation(self.hidden(features))
        return self.output(hidden)

    def parameters(self) -> list[Tensor]:
        return (
            self.embedding.parameters()
            + self.hidden.parameters()
            + self.output.parameters()
        )

    def parameter_count(self) -> int:
        return sum(
            parameter.shape[0] * parameter.shape[1]
            for parameter in self.parameters()
        )

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
        """Generate token IDs autoregressively from an existing prompt."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
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
                                     key=lambda index: scaled[index],
                                     reverse=True)[:top_k])
                scaled = [
                    value if index in allowed else float("-inf")
                    for index, value in enumerate(scaled)
                ]

            maximum = max(scaled)
            probabilities = [math.exp(value - maximum) for value in scaled]
            total = sum(probabilities)
            probabilities = [value / total for value in probabilities]
            threshold = rng.random()
            cumulative = 0.0
            next_id = len(probabilities) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability
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
                "hidden_size": self.hidden_size,
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
    def load(cls, path: str | Path) -> "TinyLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name:
            raise ValueError("Unsupported language-model checkpoint format.")
        if payload.get("version") != cls.format_version:
            raise ValueError("Unsupported language-model checkpoint version.")
        config = payload.get("config", {})
        model = cls(**config)
        saved_parameters = payload.get("parameters")
        parameters = model.parameters()
        if not isinstance(saved_parameters, list) or len(saved_parameters) != len(parameters):
            raise ValueError("Checkpoint parameter list is invalid.")
        for parameter, saved in zip(parameters, saved_parameters):
            restored = Tensor(saved, requires_grad=parameter.requires_grad)
            if restored.shape != parameter.shape:
                raise ValueError("Checkpoint parameter shape does not match model config.")
            parameter.data = restored.data
        return model


def train_language_model(
    model: TinyLanguageModel,
    dataset: NextTokenDataset,
    optimizer: SGD,
    *,
    epochs: int,
    batch_size: int = 16,
    seed: int = 0,
) -> list[float]:
    """Train on next-token examples and return the mean loss per epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not len(dataset):
        raise ValueError("Dataset has no next-token examples.")
    if len(optimizer.parameters) != len(model.parameters()):
        raise ValueError("Optimizer parameters do not match model parameters.")
    if any(left is not right for left, right in zip(
        optimizer.parameters, model.parameters()
    )):
        raise ValueError("Optimizer must contain this model's parameters in order.")

    criterion = CrossEntropyLoss()
    history: list[float] = []
    for epoch in range(epochs):
        weighted_loss = 0.0
        examples_seen = 0
        batches = dataset.batches(
            batch_size, shuffle=True, seed=seed + epoch
        )
        for inputs, targets in batches:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss_value = loss.item()
            loss.backward()
            optimizer.step()
            weighted_loss += loss_value * len(targets)
            examples_seen += len(targets)
        history.append(weighted_loss / examples_seen)
    return history


def load_language_model(path: str | Path):
    """Load a supported model checkpoint by its self-described architecture."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") == TinyLanguageModel.format_name:
        return TinyLanguageModel.load(path)
    if payload.get("format") == "simple-ai-tiny-transformer":
        from .transformer import TinyTransformerLanguageModel
        return TinyTransformerLanguageModel.load(path)
    raise ValueError(f"Unsupported language-model format: {payload.get('format')!r}")
