"""A tiny character-level language model for learning and smoke testing.

This is deliberately a toy model, not a useful conversational LLM.
It trains a first-order character transition table from scratch and needs
very little memory. The model's tiny size is intentional for the first
Termux milestone under the project's strict resource constraints.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random


class TinyCharModel:
    """Character-level next-token model using observed transition counts."""

    def __init__(self) -> None:
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.trained = False

    def train(self, text: str) -> dict[str, int]:
        """Train from text and return basic dataset statistics."""
        if not text:
            raise ValueError("Training text must not be empty.")

        self.transitions.clear()
        self.vocab = set(text)
        previous = "\0"

        for character in text:
            self.transitions[previous][character] += 1
            previous = character

        self.trained = True
        return {
            "characters": len(text),
            "vocabulary_size": len(self.vocab),
            "transition_contexts": len(self.transitions),
        }

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        """Generate characters from learned transitions."""
        if not self.trained:
            raise RuntimeError("Model is not trained or loaded.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        rng = random.Random(seed)
        output = list(prompt)
        previous = output[-1] if output else "\0"

        for _ in range(max_new_tokens):
            choices = self.transitions.get(previous)
            if not choices:
                choices = self.transitions.get("\0")
            if not choices:
                break

            chars = list(choices.keys())
            weights = [count ** (1.0 / temperature) for count in choices.values()]
            next_character = rng.choices(chars, weights=weights, k=1)[0]
            output.append(next_character)
            previous = next_character

        return "".join(output)

    def save(self, path: str | Path) -> None:
        """Save model counts as a portable JSON checkpoint."""
        if not self.trained:
            raise RuntimeError("Cannot save an untrained model.")

        data = {
            "format": "simple-ai-tiny-char-v1",
            "vocab": sorted(self.vocab),
            "transitions": {
                context: dict(counts)
                for context, counts in self.transitions.items()
            },
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyCharModel":
        """Load a checkpoint produced by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "simple-ai-tiny-char-v1":
            raise ValueError("Unsupported checkpoint format.")

        model = cls()
        model.vocab = set(data["vocab"])
        for context, counts in data["transitions"].items():
            model.transitions[context] = Counter(counts)
        model.trained = True
        return model
