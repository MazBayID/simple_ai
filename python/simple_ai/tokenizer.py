"""Deterministic Unicode character tokenizer for early language-model experiments.

This is a deliberately simple baseline. It is not a subword/BPE tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


class CharTokenizer:
    """Map Unicode characters to stable integer IDs."""

    format_name = "simple-ai-char-tokenizer"
    format_version = 1

    def __init__(self, characters: Iterable[str]) -> None:
        unique = set(characters)
        if any(not isinstance(char, str) or len(char) != 1 for char in unique):
            raise ValueError("Vocabulary entries must be single Unicode characters.")
        self.characters = tuple(sorted(unique))
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.token_to_id.update({
            char: len(SPECIAL_TOKENS) + index
            for index, char in enumerate(self.characters)
        })
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not isinstance(text, str):
            raise TypeError("Training text must be a string.")
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text.")
        return cls(text)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        output: list[str] = []
        special_ids = set(range(len(SPECIAL_TOKENS)))
        for token_id in token_ids:
            if not isinstance(token_id, int) or token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id!r}")
            if token_id in special_ids:
                if not skip_special_tokens and token_id == self.unk_id:
                    output.append("�")
                elif not skip_special_tokens:
                    output.append(self.id_to_token[token_id])
                continue
            output.append(self.id_to_token[token_id])
        return "".join(output)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format_name,
            "version": self.format_version,
            "special_tokens": list(SPECIAL_TOKENS),
            "characters": list(self.characters),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name:
            raise ValueError("Unsupported tokenizer format.")
        if payload.get("version") != cls.format_version:
            raise ValueError("Unsupported tokenizer version.")
        if payload.get("special_tokens") != list(SPECIAL_TOKENS):
            raise ValueError("Tokenizer special-token definition does not match.")
        characters = payload.get("characters")
        if not isinstance(characters, list):
            raise ValueError("Tokenizer file has no valid character vocabulary.")
        tokenizer = cls(characters)
        if list(tokenizer.characters) != characters:
            raise ValueError("Tokenizer vocabulary must be unique and sorted.")
        return tokenizer
