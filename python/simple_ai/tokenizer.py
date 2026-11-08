"""Deterministic Unicode character and byte-pair tokenizers.

The BPE implementation is intentionally small and educational. It learns
subword merges from a local corpus and uses no external tokenizer dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from collections.abc import Iterable, Sequence

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
        self.tokens = self.characters
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.token_to_id.update({char: len(SPECIAL_TOKENS) + index
                                 for index, char in enumerate(self.characters)})
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

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        output: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id!r}")
            token = self.id_to_token[token_id]
            if token_id < len(SPECIAL_TOKENS):
                if not skip_special_tokens and token_id == self.unk_id:
                    output.append("�")
                elif not skip_special_tokens:
                    output.append(token)
            else:
                output.append(token)
        return "".join(output)

    def to_dict(self) -> dict[str, object]:
        return {"format": self.format_name, "version": self.format_version,
                "special_tokens": list(SPECIAL_TOKENS), "characters": list(self.characters)}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported tokenizer format or version.")
        if payload.get("special_tokens") != list(SPECIAL_TOKENS):
            raise ValueError("Tokenizer special-token definition does not match.")
        characters = payload.get("characters")
        if not isinstance(characters, list):
            raise ValueError("Tokenizer file has no valid character vocabulary.")
        tokenizer = cls(characters)
        if list(tokenizer.characters) != characters:
            raise ValueError("Tokenizer vocabulary must be unique and sorted.")
        return tokenizer


class BPETokenizer:
    """A compact byte-pair-style tokenizer trained over Unicode characters.

    Each merge replaces the most frequent adjacent pair. Ties are resolved
    lexicographically, making training deterministic across runs.
    """

    format_name = "simple-ai-bpe-tokenizer"
    format_version = 1

    def __init__(self, characters: Sequence[str], merges: Sequence[Sequence[str]]) -> None:
        if any(not isinstance(char, str) or len(char) != 1 for char in characters):
            raise ValueError("BPE base vocabulary must contain single characters.")
        self.characters = tuple(sorted(set(characters)))
        if len(self.characters) != len(characters):
            raise ValueError("BPE base characters must be unique.")
        self.merges = tuple((str(pair[0]), str(pair[1])) for pair in merges)
        tokens = list(self.characters)
        for left, right in self.merges:
            merged = left + right
            if left not in tokens or right not in tokens or merged in tokens:
                raise ValueError("BPE merges must reference existing tokens and create new tokens.")
            tokens.append(merged)
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.token_to_id.update({token: len(SPECIAL_TOKENS) + i for i, token in enumerate(self.tokens)})
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @classmethod
    def train(cls, text: str, *, vocab_size: int = 256, min_frequency: int = 2) -> "BPETokenizer":
        if not isinstance(text, str):
            raise TypeError("Training text must be a string.")
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text.")
        if vocab_size < len(SPECIAL_TOKENS) + len(set(text)):
            raise ValueError("vocab_size is smaller than the special and base character vocabulary.")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2.")
        characters = tuple(sorted(set(text)))
        sequence = list(text)
        merges: list[tuple[str, str]] = []
        while len(SPECIAL_TOKENS) + len(characters) + len(merges) < vocab_size:
            counts = Counter(zip(sequence, sequence[1:]))
            if not counts:
                break
            best_count = max(counts.values())
            if best_count < min_frequency:
                break
            existing_tokens = set(characters)
            for left, right in merges:
                existing_tokens.add(left + right)
            candidates = [
                pair for pair, count in counts.items()
                if count == best_count and pair[0] + pair[1] not in existing_tokens
            ]
            if not candidates:
                break
            best_pair = min(candidates)
            left, right = best_pair
            merged = left + right
            output: list[str] = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
                    output.append(merged)
                    index += 2
                else:
                    output.append(sequence[index])
                    index += 1
            sequence = output
            merges.append(best_pair)
        return cls(characters, merges)

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

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")
        pieces = list(text)
        for left, right in self.merges:
            merged = left + right
            output: list[str] = []
            index = 0
            while index < len(pieces):
                if index + 1 < len(pieces) and pieces[index] == left and pieces[index + 1] == right:
                    output.append(merged)
                    index += 2
                else:
                    output.append(pieces[index])
                    index += 1
            pieces = output
        ids = [self.token_to_id.get(piece, self.unk_id) for piece in pieces]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        output: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id!r}")
            token = self.id_to_token[token_id]
            if token_id < len(SPECIAL_TOKENS):
                if not skip_special_tokens and token_id == self.unk_id:
                    output.append("�")
                elif not skip_special_tokens:
                    output.append(token)
            else:
                output.append(token)
        return "".join(output)

    def to_dict(self) -> dict[str, object]:
        return {"format": self.format_name, "version": self.format_version,
                "special_tokens": list(SPECIAL_TOKENS), "characters": list(self.characters),
                "merges": [list(pair) for pair in self.merges]}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name or payload.get("version") != cls.format_version:
            raise ValueError("Unsupported tokenizer format or version.")
        if payload.get("special_tokens") != list(SPECIAL_TOKENS):
            raise ValueError("Tokenizer special-token definition does not match.")
        return cls(payload.get("characters", []), payload.get("merges", []))


def load_tokenizer(path: str | Path) -> CharTokenizer | BPETokenizer:
    """Load a supported tokenizer using its self-described file format."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer_format = payload.get("format")
    if tokenizer_format == CharTokenizer.format_name:
        return CharTokenizer.load(path)
    if tokenizer_format == BPETokenizer.format_name:
        return BPETokenizer.load(path)
    raise ValueError(f"Unsupported tokenizer format: {tokenizer_format!r}")
