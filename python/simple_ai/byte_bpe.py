"""Fast byte-level BPE tokenizer backed by Hugging Face's Rust tokenizers.

The educational Python BPE implementation remains available in tokenizer.py.
This adapter is intended for larger cloud-training corpora.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

FORMAT_NAME = "simple-ai-byte-bpe-tokenizer"
FORMAT_VERSION = 1
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _tokenizers_api():
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError(
            "Byte-level BPE requires the optional 'tokenizers' package. "
            "Install it with: python -m pip install tokenizers"
        ) from exc
    return Tokenizer, decoders, models, pre_tokenizers, trainers


class ByteBPETokenizer:
    """Unicode-safe byte-level BPE tokenizer with explicit document tokens."""

    format_name = FORMAT_NAME
    format_version = FORMAT_VERSION

    def __init__(self, backend) -> None:
        self._backend = backend
        self._special_ids = {
            token: backend.token_to_id(token) for token in SPECIAL_TOKENS
        }
        if any(value is None for value in self._special_ids.values()):
            raise ValueError("Tokenizer backend is missing required special tokens.")

    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        *,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if vocab_size < 300:
            raise ValueError("Byte-level BPE vocab_size must be at least 300.")
        if min_frequency < 2:
            raise ValueError("min_frequency must be at least 2.")

        Tokenizer, decoders, models, pre_tokenizers, trainers = _tokenizers_api()
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            use_regex=True,
        )
        backend.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        materialized = [item for item in documents if isinstance(item, str) and item]
        if not materialized:
            raise ValueError("Cannot train a tokenizer on an empty document set.")
        backend.train_from_iterator(
            materialized,
            trainer=trainer,
            length=len(materialized),
        )
        return cls(backend)

    @property
    def pad_id(self) -> int:
        return int(self._special_ids["<pad>"])

    @property
    def unk_id(self) -> int:
        return int(self._special_ids["<unk>"])

    @property
    def bos_id(self) -> int:
        return int(self._special_ids["<bos>"])

    @property
    def eos_id(self) -> int:
        return int(self._special_ids["<eos>"])

    @property
    def vocab_size(self) -> int:
        return self._backend.get_vocab_size(with_added_tokens=True)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")
        ids = self._backend.encode(text, add_special_tokens=False).ids
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
        values = [int(token_id) for token_id in token_ids]
        return self._backend.decode(values, skip_special_tokens=skip_special_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format_name,
            "version": self.format_version,
            "special_tokens": list(SPECIAL_TOKENS),
            "backend": json.loads(self._backend.to_str()),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != cls.format_name:
            raise ValueError("Unsupported byte-BPE tokenizer format.")
        if payload.get("version") != cls.format_version:
            raise ValueError("Unsupported byte-BPE tokenizer version.")
        if payload.get("special_tokens") != list(SPECIAL_TOKENS):
            raise ValueError("Byte-BPE special-token definition does not match.")

        Tokenizer, *_ = _tokenizers_api()
        backend_data = payload.get("backend")
        if not isinstance(backend_data, dict):
            raise ValueError("Tokenizer file has no valid backend model.")
        backend = Tokenizer.from_str(
            json.dumps(backend_data, ensure_ascii=False)
        )
        return cls(backend)
