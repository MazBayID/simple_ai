"""Document-aware and packed-stream next-token datasets for PyTorch."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_documents(path: str | Path) -> list[str]:
    """Load documents from JSONL or the legacy plain-text corpus format.

    JSONL files should contain one JSON object per line with a "text" field
    and an optional "title" field. Plain-text files use blank lines as
    document separators; JSONL is preferred because it preserves article
    boundaries exactly.
    """
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        documents: list[str] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {source}."
                    ) from exc
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(
                        f"Each JSONL record in {source} must contain string field 'text'."
                    )
                title = record.get("title", "")
                if title is not None and not isinstance(title, str):
                    raise ValueError(
                        f"Document title on line {line_number} must be a string."
                    )
                content = f"{title.strip()}\n{record['text'].strip()}".strip()
                if content:
                    documents.append(content)
        return documents

    text = source.read_text(encoding="utf-8")
    return [part.strip() for part in text.split("\n\n") if part.strip()]


class DocumentTokenDataset(Dataset):
    """Fixed-length windows that never cross individual document edges.

    Retained for the original v1 training pipeline and compatibility tests.
    """

    def __init__(
        self,
        documents: list[str],
        tokenizer: Any,
        *,
        context_size: int,
    ) -> None:
        if context_size < 1:
            raise ValueError("context_size must be >= 1.")

        self.context_size = context_size
        self.encoded_documents: list[list[int]] = []
        self.cumulative_windows: list[int] = []
        total_windows = 0
        self.skipped_documents = 0

        for document in documents:
            if not isinstance(document, str):
                raise TypeError("Every document must be a string.")
            if not document.strip():
                self.skipped_documents += 1
                continue

            token_ids = tokenizer.encode(
                document, add_bos=True, add_eos=True
            )
            windows = len(token_ids) - context_size
            if windows < 1:
                self.skipped_documents += 1
                continue

            self.encoded_documents.append(token_ids)
            total_windows += windows
            self.cumulative_windows.append(total_windows)

        self.total_windows = total_windows

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total_windows
        if index < 0 or index >= self.total_windows:
            raise IndexError("DocumentTokenDataset index out of range.")

        document_index = bisect_right(self.cumulative_windows, index)
        previous_total = (
            self.cumulative_windows[document_index - 1]
            if document_index > 0 else 0
        )
        start = index - previous_total
        tokens = self.encoded_documents[document_index]

        inputs = torch.tensor(
            tokens[start:start + self.context_size], dtype=torch.long
        )
        targets = torch.tensor(
            tokens[start + 1:start + self.context_size + 1],
            dtype=torch.long,
        )
        return inputs, targets


class PackedTokenDataset(Dataset):
    """Pack all documents into one token stream separated by EOS/BOS markers.

    Windows use a configurable stride. The default stride equals context_size,
    so each epoch processes nearly every token once instead of creating a
    heavily overlapping window for every token position. A window may span a
    document boundary, but the transition is explicit: EOS followed by BOS.
    """

    def __init__(
        self,
        documents: list[str],
        tokenizer: Any,
        *,
        context_size: int,
        stride: int | None = None,
    ) -> None:
        if context_size < 1:
            raise ValueError("context_size must be >= 1.")
        if stride is None:
            stride = context_size
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        self.context_size = context_size
        self.stride = stride
        self.skipped_documents = 0
        self.document_count = 0
        token_stream: list[int] = []

        for document in documents:
            if not isinstance(document, str):
                raise TypeError("Every document must be a string.")
            if not document.strip():
                self.skipped_documents += 1
                continue
            token_stream.extend(
                tokenizer.encode(document, add_bos=True, add_eos=True)
            )
            self.document_count += 1

        self.tokens = token_stream
        self.total_tokens = len(token_stream)
        self.max_start = self.total_tokens - context_size - 1
        if self.max_start >= 0:
            regular_windows = self.max_start // stride + 1
            last_regular_start = (regular_windows - 1) * stride
            self.has_tail_window = last_regular_start < self.max_start
            self.total_windows = regular_windows + int(self.has_tail_window)
        else:
            self.has_tail_window = False
            self.total_windows = 0

        # A final overlapping window ensures the stream's last token (normally
        # EOS) is included as a next-token target instead of being discarded.
        self.dropped_tokens = 0 if self.total_windows else self.total_tokens

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.total_windows
        if index < 0 or index >= self.total_windows:
            raise IndexError("PackedTokenDataset index out of range.")

        if self.has_tail_window and index == self.total_windows - 1:
            start = self.max_start
        else:
            start = index * self.stride
        inputs = torch.tensor(
            self.tokens[start:start + self.context_size], dtype=torch.long
        )
        targets = torch.tensor(
            self.tokens[start + 1:start + self.context_size + 1],
            dtype=torch.long,
        )
        return inputs, targets
