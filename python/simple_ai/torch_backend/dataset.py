"""Document-aware next-token batches for the PyTorch backend."""

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
    # Legacy corpus files separate records with blank lines. This fallback
    # may also split long articles at paragraph boundaries.
    return [part.strip() for part in text.split("\n\n") if part.strip()]


class DocumentTokenDataset(Dataset):
    """Fixed-length language-model windows that never cross document edges.

    For every window, input_ids contains tokens [i:i+context_size] and
    targets contains the one-token-shifted sequence [i+1:i+context_size+1].
    The tokenizer should add BOS and EOS independently to each document.
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
