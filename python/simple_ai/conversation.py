"""Conversation-data utilities for the experimental character language model.

The format is intentionally plain text: it teaches dialogue boundaries as
characters, not semantic roles or instruction-following guarantees.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

USER_PREFIX = "Pengguna: "
ASSISTANT_PREFIX = "Asisten: "


def format_exchange(user: str, assistant: str) -> str:
    if not isinstance(user, str) or not user.strip():
        raise ValueError("Each conversation needs a non-empty 'user' string.")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError("Each conversation needs a non-empty 'assistant' string.")
    return f"{USER_PREFIX}{user.strip()}\n{ASSISTANT_PREFIX}{assistant.strip()}\n\n"


def load_conversations(path: str | Path) -> list[dict[str, str]]:
    """Read UTF-8 JSONL records with required user and assistant fields."""
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object.")
        try:
            rendered = format_exchange(item.get("user"), item.get("assistant"))
        except ValueError as error:
            raise ValueError(f"Invalid conversation on line {line_number}: {error}") from error
        records.append({"user": item["user"].strip(), "assistant": item["assistant"].strip(), "text": rendered})
    if not records:
        raise ValueError("Conversation dataset contains no valid records.")
    return records


def render_conversations(records: Iterable[dict[str, str]]) -> str:
    return "".join(format_exchange(record["user"], record["assistant"]) for record in records)


def build_chat_prompt(history: Iterable[tuple[str, str]], message: str) -> str:
    """Render prior turns plus the unfinished assistant prefix."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must not be empty.")
    parts = [format_exchange(user, assistant) for user, assistant in history]
    parts.append(f"{USER_PREFIX}{message.strip()}\n{ASSISTANT_PREFIX}")
    return "".join(parts)
