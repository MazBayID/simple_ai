import json

import pytest

from simple_ai.conversation import (
    ASSISTANT_PREFIX,
    USER_PREFIX,
    build_chat_prompt,
    format_exchange,
    load_conversations,
    render_conversations,
)


def test_format_exchange_has_stable_indonesian_roles():
    assert format_exchange(" Halo ", " Hai ") == (
        f"{USER_PREFIX}Halo\n{ASSISTANT_PREFIX}Hai\n\n"
    )


def test_load_and_render_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"user": "Apa kabar?", "assistant": "Baik."}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = load_conversations(path)
    assert len(records) == 1
    assert "Pengguna: Apa kabar?" in render_conversations(records)


def test_invalid_jsonl_has_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_conversations(path)


def test_empty_conversation_dataset_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no valid records"):
        load_conversations(path)


def test_empty_messages_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        format_exchange("", "jawaban")


def test_chat_prompt_includes_history_and_open_assistant_prefix():
    prompt = build_chat_prompt([("Halo", "Hai")], "Apa kabar?")
    assert prompt.endswith("Pengguna: Apa kabar?\nAsisten: ")
    assert "Pengguna: Halo\nAsisten: Hai" in prompt
