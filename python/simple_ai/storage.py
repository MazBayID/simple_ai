"""Small SQLite storage layer; works with Python's standard library."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class LocalStore:
    """Persist simple training runs and conversation text locally."""

    def __init__(self, path: str | Path = "simple_ai.db") -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    source_name TEXT NOT NULL,
                    characters INTEGER NOT NULL,
                    vocabulary_size INTEGER NOT NULL,
                    transition_contexts INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL
                );
                """
            )

    def record_training(
        self,
        source_name: str,
        stats: dict[str, int],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO training_runs
                (source_name, characters, vocabulary_size, transition_contexts)
                VALUES (?, ?, ?, ?)
                """,
                (
                    source_name,
                    stats["characters"],
                    stats["vocabulary_size"],
                    stats["transition_contexts"],
                ),
            )
            return int(cursor.lastrowid)

    def add_message(self, role: str, content: str) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be user, assistant, or system")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (role, content) VALUES (?, ?)",
                (role, content),
            )
            return int(cursor.lastrowid)

    def recent_messages(self, limit: int = 20) -> list[tuple[str, str]]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM messages
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(reversed(rows))
