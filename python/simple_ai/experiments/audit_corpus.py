"""Inspect Unicode and text quality in a local training corpus.

This tool is diagnostic only: it never rewrites or filters the source corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata


def _script_name(character: str) -> str:
    name = unicodedata.name(character, "")
    if not name:
        return "UNKNOWN"
    first = name.split(" ", 1)[0]
    if first in {"LATIN", "GREEK", "CYRILLIC", "HEBREW", "ARABIC", "HIRAGANA",
                 "KATAKANA", "HANGUL", "CJK"}:
        return first
    if first in {"COMBINING"}:
        return "COMBINING"
    if character.isspace():
        return "WHITESPACE"
    return first


def analyze_text(text: str, *, top_n: int = 30) -> dict:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    categories = Counter(unicodedata.category(char) for char in text)
    scripts = Counter(_script_name(char) for char in text if not char.isspace())
    characters = Counter(text)
    non_latin = [
        {
            "character": char,
            "codepoint": f"U+{ord(char):04X}",
            "name": unicodedata.name(char, "UNNAMED"),
            "count": count,
        }
        for char, count in characters.most_common()
        if char.isalpha() and not unicodedata.name(char, "").startswith("LATIN ")
    ]
    suspicious = [
        {
            "character": char,
            "codepoint": f"U+{ord(char):04X}",
            "name": unicodedata.name(char, "UNNAMED"),
            "category": unicodedata.category(char),
            "count": count,
        }
        for char, count in characters.most_common()
        if unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"}
        or char == "\ufffd"
    ]
    return {
        "characters": len(text),
        "unique_characters": len(characters),
        "alphabetic_characters": sum(count for char, count in characters.items() if char.isalpha()),
        "whitespace_characters": sum(count for char, count in characters.items() if char.isspace()),
        "unicode_categories": dict(sorted(categories.items())),
        "script_prefixes": dict(scripts.most_common()),
        "top_characters": [
            {"character": char, "codepoint": f"U+{ord(char):04X}",
             "name": unicodedata.name(char, "UNNAMED"), "count": count}
            for char, count in characters.most_common(top_n)
        ],
        "non_latin_letters": non_latin[:top_n],
        "suspicious_characters": suspicious[:top_n],
        "warning": (
            "Diagnostic only. Review samples and provenance before filtering; "
            "some non-Latin characters may be legitimate names, quotations, or symbols."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Unicode composition of a text corpus.")
    parser.add_argument("--input", required=True, help="UTF-8 text file to inspect.")
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument("--top", type=int, default=30, help="Number of frequent characters to show.")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive.")

    source = Path(args.input)
    text = source.read_text(encoding="utf-8")
    report = {"source": str(source), **analyze_text(text, top_n=args.top)}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
