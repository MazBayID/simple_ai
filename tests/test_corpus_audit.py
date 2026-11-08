import pytest

from simple_ai.experiments.audit_corpus import analyze_text


def test_audit_counts_characters_and_unicode_categories():
    report = analyze_text("Halo, dunia!\n")
    assert report["characters"] == 13
    assert report["unique_characters"] > 1
    assert report["whitespace_characters"] == 2
    assert report["unicode_categories"]["Lu"] == 1
    assert report["script_prefixes"]["LATIN"] == 9


def test_audit_reports_non_latin_letters_without_filtering():
    report = analyze_text("Halo שלום")
    assert any(item["character"] == "ש" for item in report["non_latin_letters"])


def test_audit_reports_format_controls():
    report = analyze_text("A\u200bB")
    assert report["suspicious_characters"][0]["codepoint"] == "U+200B"


def test_audit_rejects_invalid_top_n():
    with pytest.raises(ValueError, match="top_n"):
        analyze_text("text", top_n=0)
