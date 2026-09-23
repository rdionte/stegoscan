"""Tests for text_scan.py: trailing-whitespace and zero-width Unicode steganography."""

from pathlib import Path

import numpy as np
import pytest

from scripts.make_samples import HIDDEN_MESSAGE, PAYLOAD_MZ, generate_text_samples
from stegoscan.bits import bits_to_bytes
from stegoscan.decoding import looks_like_message
from stegoscan.report import CLEAN, LIKELY_PAYLOAD, SUSPICIOUS, UNREADABLE_CHECK
from stegoscan.text_scan import (
    decode_whitespace,
    decode_zero_width,
    find_zero_width,
    scan_text,
    trailing_whitespace,
    whitespace_stats_finding,
)

# Built with chr() so this file contains no invisible characters.
ZWSP, ZWNJ, ZWJ, BOM = chr(0x200B), chr(0x200C), chr(0x200D), chr(0xFEFF)
EMOJI_MAN, EMOJI_WOMAN = chr(0x1F468), chr(0x1F469)


@pytest.fixture(scope="module")
def text_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("text")
    generate_text_samples(directory)
    return directory


def _checks(report):
    return {f.check for f in report.findings}


# --- bits_to_bytes ----------------------------------------------------------------

def test_bits_to_bytes_msb_first_and_drops_leftovers():
    bits = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)  # 'M' + 2 leftover bits
    assert bits_to_bytes(bits) == b"M"


def test_bits_to_bytes_respects_max_bytes():
    assert bits_to_bytes(np.ones(32, dtype=np.uint8), max_bytes=2) == b"\xff\xff"


# --- trailing whitespace --------------------------------------------------------

def test_trailing_whitespace_ignores_crlf():
    assert trailing_whitespace("a \t\r\nb\nc\t") == [" \t", "", "\t"]


def test_decode_whitespace_both_mappings():
    runs = [" \t  ", "\t\t \t"]  # 01001101 = 'M' with space=0
    decoded = decode_whitespace(runs)
    assert decoded["space=0/tab=1"] == b"M"
    assert decoded["tab=0/space=1"] == bytes([0b10110010])


def test_markdown_style_breaks_are_formatting():
    finding = whitespace_stats_finding(["  "] * 30, explained=False)
    assert finding.check == "whitespace_formatting"
    assert finding.severity == "info"


def test_repeated_mixed_indentation_is_formatting():
    """Many mixed space/tab lines, but all the same pattern: indentation, not data."""
    finding = whitespace_stats_finding(["\t  "] * 30, explained=False)
    assert finding.severity == "info"


def test_varied_mixed_whitespace_is_data_like():
    rng = np.random.default_rng(0)
    runs = ["".join(rng.choice([" ", "\t"], size=8)) for _ in range(40)]
    finding = whitespace_stats_finding(runs, explained=False)
    assert (finding.check, finding.severity) == ("whitespace_data_like", "high")
    assert whitespace_stats_finding(runs, explained=True).severity == "info"


# --- zero-width -------------------------------------------------------------------

def test_zero_width_locations():
    text = "ab" + ZWSP + "c\nde" + ZWNJ + "f"
    hits = find_zero_width(text)
    assert [(h.line, h.column, h.char) for h in hits] == [(1, 3, ZWSP), (2, 3, ZWNJ)]
    assert not any(h.legitimate for h in hits)


def test_bom_only_legitimate_at_start():
    hits = find_zero_width(BOM + "hello" + BOM + "world")
    assert [h.legitimate for h in hits] == [True, False]


def test_zwj_between_emoji_is_legitimate_but_not_between_letters():
    hits = find_zero_width(EMOJI_MAN + ZWJ + EMOJI_WOMAN + " a" + ZWJ + "b")
    assert [h.legitimate for h in hits] == [True, False]


def test_decode_zero_width_tries_both_orders():
    bits = "01001101"  # 'M'
    text = "".join(ZWNJ if b == "1" else ZWSP for b in bits)
    decoded = decode_zero_width(find_zero_width(text))
    assert decoded["U+200B=0/U+200C=1"] == b"M"
    assert decoded["U+200C=0/U+200B=1"] == bytes([0b10110010])


# --- looks_like_message ---------------------------------------------------------

def test_looks_like_message():
    assert looks_like_message(b"meet me later\n")
    assert not looks_like_message(b"hi")  # too short
    assert not looks_like_message(bytes(range(128, 160)))  # not printable


# --- scan_text on samples ---------------------------------------------------------

@pytest.mark.parametrize("name", [
    "clean_prose.txt", "clean_markdown.md", "clean_code.py", "clean_crlf.txt", "clean_emoji.txt",
])
def test_clean_samples_are_clean(text_dir, name):
    report = scan_text(text_dir / name)
    assert report.verdict == CLEAN
    assert report.score == 0
    assert report.extracted is None


def test_emoji_and_persian_joiners_are_ignored(text_dir):
    report = scan_text(text_dir / "clean_emoji.txt")
    zero_width = next(f for f in report.findings if f.check == "zero_width_chars")
    assert zero_width.evidence["suspicious_count"] == 0
    assert zero_width.evidence["legitimate_count"] >= 8  # would trip the check if not ignored


@pytest.mark.parametrize("name, method", [
    ("ws_payload.txt", "whitespace:space=0/tab=1"),
    ("ws_payload_reversed.txt", "whitespace:tab=0/space=1"),
    ("ws_partial_10.txt", "whitespace:space=0/tab=1"),
    ("zw_payload.txt", "zero_width:U+200B=0/U+200C=1"),
])
def test_payloads_extracted_byte_for_byte(text_dir, name, method):
    report = scan_text(text_dir / name)
    assert report.verdict == LIKELY_PAYLOAD
    assert report.extracted == PAYLOAD_MZ
    assert report.extraction_method == method
    assert "signature_mz" in _checks(report)


@pytest.mark.parametrize("name", ["ws_message.txt", "zw_message.txt"])
def test_hidden_messages_extracted_and_suspicious(text_dir, name):
    report = scan_text(text_dir / name)
    assert report.verdict == SUSPICIOUS
    assert report.extracted == HIDDEN_MESSAGE


def test_partial_embedding_uses_about_ten_percent_of_lines(text_dir):
    text = (text_dir / "ws_partial_10.txt").read_text()
    runs = trailing_whitespace(text)
    used = sum(1 for run in runs if run) / len([line for line in text.split("\n") if line])
    assert 0.05 <= used <= 0.15


def test_random_whitespace_flagged_without_decoding(text_dir):
    report = scan_text(text_dir / "ws_random.txt")
    assert report.verdict == SUSPICIOUS
    assert report.extracted is None
    assert "whitespace_data_like" in _checks(report)


def test_decoded_anomaly_is_not_scored_twice(text_dir):
    """The whitespace statistics become 'info' once the whitespace decodes."""
    report = scan_text(text_dir / "ws_message.txt")
    data_like = next(f for f in report.findings if f.check == "whitespace_data_like")
    assert data_like.severity == "info"


def test_zero_width_location_evidence(text_dir):
    report = scan_text(text_dir / "zw_payload.txt")
    zero_width = next(f for f in report.findings if f.check == "zero_width_chars")
    text = (text_dir / "zw_payload.txt").read_text(encoding="utf-8")
    lines = text.split("\n")
    for location in zero_width.evidence["locations"]:
        char = lines[location["line"] - 1][location["column"] - 1]
        assert f"U+{ord(char):04X}" == location["char"]


# --- edge cases --------------------------------------------------------------------

def test_empty_file_is_clean(text_dir):
    report = scan_text(text_dir / "empty.txt")
    assert report.verdict == CLEAN
    assert "empty_file" in _checks(report)


def test_binary_file_is_unreadable(text_dir):
    assert UNREADABLE_CHECK in _checks(scan_text(text_dir / "binary.txt"))


def test_invalid_utf8_is_unreadable(tmp_path):
    bad = tmp_path / "latin1.txt"
    bad.write_bytes("caf\xe9 au lait".encode("latin-1"))
    assert UNREADABLE_CHECK in _checks(scan_text(bad))


def test_missing_file_is_unreadable(tmp_path):
    assert UNREADABLE_CHECK in _checks(scan_text(tmp_path / "gone.txt"))
