"""Tests for the CLI (__main__.py), file-type routing (scanner.py), and report helpers."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.make_samples import PAYLOAD_MZ, generate_all, generate_text_samples
from stegoscan.__main__ import EXIT_CLEAN, EXIT_ERROR, EXIT_FLAGGED, main
from stegoscan.image_scan import scan_image
from stegoscan.report import UNREADABLE_CHECK, hex_preview, is_unreadable, report_to_dict
from stegoscan.scanner import IMAGE, PCAP, TEXT, UNKNOWN, detect_file_type

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def samples_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("images")
    generate_all(directory)
    return directory


# --- detect_file_type ---------------------------------------------------------

@pytest.mark.parametrize("name", ["clean_rgb.png", "clean.jpg"])
def test_detects_images_by_magic(samples_dir, name):
    assert detect_file_type(samples_dir / name) == IMAGE


def test_renamed_png_is_still_an_image(samples_dir, tmp_path):
    disguised = tmp_path / "notes.txt"
    disguised.write_bytes((samples_dir / "clean_rgb.png").read_bytes())
    assert detect_file_type(disguised) == IMAGE


@pytest.mark.parametrize("magic", [b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a"])
def test_detects_pcap_by_magic(tmp_path, magic):
    capture = tmp_path / "capture.bin"
    capture.write_bytes(magic + b"\x00" * 20)
    assert detect_file_type(capture) == PCAP


def test_detects_utf8_text(tmp_path):
    note = tmp_path / "note.dat"
    note.write_text("hello world \t\nzero width\u200b here\n", encoding="utf-8")
    assert detect_file_type(note) == TEXT


def test_unknown_binary(tmp_path):
    blob = tmp_path / "blob.xyz"
    blob.write_bytes(bytes(range(256)))
    assert detect_file_type(blob) == UNKNOWN


def test_real_bmp_is_an_image(tmp_path):
    bmp = tmp_path / "picture.dat"
    Image.new("RGB", (8, 8), "red").save(bmp, format="BMP")
    assert detect_file_type(bmp) == IMAGE


def test_text_starting_with_bm_is_text(tmp_path):
    note = tmp_path / "notes.dat"
    note.write_text("BMW service notes: oil change due at 60k miles.\n")
    assert detect_file_type(note) == TEXT


def test_empty_file_falls_back_to_extension(samples_dir):
    assert detect_file_type(samples_dir / "empty.png") == IMAGE


# --- is_unreadable ------------------------------------------------------------

def test_is_unreadable_uses_shared_check_name(samples_dir):
    report = scan_image(samples_dir / "corrupt.png")
    assert UNREADABLE_CHECK in {f.check for f in report.findings}
    assert is_unreadable(report)
    assert not is_unreadable(scan_image(samples_dir / "clean_rgb.png"))


# --- hex_preview --------------------------------------------------------------

def test_hex_preview_shows_hex_and_ascii():
    preview = hex_preview(b"MZ\x00\x01hello")
    assert preview.startswith("00000000  4d 5a 00 01 68 65 6c 6c")
    assert preview.endswith("|MZ..hello|")


def test_hex_preview_truncates():
    assert "(36 more bytes)" in hex_preview(b"A" * 100, length=64)


# --- main(): verdicts and exit codes ------------------------------------------

def test_clean_image_exits_zero(samples_dir, capsys):
    assert main(["scan", str(samples_dir / "clean_rgb.png")]) == EXIT_CLEAN
    assert "CLEAN" in capsys.readouterr().out


def test_stego_image_exits_one_with_preview(samples_dir, capsys):
    assert main(["scan", str(samples_dir / "stego_full_rgb.png")]) == EXIT_FLAGGED
    out = capsys.readouterr().out
    assert "LIKELY_PAYLOAD" in out
    assert "|MZSTEGOSCAN TEST|" in out


@pytest.mark.parametrize("name", ["corrupt.png", "empty.png"])
def test_unreadable_image_is_an_error_not_clean(samples_dir, capsys, name):
    assert main(["scan", str(samples_dir / name)]) == EXIT_ERROR
    captured = capsys.readouterr()
    assert "CLEAN" not in captured.out
    assert "could not analyze" in captured.err


def test_missing_file(tmp_path, capsys):
    assert main(["scan", str(tmp_path / "nope.png")]) == EXIT_ERROR
    assert "file not found" in capsys.readouterr().err


def test_clean_text_exits_zero(tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text("hello\n")
    assert main(["scan", str(note)]) == EXIT_CLEAN
    assert "(text)" in capsys.readouterr().out


def test_text_payload_extracted_via_cli(tmp_path, capsys):
    text_dir = tmp_path / "text"
    generate_text_samples(text_dir)
    out_dir = tmp_path / "out"
    code = main(["scan", str(text_dir / "ws_payload.txt"), "--extract", str(out_dir)])
    assert code == EXIT_FLAGGED
    (saved,) = out_dir.iterdir()
    assert saved.read_bytes() == PAYLOAD_MZ
    assert saved.name == "ws_payload.whitespace_space_0_tab_1.bin"


def test_binary_text_file_is_an_error(tmp_path, capsys):
    blob = tmp_path / "data.txt"
    blob.write_bytes(b"\x00\x01 not text")
    assert main(["scan", str(blob)]) == EXIT_ERROR
    assert "could not analyze" in capsys.readouterr().err


def test_pcap_not_supported_yet(tmp_path, capsys):
    capture = tmp_path / "capture.pcap"
    capture.write_bytes(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)
    assert main(["scan", str(capture)]) == EXIT_ERROR
    assert "Phase 3" in capsys.readouterr().err


def test_unknown_type_is_an_error(tmp_path, capsys):
    blob = tmp_path / "blob.xyz"
    blob.write_bytes(bytes(range(256)))
    assert main(["scan", str(blob)]) == EXIT_ERROR
    assert "Unrecognized" in capsys.readouterr().err


# --- --json -------------------------------------------------------------------

def test_json_output(samples_dir, capsys):
    main(["scan", str(samples_dir / "stego_full_rgb.png"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "LIKELY_PAYLOAD"
    assert data["file_type"] == "image"
    assert data["extracted_length"] == len(PAYLOAD_MZ)
    assert data["extracted_sha256"] == hashlib.sha256(PAYLOAD_MZ).hexdigest()
    assert "extracted" not in data  # raw bytes never go in JSON
    assert {"check", "severity", "detail", "evidence"} <= set(data["findings"][0])


def test_report_dict_is_strict_json_for_every_sample(samples_dir):
    """Plain json.dumps (no default=str fallback) must accept every report, so a
    stray numpy type in evidence fails here instead of being silently stringified."""
    for sample in sorted(samples_dir.iterdir()):
        json.dumps(report_to_dict(scan_image(sample)))


# --- --extract ----------------------------------------------------------------

def test_extract_saves_payload_byte_for_byte(samples_dir, tmp_path, capsys):
    out_dir = tmp_path / "out"
    main(["scan", str(samples_dir / "stego_full_rgb.png"), "--extract", str(out_dir)])

    saved = list(out_dir.iterdir())
    assert len(saved) == 1
    payload = saved[0]
    assert payload.suffix == ".bin"
    assert payload.read_bytes() == scan_image(samples_dir / "stego_full_rgb.png").extracted
    assert payload.read_bytes() == PAYLOAD_MZ
    assert not os.access(payload, os.X_OK)
    assert "Saved payload" in capsys.readouterr().out


def test_extract_with_json_keeps_stdout_pure(samples_dir, tmp_path, capsys):
    main(["scan", str(samples_dir / "appended_png.png"), "--json", "--extract", str(tmp_path)])
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "Saved payload" in captured.err


def test_extract_clean_image_writes_nothing(samples_dir, tmp_path, capsys):
    out_dir = tmp_path / "out"
    main(["scan", str(samples_dir / "clean_rgb.png"), "--extract", str(out_dir)])
    assert not out_dir.exists()
    assert "Nothing to extract" in capsys.readouterr().out


# --- end to end ---------------------------------------------------------------

def test_python_dash_m_runs(samples_dir):
    result = subprocess.run(
        [sys.executable, "-m", "stegoscan", "scan", str(samples_dir / "stego_full_rgb.png")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == EXIT_FLAGGED
    assert "LIKELY_PAYLOAD" in result.stdout
