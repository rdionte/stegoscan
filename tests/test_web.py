"""Tests for the Flask web UI (web.py): display, upload limits, and safety rules."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.make_samples import (
    PAYLOAD_MZ, embed_whitespace, generate_all, generate_pcap_samples,
    generate_text_samples, make_cover_lines,
)
from stegoscan import web


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> Path:
    """All image, text, and pcap fixtures in one directory."""
    directory = tmp_path_factory.mktemp("web_samples")
    generate_all(directory)
    generate_text_samples(directory)
    generate_pcap_samples(directory)
    return directory


@pytest.fixture
def client():
    return web.create_app().test_client()


def upload(client, path: Path, url: str = "/scan", name: str | None = None):
    """POST a file the way a browser form would."""
    data = {"file": (io.BytesIO(path.read_bytes()), name or path.name)}
    return client.post(url, data=data, content_type="multipart/form-data")


def plane_count(html: str) -> int:
    return html.count('alt="Bit-plane')


# --- pages ----------------------------------------------------------------------

def test_index_shows_upload_form(client):
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'type="file"' in html and 'enctype="multipart/form-data"' in html
    assert "10 MB" in html


def test_clean_png_is_clean_with_bit_planes(client, samples):
    html = upload(client, samples / "stats_clean.png").get_data(as_text=True)
    assert "verdict-clean" in html
    assert "severity-high" not in html
    assert plane_count(html) == 8


def test_stego_png_shows_payload_and_hex_preview(client, samples):
    response = upload(client, samples / "stego_full_rgb.png")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "verdict-likely_payload" in html
    assert "|MZSTEGOSCAN TEST" in html  # the ASCII column of the hex dump
    assert "4d 5a" in html
    assert plane_count(html) == 8


@pytest.mark.parametrize("name", ["ws_payload.txt", "zw_message.txt", "icmp_payload.pcap"])
def test_text_and_pcap_stego_are_flagged_without_planes(client, samples, name):
    html = upload(client, samples / name).get_data(as_text=True)
    assert "verdict-clean" not in html
    assert "verdict-" in html
    assert plane_count(html) == 0


def test_jpeg_notes_lsb_limit_and_has_no_planes(client, samples):
    html = upload(client, samples / "clean.jpg").get_data(as_text=True)
    assert "lsb_checks_skipped" in html
    assert plane_count(html) == 0


# --- errors ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["empty.png", "corrupt.png"])
def test_unreadable_files_fail_gracefully(client, samples, name):
    response = upload(client, samples / name)
    assert response.status_code == 422
    assert "Could not analyze" in response.get_data(as_text=True)


def test_unknown_type_is_rejected(client, tmp_path):
    blob = tmp_path / "mystery.bin"
    blob.write_bytes(bytes(range(256)) * 4)
    response = upload(client, blob)
    assert response.status_code == 400
    assert "Unrecognized file type" in response.get_data(as_text=True)


def test_missing_file_is_rejected(client):
    response = client.post("/scan", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No file was uploaded" in response.get_data(as_text=True)


def test_upload_over_limit_is_rejected(tmp_path):
    small_client = web.create_app(max_upload_mb=1).test_client()
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (1024 * 1024 + 1))
    response = upload(small_client, big)
    assert response.status_code == 413
    assert "1 MB upload limit" in response.get_data(as_text=True)


# --- safety ---------------------------------------------------------------------

def test_hidden_html_is_escaped_not_rendered(client, tmp_path):
    """A decoded message is attacker-controlled; it must show as text, not run as HTML."""
    rng = np.random.default_rng(99)
    message = b"<script>alert('stegoscan')</script> hidden note"
    lines = embed_whitespace(make_cover_lines(60, rng), message, rng)
    carrier = tmp_path / "xss.txt"
    carrier.write_text("\n".join(lines) + "\n")

    html = upload(client, carrier).get_data(as_text=True)
    assert "verdict-suspicious" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_upload_is_deleted_after_scan(client, samples, tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(upload_root))
    upload(client, samples / "stego_full_rgb.png")
    upload(client, samples / "empty.png")
    assert list(upload_root.iterdir()) == []


def test_path_traversal_filename_stays_in_temp_dir(client, samples, tmp_path, monkeypatch):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(upload_root))
    response = upload(client, samples / "clean_rgb.png", name="../../escape.png")
    assert response.status_code == 200
    assert not (tmp_path / "escape.png").exists()
    assert list(upload_root.iterdir()) == []


def test_security_headers_present(client):
    response = client.get("/")
    assert response.headers["Content-Security-Policy"] == web.CONTENT_SECURITY_POLICY
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_main_binds_localhost_without_debug(monkeypatch):
    calls = {}
    monkeypatch.setattr(web.Flask, "run", lambda self, **kwargs: calls.update(kwargs))
    web.main([])
    assert calls == {"host": "127.0.0.1", "port": 5000, "debug": False}
    web.main(["--port", "5001"])
    assert calls == {"host": "127.0.0.1", "port": 5001, "debug": False}


# --- JSON API -------------------------------------------------------------------

def test_api_returns_report_without_raw_bytes(client, samples):
    response = upload(client, samples / "stego_full_rgb.png", url="/api/scan")
    body = response.get_json()
    assert response.status_code == 200
    assert body["verdict"] == "LIKELY_PAYLOAD"
    assert body["file_type"] == "image"
    assert body["extracted_length"] == len(PAYLOAD_MZ)
    assert "extracted" not in body


def test_api_errors_are_json(client, tmp_path):
    response = client.post("/api/scan", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json() == {"error": "No file was uploaded."}
