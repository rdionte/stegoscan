"""Tests for signatures.py: known payload signature detection."""

import base64

from stegoscan.signatures import find_signatures


def _checks(findings):
    return {f.check for f in findings}


def test_mz_header_detected():
    findings = find_signatures(b"MZ" + b"\x00" * 10)
    assert "signature_mz" in _checks(findings)


def test_elf_header_detected():
    findings = find_signatures(b"\x7fELF" + b"\x00" * 10)
    assert "signature_elf" in _checks(findings)


def test_shebang_detected():
    findings = find_signatures(b"#!/bin/sh\necho hi\n")
    assert "signature_shebang" in _checks(findings)


def test_zip_header_detected():
    findings = find_signatures(b"PK\x03\x04" + b"\x00" * 10)
    assert "signature_zip" in _checks(findings)


def test_powershell_keyword_detected_mid_blob():
    data = b"junk junk FromBase64String(junk) more junk"
    findings = find_signatures(data)
    assert "signature_powershell" in _checks(findings)


def test_valid_base64_run_detected_with_decoded_preview():
    decoded = b"STEGOSCAN TEST PAYLOAD - harmless, this is a fake binary blob"
    encoded = base64.b64encode(decoded)
    data = b"noise noise " + encoded + b" more noise"

    findings = find_signatures(data)
    base64_findings = [f for f in findings if f.check == "signature_base64"]

    assert base64_findings
    assert decoded[:40].decode() in base64_findings[0].evidence["decoded_preview"]


def test_no_signatures_in_plain_noise():
    data = bytes([0xFF, 0x00] * 128)
    assert find_signatures(data) == []


def test_empty_bytes_has_no_findings_and_does_not_crash():
    assert find_signatures(b"") == []
