"""Phase 1: image steganalysis (LSB, chi-square, RS, appended data, metadata, bit-planes).

Part 1 (this file so far): LSB extraction across multiple bit orders,
appended-data detection, metadata checks, and bit-plane rendering.
Chi-square/RS statistical estimators and the top-level scan_image()
orchestrator come in Part 2.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from stegoscan.report import Finding
from stegoscan.signatures import find_signatures

MAX_LSB_BYTES = 4096


def read_image_safely(path: Path) -> Image.Image | None:
    """Open and fully decode an image. Returns None if it can't be parsed."""
    try:
        image = Image.open(path)
        image.load()
        return image
    except Exception:
        return None


def is_lossless(image: Image.Image) -> bool:
    """True for formats where LSB checks apply (PNG, BMP); False for lossy JPEG."""
    return image.format in ("PNG", "BMP")


def _channel_options(mode: str) -> list[str]:
    """Which channel selections make sense for this image mode."""
    if mode == "L":
        return ["interleaved"]
    return ["interleaved"] + list(mode)


def extract_lsb_bits(array: np.ndarray, channel: str, traversal: str) -> np.ndarray:
    """Pull the least-significant bits for one (channel, traversal) combo as a flat bit array."""
    if channel == "interleaved":
        selected = array
    else:
        channel_index = "RGBA".index(channel)
        selected = array[:, :, channel_index]

    if traversal == "column":
        selected = selected.transpose(1, 0, 2) if selected.ndim == 3 else selected.T

    return (selected.reshape(-1) & 1).astype(np.uint8)


def _bytes_from_bits(bits: np.ndarray, max_bytes: int) -> bytes:
    usable_bits = (min(bits.shape[0], max_bytes * 8) // 8) * 8
    return np.packbits(bits[:usable_bits]).tobytes()


def lsb_candidates(image: Image.Image, max_bytes: int = MAX_LSB_BYTES) -> dict[str, bytes]:
    """Try every (channel, traversal) combo. For each, return the raw LSB bytes and,
    if the first 4 bytes look like a plausible length prefix, that interpretation too."""
    array = np.array(image)
    candidates: dict[str, bytes] = {}

    for channel in _channel_options(image.mode):
        for traversal in ("row", "column"):
            label = f"{channel}-{traversal}"
            raw = _bytes_from_bits(extract_lsb_bits(array, channel, traversal), max_bytes)
            candidates[label] = raw

            if len(raw) >= 4:
                length = int.from_bytes(raw[:4], "big")
                if 0 < length <= len(raw) - 4:
                    candidates[f"{label}-len_prefixed"] = raw[4 : 4 + length]

    return candidates


def lsb_findings(image: Image.Image) -> tuple[list[Finding], bytes | None, str | None]:
    """Run lsb_candidates + signature checks on each. Returns (findings, best
    extracted payload, the order/interpretation label that produced it)."""
    findings: list[Finding] = []
    best_payload: bytes | None = None
    best_label: str | None = None

    for label, data in lsb_candidates(image).items():
        sig_findings = find_signatures(data)
        for f in sig_findings:
            findings.append(Finding(
                f.check, f.severity,
                f"{f.detail} (LSB order: {label})",
                {**f.evidence, "order": label},
            ))
        if sig_findings and best_payload is None:
            best_payload = data
            best_label = label

    return findings, best_payload, best_label


def find_appended_data(path: Path, fmt: str) -> tuple[list[Finding], bytes | None]:
    """Look for bytes after the PNG IEND chunk or JPEG FFD9 marker; signature-check them."""
    data = path.read_bytes()

    if fmt == "PNG":
        idx = data.rfind(b"IEND")
        if idx == -1:
            return [], None
        end = idx + len(b"IEND") + 4  # skip the chunk's trailing 4-byte CRC
    elif fmt == "JPEG":
        idx = data.rfind(b"\xff\xd9")
        if idx == -1:
            return [], None
        end = idx + 2
    else:
        return [], None

    trailing = data[end:]
    if not trailing:
        return [], None

    sig_findings = find_signatures(trailing)
    severity = "high" if sig_findings else "medium"
    findings = [Finding(
        "appended_data", severity,
        f"Found {len(trailing)} bytes appended after the {fmt} end marker.",
        {"length": len(trailing), "preview": trailing[:40].decode("ascii", errors="replace")},
    )]
    findings.extend(sig_findings)
    return findings, trailing


def find_metadata_findings(image: Image.Image) -> list[Finding]:
    """Flag oversized or odd PNG text chunks and EXIF metadata."""
    findings: list[Finding] = []

    for key, value in getattr(image, "text", {}).items():
        if len(value) > 200:
            findings.append(Finding(
                "metadata_oversized_text", "medium",
                f"PNG text chunk '{key}' is unusually large ({len(value)} chars).",
                {"key": key, "length": len(value)},
            ))

    exif = image.info.get("exif")
    if exif and len(exif) > 2048:
        findings.append(Finding(
            "metadata_oversized_exif", "medium",
            f"EXIF metadata is unusually large ({len(exif)} bytes).",
            {"length": len(exif)},
        ))

    return findings


def render_bit_planes(image: Image.Image) -> dict[int, Image.Image]:
    """Render grayscale bit-plane images 0 (LSB) through 7 (MSB) for the web UI."""
    gray = np.array(image.convert("L"))
    return {
        bit: Image.fromarray((((gray >> bit) & 1) * 255).astype(np.uint8), mode="L")
        for bit in range(8)
    }
