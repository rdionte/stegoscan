"""Phase 1: image steganalysis (LSB, chi-square, RS, appended data, metadata, bit-planes).

Structural checks (LSB extraction in multiple bit orders, appended data,
metadata) catch payloads with a recognizable signature. The statistical
checks (chi-square, RS) catch LSB embedding even when the payload is
encrypted and has no signature. scan_image() combines them into a Report.

Chi-square and RS formulas follow Fridrich, Goljan & Du, "Practical
Steganalysis of Digital Images - State of the Art" (SPIE 2002), sections 2.2 and 3.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chi2

from stegoscan.report import Finding, Report, build_report
from stegoscan.signatures import find_signatures

MAX_LSB_BYTES = 4096
# Calibrated on 40 synthetic clean covers per size: RS clean-image bias peaked
# at 24% for 64x64, 9.3% for >=128x128; chi-square clean max was 0% at >=128.
CHI_THRESHOLD = 0.05
RS_THRESHOLD = 0.10
MIN_STATS_PIXELS = 128 * 128
RS_MASK = np.array([0, 1, 1, 0])


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


def chi_square_p_value(values: np.ndarray) -> float:
    """Pairs-of-values chi-square test. p near 1 = pairs look equalized (embedded).

    chi2 = sum over pairs i of (n_i - n_i')^2 / n_i', with k-1 degrees of freedom,
    where n_i = count of the even value 2i and n_i' = half the count of {2i, 2i+1}.
    """
    counts = np.bincount(values.reshape(-1), minlength=256).astype(float)
    observed_even = counts[0::2]
    expected = (counts[0::2] + counts[1::2]) / 2
    used = expected > 0  # a pair that never occurs has an undefined term
    k = int(used.sum())
    if k < 2:
        return 0.0
    statistic = float(((observed_even[used] - expected[used]) ** 2 / expected[used]).sum())
    return float(chi2.sf(statistic, k - 1))


def chi_square_scan(values: np.ndarray, step: float = 0.01) -> tuple[float, list[float]]:
    """Run the chi-square test on growing prefixes (1%, 2%, ... of values).

    Sequential embedding shows p near 1 until the message ends, then a sharp drop.
    Returns (estimated embedded fraction, p-value per prefix).
    """
    flat = values.reshape(-1)
    steps = int(round(1 / step))
    p_values = [chi_square_p_value(flat[: int(flat.size * i / steps)]) for i in range(1, steps + 1)]

    estimate = 0.0
    for i, p in enumerate(p_values, start=1):
        if p <= 0.5:
            break
        estimate = i / steps
    return estimate, p_values


def _apply_mask(groups: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """F1 where mask is 1 (x XOR 1), F-1 where mask is -1 (F1(x+1) - 1), identity where 0."""
    flip_positive = groups ^ 1
    flip_negative = ((groups + 1) ^ 1) - 1
    return np.where(mask == 1, flip_positive, np.where(mask == -1, flip_negative, groups))


def _regular_minus_singular(groups: np.ndarray, mask: np.ndarray) -> float:
    """R - S: fraction of groups made noisier by flipping minus fraction made smoother."""
    smoothness_before = np.abs(np.diff(groups, axis=1)).sum(axis=1)
    smoothness_after = np.abs(np.diff(_apply_mask(groups, mask), axis=1)).sum(axis=1)
    return float((smoothness_after > smoothness_before).mean() - (smoothness_after < smoothness_before).mean())


def rs_estimate(channel: np.ndarray) -> float:
    """Estimate the fraction (0-1) of pixels carrying message bits via RS analysis.

    Solves 2(d1 + d0)z^2 + (d-0 - d-1 - d1 - 3*d0)z + d0 - d-0 = 0 and returns
    p = z / (z - 1/2) for the root with the smaller |z|. d0/d-0 are measured on
    the image as-is; d1/d-1 on the same image with every LSB flipped.
    """
    values = channel.astype(np.int16).reshape(-1)
    usable = (values.size // RS_MASK.size) * RS_MASK.size
    if usable == 0:
        return 0.0
    groups = values[:usable].reshape(-1, RS_MASK.size)
    all_lsbs_flipped = groups ^ 1

    d0 = _regular_minus_singular(groups, RS_MASK)
    d_neg0 = _regular_minus_singular(groups, -RS_MASK)
    d1 = _regular_minus_singular(all_lsbs_flipped, RS_MASK)
    d_neg1 = _regular_minus_singular(all_lsbs_flipped, -RS_MASK)

    a = 2 * (d1 + d0)
    b = d_neg0 - d_neg1 - d1 - 3 * d0
    c = d0 - d_neg0
    if a == 0 and b == 0:
        return 0.0

    roots = np.roots([a, b, c])
    real_roots = roots[np.abs(roots.imag) < 1e-12].real
    if real_roots.size == 0:
        # Both measurement points collapse to R_M = S_M only when the LSB plane
        # is fully randomized, so the paper's curves never cross: report 100%.
        return 1.0
    z = real_roots[np.argmin(np.abs(real_roots))]
    return float(np.clip(z / (z - 0.5), 0.0, 1.0))


def statistical_findings(array: np.ndarray) -> list[Finding]:
    """Chi-square (sequential, interleaved order) and RS (per channel, averaged) findings.

    Estimates below threshold are still reported at 'info' severity so the
    report shows the measured numbers.
    """
    channels = [array] if array.ndim == 2 else [array[:, :, c] for c in range(array.shape[2])]
    pixel_count = channels[0].size
    if pixel_count < MIN_STATS_PIXELS:
        return [Finding(
            "stats_skipped", "info",
            f"Image too small ({pixel_count} pixels) for reliable chi-square/RS statistics.",
            {"pixels": pixel_count, "minimum": MIN_STATS_PIXELS},
        )]

    chi_estimate, _ = chi_square_scan(array)
    rs_per_channel = [rs_estimate(ch) for ch in channels]
    rs_mean = float(np.mean(rs_per_channel))

    return [
        Finding(
            "lsb_chi_square", "high" if chi_estimate >= CHI_THRESHOLD else "info",
            f"Chi-square pair test: ~{chi_estimate:.0%} of values (in sequential order) look LSB-embedded.",
            {"embedding_estimate": round(chi_estimate, 3)},
        ),
        Finding(
            "lsb_rs_analysis", "high" if rs_mean >= RS_THRESHOLD else "info",
            f"RS analysis: estimated {rs_mean:.1%} of pixels carry embedded bits.",
            {"embedding_estimate": round(rs_mean, 3), "per_channel": [round(x, 3) for x in rs_per_channel]},
        ),
    ]


def scan_image(path: Path) -> Report:
    """Scan one image file and return a Report. Never raises on bad input."""
    image = read_image_safely(path)
    if image is None:
        return build_report([Finding(
            "unreadable_file", "low",
            "File could not be analyzed: it is empty, corrupt, or not a supported image.",
            {"file": path.name},
        )])

    findings: list[Finding] = []
    extracted: bytes | None = None
    method: str | None = None

    findings.extend(find_metadata_findings(image))
    appended, trailing = find_appended_data(path, image.format)
    findings.extend(appended)

    if not is_lossless(image):
        findings.append(Finding(
            "lsb_checks_skipped", "info",
            f"LSB checks skipped for {image.format}: lossy compression destroys LSB data, "
            "so only appended-data and metadata checks were run.",
            {"format": image.format},
        ))
        return build_report(findings, trailing, "appended_data" if trailing else None)

    if image.mode not in ("L", "RGB", "RGBA"):
        converted_mode = "RGBA" if image.mode.endswith("A") else "RGB"
        findings.append(Finding(
            "mode_converted", "info",
            f"Converted {image.mode} image to {converted_mode} for analysis.",
            {"from": image.mode, "to": converted_mode},
        ))
        image = image.convert(converted_mode)

    lsb, payload, label = lsb_findings(image)
    findings.extend(lsb)
    if payload is not None:
        findings.append(Finding(
            "lsb_payload_extracted", "high",
            f"Extracted a {len(payload)}-byte payload with a known signature from the LSBs (order: {label}).",
            {"order": label, "length": len(payload)},
        ))
        extracted, method = payload, f"lsb:{label}"
    elif trailing:
        extracted, method = trailing, "appended_data"

    findings.extend(statistical_findings(np.array(image)))
    return build_report(findings, extracted, method)
