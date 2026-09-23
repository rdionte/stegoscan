"""Tests for image_scan.py Part 2: chi-square, RS analysis, and scan_image()."""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from scripts.make_samples import PAYLOAD_MZ, PAYLOAD_SHEBANG, generate_all
from stegoscan.image_scan import chi_square_p_value, rs_estimate, scan_image
from stegoscan.report import CLEAN, LIKELY_PAYLOAD


@pytest.fixture(scope="module")
def samples_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("images")
    generate_all(directory)
    return directory


def _finding(report, check):
    return next(f for f in report.findings if f.check == check)


def _checks(report):
    return {f.check for f in report.findings}


# --- chi-square formula, checked against hand-computed values ---

def test_chi_square_matches_hand_computed_value():
    # Pair (0,1) counts 60/40 -> n' = 50, (60-50)^2/50 = 2. Pair (2,3) 50/50 -> 0.
    # chi2 = 2 with k-1 = 1 degree of freedom.
    values = np.array([0] * 60 + [1] * 40 + [2] * 50 + [3] * 50, dtype=np.uint8)
    assert chi_square_p_value(values) == pytest.approx(chi2.sf(2.0, 1))


def test_chi_square_equal_pairs_look_embedded():
    values = np.array([0, 1, 2, 3] * 100, dtype=np.uint8)
    assert chi_square_p_value(values) == pytest.approx(1.0)


def test_chi_square_lopsided_pairs_look_clean():
    values = np.array([0] * 200 + [2] * 200, dtype=np.uint8)
    assert chi_square_p_value(values) < 0.001


def test_rs_estimate_handles_flat_and_tiny_input():
    assert 0.0 <= rs_estimate(np.full((16, 16), 128, dtype=np.uint8)) <= 1.0
    assert rs_estimate(np.array([[5, 6]], dtype=np.uint8)) == 0.0


# --- statistical fixtures (256x256, random payload bits, no signature) ---

def test_stats_clean_is_clean(samples_dir):
    report = scan_image(samples_dir / "stats_clean.png")
    assert report.verdict == CLEAN
    assert _finding(report, "lsb_chi_square").evidence["embedding_estimate"] < 0.05
    assert _finding(report, "lsb_rs_analysis").evidence["embedding_estimate"] < 0.10


def test_sequential_40_caught_by_chi_square(samples_dir):
    report = scan_image(samples_dir / "stats_sequential_40.png")
    assert report.verdict != CLEAN
    assert 0.35 <= _finding(report, "lsb_chi_square").evidence["embedding_estimate"] <= 0.45


@pytest.mark.parametrize(
    "name,low,high",
    [("stats_scattered_40.png", 0.30, 0.55), ("stats_scattered_10.png", 0.10, 0.20)],
)
def test_scattered_embedding_caught_by_rs(samples_dir, name, low, high):
    report = scan_image(samples_dir / name)
    assert report.verdict != CLEAN
    assert low <= _finding(report, "lsb_rs_analysis").evidence["embedding_estimate"] <= high


def test_full_embedding_caught_by_both(samples_dir):
    report = scan_image(samples_dir / "stats_full.png")
    assert report.verdict != CLEAN
    assert _finding(report, "lsb_chi_square").evidence["embedding_estimate"] == 1.0
    assert _finding(report, "lsb_rs_analysis").evidence["embedding_estimate"] >= 0.9


# --- scan_image() end-to-end on the Part 1 fixtures ---

@pytest.mark.parametrize(
    "name", ["clean_rgb.png", "clean_grayscale.png", "clean_alpha.png", "clean_tiny.png"]
)
def test_small_clean_images_are_clean_and_skip_stats(samples_dir, name):
    report = scan_image(samples_dir / name)
    assert report.verdict == CLEAN
    assert "stats_skipped" in _checks(report)


def test_jpeg_is_clean_and_states_limitation(samples_dir):
    report = scan_image(samples_dir / "clean.jpg")
    assert report.verdict == CLEAN
    assert "lsb_checks_skipped" in _checks(report)


@pytest.mark.parametrize(
    "name,payload",
    [
        ("stego_full_rgb.png", PAYLOAD_MZ),
        ("stego_partial_rgb.png", PAYLOAD_MZ),
        ("stego_grayscale.png", PAYLOAD_SHEBANG),
        ("stego_alpha.png", PAYLOAD_MZ),
        ("stego_row_order.png", PAYLOAD_SHEBANG),
        ("appended_png.png", PAYLOAD_MZ),
        ("appended_jpeg.jpg", PAYLOAD_SHEBANG),
    ],
)
def test_payload_fixtures_are_likely_payload_and_extracted_exactly(samples_dir, name, payload):
    report = scan_image(samples_dir / name)
    assert report.verdict == LIKELY_PAYLOAD
    assert report.extracted == payload
    assert report.extracted_sha256 is not None


@pytest.mark.parametrize("name", ["empty.png", "corrupt.png"])
def test_unreadable_files_fail_gracefully(samples_dir, name):
    report = scan_image(samples_dir / name)
    assert report.verdict == CLEAN
    assert "unreadable_file" in _checks(report)
