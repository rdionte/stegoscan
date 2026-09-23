"""Tests for image_scan.py Part 1: LSB extraction, appended data, metadata, bit-planes."""

from pathlib import Path

import pytest

from scripts.make_samples import PAYLOAD_MZ, PAYLOAD_SHEBANG, generate_all
from stegoscan.image_scan import (
    find_appended_data,
    find_metadata_findings,
    is_lossless,
    lsb_findings,
    read_image_safely,
    render_bit_planes,
)


@pytest.fixture(scope="module")
def samples_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("images")
    generate_all(directory)
    return directory


def _checks(findings):
    return {f.check for f in findings}


@pytest.mark.parametrize(
    "name", ["clean_rgb.png", "clean_grayscale.png", "clean_alpha.png", "clean_tiny.png"]
)
def test_clean_images_have_no_signature_findings(samples_dir, name):
    image = read_image_safely(samples_dir / name)
    findings, extracted, label = lsb_findings(image)
    assert findings == []
    assert extracted is None


def test_clean_images_have_no_metadata_findings(samples_dir):
    image = read_image_safely(samples_dir / "clean_rgb.png")
    assert find_metadata_findings(image) == []


def test_jpeg_is_not_lossless(samples_dir):
    image = read_image_safely(samples_dir / "clean.jpg")
    assert is_lossless(image) is False


def test_png_is_lossless(samples_dir):
    image = read_image_safely(samples_dir / "clean_rgb.png")
    assert is_lossless(image) is True


def test_stego_full_rgb_detected(samples_dir):
    image = read_image_safely(samples_dir / "stego_full_rgb.png")
    findings, extracted, label = lsb_findings(image)
    assert "signature_mz" in _checks(findings)
    assert extracted == PAYLOAD_MZ
    assert label == "interleaved-row-len_prefixed"


def test_stego_partial_rgb_detected_same_as_full(samples_dir):
    image = read_image_safely(samples_dir / "stego_partial_rgb.png")
    findings, extracted, label = lsb_findings(image)
    assert "signature_mz" in _checks(findings)
    assert extracted == PAYLOAD_MZ


def test_stego_grayscale_detected(samples_dir):
    image = read_image_safely(samples_dir / "stego_grayscale.png")
    findings, extracted, label = lsb_findings(image)
    assert "signature_shebang" in _checks(findings)
    assert extracted == PAYLOAD_SHEBANG


def test_stego_alpha_detected(samples_dir):
    image = read_image_safely(samples_dir / "stego_alpha.png")
    findings, extracted, label = lsb_findings(image)
    assert "signature_mz" in _checks(findings)
    assert extracted == PAYLOAD_MZ


def test_stego_row_order_only_found_via_red_channel(samples_dir):
    image = read_image_safely(samples_dir / "stego_row_order.png")
    findings, extracted, label = lsb_findings(image)
    assert "signature_shebang" in _checks(findings)
    assert extracted == PAYLOAD_SHEBANG
    assert label == "R-row-len_prefixed"


def test_appended_png_detected(samples_dir):
    findings, trailing = find_appended_data(samples_dir / "appended_png.png", "PNG")
    assert trailing == PAYLOAD_MZ
    assert "signature_mz" in _checks(findings)


def test_appended_jpeg_detected(samples_dir):
    findings, trailing = find_appended_data(samples_dir / "appended_jpeg.jpg", "JPEG")
    assert trailing == PAYLOAD_SHEBANG
    assert "signature_shebang" in _checks(findings)


def test_clean_png_has_no_appended_data(samples_dir):
    findings, trailing = find_appended_data(samples_dir / "clean_rgb.png", "PNG")
    assert findings == []
    assert trailing is None


def test_empty_and_corrupt_files_fail_gracefully(samples_dir):
    assert read_image_safely(samples_dir / "empty.png") is None
    assert read_image_safely(samples_dir / "corrupt.png") is None


def test_render_bit_planes_returns_eight_grayscale_images(samples_dir):
    image = read_image_safely(samples_dir / "clean_rgb.png")
    planes = render_bit_planes(image)
    assert set(planes.keys()) == set(range(8))
    for plane in planes.values():
        assert plane.mode == "L"
        assert plane.size == image.size
