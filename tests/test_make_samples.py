"""Sanity tests for scripts/make_samples.py.

These check that the fixture generator itself is correct, independent of
image_scan.py (which doesn't exist yet). The bit-reader below mirrors
lsb_embed's length-prefixed format purely for verification.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.make_samples import PAYLOAD_MZ, PAYLOAD_SHEBANG, generate_all

EXPECTED_FILES = {
    "clean_rgb.png", "clean_grayscale.png", "clean_alpha.png",
    "clean_tiny.png", "clean.jpg",
    "stego_full_rgb.png", "stego_partial_rgb.png",
    "stego_grayscale.png", "stego_alpha.png", "stego_row_order.png",
    "appended_png.png", "appended_jpeg.jpg",
    "empty.png", "corrupt.png",
}


def _extract_payload(array: np.ndarray, order: str) -> bytes:
    if order == "interleaved":
        flat = array.reshape(-1)
    else:
        channel_index = "RGBA".index(order.split(":")[1])
        flat = array[:, :, channel_index].reshape(-1)
    bits = flat & 1
    length = int.from_bytes(np.packbits(bits[:32]).tobytes(), "big")
    payload_bits = bits[32 : 32 + length * 8]
    return np.packbits(payload_bits).tobytes()


def test_generate_all_writes_expected_files(tmp_path: Path):
    written = generate_all(tmp_path)
    assert EXPECTED_FILES <= {p.name for p in written}


def test_clean_images_have_expected_shape_and_mode(tmp_path: Path):
    generate_all(tmp_path)
    rgb = Image.open(tmp_path / "clean_rgb.png")
    assert rgb.mode == "RGB" and rgb.size == (64, 64)

    gray = Image.open(tmp_path / "clean_grayscale.png")
    assert gray.mode == "L" and gray.size == (64, 64)

    alpha = Image.open(tmp_path / "clean_alpha.png")
    assert alpha.mode == "RGBA" and alpha.size == (64, 64)

    tiny = Image.open(tmp_path / "clean_tiny.png")
    assert tiny.size == (2, 2)


@pytest.mark.parametrize(
    "filename,order,payload",
    [
        ("stego_full_rgb.png", "interleaved", PAYLOAD_MZ),
        ("stego_partial_rgb.png", "interleaved", PAYLOAD_MZ),
        ("stego_grayscale.png", "interleaved", PAYLOAD_SHEBANG),
        ("stego_alpha.png", "interleaved", PAYLOAD_MZ),
        ("stego_row_order.png", "channel:R", PAYLOAD_SHEBANG),
    ],
)
def test_stego_images_roundtrip_payload(tmp_path: Path, filename: str, order: str, payload: bytes):
    generate_all(tmp_path)
    array = np.array(Image.open(tmp_path / filename))
    assert _extract_payload(array, order) == payload


def test_appended_png_has_payload_after_iend(tmp_path: Path):
    generate_all(tmp_path)
    assert (tmp_path / "appended_png.png").read_bytes().endswith(PAYLOAD_MZ)


def test_appended_jpeg_has_payload_after_ffd9(tmp_path: Path):
    generate_all(tmp_path)
    assert (tmp_path / "appended_jpeg.jpg").read_bytes().endswith(PAYLOAD_SHEBANG)


def test_empty_file_is_empty(tmp_path: Path):
    generate_all(tmp_path)
    assert (tmp_path / "empty.png").stat().st_size == 0


def test_corrupt_file_is_not_a_valid_png(tmp_path: Path):
    generate_all(tmp_path)
    with pytest.raises(Exception):
        Image.open(tmp_path / "corrupt.png").load()
