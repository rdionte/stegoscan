"""Generate clean + stego test images with HARMLESS fake payloads.

Produces the Phase 1 (image) fixtures used to test image_scan.py:
clean baselines, LSB-embedded stego images at 100% and ~10% embedding,
appended-data samples, and edge cases (tiny, grayscale, alpha, empty,
corrupt). See CLAUDE.md for the full plan. Payloads are fake signature
bytes only -- never real malware.
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "images"

PAYLOAD_MZ = b"MZ" + b"STEGOSCAN TEST PAYLOAD - harmless"
PAYLOAD_SHEBANG = b"#!/bin/sh\nSTEGOSCAN TEST PAYLOAD - harmless"

IMAGE_SIZE = (64, 64)  # (width, height)


def make_clean_image(size: tuple[int, int], mode: str, rng: np.random.Generator) -> Image.Image:
    """Build a base image of random pixel noise, with no hidden data."""
    width, height = size
    shape = (height, width) if mode == "L" else (height, width, len(mode))
    array = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return Image.fromarray(array, mode=mode)


def _bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def lsb_embed(array: np.ndarray, payload: bytes, fraction: float, order: str) -> np.ndarray:
    """Embed a length-prefixed payload into the least-significant bits of array.

    order is "interleaved" (row-major over all channels, matching how the
    pixels are stored) or "channel:R"/"channel:G"/"channel:B"/"channel:A"
    (row-major over a single channel only).

    fraction controls how much of the LSB capacity gets used: 1.0 repeats
    the length-prefixed payload to fill the entire capacity (simulating a
    100% embedding rate); a value like 0.1 writes the payload only once,
    into the first ~10% of capacity, leaving the rest of the carrier's
    LSBs at their original values.
    """
    out = array.copy()
    if order == "interleaved":
        flat = out.reshape(-1).copy()
    else:
        channel_index = "RGBA".index(order.split(":")[1])
        flat = out[:, :, channel_index].reshape(-1).copy()

    capacity = flat.shape[0]
    unit_bits = _bits_from_bytes(len(payload).to_bytes(4, "big") + payload)
    unit_len = unit_bits.shape[0]
    if unit_len > capacity:
        raise ValueError("payload too large for image capacity")

    if fraction >= 1.0:
        reps, remainder = divmod(capacity, unit_len)
        bits = np.concatenate([np.tile(unit_bits, reps), unit_bits[:remainder]])
    else:
        max_slots = int(capacity * fraction)
        if unit_len > max_slots:
            raise ValueError(f"payload needs {unit_len} bits but fraction={fraction} only allows {max_slots}")
        bits = unit_bits

    n = bits.shape[0]
    flat[:n] = (flat[:n] & 0xFE) | bits

    if order == "interleaved":
        out = flat.reshape(array.shape)
    else:
        out[:, :, channel_index] = flat.reshape(array.shape[0], array.shape[1])
    return out


def append_after_marker(path: Path, marker: bytes, payload: bytes) -> None:
    """Append payload bytes right after the given end-of-file marker (PNG IEND or JPEG FFD9)."""
    data = path.read_bytes()
    idx = data.rfind(marker)
    if idx == -1:
        raise ValueError(f"marker {marker!r} not found in {path}")
    end = idx + len(marker)
    if marker == b"IEND":
        end += 4  # skip the chunk's 4-byte CRC, which follows the type field
    path.write_bytes(data[:end] + payload)


def corrupt_file(path: Path) -> None:
    """Truncate and flip bytes in a file to simulate a corrupt/malformed image."""
    data = bytearray(path.read_bytes())
    cutoff = max(1, len(data) // 3)
    data = data[:cutoff]
    for i in range(0, len(data), 7):
        data[i] ^= 0xFF
    path.write_bytes(bytes(data))


def generate_all(output_dir: Path) -> list[Path]:
    """Generate every Phase 1 image fixture into output_dir. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    written: list[Path] = []

    def save(image: Image.Image, name: str, **kwargs) -> Path:
        path = output_dir / name
        image.save(path, **kwargs)
        written.append(path)
        return path

    # Clean baselines -- must scan CLEAN
    save(make_clean_image(IMAGE_SIZE, "RGB", rng), "clean_rgb.png")
    save(make_clean_image(IMAGE_SIZE, "L", rng), "clean_grayscale.png")
    save(make_clean_image(IMAGE_SIZE, "RGBA", rng), "clean_alpha.png")
    save(make_clean_image((2, 2), "RGB", rng), "clean_tiny.png")
    save(make_clean_image(IMAGE_SIZE, "RGB", rng), "clean.jpg", format="JPEG", quality=90)

    # LSB stego images
    full_arr = lsb_embed(np.array(make_clean_image(IMAGE_SIZE, "RGB", rng)), PAYLOAD_MZ, 1.0, "interleaved")
    save(Image.fromarray(full_arr, "RGB"), "stego_full_rgb.png")

    partial_arr = lsb_embed(np.array(make_clean_image(IMAGE_SIZE, "RGB", rng)), PAYLOAD_MZ, 0.1, "interleaved")
    save(Image.fromarray(partial_arr, "RGB"), "stego_partial_rgb.png")

    gray_arr = lsb_embed(np.array(make_clean_image(IMAGE_SIZE, "L", rng)), PAYLOAD_SHEBANG, 1.0, "interleaved")
    save(Image.fromarray(gray_arr, "L"), "stego_grayscale.png")

    alpha_arr = lsb_embed(np.array(make_clean_image(IMAGE_SIZE, "RGBA", rng)), PAYLOAD_MZ, 1.0, "interleaved")
    save(Image.fromarray(alpha_arr, "RGBA"), "stego_alpha.png")

    row_arr = lsb_embed(np.array(make_clean_image(IMAGE_SIZE, "RGB", rng)), PAYLOAD_SHEBANG, 1.0, "channel:R")
    save(Image.fromarray(row_arr, "RGB"), "stego_row_order.png")

    # Appended-data samples
    png_path = save(make_clean_image(IMAGE_SIZE, "RGB", rng), "appended_png.png")
    append_after_marker(png_path, b"IEND", PAYLOAD_MZ)

    jpg_path = save(make_clean_image(IMAGE_SIZE, "RGB", rng), "appended_jpeg.jpg", format="JPEG", quality=90)
    append_after_marker(jpg_path, b"\xff\xd9", PAYLOAD_SHEBANG)

    # Edge cases
    empty_path = output_dir / "empty.png"
    empty_path.write_bytes(b"")
    written.append(empty_path)

    corrupt_path = save(make_clean_image(IMAGE_SIZE, "RGB", rng), "corrupt.png")
    corrupt_file(corrupt_path)

    return written


def main() -> None:
    written = generate_all(OUTPUT_DIR)
    print(f"Wrote {len(written)} sample files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
