"""Bit-to-byte helper shared by the image (LSB) and text (whitespace, zero-width) scanners."""

import numpy as np


def bits_to_bytes(bits: np.ndarray, max_bytes: int | None = None) -> bytes:
    """Pack 0/1 values into bytes, most significant bit first.

    Leftover bits that don't fill a whole byte are dropped.
    """
    usable = bits.shape[0] if max_bytes is None else min(bits.shape[0], max_bytes * 8)
    usable = (usable // 8) * 8
    return np.packbits(bits[:usable].astype(np.uint8)).tobytes()
