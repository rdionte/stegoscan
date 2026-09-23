"""Routes a file to the right scanner (image / text / pcap).

Lives in the core (not the CLI) so the web UI can reuse the same routing.
"""

from pathlib import Path

from stegoscan.image_scan import scan_image
from stegoscan.report import Report
from stegoscan.text_scan import scan_text

IMAGE = "image"
TEXT = "text"
PCAP = "pcap"
UNKNOWN = "unknown"

# Magic bytes are checked before the extension: a renamed file keeps its real header.
_MAGIC_TYPES = [
    (b"\x89PNG\r\n\x1a\n", IMAGE),
    (b"\xff\xd8\xff", IMAGE),                   # JPEG
    (b"GIF87a", IMAGE),
    (b"GIF89a", IMAGE),
    (b"\xd4\xc3\xb2\xa1", PCAP),                # pcap, little-endian
    (b"\xa1\xb2\xc3\xd4", PCAP),                # pcap, big-endian
    (b"\x4d\x3c\xb2\xa1", PCAP),                # pcap, nanosecond, little-endian
    (b"\xa1\xb2\x3c\x4d", PCAP),                # pcap, nanosecond, big-endian
    (b"\x0a\x0d\x0d\x0a", PCAP),                # pcapng
]

_EXTENSION_TYPES = {
    ".png": IMAGE, ".jpg": IMAGE, ".jpeg": IMAGE, ".bmp": IMAGE, ".gif": IMAGE,
    ".txt": TEXT, ".md": TEXT, ".csv": TEXT, ".log": TEXT,
    ".pcap": PCAP, ".pcapng": PCAP, ".cap": PCAP,
}

_SNIFF_BYTES = 8192

# Valid BMP info-header sizes (BITMAPCOREHEADER ... BITMAPV5HEADER).
_BMP_DIB_HEADER_SIZES = {12, 40, 52, 56, 64, 108, 124}


def detect_file_type(path: Path) -> str:
    """Guess the carrier type from magic bytes, then UTF-8 text, then extension."""
    with open(path, "rb") as handle:
        head = handle.read(_SNIFF_BYTES)

    for magic, file_type in _MAGIC_TYPES:
        if head.startswith(magic):
            return file_type
    if _looks_like_bmp(head):
        return IMAGE

    if head and _looks_like_text(head):
        return TEXT

    return _EXTENSION_TYPES.get(path.suffix.lower(), UNKNOWN)


def _looks_like_bmp(head: bytes) -> bool:
    """Check the BMP header. The 2-byte "BM" magic is too weak on its own (text
    can start with it), so also require zeroed reserved bytes 6-9 and a known
    DIB header size at offset 14."""
    if len(head) < 18 or not head.startswith(b"BM"):
        return False
    dib_size = int.from_bytes(head[14:18], "little")
    return head[6:10] == b"\x00" * 4 and dib_size in _BMP_DIB_HEADER_SIZES


def _looks_like_text(head: bytes) -> bool:
    """True if the bytes decode as UTF-8 and contain no NUL bytes."""
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as error:
        # The sniff window may cut a multi-byte character in half at the end.
        return error.start >= len(head) - 3
    return True


def scan_file(path: Path) -> Report:
    """Detect the file type and run the matching scanner.

    Raises ValueError for unrecognized files.
    """
    file_type = detect_file_type(path)
    if file_type == IMAGE:
        return scan_image(path)
    if file_type == TEXT:
        return scan_text(path)
    if file_type == PCAP:
        # Imported here so image/text scans don't pay scapy's ~0.5 s import time.
        from stegoscan.net_scan import scan_pcap
        return scan_pcap(path)
    raise ValueError(f"Unrecognized file type: {path.name}")
