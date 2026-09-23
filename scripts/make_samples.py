"""Generate clean + stego test images with HARMLESS fake payloads.

Produces the Phase 1 (image) fixtures used to test image_scan.py:
clean baselines, LSB-embedded stego images at 100% and ~10% embedding,
appended-data samples, edge cases (tiny, grayscale, alpha, empty,
corrupt), and larger stats_* images with random payload bits for the
chi-square/RS tests. See CLAUDE.md for the full plan. Payloads are fake signature
bytes only -- never real malware.

Also produces the Phase 2 (text) fixtures used to test text_scan.py:
clean texts that contain legitimate whitespace and joiners (markdown line
breaks, CRLF, tab indentation, emoji, Persian), trailing-whitespace and
zero-width stego files, and edge cases (empty, binary).

Also produces the Phase 3 (network) fixtures used to test net_scan.py:
synthetic captures only, using documentation IP ranges (RFC 5737) and
reserved .test / example.* names, with clean traffic and one covert
channel per file. Nothing is ever sent on a real network.
"""

import base64
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "images"
TEXT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "text"
PCAP_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "samples" / "pcap"

PAYLOAD_MZ = b"MZ" + b"STEGOSCAN TEST PAYLOAD - harmless"
PAYLOAD_SHEBANG = b"#!/bin/sh\nSTEGOSCAN TEST PAYLOAD - harmless"
HIDDEN_MESSAGE = b"STEGOSCAN HIDDEN MESSAGE - meet at the usual place"
DNS_TUNNEL_PAYLOAD = PAYLOAD_MZ * 3  # long enough to need 10+ queries

IMAGE_SIZE = (64, 64)  # (width, height)
STATS_SIZE = (256, 256)  # statistical tests need more samples per value pair


def make_clean_image(size: tuple[int, int], mode: str, rng: np.random.Generator) -> Image.Image:
    """Build a photo-like cover image with no hidden data.

    Random noise is a bad cover: its value pairs are already balanced, so
    chi-square/RS read it as ~100% embedded. Instead: upscale random 8x8
    blobs (smooth regions), add light sensor-like noise, then contrast-stretch
    (like a levels adjustment), which leaves the uneven pair counts real photos have.
    """
    width, height = size
    color_channels = 1 if mode == "L" else 3
    low = rng.integers(0, 256, size=(8, 8, color_channels), dtype=np.uint8)
    smooth = np.stack(
        [np.array(Image.fromarray(low[:, :, c]).resize((width, height), Image.BICUBIC)) for c in range(color_channels)],
        axis=-1,
    ).astype(float)
    noisy = np.clip(smooth + rng.normal(0, 1.5, smooth.shape), 0, 255)
    array = np.clip(np.round(noisy * 0.6 + 40) * 1.5 - 60, 0, 255).astype(np.uint8)

    if mode == "L":
        array = array[:, :, 0]
    elif mode == "RGBA":
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        array = np.concatenate([array, alpha], axis=-1)
    return Image.fromarray(array, mode=mode)


def embed_random_bits(array: np.ndarray, fraction: float, scattered: bool, rng: np.random.Generator) -> np.ndarray:
    """Overwrite the LSBs of `fraction` of all values with random bits.

    Random bits stand in for an encrypted payload: no signature to find, so
    only the statistical tests can catch it. scattered=False uses the first N
    values (what chi-square is built for); scattered=True uses random
    positions (what RS is built for).
    """
    flat = array.reshape(-1).copy()
    count = int(fraction * flat.size)
    positions = rng.choice(flat.size, count, replace=False) if scattered else np.arange(count)
    flat[positions] = (flat[positions] & 0xFE) | rng.integers(0, 2, count, dtype=np.uint8)
    return flat.reshape(array.shape)


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

    # Statistical-detection fixtures: random (signature-less) payload bits
    save(make_clean_image(STATS_SIZE, "RGB", rng), "stats_clean.png")
    for name, fraction, scattered in [
        ("stats_sequential_40.png", 0.4, False),
        ("stats_scattered_40.png", 0.4, True),
        ("stats_scattered_10.png", 0.1, True),
        ("stats_full.png", 1.0, True),
    ]:
        cover = np.array(make_clean_image(STATS_SIZE, "RGB", rng))
        save(Image.fromarray(embed_random_bits(cover, fraction, scattered, rng), "RGB"), name)

    return written


# --- Phase 2: text fixtures ------------------------------------------------------
# Zero-width characters are built with chr() so no invisible characters live in this source.
ZWSP, ZWNJ, ZWJ, BOM, WORD_JOINER = (chr(c) for c in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060))

_WORDS = (
    "the report shows network traffic stayed normal during the weekly review and our "
    "team will update the dashboard after lunch before sending notes to everyone"
).split()


def make_cover_lines(count: int, rng: np.random.Generator) -> list[str]:
    """Plain sentences with no trailing whitespace."""
    return [
        " ".join(rng.choice(_WORDS, size=int(rng.integers(6, 13)))).capitalize() + "."
        for _ in range(count)
    ]


def embed_whitespace(lines: list[str], payload: bytes, rng: np.random.Generator,
                     zero: str = " ", one: str = "\t") -> list[str]:
    """Append payload bits as trailing whitespace from the top, 3-10 bits per line
    (a varying count, so the decoder can't rely on one byte per line)."""
    bits = _bits_from_bytes(payload)
    out = list(lines)
    position, line = 0, 0
    while position < len(bits):
        chunk = bits[position : position + int(rng.integers(3, 11))]
        out[line] += "".join(one if bit else zero for bit in chunk)
        position += len(chunk)
        line += 1
    assert line <= len(out), "cover text too short for payload"
    return out


def embed_zero_width(text: str, payload: bytes, rng: np.random.Generator,
                     zero: str = ZWSP, one: str = ZWNJ) -> str:
    """Scatter payload bits through text as zero-width characters, keeping bit order."""
    bits = _bits_from_bytes(payload)
    positions = np.sort(rng.integers(1, len(text), size=len(bits)))
    pieces, previous = [], 0
    for position, bit in zip(positions, bits):
        pieces.append(text[previous:position])
        pieces.append(one if bit else zero)
        previous = position
    pieces.append(text[previous:])
    return "".join(pieces)


def _persian_word() -> str:
    """'mikhaham' (I want), which is correctly written with a ZWNJ between its parts."""
    return "".join(map(chr, (0x0645, 0x06CC))) + ZWNJ + "".join(map(chr, (0x062E, 0x0648, 0x0627, 0x0647, 0x0645)))


def generate_text_samples(output_dir: Path) -> list[Path]:
    """Generate every Phase 2 text fixture into output_dir. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5678)
    written: list[Path] = []

    def save(name: str, content: str | bytes) -> Path:
        path = output_dir / name
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        written.append(path)
        return path

    def lines_to_text(lines: list[str], newline: str = "\n") -> str:
        return newline.join(lines) + newline

    # Clean baselines -- must scan CLEAN
    prose = make_cover_lines(60, rng)
    for i in (3, 17, 29, 44, 51):  # editor leftovers: a single stray space
        prose[i] += " "
    save("clean_prose.txt", lines_to_text(prose))

    markdown = ["# Weekly notes", ""] + [line + "  " for line in make_cover_lines(20, rng)]  # two-space line breaks
    markdown += ["", "- item one", "- item two"]
    save("clean_markdown.md", lines_to_text(markdown))

    code = ["def summarize(rows):"]
    for i in range(20):
        code += [f"\tvalue_{i} = rows[{i}]", "\t" if i % 2 else "\t    "]  # blank lines keeping indentation
    code += ["\treturn value_0"]
    save("clean_code.py", lines_to_text(code))

    crlf = make_cover_lines(30, rng)
    crlf[5] += " "
    save("clean_crlf.txt", lines_to_text(crlf, newline="\r\n"))

    family = chr(0x1F468) + ZWJ + chr(0x1F469) + ZWJ + chr(0x1F467)
    rainbow_flag = chr(0x1F3F3) + chr(0xFE0F) + ZWJ + chr(0x1F308)
    emoji_lines = [BOM + "Team photo captions"]
    emoji_lines += [f"Picnic day {family} with everyone {rainbow_flag}" for _ in range(4)]
    emoji_lines += [f"Persian: {_persian_word()} {_persian_word()}" for _ in range(3)]
    save("clean_emoji.txt", lines_to_text(emoji_lines))

    # Trailing-whitespace stego
    save("ws_payload.txt", lines_to_text(embed_whitespace(make_cover_lines(80, rng), PAYLOAD_MZ, rng)))
    save("ws_payload_reversed.txt", lines_to_text(
        embed_whitespace(make_cover_lines(80, rng), PAYLOAD_MZ, rng, zero="\t", one=" ")))
    save("ws_message.txt", lines_to_text(embed_whitespace(make_cover_lines(100, rng), HIDDEN_MESSAGE, rng)))
    save("ws_partial_10.txt", lines_to_text(embed_whitespace(make_cover_lines(400, rng), PAYLOAD_MZ, rng)))
    random_bytes = rng.integers(0, 256, size=64, dtype=np.uint8).tobytes()  # encrypted-like, no signature
    save("ws_random.txt", lines_to_text(embed_whitespace(make_cover_lines(120, rng), random_bytes, rng)))

    # Zero-width stego
    save("zw_payload.txt", embed_zero_width(lines_to_text(make_cover_lines(40, rng)), PAYLOAD_MZ, rng))
    save("zw_message.txt", embed_zero_width(
        lines_to_text(make_cover_lines(40, rng)), HIDDEN_MESSAGE, rng, zero=ZWJ, one=WORD_JOINER))

    # Edge cases
    save("empty.txt", b"")
    save("binary.txt", b"\x00\x01\x02 not text \xff\xfe")

    return written


# --- Phase 3: network fixtures ---------------------------------------------------
# RFC 5737 documentation addresses: never routable on the real internet.
CLIENT, SERVER, RESOLVER = "192.0.2.10", "198.51.100.20", "203.0.113.53"
COVERT_SRC, COVERT_DST = "192.0.2.66", "198.51.100.99"
TUNNEL_DOMAIN = "tunnel.example.test"
WINDOWS_PING = b"abcdefghijklmnopqrstuvwabcdefghi"
_BASE_TIME = 1_700_000_000.0
_BASE32_ALPHABET = list("abcdefghijklmnopqrstuvwxyz234567")


def _linux_ping_payload(rng: np.random.Generator) -> bytes:
    """56 bytes like Linux ping: a 16-byte timestamp, then byte i has value i."""
    return rng.bytes(16) + bytes(range(16, 56))


def make_clean_traffic(rng: np.random.Generator) -> list:
    """Normal-looking traffic: HTTPS sessions, DNS lookups, and Linux/Windows pings.

    Client IP IDs count up, the server uses ID 0 with DF set (Linux style),
    ISNs are random, and each direction keeps a steady TTL.
    """
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    from scapy.layers.inet import ICMP, IP, TCP, UDP
    from scapy.layers.l2 import Ether
    from scapy.packet import Raw

    packets = []
    client_id = int(rng.integers(0, 60000))

    def from_client(dst: str):
        nonlocal client_id
        client_id = (client_id + 1) % 65536
        return Ether() / IP(src=CLIENT, dst=dst, id=client_id, ttl=64)

    def from_server():
        return Ether() / IP(src=SERVER, dst=CLIENT, id=0, flags="DF", ttl=57)

    # HTTPS sessions: handshake, then encrypted-looking records both ways
    for _ in range(6):
        sport = int(rng.integers(40000, 60000))
        c_seq, s_seq = int(rng.integers(0, 2**32)), int(rng.integers(0, 2**32))
        packets.append(from_client(SERVER) / TCP(sport=sport, dport=443, flags="S", seq=c_seq))
        packets.append(from_server() / TCP(sport=443, dport=sport, flags="SA", seq=s_seq, ack=(c_seq + 1) % 2**32))
        c_seq, s_seq = (c_seq + 1) % 2**32, (s_seq + 1) % 2**32
        packets.append(from_client(SERVER) / TCP(sport=sport, dport=443, flags="A", seq=c_seq, ack=s_seq))
        for _ in range(20):
            request = rng.bytes(int(rng.integers(40, 400)))
            packets.append(from_client(SERVER) / TCP(sport=sport, dport=443, flags="PA", seq=c_seq, ack=s_seq) / Raw(request))
            c_seq = (c_seq + len(request)) % 2**32
            response = rng.bytes(int(rng.integers(100, 1400)))
            packets.append(from_server() / TCP(sport=443, dport=sport, flags="PA", seq=s_seq, ack=c_seq) / Raw(response))
            s_seq = (s_seq + len(response)) % 2**32

    # DNS: ordinary names, a few hash-like CDN hosts, and one legitimate TXT lookup
    names = ["www.example.com", "mail.example.com", "api.example.org", "cdn.example.net",
             "static.example.com", "login.example.org", "updates.example.com", "news.example.net"]
    names += ["".join(rng.choice(list("0123456789abcdef"), size=13)) + ".cdn.example.net" for _ in range(3)]
    queries = [(str(name), "A") for name in rng.choice(names, size=19)] + [("_dmarc.example.com", "TXT")]
    for name, qtype in queries:
        sport, query_id = int(rng.integers(40000, 60000)), int(rng.integers(0, 65536))
        packets.append(from_client(RESOLVER) / UDP(sport=sport, dport=53)
                       / DNS(id=query_id, rd=1, qd=DNSQR(qname=name, qtype=qtype)))
        answer = DNSRR(rrname=name, type="TXT", rdata="v=DMARC1; p=none") if qtype == "TXT" \
            else DNSRR(rrname=name, type="A", rdata="198.51.100.30")
        packets.append(Ether() / IP(src=RESOLVER, dst=CLIENT, id=int(rng.integers(0, 65536)), ttl=60)
                       / UDP(sport=53, dport=sport)
                       / DNS(id=query_id, qr=1, rd=1, ra=1, qd=DNSQR(qname=name, qtype=qtype), an=answer))

    # Pings: Linux style to the server, Windows style to the resolver; replies echo the payload
    for target, ttl, payloads in [
        (SERVER, 57, [_linux_ping_payload(rng) for _ in range(8)]),
        (RESOLVER, 60, [WINDOWS_PING] * 4),
    ]:
        ping_id = int(rng.integers(0, 65536))
        for seq, payload in enumerate(payloads, start=1):
            packets.append(from_client(target) / ICMP(type=8, id=ping_id, seq=seq) / Raw(payload))
            packets.append(Ether() / IP(src=target, dst=CLIENT, id=int(rng.integers(0, 65536)), ttl=ttl)
                           / ICMP(type=0, id=ping_id, seq=seq) / Raw(payload))
    return packets


def _covert_ip(rng: np.random.Generator, ip_id: int | None = None, ttl: int = 64):
    from scapy.layers.inet import IP
    from scapy.layers.l2 import Ether
    ip_id = int(rng.integers(0, 65536)) if ip_id is None else ip_id
    return Ether() / IP(src=COVERT_SRC, dst=COVERT_DST, id=ip_id, ttl=ttl)


def covert_ip_id_packets(payload: bytes, rng: np.random.Generator) -> list:
    """covert_tcp style: one byte per packet in the IP ID high byte (ID = byte * 256)."""
    from scapy.layers.inet import TCP
    return [_covert_ip(rng, ip_id=byte << 8)
            / TCP(sport=int(rng.integers(1024, 65536)), dport=80, flags="S", seq=int(rng.integers(0, 2**32)))
            for byte in payload]


def covert_isn_packets(payload: bytes, rng: np.random.Generator) -> list:
    """covert_tcp style: one byte per SYN in the top byte of the sequence number."""
    from scapy.layers.inet import TCP
    return [_covert_ip(rng) / TCP(sport=int(rng.integers(1024, 65536)), dport=80, flags="S", seq=byte << 24)
            for byte in payload]


def covert_reserved_bits_packets(payload: bytes, rng: np.random.Generator) -> list:
    """3 payload bits per packet in the TCP reserved bits (zero-padded to a multiple of 3)."""
    from scapy.layers.inet import TCP
    from scapy.packet import Raw
    bits = list(_bits_from_bytes(payload)) + [0] * (-len(payload) * 8 % 3)
    # int(): scapy silently drops numpy integer types for this bit field
    values = [int(bits[i] << 2 | bits[i + 1] << 1 | bits[i + 2]) for i in range(0, len(bits), 3)]
    sport = int(rng.integers(1024, 65536))
    return [_covert_ip(rng, ip_id=i + 1) / TCP(sport=sport, dport=80, flags="PA", reserved=value) / Raw(b"ok")
            for i, value in enumerate(values)]


def covert_urgent_pointer_packets(payload: bytes, rng: np.random.Generator) -> list:
    """2 payload bytes per packet in the urgent pointer, with the URG flag NOT set."""
    from scapy.layers.inet import TCP
    from scapy.packet import Raw
    padded = payload + b"\x00" * (len(payload) % 2)
    sport = int(rng.integers(1024, 65536))
    return [_covert_ip(rng, ip_id=i + 1)
            / TCP(sport=sport, dport=80, flags="PA", urgptr=int.from_bytes(padded[i * 2 : i * 2 + 2], "big")) / Raw(b"ok")
            for i in range(len(padded) // 2)]


def covert_ttl_packets(payload: bytes, rng: np.random.Generator) -> list:
    """One bit per packet: TTL 64 = 0, TTL 65 = 1."""
    from scapy.layers.inet import UDP
    return [_covert_ip(rng, ip_id=i + 1, ttl=64 + int(bit)) / UDP(sport=40000, dport=33434)
            for i, bit in enumerate(_bits_from_bytes(payload))]


def covert_dns_packets(labels: list[str], domain: str, qtype: str, rng: np.random.Generator) -> list:
    """One query per label: <label>.<domain>, with a short answer."""
    from scapy.layers.dns import DNS, DNSQR, DNSRR
    from scapy.layers.inet import IP, UDP
    from scapy.layers.l2 import Ether
    packets = []
    for i, label in enumerate(labels):
        name = f"{label}.{domain}"
        query_id = int(rng.integers(0, 65536))
        packets.append(Ether() / IP(src=COVERT_SRC, dst=RESOLVER, id=i + 1, ttl=64) / UDP(sport=50000 + i, dport=53)
                       / DNS(id=query_id, rd=1, qd=DNSQR(qname=name, qtype=qtype)))
        answer = DNSRR(rrname=name, type="TXT", rdata="ok") if qtype == "TXT" else DNSRR(rrname=name, type="A", rdata="198.51.100.31")
        packets.append(Ether() / IP(src=RESOLVER, dst=COVERT_SRC, id=int(rng.integers(0, 65536)), ttl=60)
                       / UDP(sport=53, dport=50000 + i)
                       / DNS(id=query_id, qr=1, rd=1, ra=1, qd=DNSQR(qname=name, qtype=qtype), an=answer))
    return packets


def covert_icmp_packets(payload: bytes, rng: np.random.Generator, chunk: int = 8) -> list:
    """Payload split across echo requests; replies echo each chunk back."""
    from scapy.layers.inet import ICMP, IP
    from scapy.layers.l2 import Ether
    from scapy.packet import Raw
    packets, ping_id = [], int(rng.integers(0, 65536))
    for seq, start in enumerate(range(0, len(payload), chunk), start=1):
        data = payload[start : start + chunk]
        packets.append(_covert_ip(rng, ip_id=seq) / ICMP(type=8, id=ping_id, seq=seq) / Raw(data))
        packets.append(Ether() / IP(src=COVERT_DST, dst=COVERT_SRC, id=int(rng.integers(0, 65536)), ttl=64)
                       / ICMP(type=0, id=ping_id, seq=seq) / Raw(data))
    return packets


def _base32_labels(data: bytes, size: int) -> list[str]:
    encoded = base64.b32encode(data).decode().rstrip("=").lower()
    return [encoded[i : i + size] for i in range(0, len(encoded), size)]


def generate_pcap_samples(output_dir: Path) -> list[Path]:
    """Generate every Phase 3 pcap fixture into output_dir. Returns the paths written."""
    from scapy.layers.l2 import ARP, Ether
    from scapy.utils import wrpcap

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9012)
    written: list[Path] = []

    def save(name: str, packets: list) -> Path:
        for i, packet in enumerate(packets):
            packet.time = _BASE_TIME + i * 0.005
        path = output_dir / name
        wrpcap(str(path), packets, linktype=1)  # 1 = Ethernet
        written.append(path)
        return path

    clean = make_clean_traffic(rng)
    save("clean_mixed.pcap", clean)

    # One covert channel per file (100%: the covert flow is all there is)
    save("ipid_payload.pcap", covert_ip_id_packets(PAYLOAD_MZ, rng))
    save("isn_payload.pcap", covert_isn_packets(PAYLOAD_MZ, rng))
    save("tcp_reserved.pcap", covert_reserved_bits_packets(PAYLOAD_MZ, rng))
    save("urgptr_payload.pcap", covert_urgent_pointer_packets(PAYLOAD_MZ, rng))
    save("ttl_channel.pcap", covert_ttl_packets(HIDDEN_MESSAGE, rng))
    save("dns_tunnel.pcap", covert_dns_packets(_base32_labels(DNS_TUNNEL_PAYLOAD, 16), TUNNEL_DOMAIN, "TXT", rng))
    random_labels = ["".join(rng.choice(_BASE32_ALPHABET, size=16)) for _ in range(15)]
    save("dns_random.pcap", covert_dns_packets(random_labels, "sync.example.test", "A", rng))
    save("icmp_payload.pcap", covert_icmp_packets(PAYLOAD_MZ, rng))

    # ~10%: a covert IP-ID flow hidden among normal traffic
    covert = covert_ip_id_packets(PAYLOAD_MZ, rng)
    step = len(clean) // len(covert)
    mixed = list(clean)
    for i, packet in enumerate(covert):
        mixed.insert(i * (step + 1), packet)
    save("mixed_10.pcap", mixed)

    # Edge cases
    (output_dir / "empty.pcap").write_bytes(b"")
    written.append(output_dir / "empty.pcap")
    save("header_only.pcap", [])
    header = (output_dir / "header_only.pcap").read_bytes()
    (output_dir / "corrupt.pcap").write_bytes(header + b"\xff" * 16 + b"garbage record")
    written.append(output_dir / "corrupt.pcap")
    save("arp_only.pcap", [Ether() / ARP(psrc=CLIENT, pdst=SERVER) for _ in range(10)])

    return written


def main() -> None:
    written = generate_all(OUTPUT_DIR)
    print(f"Wrote {len(written)} sample files to {OUTPUT_DIR}")
    written_text = generate_text_samples(TEXT_OUTPUT_DIR)
    print(f"Wrote {len(written_text)} sample files to {TEXT_OUTPUT_DIR}")
    written_pcap = generate_pcap_samples(PCAP_OUTPUT_DIR)
    print(f"Wrote {len(written_pcap)} sample files to {PCAP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
