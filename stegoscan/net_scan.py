"""Phase 3: covert channels in pcap files (IP ID, TCP, TTL, DNS, ICMP). Offline only.

Covert channels hide data in header fields that the network carries but
nobody checks closely. Each check asks "does this field behave like normal
traffic, or like data?" and then tries to decode it:
- IP ID: normally counts up, stays 0, or is random; data looks like text bytes.
- TCP ISN: normally random; covert_tcp puts one character in the top byte.
- TCP reserved bits / urgent pointer without URG: normal stacks leave them 0.
- TTL: steady within a flow; flipping between two values is a bit stream.
- DNS: many long, high-entropy subdomains or mostly TXT queries to one domain.
- ICMP echo: payloads matching no known ping pattern, or replies that don't echo.

Scoring follows text_scan: a decoded payload/message is 'high'; an anomaly
that doesn't decode is 'high'; an anomaly that did decode drops to 'info'.

Captures are read from disk with scapy. Nothing here sniffs live traffic.
"""

import base64
import binascii
import logging
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)  # quiet scapy's import-time warnings
from scapy.layers.dns import DNS  # noqa: E402
from scapy.layers.inet import ICMP, IP, TCP  # noqa: E402
from scapy.packet import Packet, Raw  # noqa: E402
from scapy.utils import rdpcap  # noqa: E402

from stegoscan.bits import bits_to_bytes  # noqa: E402
from stegoscan.decoding import Decoded, check_candidates, pick_best  # noqa: E402
from stegoscan.report import UNREADABLE_CHECK, Finding, Report, build_report  # noqa: E402

MAX_PACKETS = 200_000  # stop reading here so a huge capture can't hang the scan
MIN_FLOW_PACKETS = 16  # shorter flows are too small to judge IP ID / TTL patterns
PRINTABLE_RATIO = 0.9

# IP ID counters step forward by small amounts (other flows from the same host
# may take IDs in between).
SEQUENTIAL_MAX_STEP = 1024
SEQUENTIAL_RATIO = 0.8

MIN_SYNS = 8
MIN_ODD_TCP = 8  # packets with reserved bits / stray urgent pointer before 'high'
TTL_SWITCH_RATIO = 0.2  # a route change switches once; a TTL channel switches constantly

# DNS tunneling: calibrated on the clean capture (ordinary names plus a few
# hash-like CDN hosts). Base32 labels of 16 chars score ~3.5-4 bits/char.
MIN_DNS_SUBDOMAINS = 10
MIN_DNS_LABEL_LENGTH = 12
DNS_ENTROPY_THRESHOLD = 3.0
TXT_RATIO = 0.5
DNS_TYPE_TXT = 16

MIN_ICMP_ODD = 3
WINDOWS_PING_PATTERN = b"abcdefghijklmnopqrstuvw"
PING_TIMESTAMP_BYTES = 16

_PCAP_MAGIC_ENDIAN = {
    b"\xd4\xc3\xb2\xa1": "<", b"\x4d\x3c\xb2\xa1": "<",
    b"\xa1\xb2\xc3\xd4": ">", b"\xa1\xb2\x3c\x4d": ">",
}

FlowKey = tuple[str, str, int]  # (source IP, destination IP, IP protocol number)


# --- Reading ----------------------------------------------------------------------

def pcap_records_intact(path: Path) -> bool:
    """Walk classic-pcap record headers; False if any record runs past the end of the file.

    scapy silently returns a truncated final record as a packet, so a corrupt
    capture would otherwise look like (clean) traffic. pcapng is left to scapy.
    """
    data = path.read_bytes()
    endian = _PCAP_MAGIC_ENDIAN.get(data[:4])
    if endian is None:
        return True
    offset = 24  # global header
    while offset < len(data):
        if offset + 16 > len(data):
            return False
        _, _, captured_length, _ = struct.unpack(endian + "IIII", data[offset : offset + 16])
        offset += 16 + captured_length
    return offset == len(data)


def read_packets(path: Path) -> list[Packet] | None:
    """Read up to MAX_PACKETS packets. None if the capture is empty, truncated, or corrupt."""
    try:
        if not pcap_records_intact(path):
            return None
        return list(rdpcap(str(path), count=MAX_PACKETS))
    except Exception:  # scapy raises several unrelated exception types for bad captures
        return None


def group_flows(packets: list[Packet]) -> dict[FlowKey, list[Packet]]:
    """Group IPv4 packets by (source, destination, protocol), keeping capture order."""
    flows: dict[FlowKey, list[Packet]] = defaultdict(list)
    for packet in packets:
        ip = packet[IP]
        flows[(ip.src, ip.dst, ip.proto)].append(packet)
    return dict(flows)


def _flow_label(key: FlowKey) -> str:
    return f"{key[0]} -> {key[1]}"


def _printable_ratio(values: np.ndarray) -> float:
    return float(np.mean((values >= 32) & (values < 127)))


def _judge(check: str, detail: str, evidence: dict, candidates: dict[str, bytes],
           source: str, description: str) -> Decoded:
    """Decode an anomaly; it is 'high' unless decoding explained it (then 'info')."""
    decoded = check_candidates(candidates, source, description)
    if decoded.data is not None:
        decoded.findings.append(Finding(check, "info", f"{detail}; it decoded (see above).", evidence))
    else:
        decoded.findings.append(Finding(
            check, "high", f"{detail}, but it did not decode to a known payload or message.", evidence,
        ))
    return decoded


# --- IP ID --------------------------------------------------------------------------

def classify_ip_ids(ids: list[int]) -> str:
    """'constant', 'sequential', 'data_like', or 'random'.

    Sequential is checked before data_like: a counter's low byte can pass
    through the printable range for a while without being data.
    """
    values = np.array(ids, dtype=np.int64)
    if len(set(ids)) == 1:
        return "constant"
    steps = np.diff(values) % 65536
    if np.mean((steps >= 1) & (steps <= SEQUENTIAL_MAX_STEP)) >= SEQUENTIAL_RATIO:
        return "sequential"
    if max(_printable_ratio(values >> 8), _printable_ratio(values & 0xFF)) >= PRINTABLE_RATIO:
        return "data_like"
    return "random"


def decode_ip_ids(ids: list[int]) -> dict[str, bytes]:
    """Read one byte per packet from the high or low byte, or both bytes."""
    return {
        "high byte": bytes(i >> 8 for i in ids),
        "low byte": bytes(i & 0xFF for i in ids),
        "both bytes": b"".join(i.to_bytes(2, "big") for i in ids),
    }


def ip_id_findings(flows: dict[FlowKey, list[Packet]]) -> list[Decoded]:
    results = []
    for key, packets in flows.items():
        if len(packets) < MIN_FLOW_PACKETS:
            continue
        ids = [packet[IP].id for packet in packets]
        if classify_ip_ids(ids) != "data_like":
            continue
        results.append(_judge(
            "ip_id_data_like",
            f"IP ID values in {_flow_label(key)} ({len(ids)} packets) look like data, "
            "not a counter or random numbers",
            {"flow": _flow_label(key), "packets": len(ids), "first_ids": [hex(i) for i in ids[:8]]},
            decode_ip_ids(ids), "ip_id", f"IP ID field of {_flow_label(key)}",
        ))
    return results


# --- TCP -----------------------------------------------------------------------------

def _is_syn(packet: Packet) -> bool:
    flags = packet[TCP].flags
    return bool(flags.S) and not flags.A


def isns_look_like_data(isns: list[int]) -> bool:
    """Random ISNs almost never have three zero low bytes or a printable top byte every time."""
    values = np.array(isns, dtype=np.uint64)
    low_bytes_zero = float(np.mean((values & 0xFFFFFF) == 0))
    return max(low_bytes_zero, _printable_ratio(values >> 24)) >= PRINTABLE_RATIO


def decode_isns(isns: list[int]) -> dict[str, bytes]:
    return {
        "top byte": bytes(i >> 24 for i in isns),
        "all 4 bytes": b"".join(i.to_bytes(4, "big") for i in isns),
    }


def isn_findings(packets: list[Packet]) -> list[Decoded]:
    """Group SYNs by source host (each connection is a new flow) and judge their ISNs."""
    syns_by_host: dict[str, list[int]] = defaultdict(list)
    for packet in packets:
        if packet.haslayer(TCP) and _is_syn(packet):
            syns_by_host[packet[IP].src].append(packet[TCP].seq)

    results = []
    for host, isns in syns_by_host.items():
        if len(isns) < MIN_SYNS or not isns_look_like_data(isns):
            continue
        results.append(_judge(
            "tcp_isn_data_like",
            f"Initial sequence numbers of {len(isns)} SYNs from {host} look like data, not random",
            {"host": host, "syns": len(isns), "first_isns": [hex(i) for i in isns[:8]]},
            decode_isns(isns), "tcp_isn", f"TCP ISNs from {host}",
        ))
    return results


def decode_reserved_bits(values: list[int]) -> dict[str, bytes]:
    """3 bits per packet, most significant first."""
    bits = np.array([(value >> shift) & 1 for value in values for shift in (2, 1, 0)], dtype=np.uint8)
    return {"3 bits per packet": bits_to_bytes(bits)}


def decode_urgent_pointers(values: list[int]) -> dict[str, bytes]:
    """2 bytes per packet. Trailing zero bytes are treated as padding and dropped."""
    return {"2 bytes per packet": b"".join(v.to_bytes(2, "big") for v in values).rstrip(b"\x00")}


def tcp_header_findings(flows: dict[FlowKey, list[Packet]]) -> list[Decoded]:
    """Reserved bits set, or an urgent pointer without the URG flag."""
    results = []
    for key, packets in flows.items():
        tcp_packets = [packet for packet in packets if packet.haslayer(TCP)]
        # Every packet's value is decoded (a zero value can be data too); the
        # count of non-zero values decides whether the flow is suspicious.
        reserved = [p[TCP].reserved for p in tcp_packets]
        urgent = [p[TCP].urgptr for p in tcp_packets if not p[TCP].flags.U]

        for values, check, what, decoder, source in (
            (reserved, "tcp_reserved_bits", "have the reserved header bits set", decode_reserved_bits, "tcp_reserved"),
            (urgent, "tcp_urgent_pointer", "carry an urgent pointer without the URG flag",
             decode_urgent_pointers, "tcp_urgent"),
        ):
            odd_count = sum(1 for value in values if value)
            if not odd_count:
                continue
            detail = f"{odd_count} TCP packets in {_flow_label(key)} {what}"
            evidence = {"flow": _flow_label(key), "packets": odd_count}
            if odd_count < MIN_ODD_TCP:
                results.append(Decoded(findings=[Finding(check, "info", detail + " (too few to judge).", evidence)]))
            else:
                results.append(_judge(check, detail, evidence, decoder(values), source,
                                      f"{check.replace('_', ' ')} of {_flow_label(key)}"))
    return results


# --- TTL -------------------------------------------------------------------------------

def ttl_findings(flows: dict[FlowKey, list[Packet]]) -> list[Decoded]:
    """Flag flows whose TTL keeps switching; decode two-value flows as bits."""
    results = []
    for key, packets in flows.items():
        if len(packets) < MIN_FLOW_PACKETS:
            continue
        ttls = np.array([packet[IP].ttl for packet in packets])
        distinct = sorted(set(ttls.tolist()))
        switch_ratio = float(np.mean(np.diff(ttls) != 0))
        if len(distinct) < 2 or switch_ratio < TTL_SWITCH_RATIO:
            continue
        candidates = {}
        if len(distinct) == 2:
            low, high = distinct
            candidates[f"TTL {low}=0/{high}=1"] = bits_to_bytes((ttls == high).astype(np.uint8))
            candidates[f"TTL {high}=0/{low}=1"] = bits_to_bytes((ttls == low).astype(np.uint8))
        results.append(_judge(
            "ttl_switching",
            f"TTL in {_flow_label(key)} switches between {distinct} on {switch_ratio:.0%} of "
            "packets (a normal flow keeps a steady TTL)",
            {"flow": _flow_label(key), "packets": len(packets), "ttl_values": distinct,
             "switch_ratio": round(switch_ratio, 3)},
            candidates, "ttl", f"TTL of {_flow_label(key)}",
        ))
    return results


# --- DNS ---------------------------------------------------------------------------------

def shannon_entropy(text: str) -> float:
    """Average bits of information per character (0 for 'aaaa', 2 for 'abcd')."""
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def split_domain(name: str) -> tuple[str, str]:
    """Split into (subdomain, base domain), using the last two labels as the base.

    Simplification: suffixes like .co.uk get the wrong base domain (see README).
    """
    labels = name.rstrip(".").lower().split(".")
    return ".".join(labels[:-2]), ".".join(labels[-2:])


def strip_shared_labels(subdomains: list[str]) -> list[str]:
    """Remove trailing labels every subdomain shares ('x.tunnel', 'y.tunnel' -> 'x', 'y').

    Those labels are part of the tunnel's domain, not the data.
    """
    parts = [sub.split(".") for sub in subdomains]
    while parts and all(len(p) > 1 for p in parts) and len({p[-1] for p in parts}) == 1:
        parts = [p[:-1] for p in parts]
    return ["".join(p) for p in parts]


def dns_queries(packets: list[Packet]) -> list[tuple[str, int]]:
    """(query name, query type) for every DNS question, in capture order."""
    queries = []
    for packet in packets:
        if not packet.haslayer(DNS) or packet[DNS].qr != 0:
            continue
        for question in packet[DNS].qd or []:
            name = question.qname.decode("ascii", errors="replace") if isinstance(question.qname, bytes) \
                else str(question.qname)
            queries.append((name, int(question.qtype)))
    return queries


def _pad(text: str, block: int) -> str:
    return text + "=" * (-len(text) % block)


def decode_dns_labels(labels: list[str]) -> dict[str, bytes]:
    """Join subdomain labels in order and try common tunnel encodings."""
    joined = "".join(labels)
    decoders = {
        "base32": lambda s: base64.b32decode(_pad(s.upper(), 8)),
        "hex": bytes.fromhex,
        "base64url": lambda s: base64.urlsafe_b64decode(_pad(s, 4)),
    }
    candidates = {}
    for label, decoder in decoders.items():
        try:
            data = decoder(joined)
        except (ValueError, binascii.Error):
            continue
        if data:
            candidates[label] = data
    return candidates


def dns_findings(packets: list[Packet]) -> list[Decoded]:
    by_domain: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, qtype in dns_queries(packets):
        subdomain, base = split_domain(name)
        by_domain[base].append((subdomain, qtype))

    results = []
    for base, queries in by_domain.items():
        # Decode every query in order, dropping only immediate repeats (retransmissions);
        # data can legitimately repeat later. Statistics use the unique labels.
        in_order = [sub for i, (sub, _) in enumerate(queries) if sub and (i == 0 or sub != queries[i - 1][0])]
        stream = strip_shared_labels(in_order)
        labels = list(dict.fromkeys(stream))
        unique = list(dict.fromkeys(in_order))
        mean_length = float(np.mean([len(label) for label in labels])) if labels else 0.0
        mean_entropy = float(np.mean([shannon_entropy(label) for label in labels])) if labels else 0.0
        txt_ratio = sum(qtype == DNS_TYPE_TXT for _, qtype in queries) / len(queries)

        reasons = []
        if (len(labels) >= MIN_DNS_SUBDOMAINS and mean_length >= MIN_DNS_LABEL_LENGTH
                and mean_entropy >= DNS_ENTROPY_THRESHOLD):
            reasons.append(f"{len(labels)} long, high-entropy subdomains")
        if len(queries) >= MIN_DNS_SUBDOMAINS and txt_ratio >= TXT_RATIO:
            reasons.append(f"{txt_ratio:.0%} TXT queries")
        if not reasons:
            continue
        results.append(_judge(
            "dns_tunneling",
            f"DNS queries to {base} look like tunneling ({', '.join(reasons)})",
            {"domain": base, "queries": len(queries), "unique_subdomains": len(labels),
             "mean_label_length": round(mean_length, 1), "mean_entropy": round(mean_entropy, 2),
             "txt_ratio": round(txt_ratio, 2), "examples": unique[:3]},
            decode_dns_labels(stream), "dns", f"DNS subdomains of {base}",
        ))
    return results


# --- ICMP ---------------------------------------------------------------------------------

def is_standard_ping(payload: bytes) -> bool:
    """Empty, the Windows alphabet pattern, or Linux/BSD (timestamp, then byte i = i).

    Payloads of 1-16 bytes can't be verified against either pattern, so they
    count as non-standard.
    """
    if not payload:
        return True
    windows = (WINDOWS_PING_PATTERN * (len(payload) // len(WINDOWS_PING_PATTERN) + 1))[: len(payload)]
    if payload == windows:
        return True
    tail = payload[PING_TIMESTAMP_BYTES:]
    return bool(tail) and tail == bytes(i & 0xFF for i in range(PING_TIMESTAMP_BYTES, len(payload)))


def icmp_findings(packets: list[Packet]) -> list[Decoded]:
    """Flag echo traffic with non-standard payloads or replies that don't echo the request."""
    requests: dict[tuple, bytes] = {}
    odd_payloads: dict[tuple[str, str], list[bytes]] = defaultdict(list)
    mismatches: Counter = Counter()

    for packet in packets:
        if not packet.haslayer(ICMP) or packet[ICMP].type not in (0, 8):
            continue
        icmp, src, dst = packet[ICMP], packet[IP].src, packet[IP].dst
        payload = packet[Raw].load if packet.haslayer(Raw) else b""  # Raw excludes Ethernet padding
        if icmp.type == 8:
            requests[(src, dst, icmp.id, icmp.seq)] = payload
            if not is_standard_ping(payload):
                odd_payloads[(src, dst)].append(payload)
            continue
        request = requests.get((dst, src, icmp.id, icmp.seq))
        if request is not None and request != payload:
            mismatches[(src, dst)] += 1
            odd_payloads[(src, dst)].append(payload)
        elif request is None and not is_standard_ping(payload):
            odd_payloads[(src, dst)].append(payload)  # unsolicited reply

    results = []
    for (src, dst), payloads in odd_payloads.items():
        if len(payloads) < MIN_ICMP_ODD and not mismatches[(src, dst)]:
            continue
        label = f"{src} -> {dst}"
        results.append(_judge(
            "icmp_payload",
            f"{len(payloads)} ICMP echo packets in {label} carry non-standard payloads"
            + (f" ({mismatches[(src, dst)]} replies don't match their request)" if mismatches[(src, dst)] else ""),
            {"flow": label, "packets": len(payloads), "reply_mismatches": mismatches[(src, dst)],
             "max_payload_bytes": max(len(p) for p in payloads)},
            {"echo payloads": b"".join(payloads)}, "icmp", f"ICMP payloads of {label}",
        ))
    return results


# --- Scan ------------------------------------------------------------------------------------

def scan_pcap(path: Path) -> Report:
    """Scan one capture file and return a Report. Never raises on bad input."""
    packets = read_packets(path)
    if packets is None:
        return build_report([Finding(
            UNREADABLE_CHECK, "low",
            "File could not be analyzed: the capture is empty, truncated, or corrupt.",
            {"file": path.name},
        )])
    if not packets:
        return build_report([Finding("no_packets", "info", "Capture contains no packets.", {})])

    ip_packets = [packet for packet in packets if packet.haslayer(IP)]
    summary = Finding(
        "traffic_summary", "info",
        f"Analyzed {len(ip_packets)} IPv4 packets out of {len(packets)} "
        "(IPv6 and non-IP traffic are not analyzed).",
        {"packets": len(packets), "ipv4_packets": len(ip_packets), "truncated_at_limit": len(packets) >= MAX_PACKETS},
    )
    flows = group_flows(ip_packets)
    results = [
        *ip_id_findings(flows),
        *isn_findings(ip_packets),
        *tcp_header_findings(flows),
        *ttl_findings(flows),
        *dns_findings(ip_packets),
        *icmp_findings(ip_packets),
    ]
    findings = [finding for result in results for finding in result.findings] + [summary]
    best = pick_best(results)
    if best is None:
        return build_report(findings)
    return build_report(findings, best.data, best.method)
