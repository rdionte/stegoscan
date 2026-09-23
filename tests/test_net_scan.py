"""Tests for net_scan.py: covert channels in pcap files (offline, synthetic captures only)."""

import base64
from pathlib import Path

import pytest

from scripts.make_samples import DNS_TUNNEL_PAYLOAD, HIDDEN_MESSAGE, PAYLOAD_MZ, generate_pcap_samples
from stegoscan.net_scan import (
    classify_ip_ids,
    decode_dns_labels,
    decode_isns,
    decode_reserved_bits,
    decode_urgent_pointers,
    is_standard_ping,
    isns_look_like_data,
    pcap_records_intact,
    scan_pcap,
    shannon_entropy,
    split_domain,
    strip_shared_labels,
)
from stegoscan.report import CLEAN, LIKELY_PAYLOAD, SUSPICIOUS, UNREADABLE_CHECK


@pytest.fixture(scope="module")
def pcap_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("pcap")
    generate_pcap_samples(directory)
    return directory


def _checks(report):
    return {f.check for f in report.findings}


def _severity(report, check):
    return next(f.severity for f in report.findings if f.check == check)


# --- IP ID -----------------------------------------------------------------------

def test_classify_ip_ids():
    assert classify_ip_ids([0] * 20) == "constant"
    assert classify_ip_ids(list(range(1000, 1020))) == "sequential"
    assert classify_ip_ids([0x4100 + i for i in range(20)]) == "sequential"  # low byte printable, still a counter
    assert classify_ip_ids([b << 8 for b in b"MZ hidden in ip ids"]) == "data_like"
    assert classify_ip_ids([7, 60000, 1234, 45000, 300, 52000, 9000, 33000] * 2) == "random"


# --- TCP ----------------------------------------------------------------------------

def test_isn_detection_and_decoding():
    isns = [b << 24 for b in b"MZ covert"]
    assert isns_look_like_data(isns)
    assert decode_isns(isns)["top byte"] == b"MZ covert"
    assert not isns_look_like_data([0x1A2B3C4D, 0x9F8E7D6C, 0x0BADF00D, 0xDEADBEEF, 0x12345679,
                                    0x87654321, 0xCAFEBABE, 0x0F1E2D3C])


def test_reserved_bits_decoding_keeps_zero_groups():
    # 'M' = 010 011 01(0) -> groups 2, 3, 2 (last group zero-padded)
    assert decode_reserved_bits([2, 3, 2])["3 bits per packet"] == b"M"
    # a group of 000 in the middle must not be skipped: 'A' = 010 000 01(0)
    assert decode_reserved_bits([2, 0, 2])["3 bits per packet"] == b"A"


def test_urgent_pointer_decoding():
    assert decode_urgent_pointers([0x4D5A, 0x2100])["2 bytes per packet"] == b"MZ!"


# --- DNS ------------------------------------------------------------------------------

def test_shannon_entropy():
    assert shannon_entropy("aaaa") == 0
    assert shannon_entropy("abcd") == 2
    assert shannon_entropy("") == 0


def test_split_domain():
    assert split_domain("a.b.tunnel.example.test.") == ("a.b.tunnel", "example.test")
    assert split_domain("example.com") == ("", "example.com")


def test_strip_shared_labels():
    assert strip_shared_labels(["abc.t", "def.t"]) == ["abc", "def"]
    assert strip_shared_labels(["www", "mail"]) == ["www", "mail"]


def test_decode_dns_labels():
    encoded = base64.b32encode(b"MZ covert").decode().rstrip("=").lower()
    assert decode_dns_labels([encoded[:6], encoded[6:]])["base32"] == b"MZ covert"
    assert decode_dns_labels(["4d5a", "21"])["hex"] == b"MZ!"


# --- ICMP ------------------------------------------------------------------------------

def test_standard_ping_patterns():
    assert is_standard_ping(b"")
    assert is_standard_ping(b"abcdefghijklmnopqrstuvwabcdefghi")  # Windows
    assert is_standard_ping(bytes(16) + bytes(range(16, 56)))  # Linux
    assert not is_standard_ping(b"MZSTEGOS")
    assert not is_standard_ping(bytes(16) + b"not the ping pattern")


# --- Reading --------------------------------------------------------------------------

def test_truncated_record_is_not_intact(pcap_dir):
    assert pcap_records_intact(pcap_dir / "clean_mixed.pcap")
    assert not pcap_records_intact(pcap_dir / "corrupt.pcap")


# --- scan_pcap on samples ---------------------------------------------------------

@pytest.mark.parametrize("name", ["clean_mixed.pcap", "arp_only.pcap"])
def test_clean_captures_are_clean(pcap_dir, name):
    report = scan_pcap(pcap_dir / name)
    assert report.verdict == CLEAN
    assert report.score == 0
    assert report.extracted is None


@pytest.mark.parametrize("name, check, method, payload", [
    ("ipid_payload.pcap", "ip_id_data_like", "ip_id:high byte", PAYLOAD_MZ),
    ("isn_payload.pcap", "tcp_isn_data_like", "tcp_isn:top byte", PAYLOAD_MZ),
    ("tcp_reserved.pcap", "tcp_reserved_bits", "tcp_reserved:3 bits per packet", PAYLOAD_MZ),
    ("urgptr_payload.pcap", "tcp_urgent_pointer", "tcp_urgent:2 bytes per packet", PAYLOAD_MZ),
    ("dns_tunnel.pcap", "dns_tunneling", "dns:base32", DNS_TUNNEL_PAYLOAD),
    ("icmp_payload.pcap", "icmp_payload", "icmp:echo payloads", PAYLOAD_MZ),
    ("mixed_10.pcap", "ip_id_data_like", "ip_id:high byte", PAYLOAD_MZ),
])
def test_covert_payloads_extracted_byte_for_byte(pcap_dir, name, check, method, payload):
    report = scan_pcap(pcap_dir / name)
    assert report.verdict == LIKELY_PAYLOAD
    assert check in _checks(report)
    assert report.extracted == payload
    assert report.extraction_method == method
    assert _severity(report, check) == "info"  # explained by the decode, not scored twice


def test_ttl_channel_message(pcap_dir):
    report = scan_pcap(pcap_dir / "ttl_channel.pcap")
    assert report.verdict == SUSPICIOUS
    assert report.extracted == HIDDEN_MESSAGE
    assert "ttl_switching" in _checks(report)


def test_random_dns_flagged_without_decoding(pcap_dir):
    report = scan_pcap(pcap_dir / "dns_random.pcap")
    assert report.verdict == SUSPICIOUS
    assert report.extracted is None
    assert _severity(report, "dns_tunneling") == "high"


def test_mixed_10_is_about_ten_percent_covert(pcap_dir):
    from scapy.utils import rdpcap
    packets = rdpcap(str(pcap_dir / "mixed_10.pcap"))
    covert = sum(1 for p in packets if p.haslayer("IP") and p["IP"].src == "192.0.2.66")
    assert 0.05 <= covert / len(packets) <= 0.15


# --- edge cases ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["empty.pcap", "corrupt.pcap"])
def test_bad_captures_are_unreadable(pcap_dir, name):
    assert UNREADABLE_CHECK in _checks(scan_pcap(pcap_dir / name))


def test_header_only_capture_is_clean(pcap_dir):
    report = scan_pcap(pcap_dir / "header_only.pcap")
    assert report.verdict == CLEAN
    assert "no_packets" in _checks(report)


def test_missing_capture_is_unreadable(tmp_path):
    assert UNREADABLE_CHECK in _checks(scan_pcap(tmp_path / "gone.pcap"))
