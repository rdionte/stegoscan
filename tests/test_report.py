"""Tests for report.py: risk scoring and Report assembly."""

import hashlib

from stegoscan.report import (
    CLEAN,
    LIKELY_PAYLOAD,
    SUSPICIOUS,
    Finding,
    build_report,
    verdict_for_score,
)


def test_no_findings_is_clean():
    report = build_report([])
    assert report.score == 0
    assert report.verdict == CLEAN


def test_one_low_finding_stays_clean():
    report = build_report([Finding("check_a", "low", "detail", {})])
    assert report.score == 10
    assert report.verdict == CLEAN


def test_one_high_finding_is_suspicious():
    report = build_report([Finding("check_a", "high", "detail", {})])
    assert report.score == 50
    assert report.verdict == SUSPICIOUS


def test_two_high_findings_cap_at_100_and_are_likely_payload():
    findings = [Finding("a", "high", "d", {}), Finding("b", "high", "d", {})]
    report = build_report(findings)
    assert report.score == 100
    assert report.verdict == LIKELY_PAYLOAD


def test_verdict_thresholds():
    assert verdict_for_score(0) == CLEAN
    assert verdict_for_score(29) == CLEAN
    assert verdict_for_score(30) == SUSPICIOUS
    assert verdict_for_score(69) == SUSPICIOUS
    assert verdict_for_score(70) == LIKELY_PAYLOAD
    assert verdict_for_score(100) == LIKELY_PAYLOAD


def test_extracted_bytes_and_method_pass_through_with_hash():
    payload = b"MZ fake payload"
    report = build_report([], extracted=payload, extraction_method="lsb_interleaved")
    assert report.extracted == payload
    assert report.extraction_method == "lsb_interleaved"
    assert report.extracted_sha256 == hashlib.sha256(payload).hexdigest()


def test_no_extracted_bytes_means_no_hash():
    report = build_report([])
    assert report.extracted is None
    assert report.extracted_sha256 is None
