"""Shared reporting types + risk scoring used by every scanner (image/text/net).

Each scanner turns what it notices into Findings, then build_report()
combines them into a single verdict + score so the CLI, web UI, and
tests all speak the same format.
"""

import hashlib
from dataclasses import dataclass, field

SEVERITY_WEIGHTS = {"info": 0, "low": 10, "medium": 25, "high": 50}

CLEAN = "CLEAN"
SUSPICIOUS = "SUSPICIOUS"
LIKELY_PAYLOAD = "LIKELY_PAYLOAD"


@dataclass
class Finding:
    """One suspicious thing a scanner noticed."""

    check: str
    severity: str
    detail: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Report:
    """The overall result of scanning one file."""

    verdict: str
    score: int
    findings: list[Finding]
    extracted: bytes | None = None
    extraction_method: str | None = None
    extracted_sha256: str | None = None


def score_findings(findings: list[Finding]) -> int:
    """Sum per-severity weights into a 0-100 risk score."""
    total = sum(SEVERITY_WEIGHTS[f.severity] for f in findings)
    return min(total, 100)


def verdict_for_score(score: int) -> str:
    """Map a 0-100 score onto the three-tier verdict."""
    if score >= 70:
        return LIKELY_PAYLOAD
    if score >= 30:
        return SUSPICIOUS
    return CLEAN


def build_report(
    findings: list[Finding],
    extracted: bytes | None = None,
    extraction_method: str | None = None,
) -> Report:
    """Assemble a Report from findings, scoring and hashing extracted bytes as needed."""
    score = score_findings(findings)
    verdict = verdict_for_score(score)
    extracted_sha256 = hashlib.sha256(extracted).hexdigest() if extracted else None
    return Report(
        verdict=verdict,
        score=score,
        findings=findings,
        extracted=extracted,
        extraction_method=extraction_method,
        extracted_sha256=extracted_sha256,
    )
