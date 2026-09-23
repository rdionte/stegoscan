"""Shared reporting types + risk scoring used by every scanner (image/text/net).

Each scanner turns what it notices into Findings, then build_report()
combines them into a single verdict + score so the CLI, web UI, and
tests all speak the same format.
"""

import hashlib
from dataclasses import asdict, dataclass, field

SEVERITY_WEIGHTS = {"info": 0, "low": 10, "medium": 25, "high": 50}

CLEAN = "CLEAN"
SUSPICIOUS = "SUSPICIOUS"
LIKELY_PAYLOAD = "LIKELY_PAYLOAD"

# Every scanner uses this check name when it cannot parse its input, so the
# CLI/web can tell "could not analyze" apart from "analyzed and found nothing".
UNREADABLE_CHECK = "unreadable_file"


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


def is_unreadable(report: Report) -> bool:
    """True if the scanner could not parse the file at all."""
    return any(f.check == UNREADABLE_CHECK for f in report.findings)


def hex_preview(data: bytes, length: int = 64) -> str:
    """xxd-style hex + ASCII dump of the first `length` bytes (for display only)."""
    lines = []
    chunk = data[:length]
    for offset in range(0, len(chunk), 16):
        row = chunk[offset : offset + 16]
        hex_left = " ".join(f"{b:02x}" for b in row[:8])
        hex_right = " ".join(f"{b:02x}" for b in row[8:])
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        lines.append(f"{offset:08x}  {hex_left:<23}  {hex_right:<23}  |{ascii_part}|")
    if len(data) > length:
        lines.append(f"... ({len(data) - length} more bytes)")
    return "\n".join(lines)


def report_to_dict(report: Report) -> dict:
    """JSON-safe view of a Report. Raw extracted bytes are left out on purpose;
    only the length, hash, and a hex preview are included."""
    data = asdict(report)
    extracted = data.pop("extracted")
    data["extracted_length"] = len(extracted) if extracted else 0
    data["extracted_preview"] = hex_preview(extracted) if extracted else None
    return data
