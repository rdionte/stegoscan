"""Judging decoded bytes, shared by the text and network scanners.

A scanner decodes a hiding spot several ways ("candidates"). Each candidate
is signature-checked; the first one with a payload signature wins, otherwise
the first readable hidden message. Either becomes a 'high' finding.
"""

from dataclasses import dataclass

import numpy as np

from stegoscan.report import Finding
from stegoscan.signatures import find_signatures

MIN_MESSAGE_BYTES = 4
MESSAGE_PRINTABLE_RATIO = 0.9


@dataclass
class Decoded:
    """Findings from one hiding spot, plus the bytes worth extracting (if any)."""

    findings: list[Finding]
    data: bytes | None = None
    method: str | None = None
    is_payload: bool = False


def looks_like_message(data: bytes) -> bool:
    """True if data is mostly printable ASCII (plus tab/newline) and long enough to mean something."""
    if len(data) < MIN_MESSAGE_BYTES:
        return False
    values = np.frombuffer(data, dtype=np.uint8)
    printable = ((values >= 32) & (values < 127)) | np.isin(values, (9, 10, 13))
    return printable.mean() >= MESSAGE_PRINTABLE_RATIO


def check_candidates(candidates: dict[str, bytes], source: str, description: str) -> Decoded:
    """Signature-check each decoded candidate; keep the first payload, else the first message.

    `source` prefixes check names and the extraction method (e.g. "whitespace");
    `description` is the human wording used in details (e.g. "trailing whitespace").
    """
    result = Decoded(findings=[])
    message: tuple[str, bytes] | None = None

    for label, data in candidates.items():
        sig_findings = find_signatures(data)
        for f in sig_findings:
            result.findings.append(Finding(
                f.check, f.severity, f"{f.detail} ({description}: {label})",
                {**f.evidence, "encoding": label},
            ))
        if sig_findings and result.data is None:
            result.data, result.method, result.is_payload = data, f"{source}:{label}", True
        elif message is None and looks_like_message(data):
            message = (label, data)

    if result.data is not None:
        label = result.method.split(":", 1)[1]
        result.findings.append(Finding(
            f"{source}_payload_extracted", "high",
            f"Extracted a {len(result.data)}-byte payload with a known signature from "
            f"{description} ({label}).",
            {"encoding": label, "length": len(result.data)},
        ))
    elif message is not None:
        label, data = message
        preview = data[:60].decode("ascii", errors="replace")
        result.data, result.method = data, f"{source}:{label}"
        result.findings.append(Finding(
            f"{source}_hidden_message", "high",
            f"Decoded a {len(data)}-byte hidden text message from {description} ({label}): {preview!r}",
            {"encoding": label, "length": len(data), "preview": preview},
        ))
    return result


def pick_best(results: list[Decoded]) -> Decoded | None:
    """The extraction to report: the first signature payload, else the first message."""
    for result in results:
        if result.is_payload:
            return result
    return next((result for result in results if result.data is not None), None)
