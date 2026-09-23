"""Signature checks for candidate payload bytes pulled out of a carrier file.

Once a scanner extracts bytes (via LSB, appended-data, etc.), this module
answers "is this actually something, or just noise?" by checking for known
executable/script/archive magic bytes and obfuscation hints (base64,
PowerShell). Findings from here feed into report.build_report().
"""

import base64
import binascii
import re

from stegoscan.report import Finding

_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{40,}={0,2}")
_POWERSHELL_KEYWORDS = (b"powershell", b"-encodedcommand", b"frombase64string")

_MAGIC_CHECKS = [
    (b"MZ", "signature_mz", "Windows PE executable (MZ header)"),
    (b"\x7fELF", "signature_elf", "Linux ELF executable"),
    (b"#!", "signature_shebang", "shebang script"),
    (b"PK\x03\x04", "signature_zip", "ZIP archive"),
]


def _preview(data: bytes, length: int = 40) -> str:
    """A short, printable preview of raw bytes for report evidence."""
    return data[:length].decode("ascii", errors="replace")


def find_signatures(data: bytes) -> list[Finding]:
    """Scan data for known payload signatures. Returns one Finding per hit."""
    findings: list[Finding] = []

    for magic, check, label in _MAGIC_CHECKS:
        if data.startswith(magic):
            findings.append(Finding(
                check, "high", f"Data starts with a {label}.",
                {"offset": 0, "preview": _preview(data)},
            ))

    lowered = data.lower()
    for keyword in _POWERSHELL_KEYWORDS:
        idx = lowered.find(keyword)
        if idx != -1:
            findings.append(Finding(
                "signature_powershell", "medium",
                f"Found PowerShell indicator {keyword.decode()!r}.",
                {"offset": idx, "keyword": keyword.decode()},
            ))

    for match in _BASE64_RUN.finditer(data):
        candidate = match.group()
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except binascii.Error:
            continue
        findings.append(Finding(
            "signature_base64", "medium",
            f"Found a {len(candidate)}-byte base64-looking run that decodes cleanly.",
            {"offset": match.start(), "length": len(candidate), "decoded_preview": _preview(decoded)},
        ))

    return findings
