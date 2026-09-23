"""Phase 2: whitespace and zero-width Unicode steganography.

Text has two hiding spots a reader never sees:
- Trailing whitespace: spaces/tabs after the last visible character of a
  line, read as bits (space=0/tab=1, or the reverse). Stegsnow-style.
- Zero-width characters (U+200B, U+200C, U+200D, U+FEFF, U+2060): they
  take up no width on screen, so any two of them can encode 0 and 1.

Decoded bytes are signature-checked. When decoding fails, statistical
checks still flag whitespace or zero-width characters that look like data
rather than formatting. An anomaly that decoded successfully is reported
at 'info', so the decoded content isn't scored twice.
"""

import bisect
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

from stegoscan.bits import bits_to_bytes
from stegoscan.report import UNREADABLE_CHECK, Finding, Report, build_report
from stegoscan.signatures import find_signatures

ZWSP, ZWNJ, ZWJ, BOM, WORD_JOINER = (chr(c) for c in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060))
ZERO_WIDTH_CHARS = (ZWSP, ZWNJ, ZWJ, BOM, WORD_JOINER)
_ZERO_WIDTH_RE = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$")

# Data-like trailing whitespace: many lines mixing spaces and tabs, in varied
# patterns (mixed indentation repeats one pattern), with a coin-flip-ish tab
# ratio. Calibrated on the clean samples in make_samples.py.
MIN_MIXED_LINES = 8
MIN_DISTINCT_PATTERNS = 4
TAB_RATIO_RANGE = (0.2, 0.8)

MIN_ZERO_WIDTH = 8  # enough characters to carry one byte
MIN_MESSAGE_BYTES = 4
MESSAGE_PRINTABLE_RATIO = 0.9
MAX_LOCATIONS = 10  # zero-width locations listed in evidence

# Neighbors that make a joiner legitimate: emoji (ZWJ sequences like family
# emoji) and scripts that use ZWJ/ZWNJ for letter shaping (Arabic/Persian, Indic).
_EMOJI_RANGES = ((0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0xFE0F, 0xFE0F))
_JOINING_SCRIPT_RANGES = (
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFC), (0x0900, 0x0DFF),
)


@dataclass(frozen=True)
class ZeroWidthHit:
    """One zero-width character and where it sits (line/column are 1-based)."""

    char: str
    offset: int
    line: int
    column: int
    legitimate: bool


def read_text_safely(path: Path) -> str | None:
    """Decode the file as UTF-8. None for binary (NUL bytes) or invalid UTF-8."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --- Trailing whitespace ------------------------------------------------------

def trailing_whitespace(text: str) -> list[str]:
    """The trailing space/tab run of each line ('' if none). CR of CRLF is ignored."""
    runs = []
    for line in text.split("\n"):
        match = _TRAILING_WS_RE.search(line.rstrip("\r"))
        runs.append(match.group() if match else "")
    return runs


def decode_whitespace(runs: list[str]) -> dict[str, bytes]:
    """Read all trailing whitespace as one bit stream, under both mappings."""
    joined = "".join(runs)
    if not joined:
        return {}
    chars = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
    candidates = {
        "space=0/tab=1": bits_to_bytes(chars == ord("\t")),
        "tab=0/space=1": bits_to_bytes(chars == ord(" ")),
    }
    return {label: data for label, data in candidates.items() if data}


def whitespace_stats_finding(runs: list[str], explained: bool) -> Finding | None:
    """Judge whether trailing whitespace looks like data or like formatting.

    `explained` means decoding already turned it into a payload/message, so
    the anomaly is reported at 'info' instead of being scored again.
    """
    lines_with_ws = sum(1 for run in runs if run)
    if lines_with_ws == 0:
        return None

    mixed = [run for run in runs if " " in run and "\t" in run]
    joined = "".join(runs)
    tab_ratio = joined.count("\t") / len(joined)
    distinct_patterns = len(set(mixed))
    evidence = {
        "lines_with_trailing_ws": lines_with_ws,
        "mixed_space_tab_lines": len(mixed),
        "distinct_mixed_patterns": distinct_patterns,
        "tab_ratio": round(tab_ratio, 3),
    }

    data_like = (
        len(mixed) >= MIN_MIXED_LINES
        and distinct_patterns >= MIN_DISTINCT_PATTERNS
        and TAB_RATIO_RANGE[0] <= tab_ratio <= TAB_RATIO_RANGE[1]
    )
    if not data_like:
        return Finding(
            "whitespace_formatting", "info",
            f"Trailing whitespace on {lines_with_ws} lines looks like ordinary formatting.",
            evidence,
        )
    if explained:
        return Finding(
            "whitespace_data_like", "info",
            f"Trailing whitespace on {len(mixed)} lines looks like data; it decoded (see above).",
            evidence,
        )
    return Finding(
        "whitespace_data_like", "high",
        f"Trailing whitespace on {len(mixed)} lines looks like hidden data "
        f"({distinct_patterns} distinct space/tab patterns, tab ratio {tab_ratio:.2f}) "
        "but did not decode to a known payload or message; it may be encrypted or use another encoding.",
        evidence,
    )


# --- Zero-width characters ------------------------------------------------------

def _in_ranges(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in ranges)


def _is_legitimate(text: str, offset: int) -> bool:
    """A BOM at the very start, or a joiner between two emoji / two joining-script letters."""
    char = text[offset]
    if char == BOM and offset == 0:
        return True
    if char not in (ZWJ, ZWNJ) or offset == 0 or offset == len(text) - 1:
        return False
    before, after = text[offset - 1], text[offset + 1]
    if char == ZWJ and _in_ranges(before, _EMOJI_RANGES) and _in_ranges(after, _EMOJI_RANGES):
        return True
    return _in_ranges(before, _JOINING_SCRIPT_RANGES) and _in_ranges(after, _JOINING_SCRIPT_RANGES)


def find_zero_width(text: str) -> list[ZeroWidthHit]:
    """Every zero-width character with its location and whether it looks legitimate."""
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    hits = []
    for match in _ZERO_WIDTH_RE.finditer(text):
        offset = match.start()
        line_index = bisect.bisect_right(line_starts, offset) - 1
        hits.append(ZeroWidthHit(
            char=match.group(),
            offset=offset,
            line=line_index + 1,
            column=offset - line_starts[line_index] + 1,
            legitimate=_is_legitimate(text, offset),
        ))
    return hits


def _code_point(char: str) -> str:
    return f"U+{ord(char):04X}"


def decode_zero_width(hits: list[ZeroWidthHit]) -> dict[str, bytes]:
    """Try every ordered pair of the characters present as (0, 1); others are ignored."""
    present = sorted({hit.char for hit in hits})
    candidates = {}
    for zero, one in permutations(present, 2):
        sequence = [hit.char for hit in hits if hit.char in (zero, one)]
        data = bits_to_bytes(np.array([char == one for char in sequence], dtype=np.uint8))
        if data:
            candidates[f"{_code_point(zero)}=0/{_code_point(one)}=1"] = data
    return candidates


def zero_width_finding(hits: list[ZeroWidthHit], explained: bool) -> Finding | None:
    """Summarize zero-width characters: counts, locations, and how suspicious they are."""
    if not hits:
        return None
    suspicious = [hit for hit in hits if not hit.legitimate]
    evidence = {
        "suspicious_count": len(suspicious),
        "legitimate_count": len(hits) - len(suspicious),
        "characters": sorted({_code_point(hit.char) for hit in suspicious}),
        "locations": [
            {"line": hit.line, "column": hit.column, "char": _code_point(hit.char)}
            for hit in suspicious[:MAX_LOCATIONS]
        ],
    }
    if len(suspicious) < MIN_ZERO_WIDTH:
        detail = (f"{len(suspicious)} zero-width character(s) outside emoji/script joins "
                  f"(ignored {evidence['legitimate_count']} legitimate).")
        return Finding("zero_width_chars", "info", detail, evidence)
    if explained:
        return Finding("zero_width_chars", "info",
                       f"{len(suspicious)} hidden zero-width characters; they decoded (see above).",
                       evidence)
    return Finding(
        "zero_width_chars", "high",
        f"{len(suspicious)} hidden zero-width characters found (first at line "
        f"{suspicious[0].line}, column {suspicious[0].column}) but they did not decode "
        "to a known payload or message.",
        evidence,
    )


# --- Decoded content ------------------------------------------------------------

def looks_like_message(data: bytes) -> bool:
    """True if data is mostly printable ASCII (plus tab/newline) and long enough to mean something."""
    if len(data) < MIN_MESSAGE_BYTES:
        return False
    values = np.frombuffer(data, dtype=np.uint8)
    printable = ((values >= 32) & (values < 127)) | np.isin(values, (9, 10, 13))
    return printable.mean() >= MESSAGE_PRINTABLE_RATIO


@dataclass
class _Decoded:
    findings: list[Finding]
    data: bytes | None = None
    method: str | None = None
    is_payload: bool = False


def _check_candidates(candidates: dict[str, bytes], source: str, description: str) -> _Decoded:
    """Signature-check each decoded candidate; keep the first payload, else the first message."""
    result = _Decoded(findings=[])
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
        result.findings.append(Finding(
            f"{source}_payload_extracted", "high",
            f"Extracted a {len(result.data)}-byte payload with a known signature from "
            f"{description} ({result.method.split(':', 1)[1]}).",
            {"encoding": result.method.split(":", 1)[1], "length": len(result.data)},
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


def scan_text(path: Path) -> Report:
    """Scan one text file and return a Report. Never raises on bad input."""
    text = read_text_safely(path)
    if text is None:
        return build_report([Finding(
            UNREADABLE_CHECK, "low",
            "File could not be analyzed: it is not valid UTF-8 text (binary or corrupt).",
            {"file": path.name},
        )])
    if not text:
        return build_report([Finding("empty_file", "info", "File is empty: nothing to analyze.", {})])

    runs = trailing_whitespace(text)
    whitespace = _check_candidates(decode_whitespace(runs), "whitespace", "trailing whitespace")

    hits = find_zero_width(text)
    suspicious_hits = [hit for hit in hits if not hit.legitimate]
    zero_width = _check_candidates(decode_zero_width(suspicious_hits), "zero_width", "zero-width characters")

    findings = whitespace.findings + zero_width.findings
    for finding in (
        whitespace_stats_finding(runs, explained=whitespace.data is not None),
        zero_width_finding(hits, explained=zero_width.data is not None),
    ):
        if finding is not None:
            findings.append(finding)

    # Prefer a signature payload over a plain message; whitespace before zero-width.
    best = next(
        (d for d in (whitespace, zero_width) if d.is_payload),
        next((d for d in (whitespace, zero_width) if d.data is not None), None),
    )
    if best is None:
        return build_report(findings)
    return build_report(findings, best.data, best.method)
