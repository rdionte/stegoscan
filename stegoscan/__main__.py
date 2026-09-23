"""CLI entry point: python -m stegoscan scan <file> [--json] [--extract DIR].

Display only: detection happens in the core modules (scanner.scan_file).
Exit codes follow the ClamAV convention: 0 clean, 1 flagged, 2 error.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from stegoscan.report import CLEAN, Report, hex_preview, is_unreadable, report_to_dict
from stegoscan.scanner import detect_file_type, scan_file

EXIT_CLEAN = 0
EXIT_FLAGGED = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with a single `scan` subcommand."""
    parser = argparse.ArgumentParser(
        prog="stegoscan",
        description="Detect steganography used to hide malware.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    scan = subcommands.add_parser("scan", help="scan one file (image / text / pcap)")
    scan.add_argument("file", type=Path, help="file to scan")
    scan.add_argument("--json", action="store_true", help="print a machine-readable report")
    scan.add_argument("--extract", type=Path, metavar="DIR",
                      help="save any extracted payload bytes into DIR (never executed)")
    return parser


def format_report(report: Report, path: Path, file_type: str) -> str:
    """Human-readable report text."""
    lines = [
        f"File:     {path.name}  ({file_type})",
        f"Verdict:  {report.verdict}   score {report.score}/100",
        "",
        "Findings:" if report.findings else "Findings: none",
    ]
    for finding in report.findings:
        lines.append(f"  [{finding.severity.upper():<6}] {finding.check:<30} {finding.detail}")

    if report.extracted:
        lines += [
            "",
            f"Extracted payload ({report.extraction_method}, {len(report.extracted)} bytes, "
            f"sha256 {report.extracted_sha256}):",
        ]
        lines += ["  " + line for line in hex_preview(report.extracted).splitlines()]
    return "\n".join(lines)


def save_extracted(report: Report, path: Path, out_dir: Path) -> Path | None:
    """Write extracted bytes to out_dir/<stem>.<method>.bin. Returns the path, or
    None if there was nothing to extract. Always .bin, never executable."""
    if not report.extracted:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    method = re.sub(r"[^A-Za-z0-9_-]", "_", report.extraction_method or "payload")
    out_path = out_dir / f"{path.stem}.{method}.bin"
    out_path.write_bytes(report.extracted)
    out_path.chmod(0o644)  # read/write only: no execute bit
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    args = build_parser().parse_args(argv)
    path: Path = args.file

    if not path.is_file():
        print(f"stegoscan: error: file not found: {path}", file=sys.stderr)
        return EXIT_ERROR

    try:
        file_type = detect_file_type(path)
        report = scan_file(path)
    except (NotImplementedError, ValueError) as error:
        print(f"stegoscan: error: {error}", file=sys.stderr)
        return EXIT_ERROR

    unreadable = is_unreadable(report)
    if args.json:
        print(json.dumps({"file": path.name, "file_type": file_type, **report_to_dict(report)},
                         indent=2))
    elif not unreadable:
        # An unreadable file has no meaningful verdict, so skip the report.
        print(format_report(report, path, file_type))

    if args.extract is not None:
        saved = save_extracted(report, path, args.extract)
        message = f"Saved payload to {saved}" if saved else "Nothing to extract."
        # Keep stdout pure JSON when --json is used.
        print(message, file=sys.stderr if args.json else sys.stdout)

    if unreadable:
        print(f"stegoscan: error: could not analyze {path.name} (empty, corrupt, "
              "or unsupported)", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_CLEAN if report.verdict == CLEAN else EXIT_FLAGGED


if __name__ == "__main__":
    sys.exit(main())
