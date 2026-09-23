# CLAUDE.md — StegoScan

## What this project is
StegoScan detects **steganography used to hide malware**: payloads hidden in image pixel bits (LSB), commands hidden in text whitespace, and covert C2 data hidden in network packet headers. It flags suspicious carriers and extracts hidden bytes so they can be inspected (never executed).

Owner: Dionté (CIS student, cybersecurity focus). This is a **learning + portfolio project**. Explaining *why* a detection works matters as much as the code itself.

## How to work with me (read first)
- **Plan before building.** For each phase, first write a short plan covering the features, functions, and test cases, and wait for my OK before writing code.
- **Ask, don't guess.** Use the AskUserQuestion tool whenever requirements are unclear.
- **Build by hand first.** No automation loops, agents, or auto-fix runs until the manual version works and I understand it.
- **Teach as you go.** When you add detection logic, explain the concept in 2–4 plain sentences (e.g., what a chi-square test is actually measuring). Point me to the lines that matter.
- **One phase at a time.** Finish one phase (code + tests passing) before starting the next.
- **Keep context lean.** Read only the files you need, and keep summaries short.

## Stack
- Python 3.11+
- Pillow, numpy, scipy: image loading and statistics
- scapy: reading `.pcap` files (offline only, no live sniffing)
- Flask: local web UI
- pytest: tests

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS (Apple Silicon)
pip install -r requirements.txt
```

## Commands
```bash
python -m stegoscan scan <file>          # auto-detects image / text / pcap
python -m stegoscan scan <file> --json   # machine-readable report
python -m stegoscan scan <file> --extract out/   # save any extracted payload bytes
python scripts/make_samples.py           # generate clean + stego test files
python -m stegoscan.web                  # web UI at http://127.0.0.1:5000
pytest -v
```

## Project layout
```
stegoscan/
  __init__.py
  __main__.py        # CLI entry (argparse)
  scanner.py         # detect file type (magic bytes -> extension) and route to a scanner
  report.py          # Finding + Report dataclasses, risk scoring, JSON/hex helpers
  signatures.py      # payload signature checks (MZ, ELF, shebang, PowerShell, base64, ZIP)
  bits.py            # shared bits -> bytes helper
  decoding.py        # signature/message judging of decoded bytes (text + net)
  image_scan.py      # Phase 1
  text_scan.py       # Phase 2
  net_scan.py        # Phase 3
  web.py             # Flask app (thin wrapper around the core)
  templates/ static/
scripts/make_samples.py
samples/             # generated test files (gitignored except a few small ones)
tests/
```
**Rule:** all detection logic lives in the core modules. The CLI and web UI only call it and display results. The web UI never contains detection logic.

## Shared report format
Every scanner returns a `Report` object:
- `verdict`: `CLEAN` | `SUSPICIOUS` | `LIKELY_PAYLOAD`
- `score`: 0–100
- `findings`: list of `Finding(check, severity, detail, evidence)`
- `extracted`: optional bytes (plus the method used to extract them)

## Phase 1: Images (LSB focus)
- **LSB extraction:** read the last bit of each channel in multiple orders (RGB interleaved, R/G/B separately, row vs column), rebuild the bytes, and run `signatures.py` on each result.
- **Chi-square attack:** LSB embedding evens out the counts of pixel-value pairs (2k, 2k+1). Test this in chunks and report the estimated embedding %.
- **RS analysis:** a second, stronger estimate of the embedding rate.
- **Appended data:** bytes after PNG `IEND` or JPEG `FFD9`. Identify them with signatures.
- **Metadata:** oversized or odd EXIF/comment/tEXt chunks.
- **Bit-planes:** render bit layers 0–7 as PNGs for the web UI.
- LSB checks apply to lossless formats (PNG, BMP). For JPEG, run only the appended-data and metadata checks, and state that limitation in the report.

## Phase 2: Text (whitespace)
- Trailing spaces and tabs per line: decode them as bits (space = 0, tab = 1; also try the reverse) and signature-check the output.
- Zero-width Unicode (U+200B, U+200C, U+200D, U+FEFF, U+2060): count them, locate them, and decode them.
- Flag files whose trailing whitespace is statistically unusual even when decoding fails.

## Phase 3: Network (pcap, offline)
- IP ID field: sequential vs. random vs. data-like patterns.
- TCP: initial sequence numbers encoding data, reserved/unused flag bits set, and odd urgent pointers.
- TTL: unusual variation within a single flow (a possible covert channel).
- DNS tunneling: long, high-entropy subdomains, high query volume to one domain, and TXT-heavy traffic.
- ICMP: oversized or non-standard echo payloads.
- `make_samples.py` builds synthetic pcaps with scapy that contain both normal and covert-channel traffic.

## Tests (pytest)
Each phase must have tests that confirm:
- clean files come back `CLEAN`, with no false positives,
- stego files are flagged at 100% and at about 10% embedding,
- hidden messages and fake payloads are extracted byte-for-byte,
- edge cases fail gracefully: a tiny image, grayscale, alpha channel, an empty file, a corrupt file.

## Safety rules (non-negotiable)
- **Never create, download, or embed real malware.** Test payloads are harmless fakes only: e.g. `b"MZ" + b"STEGOSCAN TEST PAYLOAD - harmless"`.
- **Never execute extracted bytes.** Only save them and show a hex/ASCII preview.
- Network analysis is **offline pcap only**, on synthetic captures or traffic I own. No sniffing on shared or school networks.
- The web UI binds to `127.0.0.1` only, enforces an upload size limit, and doesn't keep uploads after a scan.

## Code style
- Type hints and short docstrings on public functions.
- Prefer small, pure functions that are easy to test.
- Use numpy vectorization for pixel work, not Python loops over pixels.
- Use clear names over clever code. This is a learning codebase.

## Out of scope (v1)
JPEG DCT-domain stego (F5, OutGuess), audio/video stego, live packet capture, and sandboxing or detonating payloads.

## Status
- [x] Phase 1: Images: detection engine + CLI + tests done
  - Done: `report.py`, `signatures.py`, `make_samples.py`, `image_scan.py` (LSB multi-order, chi-square, RS, appended data, metadata, bit-planes, `scan_image()`)
  - CLI: `scanner.py` routes by magic bytes → extension; `__main__.py` exit codes 0 clean / 1 flagged / 2 error (incl. unreadable files). `--json` never contains raw payload bytes; `--extract` writes non-executable `.bin`
  - Chi-square/RS follow Fridrich, Goljan & Du (SPIE 2002). Stats only run on images >=128x128; RS threshold 10%, chi-square 5% (calibrated on synthetic covers)
- [x] Phase 2: Text: `text_scan.py` + samples (`samples/text/`) + tests done
  - Trailing whitespace decoded both ways; data-like check = >=8 mixed space/tab lines, >=4 distinct patterns, tab ratio 0.2-0.8
  - Zero-width: all ordered char pairs tried; BOM at offset 0 and ZWJ/ZWNJ between emoji or Arabic/Indic letters are ignored as legitimate
  - Scoring: payload/message/unexplained anomaly = high; an anomaly that decoded drops to info (no double count). Messages -> SUSPICIOUS, signature payloads -> LIKELY_PAYLOAD
- [x] Phase 3: Network: `net_scan.py` + samples (`samples/pcap/`) + tests done (156 passing total)
  - Checks: IP ID (constant/sequential/data_like/random), TCP ISN top byte, reserved bits, urgent pointer w/o URG, TTL switching, DNS tunneling (entropy + TXT ratio, base32/hex/base64url decode), ICMP non-standard/mismatched echo payloads
  - `pcap_records_intact()` catches truncated captures (scapy silently returns a partial record). scapy is imported only when a pcap is scanned
  - Known limits: base domain = last two labels (.co.uk wrong); IPv6 not analyzed; 1-16 byte ping payloads count as non-standard
  - Shared decode/judging logic lives in `decoding.py` (used by text + net)
- [ ] Web UI
- [ ] README + portfolio write-up
