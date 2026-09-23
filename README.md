# StegoScan

Detects steganography used to hide malware in images (LSB), text files (whitespace), and network captures (covert channels).

> Work in progress. See `CLAUDE.md` for the full plan.

## Setup (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests
```bash
python scripts/make_samples.py   # generate clean + stego test images
pytest -v
```

## Status
- [ ] Phase 1: Images (detection + tests done, CLI in progress)
- [ ] Phase 2: Text
- [ ] Phase 3: Network
- [ ] Web UI
