"""Flask web UI: thin wrapper around the core scanners.

Display only: detection happens in the core modules (scanner.scan_file).
Uploads live in a temporary directory for the length of one scan and are
deleted afterwards. Run with `python -m stegoscan.web` (127.0.0.1 only).
"""

import argparse
import base64
import io
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from stegoscan.image_scan import is_lossless, read_image_safely, render_bit_planes
from stegoscan.report import Report, hex_preview, is_unreadable, report_to_dict
from stegoscan.scanner import IMAGE, detect_file_type, scan_file

HOST = "127.0.0.1"  # never 0.0.0.0: the UI must not be reachable from the network
PORT = 5000
DEFAULT_MAX_UPLOAD_MB = 10
MAX_PLANE_SIDE = 512  # bit-plane previews are shrunk to at most this many pixels per side

# No scripts, no third-party content. Images may be data: URIs (the bit-planes).
CONTENT_SECURITY_POLICY = "default-src 'self'; img-src 'self' data:; form-action 'self'"


class UploadError(Exception):
    """A request we can't scan (no file, unknown type). Shown to the user as a 400."""


def bit_planes_as_data_uris(path: Path) -> list[str]:
    """Bit-planes 0 (LSB) to 7 as PNG data URIs, or [] if LSB checks don't apply.

    Nearest-neighbour resizing keeps each pixel's 0/1 value, so plane 0 still
    looks like static on a stego image instead of blurring into grey.
    """
    image = read_image_safely(path)
    if image is None or not is_lossless(image):
        return []
    uris = []
    for plane in render_bit_planes(image).values():
        plane.thumbnail((MAX_PLANE_SIDE, MAX_PLANE_SIDE), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        plane.save(buffer, format="PNG")
        uris.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return uris


def scan_upload(upload: FileStorage | None) -> tuple[str, Report, list[str]]:
    """Save the upload to a temp dir, scan it, and delete it.

    Returns (file_type, report, bit-plane URIs). Raises UploadError for a
    missing file or an unrecognized type.
    """
    if upload is None or not upload.filename:
        raise UploadError("No file was uploaded.")

    # The name keeps its extension (the scanner falls back to it) but can't
    # contain path separators, so it can't escape the temp dir.
    safe_name = secure_filename(upload.filename) or "upload"
    # TemporaryDirectory removes the folder when the block exits, even if the scan raises.
    with tempfile.TemporaryDirectory(prefix="stegoscan-") as temp_dir:
        path = Path(temp_dir) / safe_name
        upload.save(path)
        file_type = detect_file_type(path)
        try:
            report = scan_file(path)
        except ValueError as error:
            raise UploadError(str(error)) from error
        planes = bit_planes_as_data_uris(path) if file_type == IMAGE else []
    return file_type, report, planes


def create_app(max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB) -> Flask:
    """Build the Flask app. Tests pass a small max_upload_mb to check the limit."""
    app = Flask(__name__)
    # Werkzeug rejects larger bodies with 413 before reading them into memory.
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024

    @app.after_request
    def add_security_headers(response):
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", max_upload_mb=max_upload_mb)

    @app.post("/scan")
    def scan():
        try:
            file_type, report, planes = scan_upload(request.files.get("file"))
        except UploadError as error:
            return render_template("error.html", message=str(error)), 400
        filename = request.files["file"].filename
        if is_unreadable(report):
            message = f"Could not analyze {filename}: it is empty, corrupt, or unsupported."
            return render_template("error.html", message=message), 422
        preview = hex_preview(report.extracted) if report.extracted else None
        return render_template("result.html", filename=filename, file_type=file_type,
                               report=report, preview=preview, planes=planes)

    @app.post("/api/scan")
    def api_scan():
        try:
            file_type, report, _ = scan_upload(request.files.get("file"))
        except UploadError as error:
            return jsonify(error=str(error)), 400
        status = 422 if is_unreadable(report) else 200
        body = {"file": request.files["file"].filename, "file_type": file_type,
                **report_to_dict(report)}
        return jsonify(body), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        message = f"File is larger than the {max_upload_mb} MB upload limit."
        if request.path.startswith("/api/"):
            return jsonify(error=message), 413
        return render_template("error.html", message=message), 413

    return app


def main(argv: list[str] | None = None) -> None:
    """Serve the UI on 127.0.0.1. Debug stays off: its console runs arbitrary Python.

    Only the port is configurable (macOS AirPlay Receiver often holds 5000).
    """
    parser = argparse.ArgumentParser(prog="python -m stegoscan.web",
                                     description="StegoScan local web UI (127.0.0.1 only).")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to listen on (default {PORT})")
    args = parser.parse_args(argv)
    create_app().run(host=HOST, port=args.port, debug=False)


if __name__ == "__main__":
    main()
