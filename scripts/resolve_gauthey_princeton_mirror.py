#!/usr/bin/env python3
"""Resolve Gauthey's official Princeton Data Commons mirror.

The paper names DOI 10.34770/s5hx-1x75 as an alternate repository for the same
raw/preprocessed data also deposited on Zenodo. This probe uses Princeton Data
Commons' JSON document export and records authoritative file/Globus coordinates.
It does not infer a path from the DOI and does not start a transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request

from dashi.io.princeton_data_commons import parse_pdc_document_export

PDC_DOI = "10.34770/s5hx-1x75"
PDC_RECORD_ID = "doi-10-34770-s5hx-1x75"
PDC_JSON_URL = (
    "https://datacommons.princeton.edu/discovery/catalog/"
    "doi-10-34770-s5hx-1x75.json"
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=PDC_JSON_URL)
    parser.add_argument(
        "--output",
        default="data/gauthey_lbm/princeton_mirror/gauthey_princeton_mirror.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    output = Path(args.output)
    req = urllib.request.Request(
        args.url,
        headers={
            "Accept": "application/json",
            "User-Agent": "dashiBRAIN-GautheyMirrorResolver/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout_seconds) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        payload = {
            "status": "princeton_mirror_metadata_blocked",
            "doi": PDC_DOI,
            "record_id": PDC_RECORD_ID,
            "metadata_url": args.url,
            "http_status": exc.code,
            "error": str(exc),
            "scientific_payment": False,
            "interpretation": (
                "The official Princeton mirror remains source-cited, but its machine-readable "
                "metadata endpoint was not accessible in this execution. No file path or "
                "Globus coordinate is inferred from the DOI alone."
            ),
        }
        _write(output, payload)
        raise SystemExit(2)
    except urllib.error.URLError as exc:
        payload = {
            "status": "princeton_mirror_metadata_transport_failure",
            "doi": PDC_DOI,
            "record_id": PDC_RECORD_ID,
            "metadata_url": args.url,
            "error": str(exc),
            "scientific_payment": False,
            "interpretation": "Mirror metadata transport failed; no source path was inferred.",
        }
        _write(output, payload)
        raise SystemExit(2)

    try:
        decoded = json.loads(raw.decode("utf-8"))
        record = parse_pdc_document_export(decoded)
    except Exception as exc:
        payload = {
            "status": "princeton_mirror_metadata_unusable",
            "doi": PDC_DOI,
            "record_id": PDC_RECORD_ID,
            "metadata_url": args.url,
            "content_type": content_type,
            "error": f"{type(exc).__name__}: {exc}",
            "scientific_payment": False,
            "interpretation": (
                "The endpoint responded, but not with the expected PDC document-export schema; "
                "no Globus or file coordinate is promoted."
            ),
        }
        _write(output, payload)
        raise SystemExit(2)

    data_zip_candidates = [
        file
        for file in record.files
        if Path(file.filename).name.lower() == "data.zip"
    ]
    payload = {
        "status": "princeton_mirror_metadata_resolved",
        "doi": record.doi,
        "record_id": record.record_id,
        "title": record.title,
        "metadata_url": args.url,
        "globus_url": record.globus_url,
        "globus_origin_id": record.globus_origin_id,
        "globus_origin_path": record.globus_origin_path,
        "files": [
            {"filename": file.filename, "size": file.size, "url": file.url}
            for file in record.files
        ],
        "data_zip_candidates": [
            {"filename": file.filename, "size": file.size, "url": file.url}
            for file in data_zip_candidates
        ],
        "data_zip_candidate_count": len(data_zip_candidates),
        "scientific_payment": False,
        "provenance_payment": True,
        "interpretation": (
            "This receipt resolves an official alternate repository coordinate only. It does not "
            "assert byte identity with the Zenodo Data.zip until a checksum/content comparison is paid."
        ),
    }
    _write(output, payload)


if __name__ == "__main__":
    main()
