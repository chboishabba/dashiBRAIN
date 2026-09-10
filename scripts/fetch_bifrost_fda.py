#!/usr/bin/env python3
"""Fetch the published BIFROST FDA template from Dryad by DOI and filename.

Scientific source:
Bella E. Brezovec, Andrew B. Berger, Yukun A. Hao et al.,
"BIFROST: A method for registering diverse imaging datasets of the Drosophila brain",
PNAS 121(47):e2322687121 (2024), DOI 10.1073/pnas.2322687121.
Dataset DOI 10.5061/dryad.8pk0p2nx1.

This script resolves the current public Dryad version and file ID dynamically,
then streams only ``nifti1_compliant_FDA.nii``. It does not download the 40+ GB
BIFROST dataset bundle or hard-code a transient Dryad file-stream URL.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.parse
import urllib.request


DRYAD_API = "https://datadryad.org/api/v2"
BIFROST_DATASET_DOI = "doi:10.5061/dryad.8pk0p2nx1"
FDA_FILENAME = "nifti1_compliant_FDA.nii"


def _json_get(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "dashiBRAIN-BIFROST/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def _absolute_api_href(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return "https://datadryad.org" + href


def resolve_public_file(doi: str, filename: str) -> dict:
    encoded = urllib.parse.quote(doi, safe="")
    dataset = _json_get(f"{DRYAD_API}/datasets/{encoded}")
    links = dataset.get("_links") or {}
    version_link = links.get("stash:version") or links.get("version")
    if not isinstance(version_link, dict) or not version_link.get("href"):
        raise RuntimeError("Dryad dataset response lacks current-version link")
    version = _json_get(_absolute_api_href(str(version_link["href"])))
    vlinks = version.get("_links") or {}
    files_link = vlinks.get("stash:files") or vlinks.get("files")
    if not isinstance(files_link, dict) or not files_link.get("href"):
        raise RuntimeError("Dryad version response lacks files link")
    files_payload = _json_get(_absolute_api_href(str(files_link["href"])))
    embedded = files_payload.get("_embedded") or {}
    files = embedded.get("stash:files") or embedded.get("files") or files_payload.get("files") or []
    matches = [item for item in files if str(item.get("path", "")) == filename]
    if len(matches) != 1:
        available = sorted(str(item.get("path", "")) for item in files)
        raise RuntimeError(
            f"expected exactly one Dryad file named {filename!r}, found {len(matches)}; "
            f"available={available}"
        )
    return matches[0]


def download_public_file(file_meta: dict, output_path: Path, *, chunk_bytes: int = 8 << 20) -> dict:
    file_id = file_meta.get("id")
    if file_id is None:
        raise RuntimeError("Dryad file metadata lacks file id")
    expected_size = int(file_meta.get("size") or 0)
    digest = str(file_meta.get("digest") or "").lower()
    digest_type = str(file_meta.get("digestType") or "").lower()
    download_url = f"{DRYAD_API}/files/{file_id}/download"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".part")
    req = urllib.request.Request(
        download_url,
        headers={"Accept": "application/octet-stream", "User-Agent": "dashiBRAIN-BIFROST/1.0"},
    )
    hasher = sha256()
    written = 0
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(chunk_bytes)
            if not chunk:
                break
            handle.write(chunk)
            hasher.update(chunk)
            written += len(chunk)

    if expected_size and written != expected_size:
        raise RuntimeError(f"downloaded {written} bytes; Dryad metadata says {expected_size}")
    if digest and digest_type == "sha-256" and hasher.hexdigest().lower() != digest:
        raise RuntimeError("download SHA-256 does not match Dryad metadata")
    tmp.replace(output_path)
    return {
        "dryad_file_id": int(file_id),
        "path": str(output_path),
        "size": written,
        "sha256": hasher.hexdigest(),
        "dryad_digest": digest or None,
        "dryad_digest_type": digest_type or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/bifrost/nifti1_compliant_FDA.nii")
    parser.add_argument("--doi", default=BIFROST_DATASET_DOI)
    parser.add_argument("--filename", default=FDA_FILENAME)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    meta = resolve_public_file(args.doi, args.filename)
    summary = {
        "dataset_doi": args.doi,
        "filename": args.filename,
        "dryad_file_id": meta.get("id"),
        "size": meta.get("size"),
        "digest": meta.get("digest"),
        "digest_type": meta.get("digestType"),
    }
    if not args.metadata_only:
        summary.update(download_public_file(meta, Path(args.output)))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
