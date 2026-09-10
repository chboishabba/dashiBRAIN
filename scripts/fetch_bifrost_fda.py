#!/usr/bin/env python3
"""Fetch the published BIFROST FDA template from Dryad by DOI and filename.

Scientific source:
Bella E. Brezovec, Andrew B. Berger, Yukun A. Hao et al.,
"BIFROST: A method for registering diverse imaging datasets of the Drosophila brain",
PNAS 121(47):e2322687121 (2024), DOI 10.1073/pnas.2322687121.
Dataset DOI 10.5061/dryad.8pk0p2nx1.

This script resolves the current public Dryad version and download resource
dynamically, then range-downloads only ``nifti1_compliant_FDA.nii`` with
retry/resume. It does not download the full BIFROST dataset bundle or rely on a
top-level file ``id`` field, which Dryad's v2 file representation may omit.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.parse
import urllib.request

from dashi.io.remote_zip import _request


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


def _file_resource(file_meta: dict) -> tuple[int | None, str]:
    """Return optional numeric file id and canonical Dryad download URL."""
    links = file_meta.get("_links") or {}
    download = links.get("stash:download") or links.get("download")
    if isinstance(download, dict) and download.get("href"):
        download_url = _absolute_api_href(str(download["href"]))
    else:
        download_url = ""

    raw_id = file_meta.get("id")
    if raw_id is None:
        self_link = links.get("self")
        if isinstance(self_link, dict) and self_link.get("href"):
            tail = str(self_link["href"]).rstrip("/").rsplit("/", 1)[-1]
            if tail.isdigit():
                raw_id = int(tail)
    file_id = int(raw_id) if raw_id is not None else None

    if not download_url and file_id is not None:
        download_url = f"{DRYAD_API}/files/{file_id}/download"
    if not download_url:
        raise RuntimeError("Dryad file metadata lacks a usable download resource")
    return file_id, download_url


def _hash_existing_prefix(path: Path) -> tuple[int, object]:
    hasher = sha256()
    size = 0
    if path.exists():
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
    return size, hasher


def download_public_file(file_meta: dict, output_path: Path, *, chunk_bytes: int = 8 << 20) -> dict:
    """Resume a public Dryad file by exact byte range and verify before promotion."""
    file_id, download_url = _file_resource(file_meta)
    expected_size = int(file_meta.get("size") or 0)
    if expected_size <= 0:
        raise RuntimeError("Dryad file metadata lacks a positive size")
    digest = str(file_meta.get("digest") or "").lower()
    digest_type = str(file_meta.get("digestType") or "").lower()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".part")
    written, hasher = _hash_existing_prefix(tmp)
    if written > expected_size:
        raise RuntimeError(
            f"partial FDA download is larger than Dryad metadata: {written} > {expected_size}"
        )

    with tmp.open("ab") as handle:
        cursor = written
        while cursor < expected_size:
            end = min(expected_size - 1, cursor + chunk_bytes - 1)
            chunk, _headers = _request(download_url, cursor, end)
            expected = end - cursor + 1
            if len(chunk) != expected:
                raise RuntimeError(
                    f"short Dryad range read at byte {cursor}: got {len(chunk)}, expected {expected}"
                )
            handle.write(chunk)
            handle.flush()
            hasher.update(chunk)
            cursor += len(chunk)

    if tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"downloaded {tmp.stat().st_size} bytes; Dryad metadata says {expected_size}"
        )
    actual_sha256 = hasher.hexdigest().lower()
    if digest and digest_type in {"sha-256", "sha256"} and actual_sha256 != digest:
        raise RuntimeError("download SHA-256 does not match Dryad metadata")
    tmp.replace(output_path)
    return {
        "dryad_file_id": file_id,
        "dryad_download_url": download_url,
        "path": str(output_path),
        "size": expected_size,
        "sha256": actual_sha256,
        "dryad_digest": digest or None,
        "dryad_digest_type": digest_type or None,
        "resume_sidecar": str(tmp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/bifrost/nifti1_compliant_FDA.nii")
    parser.add_argument("--doi", default=BIFROST_DATASET_DOI)
    parser.add_argument("--filename", default=FDA_FILENAME)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.chunk_mib <= 0:
        raise SystemExit("--chunk-mib must be positive")

    meta = resolve_public_file(args.doi, args.filename)
    file_id, download_url = _file_resource(meta)
    summary = {
        "dataset_doi": args.doi,
        "filename": args.filename,
        "dryad_file_id": file_id,
        "dryad_download_url": download_url,
        "size": meta.get("size"),
        "digest": meta.get("digest"),
        "digest_type": meta.get("digestType"),
    }
    if not args.metadata_only:
        summary.update(
            download_public_file(meta, Path(args.output), chunk_bytes=args.chunk_mib << 20)
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
