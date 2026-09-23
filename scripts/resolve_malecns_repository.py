#!/usr/bin/env python3
"""Resolve MaleCNS-adjacent repository records without downloading payloads.

The resolver separates repository metadata from artifact download. It is safe to
run under tight disk budgets because it requests only JSON metadata and emits a
machine-readable inventory of filenames, byte sizes and repository digests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request


def _json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dashiBRAIN-MaleCNS-Resolver/1.0"})
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def resolve_zenodo(record_id: str) -> dict:
    record = _json(f"https://zenodo.org/api/records/{record_id}")
    files = []
    for item in record.get("files", []):
        links = item.get("links") or {}
        checksum = item.get("checksum")
        files.append({
            "filename": item.get("key"),
            "size_bytes": item.get("size"),
            "checksum": checksum,
            "download_url": links.get("content") or links.get("self"),
        })
    return {
        "provider": "zenodo",
        "record_id": str(record.get("id", record_id)),
        "doi": record.get("doi") or (record.get("metadata") or {}).get("doi"),
        "title": (record.get("metadata") or {}).get("title"),
        "files": files,
    }


def resolve_dryad(doi: str) -> dict:
    encoded = urllib.parse.quote(f"doi:{doi}" if not doi.startswith("doi:") else doi, safe="")
    dataset = _json(f"https://datadryad.org/api/v2/datasets/{encoded}")
    versions_url = (dataset.get("_links") or {}).get("stash:versions", {}).get("href")
    if versions_url:
        versions = _json(versions_url)
        version_items = (versions.get("_embedded") or {}).get("stash:versions", [])
    else:
        versions = _json(f"https://datadryad.org/api/v2/datasets/{encoded}/versions")
        version_items = versions if isinstance(versions, list) else (versions.get("_embedded") or {}).get("stash:versions", [])
    if not version_items:
        raise RuntimeError(f"Dryad dataset has no versions: {doi}")

    latest = version_items[0]
    version_id = latest.get("id")
    files_payload = _json(f"https://datadryad.org/api/v2/versions/{version_id}/files")
    raw_files = (files_payload.get("_embedded") or {}).get("stash:files", [])
    files = []
    for item in raw_files:
        links = item.get("_links") or {}
        download = links.get("stash:download", {})
        files.append({
            "filename": item.get("path"),
            "size_bytes": item.get("size"),
            "checksum": f"{item.get('digestType', 'sha256')}:{item.get('digest')}" if item.get("digest") else None,
            "download_url": download.get("href"),
        })
    return {
        "provider": "dryad",
        "doi": doi.removeprefix("doi:"),
        "version_id": version_id,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve repository files without downloading them")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zenodo-record")
    source.add_argument("--dryad-doi")
    parser.add_argument("--output", help="Optional JSON inventory path")
    parser.add_argument("--max-bytes", type=int, help="Mark files larger than this disk budget")
    args = parser.parse_args()

    result = resolve_zenodo(args.zenodo_record) if args.zenodo_record else resolve_dryad(args.dryad_doi)
    total = 0
    for item in result["files"]:
        size = int(item.get("size_bytes") or 0)
        total += size
        item["within_budget"] = args.max_bytes is None or size <= args.max_bytes
        budget = "" if item["within_budget"] else " [OVER BUDGET]"
        print(f"{item['filename']}: {size} bytes; {item.get('checksum')}{budget}")
    result["total_listed_bytes"] = total

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote repository inventory to {out}")


if __name__ == "__main__":
    main()
