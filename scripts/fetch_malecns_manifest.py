#!/usr/bin/env python3
"""Inspect, fetch, and verify MaleCNS experiment artifacts safely.

Only authority entries marked as direct downloads are fetched automatically.
Repository/paper DOI landing pages are never written into artifact paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

from dashi.io.artifact_verification import verify_artifact
from dashi.io.malecns_manifest import MaleCNSManifest
from dashi.io.malecns_protocol import sha256_file
from dashi.io.malecns_real_data import MALECNS_REAL_AUTHORITIES


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:3.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def download_stream(url: str, dest: Path, expected_size: int | None = None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")
    import hashlib
    h = hashlib.sha256()
    downloaded = 0
    req = urllib.request.Request(url, headers={"User-Agent": "dashiBRAIN-MaleCNS-Benchmark/2.0"})
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            downloaded += len(chunk)
            if expected_size:
                pct = downloaded / expected_size * 100
                sys.stdout.write(f"\r  [{downloaded}/{expected_size} bytes ({pct:5.1f}%)]")
            else:
                sys.stdout.write(f"\r  [{format_bytes(downloaded)}]")
            sys.stdout.flush()
    print()
    temp_dest.replace(dest)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify MaleCNS experiment manifest")
    parser.add_argument("--base-dir", default="data/malecns")
    parser.add_argument("--tier", choices=["all", "connectome", "functional", "registration", "effector", "behaviour"], default="connectome")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--emit-receipts")
    args = parser.parse_args()

    manifest = MaleCNSManifest(base_dir=args.base_dir)
    tiers = None if args.tier == "all" else {args.tier}
    specs = [s for s in manifest.specs.values() if tiers is None or s.tier in tiers]
    print(f"=== MaleCNS Experiment Manifest [{args.tier.upper()}] ===")
    print(f"Target Directory: {Path(args.base_dir).resolve()}")
    print(f"Declared Expected Footprint: {format_bytes(sum(s.expected_size_bytes for s in specs))}")
    print("-" * 72)

    for spec in specs:
        auth = MALECNS_REAL_AUTHORITIES[spec.key]
        target = manifest.target_path(spec.key)
        print(f"[{spec.tier.upper():12}] {spec.key:30} {'PRESENT' if manifest.is_present(spec.key) else 'MISSING'}")
        print(f"  Target: {target}")
        print(f"  Scientific source: {spec.source.author_or_consortium}; {spec.source.title}; {spec.source.stable_identifier}")
        print(f"  Repository authority: {auth.repository_identifier}")
        print(f"  Resolved file: {auth.resolved_filename or 'UNRESOLVED'}")
        print(f"  Automatic direct download: {auth.direct_download}")

    if args.dry_run:
        print("\nDry run completed. Repository DOI pages were not fetched.")
        return

    receipts: dict[str, dict] = {}
    for spec in specs:
        auth = MALECNS_REAL_AUTHORITIES[spec.key]
        target = manifest.target_path(spec.key)

        if args.verify_only:
            verification = verify_artifact(target, auth)
            print(f"  [{verification.level.value.upper()}] {spec.key}")
            if verification.actual_sha256:
                print(f"    SHA-256: {verification.actual_sha256}")
            receipts[spec.key] = {
                "level": verification.level.value,
                "path": verification.path,
                "sha256": verification.actual_sha256,
                "repository_identifier": auth.repository_identifier,
                "resolved_filename": auth.resolved_filename,
            }
            continue

        if target.is_file() and target.stat().st_size > 0:
            print(f"  [SKIP] Already present: {spec.key}")
            digest = sha256_file(target)
        elif not auth.direct_download:
            print(f"  [RESOLVE] {spec.key}: exact repository file must be resolved before download")
            print(f"            authority={auth.repository_identifier}")
            continue
        else:
            print(f"  [FETCH] {spec.key}")
            digest = download_stream(spec.download_url, target, spec.expected_size_bytes)

        receipts[spec.key] = {
            "role": spec.role,
            "path": str(target),
            "sha256": digest,
            "dataset_version": spec.dataset_version,
            "repository_identifier": auth.repository_identifier,
            "resolved_filename": auth.resolved_filename,
            "hash_verified": bool(auth.expected_sha256 and digest.lower() == auth.expected_sha256.lower()),
        }

    if args.emit_receipts:
        out = Path(args.emit_receipts)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
        print(f"\nEmitted receipts to {out}")


if __name__ == "__main__":
    main()
