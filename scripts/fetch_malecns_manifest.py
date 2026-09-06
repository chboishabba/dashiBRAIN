#!/usr/bin/env python3
"""CLI utility to inspect, fetch, and verify MaleCNS experiment manifest artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

from dashi.io.malecns_manifest import CANONICAL_ARTIFACT_SPECS, MaleCNSManifest
from dashi.io.malecns_protocol import sha256_file


def format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def download_stream(url: str, dest: Path, expected_size: int | None = None) -> str:
    """Download a URL to destination path with progress and return SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")

    print(f"Downloading: {url} -> {dest}")
    import hashlib
    h = hashlib.sha256()
    downloaded = 0

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "dashiBRAIN-MaleCNS-Benchmark/1.0"},
    )

    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            downloaded += len(chunk)
            if expected_size:
                pct = (downloaded / expected_size) * 100
                sys.stdout.write(f"\r  [{downloaded}/{expected_size} bytes ({pct:5.1f}%)]")
            else:
                sys.stdout.write(f"\r  [{format_bytes(downloaded)}]")
            sys.stdout.flush()

    print()
    temp_dest.replace(dest)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify MaleCNS experiment manifest.")
    parser.add_argument("--base-dir", default="data/malecns", help="Target base directory for datasets")
    parser.add_argument("--tier", choices=["all", "connectome", "functional", "registration", "effector", "behaviour"],
                        default="connectome", help="Artifact tier to inspect or fetch")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest plan without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Verify SHA-256 digests of existing files")
    parser.add_argument("--emit-receipts", help="Output path for JSON manifest receipts")

    args = parser.parse_args()
    manifest = MaleCNSManifest(base_dir=args.base_dir)

    selected_tiers = None if args.tier == "all" else [args.tier]
    specs = [s for s in manifest.specs.values() if selected_tiers is None or s.tier in selected_tiers]
    total_bytes = sum(s.expected_size_bytes for s in specs)

    print(f"=== MaleCNS Experiment Manifest [{args.tier.upper()}] ===")
    print(f"Target Directory: {Path(args.base_dir).resolve()}")
    print(f"Total Artifacts: {len(specs)} | Total Expected Size: {format_bytes(total_bytes)}")
    print("-" * 72)

    for spec in specs:
        target = manifest.target_path(spec.key)
        present = manifest.is_present(spec.key)
        status = "PRESENT" if present else "MISSING"
        if present:
            actual_size = target.stat().st_size
            status += f" ({format_bytes(actual_size)})"
        print(f"[{spec.tier.upper():12}] {spec.key:30} {status}")
        print(f"  Path: {target}")
        print(f"  URL:  {spec.download_url}")
        print(f"  Size: {format_bytes(spec.expected_size_bytes)}")
        print(f"  DOI:  {spec.source.stable_identifier}")

    if args.dry_run:
        print("\nDry run completed. No downloads executed.")
        return

    receipts = {}

    if args.verify_only:
        print("\n--- Verifying Existing Artifacts ---")
        for spec in specs:
            target = manifest.target_path(spec.key)
            if not target.exists():
                print(f"  [MISSING] {spec.key}: {target}")
                continue
            h = sha256_file(target)
            print(f"  [OK] {spec.key}: {h}")
            receipts[spec.key] = {
                "role": spec.role,
                "path": str(target),
                "sha256": h,
                "dataset_version": spec.dataset_version,
            }
    else:
        print("\n--- Fetching Artifacts ---")
        for spec in specs:
            target = manifest.target_path(spec.key)
            if target.exists() and target.stat().st_size > 0:
                print(f"  [SKIP] Already present: {spec.key}")
                h = sha256_file(target)
            else:
                if not spec.download_url.startswith("http"):
                    print(f"  [MANUAL] {spec.key} requires manual download / DOI resolution: {spec.download_url}")
                    continue
                h = download_stream(spec.download_url, target, spec.expected_size_bytes)

            receipts[spec.key] = {
                "role": spec.role,
                "path": str(target),
                "sha256": h,
                "dataset_version": spec.dataset_version,
            }

    if args.emit_receipts:
        out_path = Path(args.emit_receipts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(receipts, f, indent=2)
        print(f"\nEmitted receipts to {out_path}")


if __name__ == "__main__":
    main()
