#!/usr/bin/env python3
"""Retry resumable Gauthey identity-recovery transfers without changing semantics.

This supervisor wraps ``recover_gauthey_lbm_remaining_identities.py``.  It does
not infer transient failure from a generic nonzero exit status.  Instead it
retries only when the failed invocation wrote a *fresh* typed transport
interruption receipt with ``resumable: true``.  The underlying range downloader
therefore remains authoritative for byte-exact ``.compressed.part`` resume.

Semantic, integrity, source-shape, identity, disk, and other failures that do
not produce a fresh resumable transport receipt are returned immediately.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping


DEFAULT_OUTPUT_DIR = Path("data/gauthey_lbm/reconstruction_all_available")


def _receipt_snapshot(output_dir: Path) -> Mapping[Path, str]:
    root = output_dir / "trial_recovery"
    if not root.exists():
        return {}
    out: dict[Path, str] = {}
    for path in root.rglob("gauthey_lbm_transport_interruption_*.json"):
        if path.is_file():
            try:
                out[path] = sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                # A concurrent replacement is equivalent to a changed receipt;
                # the next snapshot will observe the durable state.
                continue
    return out


def _fresh_resumable_transport_receipts(
    before: Mapping[Path, str],
    after: Mapping[Path, str],
) -> tuple[Path, ...]:
    fresh: list[Path] = []
    for path, digest in after.items():
        if before.get(path) == digest:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            payload.get("status") == "gauthey_lbm_transport_interruption"
            and payload.get("searched") is False
            and payload.get("resumable") is True
        ):
            fresh.append(path)
    return tuple(sorted(fresh))


def run_with_resumable_transport_retries(
    cmd: list[str],
    *,
    output_dir: str | Path,
    max_attempts: int = 8,
    retry_base_seconds: float = 30.0,
    retry_cap_seconds: float = 600.0,
) -> subprocess.CompletedProcess:
    """Run recovery until success or a non-resumable/non-transport wall appears."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if retry_base_seconds < 0 or retry_cap_seconds < 0:
        raise ValueError("retry delays must be non-negative")

    output = Path(output_dir)
    for attempt in range(max_attempts):
        before = _receipt_snapshot(output)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode == 0:
            return completed

        after = _receipt_snapshot(output)
        fresh = _fresh_resumable_transport_receipts(before, after)
        if not fresh or attempt + 1 >= max_attempts:
            return completed

        delay = min(retry_cap_seconds, retry_base_seconds * (2 ** attempt))
        joined = ", ".join(str(path) for path in fresh)
        print(
            f"resumable transport interruption recorded ({joined}); "
            f"retrying from persisted sidecar after {delay:.1f}s",
            flush=True,
        )
        time.sleep(delay)

    raise AssertionError("unreachable retry loop")


def _extract_output_dir(recovery_args: list[str]) -> Path:
    for index, token in enumerate(recovery_args):
        if token == "--output-dir":
            if index + 1 >= len(recovery_args):
                raise SystemExit("--output-dir requires a value")
            return Path(recovery_args[index + 1])
        if token.startswith("--output-dir="):
            return Path(token.split("=", 1)[1])
    return DEFAULT_OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry only typed resumable Gauthey transport interruptions."
    )
    parser.add_argument("--resume-max-attempts", type=int, default=8)
    parser.add_argument("--resume-base-seconds", type=float, default=30.0)
    parser.add_argument("--resume-cap-seconds", type=float, default=600.0)
    args, recovery_args = parser.parse_known_args()

    output_dir = _extract_output_dir(recovery_args)
    recover_script = Path(__file__).with_name("recover_gauthey_lbm_remaining_identities.py")
    cmd = [sys.executable, str(recover_script), *recovery_args]
    print("+", " ".join(cmd), flush=True)
    completed = run_with_resumable_transport_retries(
        cmd,
        output_dir=output_dir,
        max_attempts=args.resume_max_attempts,
        retry_base_seconds=args.resume_base_seconds,
        retry_cap_seconds=args.resume_cap_seconds,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
