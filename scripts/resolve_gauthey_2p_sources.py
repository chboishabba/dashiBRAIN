#!/usr/bin/env python3
"""Resolve source-code-declared Gauthey conventional-2p inputs inside Data.zip.

This performs central-directory inspection only. It does not download large
members. Resolution is by exact basename and rejects zero/multiple matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dashi.io.gauthey_2p_source_recovery import TWO_P_TRIAL_SPECS
from dashi.io.gauthey_compact import GAUTHEY_DATA_ZIP_URL
from dashi.io.remote_zip import list_remote_zip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/gauthey_2p_source_resolution.json")
    args = p.parse_args()

    members = list_remote_zip(GAUTHEY_DATA_ZIP_URL)
    by_basename: dict[str, list[object]] = {}
    for member in members:
        by_basename.setdefault(Path(member.name).name, []).append(member)

    required = []
    for spec in TWO_P_TRIAL_SPECS:
        required.extend((spec.data_filename, spec.label_filename))

    resolved = {}
    all_unique = True
    for basename in required:
        matches = by_basename.get(basename, [])
        if len(matches) != 1:
            all_unique = False
        resolved[basename] = [
            {
                "archive_member": m.name,
                "compressed_size": m.compressed_size,
                "uncompressed_size": m.uncompressed_size,
                "crc32": m.crc32,
            }
            for m in matches
        ]

    payload = {
        "archive_url": GAUTHEY_DATA_ZIP_URL,
        "required_source_files": required,
        "all_required_uniquely_resolved": all_unique,
        "resolved": resolved,
        "boundary": "basename resolution establishes archive identity only; it does not prove trace-row recovery",
    }
    print(json.dumps(payload, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
