#!/usr/bin/env python3
"""Materialize the compact Gauthey functional bundle by HTTP byte ranges only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dashi.io.gauthey_compact import (
    GAUTHEY_COMPACT_MEMBERS,
    GAUTHEY_DATA_ZIP_URL,
    load_audio_correlated_pickle,
    materialize_compact_bundle,
    write_bundle_receipt,
)
from dashi.io.remote_zip import list_remote_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="Selectively extract compact Gauthey 2026 artifacts")
    parser.add_argument("--output-dir", default="data/malecns/functional/gauthey_compact")
    parser.add_argument("--receipt", default="outputs/gauthey_compact_receipt.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    if args.list_only:
        wanted = set(GAUTHEY_COMPACT_MEMBERS.values())
        for m in list_remote_zip(GAUTHEY_DATA_ZIP_URL):
            if m.name in wanted:
                print(
                    f"{m.name}\tcompressed={m.compressed_size}\t"
                    f"uncompressed={m.uncompressed_size}\tcrc32={m.crc32:08x}"
                )
        return

    receipts = materialize_compact_bundle(args.output_dir)
    write_bundle_receipt(receipts, args.receipt)
    for role, receipt in receipts.items():
        print(f"[{role}] {receipt.path} sha256={receipt.sha256}")

    functional_path = Path(args.output_dir) / Path(
        GAUTHEY_COMPACT_MEMBERS["functional_audio_correlated"]
    ).name
    matrix = load_audio_correlated_pickle(functional_path)
    print(
        "functional matrix:", matrix.traces.shape,
        "identity:", matrix.identity_kind,
    )
    print(f"receipt: {args.receipt}")


if __name__ == "__main__":
    main()
