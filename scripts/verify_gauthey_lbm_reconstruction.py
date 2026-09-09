#!/usr/bin/env python3
"""Compile a fail-closed scientific receipt for one completed Gauthey LBM trial."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from dashi.analysis.gauthey_lbm_experiment import GAUTHEY_DATA_ZIP_URL, LBM_TRIALS
from dashi.io.gauthey_reconstruction_receipt import (
    build_gauthey_reconstruction_receipt,
    write_gauthey_reconstruction_receipt,
)
from dashi.io.remote_zip import list_remote_zip
from scripts.run_gauthey_lbm_remote_reconstruction import SOURCE_CONTAINER_MEMBERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, choices=LBM_TRIALS)
    parser.add_argument(
        "--deposited-selected",
        default="data/gauthey_lbm/dffs_audio_LB_corr_top05_all.pkl",
    )
    parser.add_argument(
        "--reconstruction-summary",
        required=True,
        help="Path to gauthey_lbm_reconstruction.json emitted by the reconstruction runner.",
    )
    parser.add_argument("--url", default=GAUTHEY_DATA_ZIP_URL)
    parser.add_argument("--output")
    args = parser.parse_args()

    trial_index = LBM_TRIALS.index(args.trial)
    expected_member = SOURCE_CONTAINER_MEMBERS[trial_index]
    by_name = {member.name: member for member in list_remote_zip(args.url)}
    if expected_member not in by_name:
        raise SystemExit(
            f"expected source member is absent from remote deposit: {expected_member}"
        )

    summary_path = Path(args.reconstruction_summary)
    output_path = (
        Path(args.output)
        if args.output
        else summary_path.with_name(f"{args.trial}_scientific_fixture_receipt.json")
    )
    receipt = build_gauthey_reconstruction_receipt(
        trial_id=args.trial,
        remote_url=args.url,
        remote_member=by_name[expected_member],
        deposited_selected_path=args.deposited_selected,
        reconstruction_summary_path=summary_path,
    )
    write_gauthey_reconstruction_receipt(receipt, output_path)
    print(json.dumps(asdict(receipt), indent=2))

    if not receipt.fixture_admissible:
        raise SystemExit("reconstruction produced no exact source identities; fixture not admissible")


if __name__ == "__main__":
    main()
