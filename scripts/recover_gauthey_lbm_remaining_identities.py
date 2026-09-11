#!/usr/bin/env python3
"""Resume exact Gauthey LBM identity recovery from an existing partial receipt.

The existing remote reconstruction runner remains the acquisition engine.  This
wrapper inventories which of the six aligned-trial containers are actually
present, runs exact trace recovery only for remaining available trials, and
merges the new receipt append-only with the already-paid identity map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from dashi.analysis.gauthey_lbm_experiment import GAUTHEY_DATA_ZIP_URL, LBM_TRIALS
from dashi.io.gauthey_lbm_identity_receipts import (
    load_identity_csv,
    merge_exact_identities,
    write_identity_csv,
)
from dashi.io.remote_zip import list_remote_zip

SOURCE_STEMS = (
    "GCaMP6f_04032024_a2_r1",
    "GCaMP6f_04032024_a2_r5",
    "GCaMP6f_04162024_a1_r1",
    "GCaMP6f_04192024_a1_r2",
    "GCaMP6f_04192024_a1_r6",
    "GCaMP6f_04192024_a1_r9",
)
SOURCE_CONTAINER_MEMBERS = tuple(f"Data/Dffs/Aligned/{stem}.zip" for stem in SOURCE_STEMS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-identities", required=True)
    parser.add_argument(
        "--seed-searched-trial",
        action="append",
        default=[],
        choices=LBM_TRIALS,
        help="Trial(s) already searched to produce the seed receipt. If omitted, positive-count seed trials are used.",
    )
    parser.add_argument(
        "--deposited-selected",
        default="data/gauthey_lbm/dffs_audio_LB_corr_top05_all.pkl",
    )
    parser.add_argument("--output-dir", default="data/gauthey_lbm/reconstruction_all_available")
    parser.add_argument("--scratch-dir", default="data/gauthey_lbm/scratch")
    parser.add_argument("--url", default=GAUTHEY_DATA_ZIP_URL)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()

    seed = load_identity_csv(args.seed_identities)
    seed_positive_trials = tuple(dict.fromkeys(row.trial_id for row in seed))
    seed_searched = tuple(args.seed_searched_trial or seed_positive_trials)

    remote_names = {member.name for member in list_remote_zip(args.url)}
    present = {
        trial: member in remote_names
        for trial, member in zip(LBM_TRIALS, SOURCE_CONTAINER_MEMBERS)
    }
    remaining = [trial for trial in LBM_TRIALS if trial not in set(seed_searched)]
    available_remaining = [trial for trial in remaining if present[trial]]
    missing_remaining = [trial for trial in remaining if not present[trial]]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    acquisition_dir = out / "new_remote_receipt"
    acquisition_dir.mkdir(parents=True, exist_ok=True)

    new_identities = ()
    if available_remaining:
        cmd = [
            sys.executable,
            "scripts/run_gauthey_lbm_remote_reconstruction.py",
            "--deposited-selected",
            args.deposited_selected,
            "--output-dir",
            str(acquisition_dir),
            "--scratch-dir",
            args.scratch_dir,
            "--url",
            args.url,
            "--chunk-mib",
            str(args.chunk_mib),
        ]
        for trial in available_remaining:
            cmd.extend(["--trial", trial])
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        new_identities = load_identity_csv(
            acquisition_dir / "gauthey_lbm_selected_identities_partial.csv"
        )

    searched = tuple(dict.fromkeys((*seed_searched, *available_remaining)))
    merged = merge_exact_identities(
        (seed, new_identities),
        searched_trials=searched,
        missing_source_trials=missing_remaining,
    )
    identity_path = out / "gauthey_lbm_selected_identities_accumulated.csv"
    write_identity_csv(merged, identity_path)

    coverage = merged.coverage
    payload = {
        "status": "gauthey_lbm_resumed_exact_source_identity_recovery",
        "seed_identity_receipt": str(Path(args.seed_identities)),
        "seed_searched_trials": list(seed_searched),
        "remaining_trials": remaining,
        "available_remaining_trials": available_remaining,
        "missing_source_trials": missing_remaining,
        "searched_trials": list(coverage.searched_trials),
        "deposited_selected_count": 1620,
        "resolved_selected_count": len(coverage.resolved_selected_rows),
        "unresolved_selected_count": len(coverage.unresolved_selected_rows),
        "resolved_count_by_trial": dict(coverage.resolved_count_by_trial),
        "complete_selected_identity_recovery": coverage.complete_selected_identity_recovery,
        "identity_output": str(identity_path),
        "new_remote_receipt_dir": str(acquisition_dir),
        "firewalls": {
            "unsearched_trial_zero_means_zero_selected_rows": False,
            "missing_source_container_means_zero_selected_rows": False,
            "seed_exact_identity_may_be_rewritten_by_new_receipt": False,
            "source_trace_identity_implies_neuron_identity": False,
        },
    }
    summary_path = out / "gauthey_lbm_remaining_identity_recovery.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
