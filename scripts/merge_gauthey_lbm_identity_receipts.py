#!/usr/bin/env python3
"""Merge exact Gauthey LBM identity receipts without rewriting historical partials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dashi.analysis.gauthey_lbm_experiment import LBM_TRIALS
from dashi.io.gauthey_lbm_identity_receipts import (
    load_identity_csv,
    merge_exact_identities,
    write_identity_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", action="append", required=True, help="Exact identity CSV; repeat to merge")
    parser.add_argument("--searched-trial", action="append", default=[])
    parser.add_argument("--missing-source-trial", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    for trial in (*args.searched_trial, *args.missing_source_trial):
        if trial not in LBM_TRIALS:
            raise SystemExit(f"unknown LBM trial: {trial}")

    receipts = [load_identity_csv(path) for path in args.identity]
    merged = merge_exact_identities(
        receipts,
        searched_trials=args.searched_trial,
        missing_source_trials=args.missing_source_trial,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity_path = out / "gauthey_lbm_selected_identities_accumulated.csv"
    write_identity_csv(merged, identity_path)

    coverage = merged.coverage
    payload = {
        "status": "gauthey_lbm_exact_identity_accumulation",
        "input_identity_receipts": [str(Path(path)) for path in args.identity],
        "searched_trials": list(coverage.searched_trials),
        "missing_source_trials": list(coverage.missing_source_trials),
        "resolved_selected_count": len(coverage.resolved_selected_rows),
        "unresolved_selected_count": len(coverage.unresolved_selected_rows),
        "resolved_count_by_trial": dict(coverage.resolved_count_by_trial),
        "complete_selected_identity_recovery": coverage.complete_selected_identity_recovery,
        "identity_output": str(identity_path),
        "unresolved_selected_rows": list(coverage.unresolved_selected_rows),
        "firewalls": {
            "unsearched_trial_zero_means_zero_selected_rois": False,
            "missing_source_container_means_trial_contributed_zero_rois": False,
            "conflicting_exact_identity_silently_adjudicated": False,
            "historical_partial_receipt_overwritten": False,
        },
    }
    summary_path = out / "gauthey_lbm_identity_accumulation.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
