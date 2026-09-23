#!/usr/bin/env python3
"""Reconstruct Gauthey LBM selected ROI identities from six local source trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import reconstruct_lbm_selected_from_trials

SOURCE_PICKLE_NAMES = (
    "GCaMP6f_04032024_a2_r1.pkl",
    "GCaMP6f_04032024_a2_r5.pkl",
    "GCaMP6f_04162024_a1_r1.pkl",
    "GCaMP6f_04192024_a1_r2.pkl",
    "GCaMP6f_04192024_a1_r6.pkl",
    "GCaMP6f_04192024_a1_r9.pkl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", required=True, help="Directory containing the six GCaMP6f_*.pkl source dictionaries")
    parser.add_argument("--deposited-selected", required=True, help="Path to dffs_audio_LB_corr_top05_all.pkl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    trial_paths = [trial_dir / name for name in SOURCE_PICKLE_NAMES]
    missing = [str(path) for path in trial_paths if not path.exists()]
    if missing:
        raise SystemExit("missing source trial pickles:\n" + "\n".join(missing))

    result = reconstruct_lbm_selected_from_trials(
        trial_paths,
        deposited_selected_path=args.deposited_selected,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "gauthey_lbm_selected_reconstructed.npy", result.selected_traces)
    with (out / "gauthey_lbm_selected_identities.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
        ])
        for identity in result.identities:
            writer.writerow([
                identity.selected_row,
                identity.pooled_source_row,
                identity.trial_id,
                identity.plane_index,
                identity.cluster_index,
                f"{identity.correlation:.17g}",
            ])

    summary = {
        "selected_count": len(result.identities),
        "timepoints": int(result.selected_traces.shape[1]),
        "deposited_match": result.deposited_match,
        "max_abs_difference": result.max_abs_difference,
        "identity_output": str(out / "gauthey_lbm_selected_identities.csv"),
        "trace_output": str(out / "gauthey_lbm_selected_reconstructed.npy"),
    }
    (out / "gauthey_lbm_selection_reconstruction.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if result.deposited_match is not True:
        raise SystemExit("reconstructed selected matrix does not match deposited selected matrix")


if __name__ == "__main__":
    main()
