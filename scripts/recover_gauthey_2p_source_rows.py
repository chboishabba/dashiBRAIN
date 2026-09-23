#!/usr/bin/env python3
"""Recover pooled selected Gauthey 2p ROI rows back to trial/plane/cluster identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from dashi.io.gauthey_2p_source_recovery import (
    TWO_P_TRIAL_SPECS,
    load_pooled_source_matrix,
    recover_selected_source_rows,
    write_source_recovery_csv,
)
from dashi.io.gauthey_compact import GAUTHEY_2P_TIME_SAMPLES


def _selected_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "audio_correlated" not in obj:
        raise ValueError("selected pickle must contain audio_correlated")
    return np.asarray(obj["audio_correlated"], dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selected", required=True, help="dffs_audio_2p_corr_top05_all.pkl")
    p.add_argument("--source-dir", required=True, help="directory containing the four source GCaMP6f_*.pkl files")
    p.add_argument("--output", required=True, help="output source-recovery CSV")
    p.add_argument("--receipt", help="optional JSON summary")
    p.add_argument("--atol", type=float, help="optional absolute tolerance fallback after exact misses")
    p.add_argument("--list-required", action="store_true")
    args = p.parse_args()

    if args.list_required:
        for spec in TWO_P_TRIAL_SPECS:
            print(spec.data_filename)
            print(spec.label_filename)
        return

    source_dir = Path(args.source_dir)
    trial_paths = {spec.data_filename: source_dir / spec.data_filename for spec in TWO_P_TRIAL_SPECS}
    pooled = load_pooled_source_matrix(trial_paths)
    selected = _selected_matrix(Path(args.selected))
    receipt = recover_selected_source_rows(selected, pooled, atol=args.atol)

    summary = {
        "selected_roi_count": receipt.selected_roi_count,
        "candidate_roi_count": receipt.candidate_roi_count,
        "time_sample_count": receipt.time_sample_count,
        "exact_match_count": receipt.exact_match_count,
        "ambiguous_match_count": receipt.ambiguous_match_count,
        "unmatched_count": receipt.unmatched_count,
        "complete": receipt.complete,
        "expected_source_semantics": {
            "trials": len(TWO_P_TRIAL_SPECS),
            "planes_per_trial": 47,
            "clusters_per_plane": 1000,
            "time_samples": GAUTHEY_2P_TIME_SAMPLES,
        },
    }
    print(json.dumps(summary, indent=2))

    if receipt.complete:
        write_source_recovery_csv(receipt, args.output)
    if args.receipt:
        path = Path(args.receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
