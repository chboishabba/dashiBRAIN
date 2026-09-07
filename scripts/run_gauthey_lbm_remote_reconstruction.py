#!/usr/bin/env python3
"""Reconstruct Gauthey's 1,620 selected LBM ROIs directly from Zenodo.

The six source trial dictionaries are nested inside per-trial ZIP members of the
~48 GB Data.zip deposit. This executable streams one outer member at a time,
unpickles the source dictionary directly from the inner ZIP, retains only the
trial-local top 1,620 candidates, deletes the temporary container, then performs
the exact global top-0.5% selection and compares it to the deposited
``dffs_audio_LB_corr_top05_all.pkl`` matrix.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import (
    GAUTHEY_DATA_ZIP_URL,
    LBM_EXPECTED_SELECTED,
    LBM_MIN_TIMEPOINTS,
    LBM_ROWS_PER_TRIAL,
    LBM_TRIALS,
    load_deposited_lbm_selected,
    pooled_lbm_row_to_trial_plane_cluster,
    published_lbm_stimulus_regressor,
)
from dashi.io.remote_zip import list_remote_zip, stream_remote_member_to_file

# Source dictionary/container stems differ from segmentation IDs: the source uses
# GCaMP6f_04032024_a2_r1 while the segmentation is 04032024_6f_a2_r1.
SOURCE_STEMS = (
    "GCaMP6f_04032024_a2_r1",
    "GCaMP6f_04032024_a2_r5",
    "GCaMP6f_04162024_a1_r1",
    "GCaMP6f_04192024_a1_r2",
    "GCaMP6f_04192024_a1_r6",
    "GCaMP6f_04192024_a1_r9",
)
SOURCE_CONTAINER_MEMBERS = tuple(f"Data/Dffs/Aligned/{stem}.zip" for stem in SOURCE_STEMS)


def _local_candidates_from_inner_zip(
    zip_path: Path,
    source_stem: str,
    trial_index: int,
    stimulus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with zipfile.ZipFile(zip_path) as zf:
        expected_basename = f"{source_stem}.pkl"
        matches = [name for name in zf.namelist() if Path(name).name == expected_basename]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {expected_basename} inside {zip_path}, found {matches}"
            )
        with zf.open(matches[0], "r") as handle:
            payload = pickle.load(handle)

    if not isinstance(payload, dict) or "dffs_aligned" not in payload:
        raise RuntimeError(f"{expected_basename} lacks 'dffs_aligned'")
    traces = np.asarray(payload["dffs_aligned"], dtype=float)
    if traces.ndim != 2 or traces.shape[0] != LBM_ROWS_PER_TRIAL:
        raise RuntimeError(
            f"{expected_basename} dffs_aligned shape {traces.shape}; expected "
            f"({LBM_ROWS_PER_TRIAL}, >= {LBM_MIN_TIMEPOINTS})"
        )
    if traces.shape[1] < LBM_MIN_TIMEPOINTS:
        raise RuntimeError(f"{expected_basename} has too few aligned timepoints")
    traces = traces[:, :LBM_MIN_TIMEPOINTS]

    means = traces.mean(axis=1, keepdims=True)
    stds = traces.std(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (traces - means) / stds
    stim_norm = (stimulus - stimulus.mean()) / stimulus.std()
    corr = np.dot(normalized, stim_norm.T) / normalized.shape[1]

    finite_mask = ~np.isnan(corr)
    finite_rows = np.arange(LBM_ROWS_PER_TRIAL)[finite_mask]
    finite_corr = corr[finite_mask]
    if finite_rows.size < LBM_EXPECTED_SELECTED:
        raise RuntimeError(f"trial {LBM_TRIALS[trial_index]} has too few finite traces")
    local_order = np.argsort(finite_corr)[-LBM_EXPECTED_SELECTED:]
    local_rows = finite_rows[local_order]
    pooled_rows = trial_index * LBM_ROWS_PER_TRIAL + local_rows
    candidate_corr = finite_corr[local_order].astype(float, copy=True)
    candidate_traces = np.asarray(traces[local_rows, :], dtype=float).copy()

    del payload, traces, normalized, corr
    gc.collect()
    return pooled_rows.astype(np.int64), candidate_corr, candidate_traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deposited-selected",
        default="data/gauthey_lbm/dffs_audio_LB_corr_top05_all.pkl",
    )
    parser.add_argument("--output-dir", default="data/gauthey_lbm/reconstruction")
    parser.add_argument("--scratch-dir", default="data/gauthey_lbm/scratch")
    parser.add_argument("--url", default=GAUTHEY_DATA_ZIP_URL)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()

    deposited_path = Path(args.deposited_selected)
    if not deposited_path.exists():
        raise SystemExit(f"deposited selected matrix not found: {deposited_path}")

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    by_name = {m.name: m for m in list_remote_zip(args.url)}
    missing = [name for name in SOURCE_CONTAINER_MEMBERS if name not in by_name]
    if missing:
        raise SystemExit("source trial containers missing from deposit:\n" + "\n".join(missing))

    stimulus = published_lbm_stimulus_regressor()
    candidate_rows: list[np.ndarray] = []
    candidate_corrs: list[np.ndarray] = []
    candidate_traces: list[np.ndarray] = []

    for trial_index, (source_stem, member_name) in enumerate(zip(SOURCE_STEMS, SOURCE_CONTAINER_MEMBERS)):
        member = by_name[member_name]
        tmp_zip = scratch / f"{source_stem}.zip"
        print(
            f"[{trial_index + 1}/{len(SOURCE_STEMS)}] streaming {member_name} "
            f"({member.compressed_size / 1e9:.2f} GB compressed -> "
            f"{member.uncompressed_size / 1e9:.2f} GB inner ZIP)"
        )
        stream_remote_member_to_file(
            args.url,
            member,
            tmp_zip,
            compressed_chunk_bytes=args.chunk_mib << 20,
        )
        try:
            rows, corrs, traces = _local_candidates_from_inner_zip(
                tmp_zip, source_stem, trial_index, stimulus
            )
            candidate_rows.append(rows)
            candidate_corrs.append(corrs)
            candidate_traces.append(traces)
            print(
                f"  retained local top {len(rows)}; "
                f"corr=[{float(corrs[0]):.6g}, {float(corrs[-1]):.6g}]"
            )
        finally:
            tmp_zip.unlink(missing_ok=True)
            gc.collect()

    rows = np.concatenate(candidate_rows)
    corrs = np.concatenate(candidate_corrs)
    traces = np.vstack(candidate_traces)
    global_order = np.argsort(corrs)[-LBM_EXPECTED_SELECTED:]
    selected_rows = rows[global_order]
    selected_corrs = corrs[global_order]
    selected_traces = traces[global_order, :]

    deposited = load_deposited_lbm_selected(deposited_path)
    diff = np.abs(selected_traces - deposited)
    max_abs_difference = float(np.max(diff))
    match = bool(
        np.allclose(
            selected_traces,
            deposited,
            atol=args.atol,
            rtol=args.rtol,
            equal_nan=True,
        )
    )

    identity_path = output / "gauthey_lbm_selected_identities.csv"
    with identity_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
        ])
        for selected_row, (pooled_row, corr) in enumerate(zip(selected_rows, selected_corrs)):
            trial_id, plane, cluster = pooled_lbm_row_to_trial_plane_cluster(int(pooled_row))
            writer.writerow([
                selected_row,
                int(pooled_row),
                trial_id,
                plane,
                cluster,
                f"{float(corr):.17g}",
            ])

    np.save(output / "gauthey_lbm_selected_reconstructed.npy", selected_traces)
    summary = {
        "source_supervoxel_count": int(len(SOURCE_STEMS) * LBM_ROWS_PER_TRIAL),
        "selected_count": int(selected_traces.shape[0]),
        "timepoints": int(selected_traces.shape[1]),
        "deposited_match": match,
        "max_abs_difference": max_abs_difference,
        "identity_output": str(identity_path),
        "selected_trace_output": str(output / "gauthey_lbm_selected_reconstructed.npy"),
        "scratch_containers_retained": False,
    }
    summary_path = output / "gauthey_lbm_reconstruction.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not match:
        raise SystemExit(
            "reconstructed top-0.5% matrix does not match the deposited matrix; "
            "do not use emitted identities for atlas aggregation"
        )


if __name__ == "__main__":
    main()
