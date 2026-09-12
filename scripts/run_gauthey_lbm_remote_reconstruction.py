#!/usr/bin/env python3
"""Recover exact Gauthey LBM selected-ROI source identities from Zenodo or a local trial ZIP.

The deposited ``dffs_audio_LB_corr_top05_all.pkl`` matrix is a literal row subset
of six vertically stacked per-trial ``dffs_aligned[:, :7977]`` matrices. One of
the six source containers is absent from Zenodo record 17618684, so this runner
recovers every identity that can be established from the available containers
instead of treating one missing trial as a global failure.

Large aligned-trial pickles are loaded out of core. Embedded NumPy payloads are
spilled directly to disk-backed memmaps and correlations are scored in row
blocks, so neither pickle ingestion nor normalization requires a full trial
matrix copy in RAM.

For resumable large downloads, ``--local-source-zip`` processes exactly one
explicit ``--trial`` from an already-downloaded nested source ZIP. This bypasses
all remote archive access while preserving the exact same extraction, scoring,
and source-row matching path.
"""

from __future__ import annotations

import argparse
import csv
import gc
from hashlib import sha256
import json
from pathlib import Path
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
from dashi.io.gauthey_lbm_identity_receipts import (
    LBMExactIdentity,
    checkpoint_trial_identity_receipt,
)
from dashi.io.out_of_core_numpy_pickle import load_numpy_pickle_out_of_core
from dashi.io.remote_zip import list_remote_zip, stream_remote_member_to_file

SOURCE_STEMS = (
    "GCaMP6f_04032024_a2_r1",
    "GCaMP6f_04032024_a2_r5",
    "GCaMP6f_04162024_a1_r1",
    "GCaMP6f_04192024_a1_r2",
    "GCaMP6f_04192024_a1_r6",
    "GCaMP6f_04192024_a1_r9",
)
SOURCE_CONTAINER_MEMBERS = tuple(f"Data/Dffs/Aligned/{stem}.zip" for stem in SOURCE_STEMS)


def _blockwise_zero_lag_correlations(
    traces_roi_by_time: np.ndarray,
    stimulus: np.ndarray,
    *,
    block_rows: int = 256,
) -> np.ndarray:
    """Match the source zero-lag correlation formula without a full normalized copy."""
    traces = traces_roi_by_time
    if traces.ndim != 2:
        raise ValueError("traces must be two-dimensional [roi,time]")
    if traces.shape[1] < LBM_MIN_TIMEPOINTS:
        raise ValueError("traces have fewer than the required aligned timepoints")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")

    stim = np.asarray(stimulus, dtype=float)
    if stim.shape != (LBM_MIN_TIMEPOINTS,):
        raise ValueError("stimulus does not match the aligned Gauthey time carrier")
    stim_norm = (stim - stim.mean()) / stim.std()
    out = np.empty(traces.shape[0], dtype=float)

    for start in range(0, traces.shape[0], block_rows):
        stop = min(start + block_rows, traces.shape[0])
        block = np.asarray(traces[start:stop, :LBM_MIN_TIMEPOINTS], dtype=float)
        means = block.mean(axis=1, keepdims=True)
        stds = block.std(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (block - means) / stds
        out[start:stop] = np.dot(normalized, stim_norm.T) / normalized.shape[1]
        del block, means, stds, normalized
    return out


def _local_candidates_from_inner_zip(
    zip_path: Path,
    source_stem: str,
    trial_index: int,
    stimulus: np.ndarray,
    *,
    spill_dir: Path,
    correlation_block_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract local candidate traces with bounded RAM from a nested source ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        expected_basename = f"{source_stem}.pkl"
        matches = [name for name in zf.namelist() if Path(name).name == expected_basename]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {expected_basename} inside {zip_path}, found {matches}"
            )
        with zf.open(matches[0], "r") as handle:
            with load_numpy_pickle_out_of_core(handle, spill_dir=spill_dir) as payload:
                if not isinstance(payload, dict) or "dffs_aligned" not in payload:
                    raise RuntimeError(f"{expected_basename} lacks 'dffs_aligned'")
                traces = payload["dffs_aligned"]
                if not isinstance(traces, np.ndarray):
                    raise RuntimeError(f"{expected_basename} dffs_aligned is not an ndarray")
                if traces.ndim != 2 or traces.shape[0] != LBM_ROWS_PER_TRIAL:
                    raise RuntimeError(
                        f"{expected_basename} dffs_aligned shape {traces.shape}; expected "
                        f"({LBM_ROWS_PER_TRIAL}, >= {LBM_MIN_TIMEPOINTS})"
                    )
                if traces.shape[1] < LBM_MIN_TIMEPOINTS:
                    raise RuntimeError(f"{expected_basename} has too few aligned timepoints")

                corr = _blockwise_zero_lag_correlations(
                    traces,
                    stimulus,
                    block_rows=correlation_block_rows,
                )
                finite_mask = ~np.isnan(corr)
                finite_rows = np.arange(LBM_ROWS_PER_TRIAL)[finite_mask]
                finite_corr = corr[finite_mask]
                if finite_rows.size < LBM_EXPECTED_SELECTED:
                    raise RuntimeError(f"trial {LBM_TRIALS[trial_index]} has too few finite traces")

                local_order = np.argsort(finite_corr)[-LBM_EXPECTED_SELECTED:]
                local_rows = finite_rows[local_order]
                pooled_rows = trial_index * LBM_ROWS_PER_TRIAL + local_rows
                candidate_corr = finite_corr[local_order].astype(float, copy=True)
                candidate_traces = np.asarray(
                    traces[local_rows, :LBM_MIN_TIMEPOINTS], dtype=float
                ).copy()

    gc.collect()
    return pooled_rows.astype(np.int64), candidate_corr, candidate_traces


def _row_digest(row: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(row, dtype=np.float64)
    return sha256(contiguous.view(np.uint8)).digest()


def _match_candidates_to_deposited(
    deposited: np.ndarray,
    candidate_rows: np.ndarray,
    candidate_corrs: np.ndarray,
    candidate_traces: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Return exact (deposited_row, pooled_source_row, correlation) matches."""
    deposited_index: dict[bytes, list[int]] = {}
    for i in range(deposited.shape[0]):
        deposited_index.setdefault(_row_digest(deposited[i]), []).append(i)

    matches: list[tuple[int, int, float]] = []
    used_deposited: set[int] = set()
    for pooled_row, corr, trace in zip(candidate_rows, candidate_corrs, candidate_traces):
        for dep_row in deposited_index.get(_row_digest(trace), ()):
            if dep_row in used_deposited:
                continue
            if np.array_equal(trace, deposited[dep_row], equal_nan=True):
                matches.append((dep_row, int(pooled_row), float(corr)))
                used_deposited.add(dep_row)
                break
    return matches


def _trial_exact_identities(
    trial_matches: list[tuple[int, int, float]],
) -> tuple[LBMExactIdentity, ...]:
    identities: list[LBMExactIdentity] = []
    for selected_row, pooled_source_row, correlation in sorted(trial_matches):
        trial_id, plane_index, cluster_index = pooled_lbm_row_to_trial_plane_cluster(
            pooled_source_row
        )
        identities.append(
            LBMExactIdentity(
                selected_row=selected_row,
                pooled_source_row=pooled_source_row,
                trial_id=trial_id,
                plane_index=plane_index,
                cluster_index=cluster_index,
                correlation=correlation,
            )
        )
    return tuple(identities)


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
    parser.add_argument("--correlation-block-rows", type=int, default=256)
    parser.add_argument(
        "--trial",
        action="append",
        choices=LBM_TRIALS,
        help="Recover only this segmentation-style trial ID; repeat for multiple trials. Default: all deposited trials.",
    )
    parser.add_argument(
        "--local-source-zip",
        help=(
            "Process one already-downloaded nested source ZIP and bypass all remote archive access. "
            "Requires exactly one --trial."
        ),
    )
    args = parser.parse_args()

    if args.local_source_zip is not None:
        if args.trial is None or len(args.trial) != 1:
            raise SystemExit("--local-source-zip requires exactly one --trial")
        local_source_zip = Path(args.local_source_zip)
        if not local_source_zip.is_file():
            raise SystemExit(f"local source ZIP not found: {local_source_zip}")
    else:
        local_source_zip = None

    deposited_path = Path(args.deposited_selected)
    if not deposited_path.exists():
        raise SystemExit(f"deposited selected matrix not found: {deposited_path}")
    deposited = load_deposited_lbm_selected(deposited_path)

    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    spill_dir = scratch / "pickle_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trial_receipt_dir = output / "trial_receipts"

    requested = set(args.trial or LBM_TRIALS)
    stimulus = published_lbm_stimulus_regressor()
    by_name = {} if local_source_zip is not None else {m.name: m for m in list_remote_zip(args.url)}

    recovered: list[tuple[int, int, float]] = []
    processed_trials: list[str] = []
    missing_trials: list[str] = []

    for trial_index, (trial_id, source_stem, member_name) in enumerate(
        zip(LBM_TRIALS, SOURCE_STEMS, SOURCE_CONTAINER_MEMBERS)
    ):
        if trial_id not in requested:
            continue

        if local_source_zip is not None:
            tmp_zip = local_source_zip
            print(f"[{trial_id}] using explicit local source container {tmp_zip}")
        else:
            if member_name not in by_name:
                print(f"[{trial_id}] source container absent: {member_name}")
                missing_trials.append(trial_id)
                continue
            member = by_name[member_name]
            tmp_zip = scratch / f"{source_stem}.zip"
            if tmp_zip.exists():
                print(f"[{trial_id}] reusing existing source container {tmp_zip}")
            else:
                print(
                    f"[{trial_id}] streaming {member_name} "
                    f"({member.compressed_size / 1e9:.2f} GB compressed -> "
                    f"{member.uncompressed_size / 1e9:.2f} GB inner ZIP)"
                )
                stream_remote_member_to_file(
                    args.url,
                    member,
                    tmp_zip,
                    compressed_chunk_bytes=args.chunk_mib << 20,
                )

        rows, corrs, traces = _local_candidates_from_inner_zip(
            tmp_zip,
            source_stem,
            trial_index,
            stimulus,
            spill_dir=spill_dir,
            correlation_block_rows=args.correlation_block_rows,
        )
        trial_matches = _match_candidates_to_deposited(deposited, rows, corrs, traces)
        trial_identities = _trial_exact_identities(trial_matches)
        checkpoint = checkpoint_trial_identity_receipt(
            trial_identities,
            trial_id=trial_id,
            output_dir=trial_receipt_dir,
            remote_archive_accessed=(local_source_zip is None),
        )
        recovered.extend(trial_matches)
        processed_trials.append(trial_id)
        print(
            f"  local candidates={len(rows)}; exact deposited matches={len(trial_matches)}; "
            f"corr=[{float(corrs[0]):.6g}, {float(corrs[-1]):.6g}]"
        )
        print(f"  durable trial receipt: {checkpoint.summary_path}")
        gc.collect()

    by_deposited: dict[int, list[tuple[int, float]]] = {}
    for dep_row, pooled_row, corr in recovered:
        by_deposited.setdefault(dep_row, []).append((pooled_row, corr))
    ambiguous = {k: v for k, v in by_deposited.items() if len(v) > 1}
    if ambiguous:
        raise SystemExit(f"ambiguous exact source-trace matches detected: {ambiguous}")

    identity_path = output / "gauthey_lbm_selected_identities_partial.csv"
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
        for dep_row in sorted(by_deposited):
            pooled_row, corr = by_deposited[dep_row][0]
            trial_id, plane, cluster = pooled_lbm_row_to_trial_plane_cluster(pooled_row)
            writer.writerow([dep_row, pooled_row, trial_id, plane, cluster, f"{corr:.17g}"])

    resolved_rows = sorted(by_deposited)
    unresolved_rows = sorted(set(range(LBM_EXPECTED_SELECTED)) - set(resolved_rows))
    unresolved_path = output / "gauthey_lbm_selected_unresolved_rows.csv"
    with unresolved_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_row"])
        writer.writerows([[row] for row in unresolved_rows])

    summary = {
        "requested_trials": sorted(requested),
        "processed_trials": processed_trials,
        "missing_requested_trials": missing_trials,
        "deposited_selected_count": LBM_EXPECTED_SELECTED,
        "exact_source_identities_recovered": len(resolved_rows),
        "unresolved_selected_rows": len(unresolved_rows),
        "complete_identity_recovery": len(unresolved_rows) == 0,
        "identity_output": str(identity_path),
        "unresolved_output": str(unresolved_path),
        "trial_receipt_dir": str(trial_receipt_dir),
        "per_trial_checkpointing": True,
        "source_containers_retained_for_resume": True,
        "out_of_core_pickle_ingestion": True,
        "correlation_block_rows": args.correlation_block_rows,
        "remote_archive_accessed": local_source_zip is None,
        "local_source_zip": None if local_source_zip is None else str(local_source_zip),
        "identity_semantics": "exact trace equality to deposited selected row; not neuron identity",
    }
    summary_path = output / "gauthey_lbm_reconstruction.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not resolved_rows:
        raise SystemExit("no exact deposited selected traces were recovered from requested available trials")


if __name__ == "__main__":
    main()
