#!/usr/bin/env python3
"""Resume exact Gauthey LBM identity recovery one trial at a time.

The canonical remote/local reconstruction runner remains the acquisition engine.
This wrapper inventories which of the six aligned-trial containers are actually
present, processes one available unresolved trial per subprocess, merges that
trial's durable exact identity checkpoint immediately into the accumulated
receipt, and only then advances to the next trial.

This failure geometry is intentional: a later network, disk, or memory failure
must not roll back identities already paid by an earlier completed trial.

If an accumulated receipt already exists in ``--output-dir``, it is authoritative
for resume: both its merged identity CSV and its searched-trial list are reused.
This preserves searched-zero trials, which cannot be reconstructed from positive
identity rows alone.
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


def _resolve_resume_seed(
    *,
    out: Path,
    cli_seed_identity: str,
    cli_seed_searched: tuple[str, ...],
) -> tuple[Path, tuple[str, ...]]:
    """Prefer a durable accumulated receipt over historical CLI seed state."""
    summary_path = out / "gauthey_lbm_remaining_identity_recovery.json"
    if not summary_path.is_file():
        return Path(cli_seed_identity), tuple(cli_seed_searched)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    identity_raw = payload.get("identity_output")
    searched_raw = payload.get("searched_trials")
    if not isinstance(identity_raw, str) or not isinstance(searched_raw, list):
        raise RuntimeError(f"existing accumulation receipt is malformed: {summary_path}")

    identity_path = Path(identity_raw)
    if not identity_path.is_file():
        raise RuntimeError(
            "existing accumulation receipt points to missing identity CSV: "
            f"{identity_path}"
        )

    searched = tuple(str(trial) for trial in searched_raw)
    unknown = set(searched) - set(LBM_TRIALS)
    if unknown:
        raise RuntimeError(
            f"existing accumulation receipt contains unknown searched trials: {sorted(unknown)}"
        )
    return identity_path, searched


def _write_accumulated_receipt(
    *,
    out: Path,
    seed_identity_receipt: str,
    seed_searched_trials: tuple[str, ...],
    remaining_trials: list[str],
    available_remaining_trials: list[str],
    missing_source_trials: list[str],
    searched_trials: tuple[str, ...],
    receipt_sets,
    completed_trial_receipts: list[str],
    released_source_zips: list[str],
) -> Path:
    merged = merge_exact_identities(
        receipt_sets,
        searched_trials=searched_trials,
        missing_source_trials=missing_source_trials,
    )
    identity_path = out / "gauthey_lbm_selected_identities_accumulated.csv"
    write_identity_csv(merged, identity_path)

    coverage = merged.coverage
    payload = {
        "status": "gauthey_lbm_resumed_exact_source_identity_recovery",
        "incremental_trial_checkpointing": True,
        "resume_from_accumulated_receipt": True,
        "seed_identity_receipt": str(Path(seed_identity_receipt)),
        "seed_searched_trials": list(seed_searched_trials),
        "remaining_trials": remaining_trials,
        "available_remaining_trials": available_remaining_trials,
        "missing_source_trials": missing_source_trials,
        "searched_trials": list(coverage.searched_trials),
        "completed_trial_receipts": list(completed_trial_receipts),
        "released_source_zips": list(released_source_zips),
        "deposited_selected_count": 1620,
        "resolved_selected_count": len(coverage.resolved_selected_rows),
        "unresolved_selected_count": len(coverage.unresolved_selected_rows),
        "resolved_count_by_trial": dict(coverage.resolved_count_by_trial),
        "complete_selected_identity_recovery": coverage.complete_selected_identity_recovery,
        "identity_output": str(identity_path),
        "firewalls": {
            "unsearched_trial_zero_means_zero_selected_rows": False,
            "missing_source_container_means_zero_selected_rows": False,
            "seed_exact_identity_may_be_rewritten_by_new_receipt": False,
            "source_trace_identity_implies_neuron_identity": False,
            "trial_identity_receipt_implies_replication": False,
        },
    }
    summary_path = out / "gauthey_lbm_remaining_identity_recovery.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return summary_path


def _require_durable_trial_checkpoint(
    *,
    returncode: int,
    trial_id: str,
    identity_path: Path,
    summary_path: Path,
) -> None:
    """Accept a searched-zero exit only after its checkpoint is durable."""
    if not identity_path.is_file() or not summary_path.is_file():
        raise SystemExit(
            "trial subprocess ended without durable standalone receipt: "
            f"trial={trial_id} exit={returncode} identity={identity_path} summary={summary_path}"
        )
    # The single-trial runner uses exit 1 to signal zero exact matches after it
    # has written a valid searched-zero checkpoint. Other nonzero exits remain
    # execution failures.
    if returncode not in (0, 1):
        raise SystemExit(f"trial subprocess failed: trial={trial_id} exit={returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-identities", required=True)
    parser.add_argument(
        "--seed-searched-trial",
        action="append",
        default=[],
        choices=LBM_TRIALS,
        help="Trial(s) already searched to produce the seed receipt. Existing accumulated receipt state takes precedence on resume.",
    )
    parser.add_argument(
        "--deposited-selected",
        default="data/gauthey_lbm/dffs_audio_LB_corr_top05_all.pkl",
    )
    parser.add_argument("--output-dir", default="data/gauthey_lbm/reconstruction_all_available")
    parser.add_argument("--scratch-dir", default="data/gauthey_lbm/scratch")
    parser.add_argument("--url", default=GAUTHEY_DATA_ZIP_URL)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--correlation-block-rows", type=int, default=256)
    parser.add_argument(
        "--release-source-zip-after-checkpoint",
        action="store_true",
        help=(
            "After one trial has produced a standalone exact identity checkpoint and "
            "the accumulated receipt has been rewritten, delete that trial's completed "
            "source ZIP from scratch to reclaim disk space."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cli_seed = load_identity_csv(args.seed_identities)
    cli_positive_trials = tuple(dict.fromkeys(row.trial_id for row in cli_seed))
    cli_searched = tuple(args.seed_searched_trial or cli_positive_trials)
    seed_path, seed_searched = _resolve_resume_seed(
        out=out,
        cli_seed_identity=args.seed_identities,
        cli_seed_searched=cli_searched,
    )
    seed = load_identity_csv(seed_path)

    remote_names = {member.name for member in list_remote_zip(args.url)}
    present = {
        trial: member in remote_names
        for trial, member in zip(LBM_TRIALS, SOURCE_CONTAINER_MEMBERS)
    }
    stem_by_trial = dict(zip(LBM_TRIALS, SOURCE_STEMS))
    remaining = [trial for trial in LBM_TRIALS if trial not in set(seed_searched)]
    available_remaining = [trial for trial in remaining if present[trial]]
    missing_remaining = [trial for trial in remaining if not present[trial]]

    trial_root = out / "trial_recovery"
    trial_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch_dir)

    receipt_sets = [seed]
    searched = list(seed_searched)
    completed_trial_receipts: list[str] = []
    released_source_zips: list[str] = []

    _write_accumulated_receipt(
        out=out,
        seed_identity_receipt=str(seed_path),
        seed_searched_trials=seed_searched,
        remaining_trials=remaining,
        available_remaining_trials=available_remaining,
        missing_source_trials=missing_remaining,
        searched_trials=tuple(searched),
        receipt_sets=receipt_sets,
        completed_trial_receipts=completed_trial_receipts,
        released_source_zips=released_source_zips,
    )

    for trial in available_remaining:
        trial_dir = trial_root / trial
        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/run_gauthey_lbm_remote_reconstruction.py",
            "--deposited-selected",
            args.deposited_selected,
            "--output-dir",
            str(trial_dir),
            "--scratch-dir",
            args.scratch_dir,
            "--url",
            args.url,
            "--chunk-mib",
            str(args.chunk_mib),
            "--correlation-block-rows",
            str(args.correlation_block_rows),
            "--trial",
            trial,
        ]
        print("+", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, check=False)

        stem = trial.replace("/", "_")
        checkpoint_identity = (
            trial_dir
            / "trial_receipts"
            / f"gauthey_lbm_selected_identities_{stem}.csv"
        )
        checkpoint_summary = (
            trial_dir
            / "trial_receipts"
            / f"gauthey_lbm_identity_receipt_{stem}.json"
        )
        _require_durable_trial_checkpoint(
            returncode=completed.returncode,
            trial_id=trial,
            identity_path=checkpoint_identity,
            summary_path=checkpoint_summary,
        )

        new_identities = load_identity_csv(checkpoint_identity)
        receipt_sets.append(new_identities)
        searched.append(trial)
        completed_trial_receipts.append(str(checkpoint_summary))

        _write_accumulated_receipt(
            out=out,
            seed_identity_receipt=str(seed_path),
            seed_searched_trials=seed_searched,
            remaining_trials=remaining,
            available_remaining_trials=available_remaining,
            missing_source_trials=missing_remaining,
            searched_trials=tuple(dict.fromkeys(searched)),
            receipt_sets=receipt_sets,
            completed_trial_receipts=completed_trial_receipts,
            released_source_zips=released_source_zips,
        )

        if args.release_source_zip_after_checkpoint:
            source_zip = scratch / f"{stem_by_trial[trial]}.zip"
            if source_zip.exists():
                source_zip.unlink()
                released_source_zips.append(str(source_zip))
                _write_accumulated_receipt(
                    out=out,
                    seed_identity_receipt=str(seed_path),
                    seed_searched_trials=seed_searched,
                    remaining_trials=remaining,
                    available_remaining_trials=available_remaining,
                    missing_source_trials=missing_remaining,
                    searched_trials=tuple(dict.fromkeys(searched)),
                    receipt_sets=receipt_sets,
                    completed_trial_receipts=completed_trial_receipts,
                    released_source_zips=released_source_zips,
                )


if __name__ == "__main__":
    main()
