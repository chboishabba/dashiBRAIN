"""Accumulate and classify exact Gauthey LBM selected-ROI identity receipts.

A historical identity CSV may cover only a subset of the six LBM source trials.
Zero rows for an unsearched trial are therefore missing-data zeros, not evidence
that the trial contributed no globally selected ROIs.  This module keeps search
coverage separate from resolved identities and supports append-only accumulation
of exact source-row receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_EXPECTED_SELECTED,
    LBM_TRIALS,
    pooled_lbm_row_to_trial_plane_cluster,
)


@dataclass(frozen=True)
class LBMExactIdentity:
    selected_row: int
    pooled_source_row: int
    trial_id: str
    plane_index: int
    cluster_index: int
    correlation: float


@dataclass(frozen=True)
class LBMIdentityCoverage:
    searched_trials: tuple[str, ...]
    missing_source_trials: tuple[str, ...]
    resolved_selected_rows: tuple[int, ...]
    unresolved_selected_rows: tuple[int, ...]
    resolved_count_by_trial: Mapping[str, int]
    complete_selected_identity_recovery: bool


@dataclass(frozen=True)
class LBMIdentityAccumulator:
    identities: tuple[LBMExactIdentity, ...]
    coverage: LBMIdentityCoverage


@dataclass(frozen=True)
class LBMTrialIdentityCheckpoint:
    trial_id: str
    identity_path: Path
    summary_path: Path
    exact_source_identities_recovered: int
    remote_archive_accessed: bool


def _validate_identity(identity: LBMExactIdentity) -> None:
    if not (0 <= identity.selected_row < LBM_EXPECTED_SELECTED):
        raise ValueError(f"selected_row outside deposited carrier: {identity.selected_row}")
    decoded = pooled_lbm_row_to_trial_plane_cluster(identity.pooled_source_row)
    declared = (identity.trial_id, identity.plane_index, identity.cluster_index)
    if decoded != declared:
        raise ValueError(
            "identity geometry disagrees with pooled_source_row: "
            f"selected_row={identity.selected_row} decoded={decoded!r} declared={declared!r}"
        )
    if identity.trial_id not in LBM_TRIALS:
        raise ValueError(f"unknown LBM trial_id: {identity.trial_id}")


def load_identity_csv(path: str | Path) -> tuple[LBMExactIdentity, ...]:
    out: list[LBMExactIdentity] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: identity CSV lacks required columns")
        for row in reader:
            identity = LBMExactIdentity(
                selected_row=int(row["selected_row"]),
                pooled_source_row=int(row["pooled_source_row"]),
                trial_id=str(row["trial_id"]),
                plane_index=int(row["plane_index"]),
                cluster_index=int(row["cluster_index"]),
                correlation=float(row["correlation"]),
            )
            _validate_identity(identity)
            out.append(identity)
    return tuple(out)


def merge_exact_identities(
    receipts: Iterable[Sequence[LBMExactIdentity]],
    *,
    searched_trials: Iterable[str],
    missing_source_trials: Iterable[str] = (),
) -> LBMIdentityAccumulator:
    """Merge append-only exact identity receipts without silently adjudicating conflicts."""
    searched = tuple(dict.fromkeys(str(t) for t in searched_trials))
    missing = tuple(dict.fromkeys(str(t) for t in missing_source_trials))
    unknown = (set(searched) | set(missing)) - set(LBM_TRIALS)
    if unknown:
        raise ValueError(f"unknown LBM trial(s): {sorted(unknown)}")
    overlap = set(searched) & set(missing)
    if overlap:
        raise ValueError(f"trial cannot be both searched and source-missing: {sorted(overlap)}")

    by_selected: dict[int, LBMExactIdentity] = {}
    for receipt in receipts:
        for identity in receipt:
            _validate_identity(identity)
            prior = by_selected.get(identity.selected_row)
            if prior is None:
                by_selected[identity.selected_row] = identity
                continue
            if prior != identity:
                raise ValueError(
                    "conflicting exact identities for deposited selected row "
                    f"{identity.selected_row}: {prior!r} vs {identity!r}"
                )

    identities = tuple(by_selected[i] for i in sorted(by_selected))
    resolved_rows = tuple(sorted(by_selected))
    unresolved_rows = tuple(sorted(set(range(LBM_EXPECTED_SELECTED)) - set(resolved_rows)))
    count_by_trial = {trial: 0 for trial in LBM_TRIALS}
    for identity in identities:
        count_by_trial[identity.trial_id] += 1

    coverage = LBMIdentityCoverage(
        searched_trials=searched,
        missing_source_trials=missing,
        resolved_selected_rows=resolved_rows,
        unresolved_selected_rows=unresolved_rows,
        resolved_count_by_trial=count_by_trial,
        complete_selected_identity_recovery=(len(unresolved_rows) == 0),
    )
    return LBMIdentityAccumulator(identities=identities, coverage=coverage)


def write_identity_csv(accumulator: LBMIdentityAccumulator, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
        ])
        for identity in accumulator.identities:
            writer.writerow([
                identity.selected_row,
                identity.pooled_source_row,
                identity.trial_id,
                identity.plane_index,
                identity.cluster_index,
                f"{identity.correlation:.17g}",
            ])


def checkpoint_trial_identity_receipt(
    identities: Sequence[LBMExactIdentity],
    *,
    trial_id: str,
    output_dir: str | Path,
    remote_archive_accessed: bool,
) -> LBMTrialIdentityCheckpoint:
    """Persist one searched trial immediately so its source archive may be released.

    This receipt is intentionally standalone and append-only.  It records only
    exact identities from one searched trial and does not infer anything about
    unsearched or source-missing trials.
    """
    if trial_id not in LBM_TRIALS:
        raise ValueError(f"unknown LBM trial_id: {trial_id}")
    for identity in identities:
        _validate_identity(identity)
        if identity.trial_id != trial_id:
            raise ValueError(
                "checkpoint contains identity from a different trial: "
                f"expected={trial_id}, got={identity.trial_id}"
            )

    accumulator = merge_exact_identities(
        [identities],
        searched_trials=(trial_id,),
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = trial_id.replace("/", "_")
    identity_path = out / f"gauthey_lbm_selected_identities_{stem}.csv"
    summary_path = out / f"gauthey_lbm_identity_receipt_{stem}.json"
    write_identity_csv(accumulator, identity_path)
    payload = {
        "status": "gauthey_lbm_single_trial_exact_identity_receipt",
        "trial_id": trial_id,
        "standalone_trial_receipt": True,
        "searched_trial": True,
        "exact_source_identities_recovered": len(accumulator.identities),
        "remote_archive_accessed": bool(remote_archive_accessed),
        "identity_output": str(identity_path),
        "identity_semantics": "exact trace equality to deposited selected row; not neuron identity",
        "firewalls": {
            "zero_matches_imply_zero_biological_contribution": False,
            "trial_receipt_implies_replication": False,
            "trial_receipt_implies_atlas_registration": False,
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LBMTrialIdentityCheckpoint(
        trial_id=trial_id,
        identity_path=identity_path,
        summary_path=summary_path,
        exact_source_identities_recovered=len(accumulator.identities),
        remote_archive_accessed=bool(remote_archive_accessed),
    )
