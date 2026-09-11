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
