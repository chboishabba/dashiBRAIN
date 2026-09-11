"""Composable exact source-identity ledger for Gauthey LBM selected ROIs.

This module lets independently recovered per-trial receipts be merged without
rerunning already-paid trials.  Conflicting source identities for the same
published selected row are rejected rather than adjudicated silently.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

from dashi.analysis.gauthey_lbm_experiment import LBM_EXPECTED_SELECTED, LBM_TRIALS


@dataclass(frozen=True)
class LBMSourceIdentity:
    selected_row: int
    pooled_source_row: int
    trial_id: str
    plane_index: int
    cluster_index: int
    correlation: float


@dataclass(frozen=True)
class LBMIdentityLedger:
    rows: tuple[LBMSourceIdentity, ...]

    @property
    def resolved_selected_rows(self) -> tuple[int, ...]:
        return tuple(row.selected_row for row in self.rows)

    @property
    def unresolved_selected_rows(self) -> tuple[int, ...]:
        paid = set(self.resolved_selected_rows)
        return tuple(i for i in range(LBM_EXPECTED_SELECTED) if i not in paid)

    @property
    def complete(self) -> bool:
        return len(self.rows) == LBM_EXPECTED_SELECTED

    @property
    def counts_by_trial(self) -> dict[str, int]:
        counts = {trial: 0 for trial in LBM_TRIALS}
        for row in self.rows:
            counts[row.trial_id] += 1
        return counts


def read_identity_csv(path: str | Path) -> LBMIdentityLedger:
    rows: list[LBMSourceIdentity] = []
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
            raise ValueError(f"identity CSV lacks required fields: {path}")
        for raw in reader:
            row = LBMSourceIdentity(
                selected_row=int(raw["selected_row"]),
                pooled_source_row=int(raw["pooled_source_row"]),
                trial_id=str(raw["trial_id"]),
                plane_index=int(raw["plane_index"]),
                cluster_index=int(raw["cluster_index"]),
                correlation=float(raw["correlation"]),
            )
            if row.trial_id not in LBM_TRIALS:
                raise ValueError(f"unknown LBM trial_id {row.trial_id!r}")
            if not (0 <= row.selected_row < LBM_EXPECTED_SELECTED):
                raise ValueError(f"selected_row outside deposited carrier: {row.selected_row}")
            rows.append(row)
    return merge_identity_ledgers((LBMIdentityLedger(tuple(rows)),))


def merge_identity_ledgers(ledgers: Iterable[LBMIdentityLedger]) -> LBMIdentityLedger:
    by_selected: dict[int, LBMSourceIdentity] = {}
    source_owner: dict[int, int] = {}
    for ledger in ledgers:
        for row in ledger.rows:
            prior = by_selected.get(row.selected_row)
            if prior is not None and prior != row:
                raise ValueError(
                    "conflicting exact identities for selected_row "
                    f"{row.selected_row}: {prior!r} != {row!r}"
                )
            prior_selected = source_owner.get(row.pooled_source_row)
            if prior_selected is not None and prior_selected != row.selected_row:
                raise ValueError(
                    "one pooled source row maps to multiple selected rows: "
                    f"source={row.pooled_source_row} selected={prior_selected},{row.selected_row}"
                )
            by_selected[row.selected_row] = row
            source_owner[row.pooled_source_row] = row.selected_row
    return LBMIdentityLedger(tuple(by_selected[i] for i in sorted(by_selected)))


def write_identity_csv(ledger: LBMIdentityLedger, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selected_row",
                "pooled_source_row",
                "trial_id",
                "plane_index",
                "cluster_index",
                "correlation",
            ]
        )
        for row in ledger.rows:
            writer.writerow(
                [
                    row.selected_row,
                    row.pooled_source_row,
                    row.trial_id,
                    row.plane_index,
                    row.cluster_index,
                    f"{row.correlation:.17g}",
                ]
            )
