"""Biologically meaningful holdout splits for the MaleCNS programme."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class HoldoutSplit:
    name: str
    train_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]

    @property
    def disjoint(self) -> bool:
        return set(self.train_ids).isdisjoint(self.held_out_ids)


def split_by_category(
    ids: Sequence[str],
    labels: Mapping[str, str],
    held_out_labels: set[str],
    *,
    name: str,
) -> HoldoutSplit:
    train = tuple(i for i in ids if labels.get(i, "") not in held_out_labels)
    held = tuple(i for i in ids if labels.get(i, "") in held_out_labels)
    split = HoldoutSplit(name, train, held)
    if not split.disjoint:
        raise AssertionError("biological split is not disjoint")
    if not held:
        raise ValueError("held-out biological category contains no units")
    return split


def region_holdout(ids: Sequence[str], region_by_id: Mapping[str, str], regions: set[str]) -> HoldoutSplit:
    return split_by_category(ids, region_by_id, regions, name="region_holdout")


def cell_type_holdout(ids: Sequence[str], type_by_id: Mapping[str, str], types: set[str]) -> HoldoutSplit:
    return split_by_category(ids, type_by_id, types, name="cell_type_holdout")


def dimorphism_holdout(ids: Sequence[str], dimorphism_by_id: Mapping[str, str]) -> HoldoutSplit:
    held = {v for v in dimorphism_by_id.values() if v and v.lower() not in {"none", "false", "0", "unisex", "isomorphic"}}
    return split_by_category(ids, dimorphism_by_id, held, name="dimorphism_holdout")


def trial_holdout(trial_ids: Sequence[str], held_out_trials: set[str]) -> HoldoutSplit:
    labels = {t: t for t in trial_ids}
    return split_by_category(trial_ids, labels, held_out_trials, name="trial_holdout")
