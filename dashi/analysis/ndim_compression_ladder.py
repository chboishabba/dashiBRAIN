"""Nested consumer-relative compression of NDim structural fibres.

The compression decision for an outer held-out region is made only from the
remaining regions.  Candidate fibre subsets are ranked by an inner leave-one-
region-out consumer score; the smallest subset within a declared absolute
residual tolerance of the full-family inner score is selected.  That subset is
then fitted on all outer-training pairs and evaluated on the untouched outer
region.

This is an empirical counterpart of the repo's FactorsThrough/compression
formalism: smaller is not intrinsically better, and pruning is admitted only
relative to a declared consumer/tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    evaluate_leave_one_region_out,
    fit_ndim_fibre_consumer,
    leave_one_region_out_masks,
)


@dataclass(frozen=True)
class CompressionFoldResult:
    held_out_region: str
    selected_fibres: tuple[str, ...]
    selected_size: int
    full_size: int
    inner_full_residual: float
    inner_selected_residual: float
    outer_residual: float
    outer_pair_count: int


@dataclass(frozen=True)
class NestedCompressionResult:
    tolerance: float
    folds: tuple[CompressionFoldResult, ...]

    @property
    def weighted_mean_residual(self) -> float:
        total = sum(f.outer_pair_count for f in self.folds)
        if total == 0:
            return float("nan")
        return float(sum(f.outer_residual * f.outer_pair_count for f in self.folds) / total)

    @property
    def mean_selected_size(self) -> float:
        return float(np.mean([f.selected_size for f in self.folds]))

    @property
    def selection_frequency(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for fold in self.folds:
            for name in fold.selected_fibres:
                counts[name] = counts.get(name, 0) + 1
        return counts


def subset_family(family: StructuralFibreFamily, names: Iterable[str]) -> StructuralFibreFamily:
    selected = tuple(names)
    if not selected:
        raise ValueError("fibre subset must be non-empty")
    missing = [name for name in selected if name not in family.fibres]
    if missing:
        raise KeyError(f"unknown fibre names: {missing}")
    return StructuralFibreFamily(
        family.regions,
        {name: family.fibres[name] for name in selected},
    )


def restrict_family_regions(
    family: StructuralFibreFamily,
    keep_indices: np.ndarray,
) -> StructuralFibreFamily:
    idx = np.asarray(keep_indices, dtype=int)
    return StructuralFibreFamily(
        tuple(family.regions[i] for i in idx),
        {name: np.asarray(matrix)[np.ix_(idx, idx)] for name, matrix in family.fibres.items()},
    )


def _all_nonempty_subsets(names: tuple[str, ...]):
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            yield subset


def choose_smallest_inner_adequate_subset(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    *,
    tolerance: float = 0.01,
    correlation_threshold: float = 0.98,
) -> tuple[tuple[str, ...], float, float]:
    """Choose the smallest subset within tolerance of full-family inner LORO."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    names = tuple(family.fibres.keys())
    if not names:
        raise ValueError("family must contain at least one fibre")
    full_score = evaluate_leave_one_region_out(
        family, observed, correlation_threshold=correlation_threshold
    ).weighted_mean_residual
    limit = full_score + tolerance

    best_subset: tuple[str, ...] | None = None
    best_score = float("inf")
    for subset in _all_nonempty_subsets(names):
        score = evaluate_leave_one_region_out(
            subset_family(family, subset),
            observed,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual
        if score <= limit + 1e-12:
            if best_subset is None or len(subset) < len(best_subset):
                best_subset, best_score = subset, score
            elif len(subset) == len(best_subset) and score < best_score - 1e-12:
                best_subset, best_score = subset, score
        if best_subset is not None and len(subset) > len(best_subset):
            break

    if best_subset is None:  # full family must always qualify numerically
        return names, full_score, full_score
    return best_subset, float(best_score), float(full_score)


def evaluate_nested_fibre_compression(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    *,
    tolerance: float = 0.01,
    correlation_threshold: float = 0.98,
) -> NestedCompressionResult:
    """Nested LORO fibre compression with untouched outer held-out regions."""
    obs = np.asarray(observed, dtype=float)
    n = len(family.regions)
    if obs.shape != (n, n):
        raise ValueError("observed must align with family regions")
    if n < 4:
        raise ValueError("nested compression requires at least four regions")

    folds: list[CompressionFoldResult] = []
    all_indices = np.arange(n)
    for held_i, held_region in enumerate(family.regions):
        train_indices = all_indices[all_indices != held_i]
        inner_family = restrict_family_regions(family, train_indices)
        inner_observed = obs[np.ix_(train_indices, train_indices)]
        selected, selected_score, full_score = choose_smallest_inner_adequate_subset(
            inner_family,
            inner_observed,
            tolerance=tolerance,
            correlation_threshold=correlation_threshold,
        )

        outer_fit, outer_held = leave_one_region_out_masks(n, held_i)
        outer_result = fit_ndim_fibre_consumer(
            subset_family(family, selected),
            obs,
            outer_fit,
            outer_held,
            correlation_threshold=correlation_threshold,
        )
        folds.append(
            CompressionFoldResult(
                held_out_region=held_region,
                selected_fibres=selected,
                selected_size=len(selected),
                full_size=len(family.fibres),
                inner_full_residual=float(full_score),
                inner_selected_residual=float(selected_score),
                outer_residual=outer_result.mean_residual,
                outer_pair_count=outer_result.held_out_pair_count,
            )
        )
    return NestedCompressionResult(float(tolerance), tuple(folds))
