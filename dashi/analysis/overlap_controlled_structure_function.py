"""Control soft-atlas overlap geometry before testing MaleCNS structural fibres.

The soft VFB carrier intentionally allows one selected ROI to contribute to
multiple overlapping painted domains. That source-faithful choice induces a
known covariance nuisance: even independent ROI signals produce similar domain
traces when their ROI membership vectors overlap.

This module keeps that atlas-side geometry separate from connectome structure.
For domain membership rows w_r, define the overlap kernel

    K[r,s] = <w_r,w_s> / (||w_r|| ||w_s||).

Inside each train/test split only, fit

    C_func = alpha + beta K + residual

on training pairs. The NDim structural consumer then predicts the residual.
Neither alpha nor beta is allowed to see held-out functional pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Mapping

import numpy as np

from dashi.analysis.ndim_structure_function import (
    NDimHeldoutResult,
    RegionBlockedResult,
    RegionHoldoutFoldResult,
    StructuralFibreFamily,
    fit_ndim_fibre_consumer,
    leave_one_region_out_masks,
)


@dataclass(frozen=True)
class OverlapMembership:
    regions: tuple[str, ...]
    weights: np.ndarray  # region x selected ROI


@dataclass(frozen=True)
class OverlapNuisanceFit:
    intercept: float
    coefficient: float


@dataclass(frozen=True)
class OverlapControlledHeldoutResult:
    nuisance: OverlapNuisanceFit
    residual_target: np.ndarray
    ndim: NDimHeldoutResult


@dataclass(frozen=True)
class OverlapControlledRegionFold:
    held_out_region: str
    nuisance: OverlapNuisanceFit
    mean_residual: float
    train_pair_count: int
    held_out_pair_count: int
    selected_fibres: tuple[str, ...]
    coefficients: np.ndarray


@dataclass(frozen=True)
class OverlapControlledRegionBlockedResult:
    folds: tuple[OverlapControlledRegionFold, ...]

    @property
    def weighted_mean_residual(self) -> float:
        total = sum(f.held_out_pair_count for f in self.folds)
        return float(sum(f.mean_residual * f.held_out_pair_count for f in self.folds) / total)

    @property
    def mean_fold_residual(self) -> float:
        return float(np.mean([f.mean_residual for f in self.folds]))


def load_overlap_membership_csv(path: str | Path) -> OverlapMembership:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2 or header[0] != "region":
            raise ValueError("membership CSV must start with region followed by ROI columns")
        regions: list[str] = []
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError("membership CSV row width mismatch")
            regions.append(str(row[0]))
            rows.append([float(x) for x in row[1:]])
    if not rows:
        raise ValueError("membership CSV contains no regions")
    if len(set(regions)) != len(regions):
        raise ValueError("membership CSV contains duplicate regions")
    return OverlapMembership(tuple(regions), np.asarray(rows, dtype=float))


def overlap_cosine_kernel(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must be region x ROI")
    norms = np.linalg.norm(w, axis=1)
    if np.any(norms <= 0):
        raise ValueError("every retained region needs positive membership mass")
    k = (w @ w.T) / np.outer(norms, norms)
    return np.clip(k, -1.0, 1.0)


def restrict_overlap_kernel(
    membership: OverlapMembership,
    regions: tuple[str, ...],
) -> np.ndarray:
    index = {name: i for i, name in enumerate(membership.regions)}
    missing = [r for r in regions if r not in index]
    if missing:
        raise KeyError(f"functional membership lacks common regions: {missing}")
    ii = [index[r] for r in regions]
    return overlap_cosine_kernel(membership.weights[np.asarray(ii, dtype=int)])


def fit_overlap_nuisance(
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    fit_mask: np.ndarray,
) -> OverlapNuisanceFit:
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if obs.shape != kernel.shape or obs.shape != fit.shape:
        raise ValueError("observed, overlap_kernel, and fit_mask must align")
    y = obs[fit]
    x = kernel[fit]
    if y.size == 0:
        raise ValueError("fit_mask selects no pairs")
    design = np.column_stack([np.ones(y.size), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return OverlapNuisanceFit(float(beta[0]), float(beta[1]))


def residualize_overlap_geometry(
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    nuisance: OverlapNuisanceFit,
) -> np.ndarray:
    return np.asarray(observed, dtype=float) - (
        nuisance.intercept + nuisance.coefficient * np.asarray(overlap_kernel, dtype=float)
    )


def fit_overlap_controlled_ndim(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    fit_mask: np.ndarray,
    held_out_mask: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> OverlapControlledHeldoutResult:
    nuisance = fit_overlap_nuisance(observed, overlap_kernel, fit_mask)
    residual = residualize_overlap_geometry(observed, overlap_kernel, nuisance)
    ndim = fit_ndim_fibre_consumer(
        family,
        residual,
        fit_mask,
        held_out_mask,
        correlation_threshold=correlation_threshold,
    )
    return OverlapControlledHeldoutResult(nuisance, residual, ndim)


def evaluate_overlap_controlled_leave_one_region_out(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> OverlapControlledRegionBlockedResult:
    n = len(family.regions)
    folds: list[OverlapControlledRegionFold] = []
    for i, region in enumerate(family.regions):
        fit, held = leave_one_region_out_masks(n, i)
        result = fit_overlap_controlled_ndim(
            family,
            observed,
            overlap_kernel,
            fit,
            held,
            correlation_threshold=correlation_threshold,
        )
        folds.append(
            OverlapControlledRegionFold(
                held_out_region=region,
                nuisance=result.nuisance,
                mean_residual=result.ndim.mean_residual,
                train_pair_count=result.ndim.fit_pair_count,
                held_out_pair_count=result.ndim.held_out_pair_count,
                selected_fibres=result.ndim.selected_fibres,
                coefficients=result.ndim.coefficients,
            )
        )
    return OverlapControlledRegionBlockedResult(tuple(folds))


def overlap_controlled_label_permutation_null_loro(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> tuple[float, np.ndarray, float]:
    """Permute functional identity and atlas-overlap geometry together."""
    real = evaluate_overlap_controlled_leave_one_region_out(
        family, observed, overlap_kernel, correlation_threshold=correlation_threshold
    ).weighted_mean_residual
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_null, dtype=float)
    n = len(family.regions)
    for k in range(n_null):
        p = rng.permutation(n)
        obs_p = np.asarray(observed)[np.ix_(p, p)]
        ker_p = np.asarray(overlap_kernel)[np.ix_(p, p)]
        nulls[k] = evaluate_overlap_controlled_leave_one_region_out(
            family, obs_p, ker_p, correlation_threshold=correlation_threshold
        ).weighted_mean_residual
    p_value = float((1 + np.count_nonzero(nulls <= real)) / (n_null + 1))
    return float(real), nulls, p_value
