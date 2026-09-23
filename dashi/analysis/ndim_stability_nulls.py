"""Stability diagnostics and coarse-structure-preserving nulls for Fly NDim.

These diagnostics add no predictive features.  They ask whether the selected
structural fibres/coefficients are stable across leave-one-region-out folds and
whether the observed result depends on pair-specific wiring beyond each
neuropil's coarse in/out strength and source-level signed tendency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dashi.analysis.ndim_structure_function import (
    RegionBlockedResult,
    build_ndim_structural_fibres,
    evaluate_leave_one_region_out,
)
from dashi.analysis.structure_function_real import RegionStructuralFeatures


@dataclass(frozen=True)
class FibreFoldStability:
    fibre: str
    selected_fold_count: int
    selected_fold_fraction: float
    coefficient_mean_when_selected: float
    coefficient_std_when_selected: float
    coefficient_min_when_selected: float
    coefficient_max_when_selected: float


@dataclass(frozen=True)
class FoldStabilitySummary:
    fold_count: int
    fibres: tuple[FibreFoldStability, ...]
    intercept_mean: float
    intercept_std: float


@dataclass(frozen=True)
class StructuralNullSummary:
    observed_residual: float
    null_residuals: np.ndarray
    empirical_p_value: float
    null_mean_residual: float
    null_min_residual: float
    max_row_strength_error: float
    max_column_strength_error: float


def summarize_loro_fibre_stability(blocked: RegionBlockedResult) -> FoldStabilitySummary:
    """Summarize selection and fitted standardized coefficients across LORO folds."""
    if not blocked.folds:
        raise ValueError("blocked result contains no folds")
    fold_count = len(blocked.folds)
    names: list[str] = []
    for fold in blocked.folds:
        for name in fold.selected_fibres:
            if name not in names:
                names.append(name)

    rows: list[FibreFoldStability] = []
    for name in names:
        coeffs: list[float] = []
        for fold in blocked.folds:
            if name in fold.selected_fibres:
                j = fold.selected_fibres.index(name)
                # coefficients[0] is the intercept; the rest align to selected fibres.
                coeffs.append(float(fold.coefficients[j + 1]))
        a = np.asarray(coeffs, dtype=float)
        rows.append(
            FibreFoldStability(
                fibre=name,
                selected_fold_count=int(a.size),
                selected_fold_fraction=float(a.size / fold_count),
                coefficient_mean_when_selected=float(np.mean(a)),
                coefficient_std_when_selected=float(np.std(a)),
                coefficient_min_when_selected=float(np.min(a)),
                coefficient_max_when_selected=float(np.max(a)),
            )
        )

    intercepts = np.asarray([float(f.coefficients[0]) for f in blocked.folds], dtype=float)
    return FoldStabilitySummary(
        fold_count=fold_count,
        fibres=tuple(rows),
        intercept_mean=float(np.mean(intercepts)),
        intercept_std=float(np.std(intercepts)),
    )


def _ipf_strength_scramble(
    direct: np.ndarray,
    rng: np.random.Generator,
    *,
    iterations: int = 400,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float, float]:
    """Randomize pairwise weights while retaining weighted row/column marginals.

    Iterative proportional fitting is performed on off-diagonal positive support.
    The result preserves the original nonnegative direct matrix's in/out strength
    to numerical tolerance while removing its pair-specific allocation.
    """
    a = np.asarray(direct, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("direct must be square")
    if np.any(a < 0):
        raise ValueError("strength-preserving null expects nonnegative direct weights")
    n = a.shape[0]
    target_row = np.sum(a, axis=1)
    target_col = np.sum(a, axis=0)
    if not np.isclose(np.sum(target_row), np.sum(target_col)):
        raise ValueError("row/column strength totals disagree")

    # Strictly positive off-diagonal seed gives IPF freedom to reallocate every
    # pair while retaining the no-self-edge convention.
    x = rng.gamma(shape=1.0, scale=1.0, size=(n, n))
    np.fill_diagonal(x, 0.0)

    zero_rows = target_row <= 0
    zero_cols = target_col <= 0
    x[zero_rows, :] = 0.0
    x[:, zero_cols] = 0.0

    for _ in range(iterations):
        rs = np.sum(x, axis=1)
        nz = rs > 0
        x[nz, :] *= (target_row[nz] / rs[nz])[:, None]
        cs = np.sum(x, axis=0)
        nz = cs > 0
        x[:, nz] *= (target_col[nz] / cs[nz])[None, :]
        np.fill_diagonal(x, 0.0)
        row_err = float(np.max(np.abs(np.sum(x, axis=1) - target_row)))
        col_err = float(np.max(np.abs(np.sum(x, axis=0) - target_col)))
        if max(row_err, col_err) <= tolerance:
            break

    row_err = float(np.max(np.abs(np.sum(x, axis=1) - target_row)))
    col_err = float(np.max(np.abs(np.sum(x, axis=0) - target_col)))
    return x, row_err, col_err


def _two_hop_from_direct(direct: np.ndarray) -> np.ndarray:
    denom = np.sum(np.abs(direct), axis=1)
    p = np.zeros_like(direct, dtype=float)
    nz = denom > 0
    p[nz] = direct[nz] / denom[nz, None]
    return p @ p


def _sender_polarity_tendency(
    direct: np.ndarray,
    signed_direct: np.ndarray | None,
) -> np.ndarray | None:
    if signed_direct is None:
        return None
    d = np.asarray(direct, dtype=float)
    s = np.asarray(signed_direct, dtype=float)
    if d.shape != s.shape:
        raise ValueError("signed_direct must align with direct")
    out = np.zeros(d.shape[0], dtype=float)
    denom = np.sum(np.abs(d), axis=1)
    nz = denom > 0
    out[nz] = np.sum(s, axis=1)[nz] / denom[nz]
    return np.clip(out, -1.0, 1.0)


def strength_preserving_structural_null_loro(
    structural: RegionStructuralFeatures,
    observed: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> StructuralNullSummary:
    """LORO null preserving coarse weighted marginals but scrambling wiring.

    Every null draw:
      1. preserves each region's weighted direct out/in strength numerically;
      2. retains only each source region's aggregate signed polarity tendency;
      3. recomputes two-hop/common-neighbour fibres from the scrambled graph;
      4. reselects compatible fibres and refits every LORO fold from scratch.

    Thus this null asks whether pair-specific connectome geometry adds predictive
    information beyond coarse region strength/polarity coordinates.
    """
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    observed_result = evaluate_leave_one_region_out(
        build_ndim_structural_fibres(structural),
        observed,
        correlation_threshold=correlation_threshold,
    )
    observed_residual = observed_result.weighted_mean_residual
    rng = np.random.default_rng(seed)
    polarity = _sender_polarity_tendency(structural.direct, structural.signed_direct)

    nulls = np.empty(n_null, dtype=float)
    max_row_error = 0.0
    max_col_error = 0.0
    for k in range(n_null):
        direct_null, row_error, col_error = _ipf_strength_scramble(structural.direct, rng)
        max_row_error = max(max_row_error, row_error)
        max_col_error = max(max_col_error, col_error)
        signed_null = None if polarity is None else polarity[:, None] * direct_null
        structural_null = RegionStructuralFeatures(
            structural.regions,
            direct_null,
            _two_hop_from_direct(direct_null),
            signed_null,
        )
        family_null = build_ndim_structural_fibres(structural_null)
        nulls[k] = evaluate_leave_one_region_out(
            family_null,
            observed,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    p = float((1 + np.count_nonzero(nulls <= observed_residual)) / (n_null + 1))
    return StructuralNullSummary(
        observed_residual=float(observed_residual),
        null_residuals=nulls,
        empirical_p_value=p,
        null_mean_residual=float(np.mean(nulls)),
        null_min_residual=float(np.min(nulls)),
        max_row_strength_error=float(max_row_error),
        max_column_strength_error=float(max_col_error),
    )
