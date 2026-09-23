"""Nulls for the overlap-controlled Fly structure/function target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.ndim_stability_nulls import (
    _ipf_strength_scramble,
    _sender_polarity_tendency,
    _two_hop_from_direct,
)
from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)
from dashi.analysis.structure_function_real import RegionStructuralFeatures


@dataclass(frozen=True)
class OverlapControlledStructuralNullSummary:
    observed_residual: float
    null_residuals: np.ndarray
    empirical_p_value: float
    null_mean_residual: float
    null_min_residual: float
    max_row_strength_error: float
    max_column_strength_error: float


def overlap_controlled_strength_preserving_null_loro(
    structural: RegionStructuralFeatures,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> OverlapControlledStructuralNullSummary:
    """Scramble wiring while keeping the functional overlap nuisance fixed.

    The functional observation and atlas-overlap kernel stay unchanged. Every
    null draw preserves direct weighted in/out strength, retains only sender
    polarity tendency, rebuilds all NDim structural fibres, and refits both the
    overlap nuisance and NDim consumer independently within every LORO fold.
    """
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    family = build_ndim_structural_fibres(structural)
    observed_result = evaluate_overlap_controlled_leave_one_region_out(
        family,
        observed,
        overlap_kernel,
        correlation_threshold=correlation_threshold,
    )
    real = observed_result.weighted_mean_residual
    rng = np.random.default_rng(seed)
    polarity = _sender_polarity_tendency(structural.direct, structural.signed_direct)

    nulls = np.empty(n_null, dtype=float)
    max_row = 0.0
    max_col = 0.0
    for k in range(n_null):
        direct_null, row_err, col_err = _ipf_strength_scramble(structural.direct, rng)
        max_row = max(max_row, row_err)
        max_col = max(max_col, col_err)
        signed_null = None if polarity is None else polarity[:, None] * direct_null
        structural_null = RegionStructuralFeatures(
            structural.regions,
            direct_null,
            _two_hop_from_direct(direct_null),
            signed_null,
        )
        nulls[k] = evaluate_overlap_controlled_leave_one_region_out(
            build_ndim_structural_fibres(structural_null),
            observed,
            overlap_kernel,
            correlation_threshold=correlation_threshold,
        ).weighted_mean_residual

    p = float((1 + np.count_nonzero(nulls <= real)) / (n_null + 1))
    return OverlapControlledStructuralNullSummary(
        observed_residual=float(real),
        null_residuals=nulls,
        empirical_p_value=p,
        null_mean_residual=float(np.mean(nulls)),
        null_min_residual=float(np.min(nulls)),
        max_row_strength_error=float(max_row),
        max_column_strength_error=float(max_col),
    )
