"""N-dimensional structural fibres for the MaleCNS structure/function benchmark.

This module applies the repo-wide graph-colouring/NDim lesson to Fly:
keep structurally distinct candidate fibres separate, remove only structurally
redundant candidates using training-feature geometry, then compose the surviving
family for the functional consumer. Functional observations are *not* used to
select the compatible fibre family.

The present functional consumer is symmetric region correlation. Directed
connectome structure is therefore exposed as forward/reverse fibres rather than
silently comparing one arbitrary direction to a symmetric target.

Pair-held-out evaluation is useful but not sufficient for region-level
externalization because the same neuropils can appear in both train and test
pairs. This module therefore also provides leave-one-region-out evaluation and
region-label permutation nulls that refit the complete fibre consumer inside
each null replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dashi.analysis.structure_function_real import RegionStructuralFeatures


@dataclass(frozen=True)
class StructuralFibreFamily:
    regions: tuple[str, ...]
    fibres: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class FibreCompatibilitySelection:
    selected: tuple[str, ...]
    rejected_constant: tuple[str, ...]
    rejected_redundant: tuple[tuple[str, str, float], ...]
    correlation_threshold: float


@dataclass(frozen=True)
class NDimHeldoutResult:
    selected_fibres: tuple[str, ...]
    coefficients: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    held_out_prediction: np.ndarray
    held_out_observed: np.ndarray
    held_out_residuals: np.ndarray
    fit_pair_count: int
    held_out_pair_count: int
    compatibility: FibreCompatibilitySelection

    @property
    def mean_residual(self) -> float:
        return float(np.mean(self.held_out_residuals))


@dataclass(frozen=True)
class RegionHoldoutFoldResult:
    held_out_region: str
    train_pair_count: int
    held_out_pair_count: int
    mean_residual: float
    selected_fibres: tuple[str, ...]
    coefficients: np.ndarray


@dataclass(frozen=True)
class RegionBlockedResult:
    folds: tuple[RegionHoldoutFoldResult, ...]

    @property
    def mean_fold_residual(self) -> float:
        return float(np.mean([f.mean_residual for f in self.folds]))

    @property
    def weighted_mean_residual(self) -> float:
        total = sum(f.held_out_pair_count for f in self.folds)
        if total == 0:
            return float("nan")
        return float(
            sum(f.mean_residual * f.held_out_pair_count for f in self.folds) / total
        )


@dataclass(frozen=True)
class PermutationNullResult:
    observed_residual: float
    null_residuals: np.ndarray
    empirical_p_value: float
    fold_regions: tuple[str, ...] = ()
    observed_fold_residuals: np.ndarray | None = None
    null_fold_residuals: np.ndarray | None = None
    fold_empirical_p_values: np.ndarray | None = None


def _safe_row_normalize_dense(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    denom = np.sum(np.abs(x), axis=1)
    out = np.zeros_like(x, dtype=float)
    nz = denom > 0
    out[nz] = x[nz] / denom[nz, None]
    return out


def build_ndim_structural_fibres(structural: RegionStructuralFeatures) -> StructuralFibreFamily:
    """Expose the current structural carrier as distinct consumer-facing fibres.

    No weights are assigned here. The axes are declared structural hypotheses:
    directed direct/path influence, shared-input/shared-output similarity, and
    signed directed influence when transmitter signs are available.
    """
    direct = np.asarray(structural.direct, dtype=float)
    two_hop = np.asarray(structural.two_hop, dtype=float)
    if direct.shape != two_hop.shape or direct.ndim != 2 or direct.shape[0] != direct.shape[1]:
        raise ValueError("direct and two_hop must be aligned square region matrices")

    p = _safe_row_normalize_dense(direct)
    common_input = p.T @ p
    common_output = p @ p.T

    fibres: dict[str, np.ndarray] = {
        "direct_forward": direct,
        "direct_reverse": direct.T,
        "two_hop_forward": two_hop,
        "two_hop_reverse": two_hop.T,
        "common_input": common_input,
        "common_output": common_output,
    }
    if structural.signed_direct is not None:
        signed = np.asarray(structural.signed_direct, dtype=float)
        if signed.shape != direct.shape:
            raise ValueError("signed_direct must align with direct")
        fibres["signed_forward"] = signed
        fibres["signed_reverse"] = signed.T

    return StructuralFibreFamily(tuple(structural.regions), fibres)


def _upper_values(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float)[mask]


def select_compatible_fibres(
    family: StructuralFibreFamily,
    fit_mask: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
    minimum_scale: float = 1e-12,
    priority: Sequence[str] | None = None,
) -> FibreCompatibilitySelection:
    """Select a structurally non-redundant fibre family without using outcomes."""
    if not (0.0 <= correlation_threshold <= 1.0):
        raise ValueError("correlation_threshold must be in [0,1]")
    fit = np.asarray(fit_mask, dtype=bool)
    names = tuple(priority) if priority is not None else tuple(family.fibres.keys())
    missing = [name for name in names if name not in family.fibres]
    if missing:
        raise KeyError(f"unknown fibre names in priority: {missing}")

    selected: list[str] = []
    constant: list[str] = []
    redundant: list[tuple[str, str, float]] = []
    standardized: dict[str, np.ndarray] = {}

    for name in names:
        values = _upper_values(family.fibres[name], fit)
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= minimum_scale:
            constant.append(name)
            continue
        z = (values - float(np.mean(values))) / scale
        conflict = None
        for existing in selected:
            corr = float(np.dot(z, standardized[existing]) / z.size)
            if abs(corr) >= correlation_threshold:
                conflict = (existing, corr)
                break
        if conflict is not None:
            redundant.append((name, conflict[0], conflict[1]))
            continue
        selected.append(name)
        standardized[name] = z

    if not selected:
        raise ValueError("no non-constant compatible structural fibres remain")
    return FibreCompatibilitySelection(
        tuple(selected), tuple(constant), tuple(redundant), float(correlation_threshold)
    )


def fit_ndim_fibre_consumer(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    held_out_mask: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> NDimHeldoutResult:
    """Fit only the compatible fibre composition on training pairs."""
    fit = np.asarray(fit_mask, dtype=bool)
    held = np.asarray(held_out_mask, dtype=bool)
    if np.any(fit & held):
        raise ValueError("fit and held-out masks must be disjoint")
    if not np.any(fit) or not np.any(held):
        raise ValueError("fit and held-out masks must both select region pairs")
    selection = select_compatible_fibres(
        family, fit, correlation_threshold=correlation_threshold
    )

    x_train = np.column_stack([_upper_values(family.fibres[n], fit) for n in selection.selected])
    x_held = np.column_stack([_upper_values(family.fibres[n], held) for n in selection.selected])
    y_train = np.asarray(observed, dtype=float)[fit]
    y_held = np.asarray(observed, dtype=float)[held]

    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales[scales <= 1e-12] = 1.0
    z_train = (x_train - means) / scales
    z_held = (x_held - means) / scales

    design_train = np.column_stack([np.ones(z_train.shape[0]), z_train])
    beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)
    prediction = np.column_stack([np.ones(z_held.shape[0]), z_held]) @ beta

    return NDimHeldoutResult(
        selected_fibres=selection.selected,
        coefficients=np.asarray(beta, dtype=float),
        feature_means=np.asarray(means, dtype=float),
        feature_scales=np.asarray(scales, dtype=float),
        held_out_prediction=np.asarray(prediction, dtype=float),
        held_out_observed=np.asarray(y_held, dtype=float),
        held_out_residuals=np.abs(prediction - y_held),
        fit_pair_count=int(np.sum(fit)),
        held_out_pair_count=int(np.sum(held)),
        compatibility=selection,
    )


def leave_one_region_out_masks(n: int, held_out_region_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Train on pairs excluding one region; test on every pair touching it."""
    if n < 3:
        raise ValueError("at least three regions are required")
    if not (0 <= held_out_region_index < n):
        raise IndexError("held_out_region_index out of range")
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    ii, jj = np.indices((n, n))
    touches = (ii == held_out_region_index) | (jj == held_out_region_index)
    held = upper & touches
    fit = upper & ~touches
    return fit, held


def evaluate_leave_one_region_out(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> RegionBlockedResult:
    """Evaluate externalization to each completely unseen neuropil in turn."""
    n = len(family.regions)
    obs = np.asarray(observed, dtype=float)
    if obs.shape != (n, n):
        raise ValueError("observed must align with family regions")
    folds: list[RegionHoldoutFoldResult] = []
    for region_i, region in enumerate(family.regions):
        fit, held = leave_one_region_out_masks(n, region_i)
        result = fit_ndim_fibre_consumer(
            family,
            obs,
            fit,
            held,
            correlation_threshold=correlation_threshold,
        )
        folds.append(
            RegionHoldoutFoldResult(
                held_out_region=region,
                train_pair_count=result.fit_pair_count,
                held_out_pair_count=result.held_out_pair_count,
                mean_residual=result.mean_residual,
                selected_fibres=result.selected_fibres,
                coefficients=result.coefficients,
            )
        )
    return RegionBlockedResult(tuple(folds))


def _permute_symmetric_observation(observed: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observed, dtype=float)
    return obs[np.ix_(permutation, permutation)]


def region_label_permutation_null_pair_holdout(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    held_out_mask: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> PermutationNullResult:
    """Permutation null that refits the NDim consumer for every region relabeling."""
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    observed_result = fit_ndim_fibre_consumer(
        family,
        observed,
        fit_mask,
        held_out_mask,
        correlation_threshold=correlation_threshold,
    )
    rng = np.random.default_rng(seed)
    n = len(family.regions)
    nulls = np.empty(n_null, dtype=float)
    for k in range(n_null):
        permuted = _permute_symmetric_observation(observed, rng.permutation(n))
        nulls[k] = fit_ndim_fibre_consumer(
            family,
            permuted,
            fit_mask,
            held_out_mask,
            correlation_threshold=correlation_threshold,
        ).mean_residual
    p = float((1 + np.count_nonzero(nulls <= observed_result.mean_residual)) / (n_null + 1))
    return PermutationNullResult(observed_result.mean_residual, nulls, p)


def region_label_permutation_null_leave_one_region_out(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
    correlation_threshold: float = 0.98,
) -> PermutationNullResult:
    """Region-label null for unseen-region generalization, refitting every fold/null.

    Besides the global weighted LORO score, retain each null draw's residual for
    each held-out region. This lets the empirical result distinguish a broad
    cross-neuropil advantage from a global score driven by a few easy folds.
    """
    if n_null < 1:
        raise ValueError("n_null must be >= 1")
    observed_result = evaluate_leave_one_region_out(
        family, observed, correlation_threshold=correlation_threshold
    )
    rng = np.random.default_rng(seed)
    n = len(family.regions)
    observed_folds = np.asarray([f.mean_residual for f in observed_result.folds], dtype=float)
    null_folds = np.empty((n_null, n), dtype=float)
    nulls = np.empty(n_null, dtype=float)
    for k in range(n_null):
        permuted = _permute_symmetric_observation(observed, rng.permutation(n))
        null_result = evaluate_leave_one_region_out(
            family,
            permuted,
            correlation_threshold=correlation_threshold,
        )
        null_folds[k, :] = [f.mean_residual for f in null_result.folds]
        nulls[k] = null_result.weighted_mean_residual
    p = float((1 + np.count_nonzero(nulls <= observed_result.weighted_mean_residual)) / (n_null + 1))
    fold_p = np.asarray(
        [
            (1 + np.count_nonzero(null_folds[:, i] <= observed_folds[i])) / (n_null + 1)
            for i in range(n)
        ],
        dtype=float,
    )
    return PermutationNullResult(
        observed_result.weighted_mean_residual,
        nulls,
        p,
        fold_regions=tuple(family.regions),
        observed_fold_residuals=observed_folds,
        null_fold_residuals=null_folds,
        fold_empirical_p_values=fold_p,
    )
