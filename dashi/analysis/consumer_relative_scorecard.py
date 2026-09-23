"""Terminal metrics for an already-declared consumer-relative representation.

This module deliberately sits *after* observer construction, admissibility,
consumer adequacy, and holdout design.  It does not select structural fibres and
it does not promote a low terminal loss into representation equivalence or
consumer sufficiency.

The scorecard makes the current MaleCNS structure/function benchmark easier to
interpret by reporting standard held-out prediction metrics together with
explicit baseline-normalized gains.  For the overlap-controlled LORO consumer,
``collect_overlap_controlled_loro_predictions`` reuses the exact existing fold
fit and concatenates its held-out prediction/target vectors; therefore its MAE
is the existing pair-count-weighted LORO residual, not a new scoring protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    leave_one_region_out_masks,
)
from dashi.analysis.overlap_controlled_structure_function import (
    fit_overlap_controlled_ndim,
)


@dataclass(frozen=True)
class PredictionMetrics:
    pair_count: int
    mae: float
    r2: float
    pearson_r: float
    spearman_rho: float


@dataclass(frozen=True)
class BaselineComparison:
    baseline_name: str
    baseline_metrics: PredictionMetrics
    mae_absolute_improvement: float
    mae_fractional_improvement: float
    r2_delta: float
    pearson_r_delta: float
    spearman_rho_delta: float


@dataclass(frozen=True)
class ConsumerRelativeScorecard:
    candidate_name: str
    target_semantics: str
    candidate_metrics: PredictionMetrics
    baselines: tuple[BaselineComparison, ...]
    latent_dimension: int | None
    description_length: float | None
    description_length_unit: str | None
    consumer_sufficiency_certified: bool = False
    physical_representation_identity_certified: bool = False


@dataclass(frozen=True)
class LOROPredictionBundle:
    prediction: np.ndarray
    observed: np.ndarray
    zero_baseline: np.ndarray
    fold_mean_baseline: np.ndarray
    fold_regions: tuple[str, ...]
    selected_fibres_by_fold: tuple[tuple[str, ...], ...]
    held_out_pair_counts: tuple[int, ...]


def _as_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    xnorm = float(np.linalg.norm(xc))
    ynorm = float(np.linalg.norm(yc))
    if xnorm <= 1e-15 or ynorm <= 1e-15:
        return float("nan")
    return float(np.dot(xc, yc) / (xnorm * ynorm))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with exact-tie handling, without SciPy."""
    x = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and x[order[stop]] == x[order[start]]:
            stop += 1
        # 1-based ranks are start+1 .. stop; average = (start+1+stop)/2.
        average = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def score_prediction_vector(
    prediction: np.ndarray,
    observed: np.ndarray,
) -> PredictionMetrics:
    pred = _as_finite_vector(prediction, name="prediction")
    obs = _as_finite_vector(observed, name="observed")
    if pred.shape != obs.shape:
        raise ValueError("prediction and observed must have the same shape")

    residual = pred - obs
    mae = float(np.mean(np.abs(residual)))
    sse = float(np.dot(residual, residual))
    centred = obs - float(np.mean(obs))
    sst = float(np.dot(centred, centred))
    r2 = float("nan") if sst <= 1e-15 else float(1.0 - sse / sst)
    pearson = _pearson(pred, obs)
    spearman = _pearson(_average_ranks(pred), _average_ranks(obs))
    return PredictionMetrics(
        pair_count=int(obs.size),
        mae=mae,
        r2=r2,
        pearson_r=pearson,
        spearman_rho=spearman,
    )


def _delta(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline):
        return float("nan")
    return float(candidate - baseline)


def build_consumer_relative_scorecard(
    prediction: np.ndarray,
    observed: np.ndarray,
    *,
    baseline_predictions: Mapping[str, np.ndarray],
    candidate_name: str,
    target_semantics: str,
    latent_dimension: int | None = None,
    description_length: float | None = None,
    description_length_unit: str | None = None,
) -> ConsumerRelativeScorecard:
    if latent_dimension is not None and latent_dimension < 1:
        raise ValueError("latent_dimension must be positive when declared")
    if description_length is not None and description_length < 0:
        raise ValueError("description_length must be non-negative when declared")
    if description_length is not None and not description_length_unit:
        raise ValueError("description_length_unit is required with description_length")

    candidate = score_prediction_vector(prediction, observed)
    comparisons: list[BaselineComparison] = []
    for name, baseline_prediction in baseline_predictions.items():
        baseline = score_prediction_vector(baseline_prediction, observed)
        absolute = float(baseline.mae - candidate.mae)
        fractional = (
            float(absolute / baseline.mae)
            if baseline.mae > 1e-15
            else float("nan")
        )
        comparisons.append(
            BaselineComparison(
                baseline_name=str(name),
                baseline_metrics=baseline,
                mae_absolute_improvement=absolute,
                mae_fractional_improvement=fractional,
                r2_delta=_delta(candidate.r2, baseline.r2),
                pearson_r_delta=_delta(candidate.pearson_r, baseline.pearson_r),
                spearman_rho_delta=_delta(candidate.spearman_rho, baseline.spearman_rho),
            )
        )

    return ConsumerRelativeScorecard(
        candidate_name=str(candidate_name),
        target_semantics=str(target_semantics),
        candidate_metrics=candidate,
        baselines=tuple(comparisons),
        latent_dimension=latent_dimension,
        description_length=description_length,
        description_length_unit=description_length_unit,
        consumer_sufficiency_certified=False,
        physical_representation_identity_certified=False,
    )


def collect_overlap_controlled_loro_predictions(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> LOROPredictionBundle:
    """Concatenate the exact existing overlap-controlled LORO held-out vectors.

    Each unordered region pair appears in the two folds corresponding to its two
    endpoint regions, exactly matching the weighting implicit in the historical
    fold aggregate.  The observed vector is the fold-specific overlap-controlled
    residual target, because that is what the structural consumer actually fits.
    """
    n = len(family.regions)
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    if obs.shape != (n, n) or kernel.shape != (n, n):
        raise ValueError("observed and overlap_kernel must align with family regions")

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    zeros: list[np.ndarray] = []
    means: list[np.ndarray] = []
    selected: list[tuple[str, ...]] = []
    counts: list[int] = []

    for region_i, _region in enumerate(family.regions):
        fit_mask, held_mask = leave_one_region_out_masks(n, region_i)
        result = fit_overlap_controlled_ndim(
            family,
            obs,
            kernel,
            fit_mask,
            held_mask,
            correlation_threshold=correlation_threshold,
        )
        held_prediction = np.asarray(result.ndim.held_out_prediction, dtype=float)
        held_target = np.asarray(result.ndim.held_out_observed, dtype=float)
        train_mean = float(np.mean(np.asarray(result.residual_target, dtype=float)[fit_mask]))

        predictions.append(held_prediction)
        targets.append(held_target)
        zeros.append(np.zeros_like(held_target))
        means.append(np.full_like(held_target, train_mean))
        selected.append(tuple(result.ndim.selected_fibres))
        counts.append(int(result.ndim.held_out_pair_count))

    return LOROPredictionBundle(
        prediction=np.concatenate(predictions),
        observed=np.concatenate(targets),
        zero_baseline=np.concatenate(zeros),
        fold_mean_baseline=np.concatenate(means),
        fold_regions=tuple(family.regions),
        selected_fibres_by_fold=tuple(selected),
        held_out_pair_counts=tuple(counts),
    )


def evaluate_overlap_controlled_loro_scorecard(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    candidate_name: str,
    baseline_families: Mapping[str, StructuralFibreFamily] | None = None,
    correlation_threshold: float = 0.98,
    latent_dimension: int | None = None,
    description_length: float | None = None,
    description_length_unit: str | None = None,
) -> ConsumerRelativeScorecard:
    """Score one admissible candidate against fold-identical declared baselines."""
    candidate = collect_overlap_controlled_loro_predictions(
        family,
        observed,
        overlap_kernel,
        correlation_threshold=correlation_threshold,
    )
    baseline_predictions: dict[str, np.ndarray] = {
        "zero": candidate.zero_baseline,
        "fold_train_mean": candidate.fold_mean_baseline,
    }
    for name, baseline_family in (baseline_families or {}).items():
        baseline = collect_overlap_controlled_loro_predictions(
            baseline_family,
            observed,
            overlap_kernel,
            correlation_threshold=correlation_threshold,
        )
        if baseline.fold_regions != candidate.fold_regions:
            raise ValueError(f"baseline {name!r} uses a different LORO region carrier")
        if not np.array_equal(baseline.observed, candidate.observed):
            raise ValueError(f"baseline {name!r} does not share the same controlled target")
        baseline_predictions[str(name)] = baseline.prediction

    return build_consumer_relative_scorecard(
        candidate.prediction,
        candidate.observed,
        baseline_predictions=baseline_predictions,
        candidate_name=candidate_name,
        target_semantics=(
            "published-stimulus-residualized functional correlation with foldwise "
            "atlas-overlap control; metrics are terminal consumer coordinates"
        ),
        latent_dimension=latent_dimension,
        description_length=description_length,
        description_length_unit=description_length_unit,
    )


def prediction_metrics_to_dict(metrics: PredictionMetrics) -> dict[str, float | int]:
    return {
        "pair_count": metrics.pair_count,
        "mae": metrics.mae,
        "r2": metrics.r2,
        "pearson_r": metrics.pearson_r,
        "spearman_rho": metrics.spearman_rho,
    }


def scorecard_to_dict(scorecard: ConsumerRelativeScorecard) -> dict[str, object]:
    return {
        "candidate_name": scorecard.candidate_name,
        "target_semantics": scorecard.target_semantics,
        "candidate_metrics": prediction_metrics_to_dict(scorecard.candidate_metrics),
        "baselines": [
            {
                "baseline_name": comparison.baseline_name,
                "baseline_metrics": prediction_metrics_to_dict(comparison.baseline_metrics),
                "mae_absolute_improvement": comparison.mae_absolute_improvement,
                "mae_fractional_improvement": comparison.mae_fractional_improvement,
                "r2_delta": comparison.r2_delta,
                "pearson_r_delta": comparison.pearson_r_delta,
                "spearman_rho_delta": comparison.spearman_rho_delta,
            }
            for comparison in scorecard.baselines
        ],
        "complexity": {
            "latent_dimension": scorecard.latent_dimension,
            "description_length": scorecard.description_length,
            "description_length_unit": scorecard.description_length_unit,
        },
        "firewalls": {
            "consumer_sufficiency_certified": scorecard.consumer_sufficiency_certified,
            "physical_representation_identity_certified": (
                scorecard.physical_representation_identity_certified
            ),
        },
    }
