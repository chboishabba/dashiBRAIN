"""Freeze structural latent encoders for cross-recording evaluation.

The MaleCNS structural carrier is shared across Gauthey recordings.  Therefore a
consumer-relative latent encoder can be fit once from structural training-pair
geometry alone and reused unchanged on every independently registered functional
recording.  This module makes that freeze explicit so replication cannot silently
reselect fibres, scales, or PCA components from the replicate outcome.

Functional outcomes are used only downstream: foldwise overlap nuisance fitting
and linear consumer coefficients are still trained within the evaluated recording.
This tests transfer of the *representation*, not literal transfer of regression
coefficients or a mechanistic dynamical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.consumer_relative_scorecard import (
    build_consumer_relative_scorecard,
    collect_overlap_controlled_loro_predictions,
    score_prediction_vector,
)
from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    leave_one_region_out_masks,
)
from dashi.analysis.overlap_controlled_structure_function import (
    fit_overlap_nuisance,
    residualize_overlap_geometry,
)
from dashi.analysis.structural_latent_ladder import (
    FoldStructuralLatentGeometry,
    LatentDimensionScore,
    StructuralLatentLadderResult,
    fit_structural_latent_geometry,
)


@dataclass(frozen=True)
class FrozenStructuralLOROEncoder:
    regions: tuple[str, ...]
    folds: tuple[FoldStructuralLatentGeometry, ...]
    train_latent_full_by_fold: tuple[np.ndarray, ...]
    held_latent_full_by_fold: tuple[np.ndarray, ...]
    common_max_dimension: int
    correlation_threshold: float
    functional_outcomes_used_to_fit_encoder: bool = False


def fit_frozen_structural_loro_encoder(
    family: StructuralFibreFamily,
    *,
    correlation_threshold: float = 0.98,
) -> FrozenStructuralLOROEncoder:
    n = len(family.regions)
    if n < 4:
        raise ValueError("frozen latent encoder requires at least four regions")

    folds: list[FoldStructuralLatentGeometry] = []
    train_latents: list[np.ndarray] = []
    held_latents: list[np.ndarray] = []
    for region_i, region in enumerate(family.regions):
        fit_mask, held_mask = leave_one_region_out_masks(n, region_i)
        geometry, train_full, held_full = fit_structural_latent_geometry(
            family,
            fit_mask,
            held_mask,
            held_out_region=region,
            correlation_threshold=correlation_threshold,
        )
        folds.append(geometry)
        train_latents.append(np.asarray(train_full, dtype=float))
        held_latents.append(np.asarray(held_full, dtype=float))

    common_max_dimension = min(fold.source_dimension for fold in folds)
    return FrozenStructuralLOROEncoder(
        regions=tuple(family.regions),
        folds=tuple(folds),
        train_latent_full_by_fold=tuple(train_latents),
        held_latent_full_by_fold=tuple(held_latents),
        common_max_dimension=int(common_max_dimension),
        correlation_threshold=float(correlation_threshold),
        functional_outcomes_used_to_fit_encoder=False,
    )


def _fit_latent_consumer(
    train_latent: np.ndarray,
    held_latent: np.ndarray,
    train_target: np.ndarray,
) -> np.ndarray:
    train_design = np.column_stack(
        [np.ones(train_latent.shape[0], dtype=float), train_latent]
    )
    beta, *_ = np.linalg.lstsq(train_design, train_target, rcond=None)
    held_design = np.column_stack(
        [np.ones(held_latent.shape[0], dtype=float), held_latent]
    )
    return np.asarray(held_design @ beta, dtype=float)


def _variance_fraction(singular_values: np.ndarray, dimension: int) -> float:
    power = np.square(np.asarray(singular_values, dtype=float))
    total = float(np.sum(power))
    if total <= 1e-15:
        return float("nan")
    return float(np.sum(power[:dimension]) / total)


def evaluate_frozen_overlap_controlled_latent_ladder(
    encoder: FrozenStructuralLOROEncoder,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    full_reference_family: StructuralFibreFamily | None = None,
) -> StructuralLatentLadderResult:
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(encoder.regions)
    if obs.shape != (n, n) or kernel.shape != (n, n):
        raise ValueError("observed and overlap kernel must align with frozen region carrier")
    if len(encoder.folds) != n:
        raise ValueError("frozen encoder fold count does not match region carrier")

    predictions_by_dimension: list[list[np.ndarray]] = [
        [] for _ in range(encoder.common_max_dimension)
    ]
    targets: list[np.ndarray] = []
    zeros: list[np.ndarray] = []
    means: list[np.ndarray] = []
    fractions_by_dimension: list[list[float]] = [
        [] for _ in range(encoder.common_max_dimension)
    ]

    for region_i, geometry in enumerate(encoder.folds):
        if geometry.held_out_region != encoder.regions[region_i]:
            raise ValueError("frozen encoder fold order disagrees with region carrier")
        fit_mask, held_mask = leave_one_region_out_masks(n, region_i)
        nuisance = fit_overlap_nuisance(obs, kernel, fit_mask)
        residual = residualize_overlap_geometry(obs, kernel, nuisance)
        train_target = np.asarray(residual[fit_mask], dtype=float)
        held_target = np.asarray(residual[held_mask], dtype=float)
        targets.append(held_target)
        zeros.append(np.zeros_like(held_target))
        means.append(np.full_like(held_target, float(np.mean(train_target))))

        train_full = encoder.train_latent_full_by_fold[region_i]
        held_full = encoder.held_latent_full_by_fold[region_i]
        for dimension in range(1, encoder.common_max_dimension + 1):
            predictions_by_dimension[dimension - 1].append(
                _fit_latent_consumer(
                    train_full[:, :dimension],
                    held_full[:, :dimension],
                    train_target,
                )
            )
            fractions_by_dimension[dimension - 1].append(
                _variance_fraction(geometry.singular_values, dimension)
            )

    target = np.concatenate(targets)
    zero = np.concatenate(zeros)
    mean = np.concatenate(means)

    if full_reference_family is not None:
        if tuple(full_reference_family.regions) != encoder.regions:
            raise ValueError("full reference family does not match frozen region carrier")
        reference_bundle = collect_overlap_controlled_loro_predictions(
            full_reference_family,
            obs,
            kernel,
            correlation_threshold=encoder.correlation_threshold,
        )
        if not np.array_equal(reference_bundle.observed, target):
            raise RuntimeError("full reference uses a different controlled held-out target")
        reference_prediction = reference_bundle.prediction
    else:
        # Full structural span of the frozen basis is the natural reference when
        # every fold shares the declared common maximum dimension.
        reference_prediction = np.concatenate(
            predictions_by_dimension[encoder.common_max_dimension - 1]
        )

    full_reference = score_prediction_vector(reference_prediction, target)

    scores: list[LatentDimensionScore] = []
    for dimension in range(1, encoder.common_max_dimension + 1):
        prediction = np.concatenate(predictions_by_dimension[dimension - 1])
        scorecard = build_consumer_relative_scorecard(
            prediction,
            target,
            baseline_predictions={
                "zero": zero,
                "fold_train_mean": mean,
                "full_selected_feature_consumer": reference_prediction,
            },
            candidate_name=f"frozen training-structural PCA latent d={dimension}",
            target_semantics=(
                "foldwise overlap-controlled functional target evaluated with a "
                "structural latent encoder frozen independently of functional outcome"
            ),
            latent_dimension=dimension,
        )
        fractions = fractions_by_dimension[dimension - 1]
        scores.append(
            LatentDimensionScore(
                dimension=dimension,
                metrics=scorecard.candidate_metrics,
                baselines=scorecard.baselines,
                mean_structural_variance_fraction=float(np.mean(fractions)),
                minimum_structural_variance_fraction=float(np.min(fractions)),
                consumer_sufficiency_certified=False,
            )
        )

    return StructuralLatentLadderResult(
        fold_geometries=encoder.folds,
        dimensions=tuple(scores),
        common_max_dimension=encoder.common_max_dimension,
        full_feature_reference=full_reference,
        target_semantics=(
            "same overlap-controlled LORO consumer with frozen structural latent encoder"
        ),
        independent_trial_replication_paid=False,
        physical_stalk_dimension_identified=False,
    )
