"""Training-structure-only low-dimensional latent ladder for MaleCNS fibres.

This module asks a narrower question than the hand-designed fibre-compression
ladder: after the admissible structural fibre family is selected *without using
functional outcomes*, how many orthogonal structural coordinates are needed by
the declared overlap-controlled LORO consumer?

For each held-out region:
1. select structurally compatible fibres from training-pair geometry only;
2. standardize those training structural features;
3. fit an SVD/PCA basis using structure only;
4. project training and untouched held-out structural pairs into Z_d;
5. fit the declared functional consumer on training targets only;
6. evaluate on held-out pairs.

The resulting dimension ladder is a discovery-recording diagnostic.  It does
not certify a universally sufficient latent, does not identify physical stalk
dimension, and does not pay independent-trial replication.  A future frozen
cross-trial encoder is a separate gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.consumer_relative_scorecard import (
    BaselineComparison,
    PredictionMetrics,
    build_consumer_relative_scorecard,
    collect_overlap_controlled_loro_predictions,
    score_prediction_vector,
)
from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    leave_one_region_out_masks,
    select_compatible_fibres,
)
from dashi.analysis.overlap_controlled_structure_function import (
    fit_overlap_nuisance,
    residualize_overlap_geometry,
)


@dataclass(frozen=True)
class FoldStructuralLatentGeometry:
    held_out_region: str
    selected_fibres: tuple[str, ...]
    feature_means: np.ndarray
    feature_scales: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    source_dimension: int
    fit_pair_count: int
    held_out_pair_count: int


@dataclass(frozen=True)
class LatentDimensionScore:
    dimension: int
    metrics: PredictionMetrics
    baselines: tuple[BaselineComparison, ...]
    mean_structural_variance_fraction: float
    minimum_structural_variance_fraction: float
    consumer_sufficiency_certified: bool = False


@dataclass(frozen=True)
class StructuralLatentLadderResult:
    fold_geometries: tuple[FoldStructuralLatentGeometry, ...]
    dimensions: tuple[LatentDimensionScore, ...]
    common_max_dimension: int
    full_feature_reference: PredictionMetrics
    target_semantics: str
    independent_trial_replication_paid: bool = False
    physical_stalk_dimension_identified: bool = False


@dataclass(frozen=True)
class _FoldCache:
    geometry: FoldStructuralLatentGeometry
    train_latent_full: np.ndarray
    held_latent_full: np.ndarray
    train_target: np.ndarray
    held_target: np.ndarray
    train_mean_baseline: np.ndarray
    zero_baseline: np.ndarray


def _feature_matrix(
    family: StructuralFibreFamily,
    names: tuple[str, ...],
    mask: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [np.asarray(family.fibres[name], dtype=float)[mask] for name in names]
    )


def fit_structural_latent_geometry(
    family: StructuralFibreFamily,
    fit_mask: np.ndarray,
    held_out_mask: np.ndarray,
    *,
    held_out_region: str,
    correlation_threshold: float = 0.98,
) -> tuple[FoldStructuralLatentGeometry, np.ndarray, np.ndarray]:
    """Fit one fold's latent basis from structural feature geometry only."""
    fit = np.asarray(fit_mask, dtype=bool)
    held = np.asarray(held_out_mask, dtype=bool)
    if np.any(fit & held):
        raise ValueError("fit and held-out masks must be disjoint")
    selection = select_compatible_fibres(
        family,
        fit,
        correlation_threshold=correlation_threshold,
    )
    names = tuple(selection.selected)
    x_train = _feature_matrix(family, names, fit)
    x_held = _feature_matrix(family, names, held)

    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.asarray(scales, dtype=float)
    scales[scales <= 1e-12] = 1.0
    z_train = (x_train - means) / scales
    z_held = (x_held - means) / scales

    _u, singular_values, components = np.linalg.svd(z_train, full_matrices=False)
    source_dimension = int(components.shape[0])
    if source_dimension < 1:
        raise ValueError("structural latent geometry has zero dimension")

    train_latent_full = z_train @ components.T
    held_latent_full = z_held @ components.T
    geometry = FoldStructuralLatentGeometry(
        held_out_region=str(held_out_region),
        selected_fibres=names,
        feature_means=np.asarray(means, dtype=float),
        feature_scales=scales,
        components=np.asarray(components, dtype=float),
        singular_values=np.asarray(singular_values, dtype=float),
        source_dimension=source_dimension,
        fit_pair_count=int(np.sum(fit)),
        held_out_pair_count=int(np.sum(held)),
    )
    return geometry, train_latent_full, held_latent_full


def _fit_linear_latent_consumer(
    train_latent: np.ndarray,
    held_latent: np.ndarray,
    train_target: np.ndarray,
) -> np.ndarray:
    design_train = np.column_stack(
        [np.ones(train_latent.shape[0], dtype=float), train_latent]
    )
    beta, *_ = np.linalg.lstsq(design_train, train_target, rcond=None)
    design_held = np.column_stack(
        [np.ones(held_latent.shape[0], dtype=float), held_latent]
    )
    return np.asarray(design_held @ beta, dtype=float)


def _variance_fraction(singular_values: np.ndarray, dimension: int) -> float:
    power = np.square(np.asarray(singular_values, dtype=float))
    total = float(np.sum(power))
    if total <= 1e-15:
        return float("nan")
    return float(np.sum(power[:dimension]) / total)


def evaluate_overlap_controlled_structural_latent_ladder(
    family: StructuralFibreFamily,
    observed: np.ndarray,
    overlap_kernel: np.ndarray,
    *,
    correlation_threshold: float = 0.98,
) -> StructuralLatentLadderResult:
    obs = np.asarray(observed, dtype=float)
    kernel = np.asarray(overlap_kernel, dtype=float)
    n = len(family.regions)
    if obs.shape != (n, n) or kernel.shape != (n, n):
        raise ValueError("observed and overlap_kernel must align with family regions")
    if n < 4:
        raise ValueError("latent ladder requires at least four regions")

    caches: list[_FoldCache] = []
    geometries: list[FoldStructuralLatentGeometry] = []
    for region_i, region in enumerate(family.regions):
        fit_mask, held_mask = leave_one_region_out_masks(n, region_i)
        geometry, train_latent_full, held_latent_full = fit_structural_latent_geometry(
            family,
            fit_mask,
            held_mask,
            held_out_region=region,
            correlation_threshold=correlation_threshold,
        )
        nuisance = fit_overlap_nuisance(obs, kernel, fit_mask)
        residual_target = residualize_overlap_geometry(obs, kernel, nuisance)
        train_target = np.asarray(residual_target[fit_mask], dtype=float)
        held_target = np.asarray(residual_target[held_mask], dtype=float)
        train_mean = float(np.mean(train_target))
        caches.append(
            _FoldCache(
                geometry=geometry,
                train_latent_full=train_latent_full,
                held_latent_full=held_latent_full,
                train_target=train_target,
                held_target=held_target,
                train_mean_baseline=np.full_like(held_target, train_mean),
                zero_baseline=np.zeros_like(held_target),
            )
        )
        geometries.append(geometry)

    common_max_dimension = min(g.source_dimension for g in geometries)
    full_bundle = collect_overlap_controlled_loro_predictions(
        family,
        obs,
        kernel,
        correlation_threshold=correlation_threshold,
    )
    full_reference = score_prediction_vector(
        full_bundle.prediction,
        full_bundle.observed,
    )

    dimension_scores: list[LatentDimensionScore] = []
    for dimension in range(1, common_max_dimension + 1):
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        zeros: list[np.ndarray] = []
        means: list[np.ndarray] = []
        fractions: list[float] = []
        for cache in caches:
            predictions.append(
                _fit_linear_latent_consumer(
                    cache.train_latent_full[:, :dimension],
                    cache.held_latent_full[:, :dimension],
                    cache.train_target,
                )
            )
            targets.append(cache.held_target)
            zeros.append(cache.zero_baseline)
            means.append(cache.train_mean_baseline)
            fractions.append(
                _variance_fraction(cache.geometry.singular_values, dimension)
            )

        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        if not np.array_equal(target, full_bundle.observed):
            raise RuntimeError(
                "latent ladder and full-feature reference disagree on controlled held-out target"
            )
        scorecard = build_consumer_relative_scorecard(
            prediction,
            target,
            baseline_predictions={
                "zero": np.concatenate(zeros),
                "fold_train_mean": np.concatenate(means),
                "full_selected_feature_consumer": full_bundle.prediction,
            },
            candidate_name=f"training-structural PCA latent d={dimension}",
            target_semantics=(
                "published-stimulus-residualized functional correlation with foldwise "
                "atlas-overlap control; latent encoder fit from training structure only"
            ),
            latent_dimension=dimension,
        )
        dimension_scores.append(
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
        fold_geometries=tuple(geometries),
        dimensions=tuple(dimension_scores),
        common_max_dimension=int(common_max_dimension),
        full_feature_reference=full_reference,
        target_semantics=(
            "same overlap-controlled LORO consumer as the current MaleCNS benchmark"
        ),
        independent_trial_replication_paid=False,
        physical_stalk_dimension_identified=False,
    )


def structural_latent_ladder_to_dict(
    result: StructuralLatentLadderResult,
) -> dict[str, object]:
    return {
        "target_semantics": result.target_semantics,
        "common_max_dimension": result.common_max_dimension,
        "full_feature_reference": {
            "pair_count": result.full_feature_reference.pair_count,
            "mae": result.full_feature_reference.mae,
            "r2": result.full_feature_reference.r2,
            "pearson_r": result.full_feature_reference.pearson_r,
            "spearman_rho": result.full_feature_reference.spearman_rho,
        },
        "dimensions": [
            {
                "dimension": score.dimension,
                "metrics": {
                    "pair_count": score.metrics.pair_count,
                    "mae": score.metrics.mae,
                    "r2": score.metrics.r2,
                    "pearson_r": score.metrics.pearson_r,
                    "spearman_rho": score.metrics.spearman_rho,
                },
                "mean_structural_variance_fraction": score.mean_structural_variance_fraction,
                "minimum_structural_variance_fraction": score.minimum_structural_variance_fraction,
                "baselines": [
                    {
                        "baseline_name": baseline.baseline_name,
                        "mae": baseline.baseline_metrics.mae,
                        "mae_absolute_improvement": baseline.mae_absolute_improvement,
                        "mae_fractional_improvement": baseline.mae_fractional_improvement,
                        "r2_delta": baseline.r2_delta,
                        "pearson_r_delta": baseline.pearson_r_delta,
                        "spearman_rho_delta": baseline.spearman_rho_delta,
                    }
                    for baseline in score.baselines
                ],
                "consumer_sufficiency_certified": score.consumer_sufficiency_certified,
            }
            for score in result.dimensions
        ],
        "fold_geometries": [
            {
                "held_out_region": geometry.held_out_region,
                "selected_fibres": list(geometry.selected_fibres),
                "source_dimension": geometry.source_dimension,
                "fit_pair_count": geometry.fit_pair_count,
                "held_out_pair_count": geometry.held_out_pair_count,
            }
            for geometry in result.fold_geometries
        ],
        "firewalls": {
            "independent_trial_replication_paid": result.independent_trial_replication_paid,
            "physical_stalk_dimension_identified": result.physical_stalk_dimension_identified,
            "lowest_discovery_mae_dimension_is_universal_minimal_latent": False,
            "pca_geometry_is_biological_mechanism": False,
        },
    }
