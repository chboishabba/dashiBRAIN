import numpy as np
import pytest

from dashi.analysis.frozen_structural_latent_encoder import (
    evaluate_frozen_overlap_controlled_latent_ladder,
    fit_frozen_structural_loro_encoder,
    load_frozen_structural_loro_encoder,
    save_frozen_structural_loro_encoder,
)
from dashi.analysis.ndim_structure_function import StructuralFibreFamily
from dashi.analysis.structural_latent_ladder import (
    evaluate_overlap_controlled_structural_latent_ladder,
)


def _synthetic(seed=11, n=7):
    rng = np.random.default_rng(seed)
    regions = tuple(f"R{i}" for i in range(n))
    fibres = {}
    for name in ("a", "b", "c", "d"):
        raw = rng.normal(size=(n, n))
        matrix = (raw + raw.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        fibres[name] = matrix
    family = StructuralFibreFamily(regions, fibres)

    ii, jj = np.indices((n, n))
    overlap = np.exp(-np.abs(ii - jj) / 2.3)
    observed = (
        0.1
        + 0.31 * fibres["a"]
        - 0.14 * fibres["b"]
        + 0.06 * fibres["c"]
        + 0.22 * overlap
    )
    observed = (observed + observed.T) / 2.0
    np.fill_diagonal(observed, 1.0)
    return family, observed, overlap


def test_frozen_encoder_fitted_from_structure_only_reproduces_discovery_ladder():
    family, observed, overlap = _synthetic()
    direct = evaluate_overlap_controlled_structural_latent_ladder(
        family, observed, overlap
    )
    encoder = fit_frozen_structural_loro_encoder(family)
    frozen = evaluate_frozen_overlap_controlled_latent_ladder(
        encoder, observed, overlap
    )

    assert encoder.regions == family.regions
    assert encoder.common_max_dimension == direct.common_max_dimension
    assert len(frozen.dimensions) == len(direct.dimensions)
    for a, b in zip(frozen.dimensions, direct.dimensions):
        assert a.dimension == b.dimension
        assert a.metrics.mae == pytest.approx(b.metrics.mae, abs=1e-12)
        assert a.metrics.r2 == pytest.approx(b.metrics.r2, abs=1e-12)
        assert a.metrics.pearson_r == pytest.approx(b.metrics.pearson_r, abs=1e-12)
        assert a.metrics.spearman_rho == pytest.approx(b.metrics.spearman_rho, abs=1e-12)


def test_same_frozen_encoder_can_score_different_functional_target_without_refitting_geometry():
    family, observed, overlap = _synthetic()
    encoder = fit_frozen_structural_loro_encoder(family)
    first_components = tuple(fold.components.copy() for fold in encoder.folds)

    changed = observed.copy()
    changed += 0.03 * np.sin(np.arange(changed.size)).reshape(changed.shape)
    changed = (changed + changed.T) / 2.0
    np.fill_diagonal(changed, 1.0)
    result = evaluate_frozen_overlap_controlled_latent_ladder(
        encoder, changed, overlap
    )

    assert len(result.dimensions) == encoder.common_max_dimension
    for before, fold in zip(first_components, encoder.folds):
        assert np.array_equal(before, fold.components)


def test_frozen_encoder_rejects_region_vocabulary_drift():
    family, observed, overlap = _synthetic()
    encoder = fit_frozen_structural_loro_encoder(family)

    with pytest.raises(ValueError, match="region carrier"):
        evaluate_frozen_overlap_controlled_latent_ladder(
            encoder,
            observed[:-1, :-1],
            overlap[:-1, :-1],
        )


def test_frozen_encoder_does_not_pay_replication_or_sufficiency_by_construction():
    family, observed, overlap = _synthetic()
    encoder = fit_frozen_structural_loro_encoder(family)
    result = evaluate_frozen_overlap_controlled_latent_ladder(
        encoder, observed, overlap
    )

    assert encoder.functional_outcomes_used_to_fit_encoder is False
    assert result.independent_trial_replication_paid is False
    assert all(x.consumer_sufficiency_certified is False for x in result.dimensions)


def test_frozen_encoder_artifact_roundtrip_is_exact_and_reuses_same_scores(tmp_path):
    family, observed, overlap = _synthetic()
    encoder = fit_frozen_structural_loro_encoder(family)
    target = tmp_path / "encoder.npz"

    save_frozen_structural_loro_encoder(encoder, target)
    restored = load_frozen_structural_loro_encoder(target)

    assert restored.regions == encoder.regions
    assert restored.common_max_dimension == encoder.common_max_dimension
    assert restored.correlation_threshold == encoder.correlation_threshold
    assert restored.functional_outcomes_used_to_fit_encoder is False
    assert len(restored.folds) == len(encoder.folds)
    for a, b in zip(restored.folds, encoder.folds):
        assert a.held_out_region == b.held_out_region
        assert a.selected_fibres == b.selected_fibres
        assert np.array_equal(a.feature_means, b.feature_means)
        assert np.array_equal(a.feature_scales, b.feature_scales)
        assert np.array_equal(a.components, b.components)
        assert np.array_equal(a.singular_values, b.singular_values)
    for a, b in zip(restored.train_latent_full_by_fold, encoder.train_latent_full_by_fold):
        assert np.array_equal(a, b)
    for a, b in zip(restored.held_latent_full_by_fold, encoder.held_latent_full_by_fold):
        assert np.array_equal(a, b)

    before = evaluate_frozen_overlap_controlled_latent_ladder(encoder, observed, overlap)
    after = evaluate_frozen_overlap_controlled_latent_ladder(restored, observed, overlap)
    for a, b in zip(before.dimensions, after.dimensions):
        assert a.dimension == b.dimension
        assert a.metrics == b.metrics
