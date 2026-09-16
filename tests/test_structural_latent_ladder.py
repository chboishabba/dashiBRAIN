import numpy as np
import pytest

from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    evaluate_leave_one_region_out,
)
from dashi.analysis.overlap_controlled_structure_function import (
    evaluate_overlap_controlled_leave_one_region_out,
)
from dashi.analysis.structural_latent_ladder import (
    evaluate_overlap_controlled_structural_latent_ladder,
)


def _synthetic(seed=7, n=7):
    rng = np.random.default_rng(seed)
    regions = tuple(f"R{i}" for i in range(n))
    fibres = {}
    for name in ("f1", "f2", "f3"):
        raw = rng.normal(size=(n, n))
        matrix = (raw + raw.T) / 2.0
        np.fill_diagonal(matrix, 0.0)
        fibres[name] = matrix
    family = StructuralFibreFamily(regions, fibres)

    ii, jj = np.indices((n, n))
    overlap = np.exp(-np.abs(ii - jj) / 2.0)
    observed = (
        0.15
        + 0.35 * fibres["f1"]
        - 0.18 * fibres["f2"]
        + 0.08 * fibres["f3"]
        + 0.25 * overlap
    )
    observed = (observed + observed.T) / 2.0
    np.fill_diagonal(observed, 1.0)
    return family, observed, overlap


def test_full_structural_reference_preserves_existing_overlap_controlled_loro_mae():
    family, observed, overlap = _synthetic()
    old = evaluate_overlap_controlled_leave_one_region_out(family, observed, overlap)
    ladder = evaluate_overlap_controlled_structural_latent_ladder(
        family, observed, overlap
    )

    assert ladder.full_feature_reference.mae == pytest.approx(
        old.weighted_mean_residual, abs=1e-12
    )
    assert ladder.common_max_dimension >= 1
    assert tuple(score.dimension for score in ladder.dimensions) == tuple(
        range(1, ladder.common_max_dimension + 1)
    )


def test_full_pca_span_matches_full_linear_consumer_when_fold_dimensions_agree():
    family, observed, overlap = _synthetic()
    ladder = evaluate_overlap_controlled_structural_latent_ladder(
        family, observed, overlap
    )
    source_dimensions = {geometry.source_dimension for geometry in ladder.fold_geometries}
    assert len(source_dimensions) == 1
    full_dimension = next(iter(source_dimensions))
    assert full_dimension == ladder.common_max_dimension

    full_latent = ladder.dimensions[-1]
    assert full_latent.metrics.mae == pytest.approx(
        ladder.full_feature_reference.mae, abs=1e-10
    )
    assert full_latent.metrics.r2 == pytest.approx(
        ladder.full_feature_reference.r2, abs=1e-10
    )


def test_outer_heldout_functional_changes_do_not_change_that_folds_structural_encoder():
    family, observed, overlap = _synthetic()
    baseline = evaluate_overlap_controlled_structural_latent_ladder(
        family, observed, overlap
    )

    changed = observed.copy()
    changed[0, 1:] += 5.0
    changed[1:, 0] += 5.0
    perturbed = evaluate_overlap_controlled_structural_latent_ladder(
        family, changed, overlap
    )

    a = baseline.fold_geometries[0]
    b = perturbed.fold_geometries[0]
    assert a.held_out_region == b.held_out_region == family.regions[0]
    assert a.selected_fibres == b.selected_fibres
    assert np.array_equal(a.feature_means, b.feature_means)
    assert np.array_equal(a.feature_scales, b.feature_scales)
    assert np.allclose(a.components, b.components, atol=0.0, rtol=0.0)
    assert np.allclose(a.singular_values, b.singular_values, atol=0.0, rtol=0.0)


def test_ladder_does_not_promote_dimension_to_consumer_sufficiency():
    family, observed, overlap = _synthetic()
    ladder = evaluate_overlap_controlled_structural_latent_ladder(
        family, observed, overlap
    )
    assert all(score.consumer_sufficiency_certified is False for score in ladder.dimensions)
    assert ladder.independent_trial_replication_paid is False
