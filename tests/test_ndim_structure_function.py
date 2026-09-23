import numpy as np

from dashi.analysis.ndim_stability_nulls import (
    strength_preserving_structural_null_loro,
    summarize_loro_fibre_stability,
)
from dashi.analysis.ndim_structure_function import (
    build_ndim_structural_fibres,
    evaluate_leave_one_region_out,
    fit_ndim_fibre_consumer,
    leave_one_region_out_masks,
    region_label_permutation_null_leave_one_region_out,
    region_label_permutation_null_pair_holdout,
    select_compatible_fibres,
)
from dashi.analysis.structure_function_real import (
    RegionStructuralFeatures,
    pairwise_train_holdout_masks,
)


def _structural():
    direct = np.array([
        [0.0, 1.0, 2.0, 0.0],
        [3.0, 0.0, 1.0, 1.0],
        [0.5, 2.0, 0.0, 4.0],
        [1.0, 0.0, 2.0, 0.0],
    ])
    two_hop = np.array([
        [0.0, .2, .4, .3],
        [.1, 0.0, .6, .2],
        [.3, .5, 0.0, .7],
        [.4, .2, .1, 0.0],
    ])
    signed = np.array([
        [0.0, .5, -.2, .1],
        [-.1, 0.0, .3, .2],
        [.4, -.2, 0.0, .6],
        [.2, .1, -.3, 0.0],
    ])
    return RegionStructuralFeatures(("A", "B", "C", "D"), direct, two_hop, signed)


def _observed():
    return np.array([
        [0.0, .2, .3, .4],
        [.2, 0.0, .6, .5],
        [.3, .6, 0.0, .8],
        [.4, .5, .8, 0.0],
    ])


def test_ndim_family_keeps_directed_and_shared_neighbour_fibres_distinct():
    family = build_ndim_structural_fibres(_structural())
    assert tuple(family.fibres) == (
        "direct_forward",
        "direct_reverse",
        "two_hop_forward",
        "two_hop_reverse",
        "common_input",
        "common_output",
        "signed_forward",
        "signed_reverse",
    )
    assert np.array_equal(family.fibres["direct_reverse"], family.fibres["direct_forward"].T)
    assert np.allclose(family.fibres["common_input"], family.fibres["common_input"].T)
    assert np.allclose(family.fibres["common_output"], family.fibres["common_output"].T)


def test_compatibility_selection_uses_feature_geometry_not_observed_outcomes():
    family = build_ndim_structural_fibres(_structural())
    fit, _ = pairwise_train_holdout_masks(4)
    a = select_compatible_fibres(family, fit, correlation_threshold=.999)
    b = select_compatible_fibres(family, fit, correlation_threshold=.999)
    assert a == b
    assert len(a.selected) >= 1


def test_exact_redundant_fibre_is_rejected_by_conflict_selection():
    family = build_ndim_structural_fibres(_structural())
    fibres = dict(family.fibres)
    fibres["duplicate_direct"] = fibres["direct_forward"].copy()
    family = type(family)(family.regions, fibres)
    fit, _ = pairwise_train_holdout_masks(4)
    selected = select_compatible_fibres(
        family,
        fit,
        correlation_threshold=.999999,
        priority=tuple(fibres),
    )
    assert "direct_forward" in selected.selected
    assert any(name == "duplicate_direct" and against == "direct_forward" for name, against, _ in selected.rejected_redundant)


def test_held_out_observations_cannot_change_selected_fibres_or_fitted_coefficients():
    family = build_ndim_structural_fibres(_structural())
    fit, held = pairwise_train_holdout_masks(4)
    observed = _observed()
    first = fit_ndim_fibre_consumer(family, observed, fit, held, correlation_threshold=.999)
    changed = observed.copy()
    changed[held] += 100.0
    changed = np.maximum(changed, changed.T)
    second = fit_ndim_fibre_consumer(family, changed, fit, held, correlation_threshold=.999)
    assert first.selected_fibres == second.selected_fibres
    assert np.allclose(first.coefficients, second.coefficients)
    assert np.allclose(first.feature_means, second.feature_means)
    assert np.allclose(first.feature_scales, second.feature_scales)
    assert not np.allclose(first.held_out_observed, second.held_out_observed)


def test_leave_one_region_out_excludes_held_region_from_every_training_pair():
    n = 5
    held_region = 2
    fit, held = leave_one_region_out_masks(n, held_region)
    ii, jj = np.indices((n, n))
    assert not np.any(fit & ((ii == held_region) | (jj == held_region)))
    assert np.all(held[np.triu(((ii == held_region) | (jj == held_region)), 1)])
    assert int(np.sum(fit)) == 6  # C(4,2)
    assert int(np.sum(held)) == 4


def test_leave_one_region_out_runs_one_fold_per_region():
    family = build_ndim_structural_fibres(_structural())
    result = evaluate_leave_one_region_out(family, _observed(), correlation_threshold=1.0)
    assert tuple(f.held_out_region for f in result.folds) == family.regions
    assert all(f.train_pair_count == 3 for f in result.folds)
    assert all(f.held_out_pair_count == 3 for f in result.folds)
    assert np.isfinite(result.weighted_mean_residual)


def test_region_label_permutation_nulls_refit_and_return_valid_p_values():
    family = build_ndim_structural_fibres(_structural())
    observed = _observed()
    fit, held = pairwise_train_holdout_masks(4)
    pair_null = region_label_permutation_null_pair_holdout(
        family,
        observed,
        fit,
        held,
        n_null=8,
        seed=11,
        correlation_threshold=1.0,
    )
    blocked_null = region_label_permutation_null_leave_one_region_out(
        family,
        observed,
        n_null=8,
        seed=12,
        correlation_threshold=1.0,
    )
    assert pair_null.null_residuals.shape == (8,)
    assert blocked_null.null_residuals.shape == (8,)
    assert 0.0 < pair_null.empirical_p_value <= 1.0
    assert 0.0 < blocked_null.empirical_p_value <= 1.0
    assert np.isfinite(pair_null.observed_residual)
    assert np.isfinite(blocked_null.observed_residual)


def test_loro_null_retains_fold_matched_refitted_distributions():
    family = build_ndim_structural_fibres(_structural())
    blocked_null = region_label_permutation_null_leave_one_region_out(
        family,
        _observed(),
        n_null=9,
        seed=21,
        correlation_threshold=1.0,
    )
    assert blocked_null.fold_regions == family.regions
    assert blocked_null.observed_fold_residuals is not None
    assert blocked_null.null_fold_residuals is not None
    assert blocked_null.fold_empirical_p_values is not None
    assert blocked_null.observed_fold_residuals.shape == (4,)
    assert blocked_null.null_fold_residuals.shape == (9, 4)
    assert blocked_null.fold_empirical_p_values.shape == (4,)
    assert np.all((blocked_null.fold_empirical_p_values > 0.0) & (blocked_null.fold_empirical_p_values <= 1.0))
    assert not np.allclose(
        blocked_null.null_fold_residuals,
        blocked_null.null_residuals[:, None],
    )


def test_loro_stability_summary_tracks_fold_selection_and_coefficients():
    family = build_ndim_structural_fibres(_structural())
    blocked = evaluate_leave_one_region_out(family, _observed(), correlation_threshold=1.0)
    stability = summarize_loro_fibre_stability(blocked)
    assert stability.fold_count == 4
    assert len(stability.fibres) >= 1
    assert all(0.0 < f.selected_fold_fraction <= 1.0 for f in stability.fibres)
    assert np.isfinite(stability.intercept_mean)
    assert np.isfinite(stability.intercept_std)


def test_strength_preserving_structural_null_refits_loro_and_preserves_marginals():
    structural = _structural()
    result = strength_preserving_structural_null_loro(
        structural,
        _observed(),
        n_null=4,
        seed=19,
        correlation_threshold=1.0,
    )
    assert result.null_residuals.shape == (4,)
    assert 0.0 < result.empirical_p_value <= 1.0
    assert np.isfinite(result.observed_residual)
    assert np.all(np.isfinite(result.null_residuals))
    assert result.max_row_strength_error < 1e-6
    assert result.max_column_strength_error < 1e-6
