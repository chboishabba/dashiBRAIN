import numpy as np

from dashi.analysis.ndim_structure_function import (
    build_ndim_structural_fibres,
    fit_ndim_fibre_consumer,
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
    observed = np.array([
        [0.0, .2, .3, .4],
        [.2, 0.0, .6, .5],
        [.3, .6, 0.0, .8],
        [.4, .5, .8, 0.0],
    ])
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
