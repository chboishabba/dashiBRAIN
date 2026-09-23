from __future__ import annotations

import numpy as np

from dashi.analysis.structure_function_real import FunctionalAssociation, RegionStructuralFeatures
from dashi.analysis.train_fitted_structure_function import fit_train_only_structure_mixture


def test_train_only_mixture_recovers_training_law_without_holdout_peeking():
    regions = ("A", "B", "C", "D")
    direct = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 4.0, 5.0],
        [2.0, 4.0, 0.0, 6.0],
        [3.0, 5.0, 6.0, 0.0],
    ])
    two_hop = np.array([
        [0.0, 2.0, 1.0, 0.5],
        [2.0, 0.0, 3.0, 1.5],
        [1.0, 3.0, 0.0, 2.5],
        [0.5, 1.5, 2.5, 0.0],
    ])
    signed = np.array([
        [0.0, -1.0, 1.0, 0.0],
        [-1.0, 0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0, -0.5],
        [0.0, 1.0, -0.5, 0.0],
    ])
    structural = RegionStructuralFeatures(regions, direct, two_hop, signed)

    true_beta = np.array([0.25, 0.70, -0.10])
    observed = true_beta[0] * direct + true_beta[1] * two_hop + true_beta[2] * signed
    np.fill_diagonal(observed, 0.0)
    functional = FunctionalAssociation(regions, observed.copy())

    upper = np.triu(np.ones((4, 4), dtype=bool), 1)
    held = np.zeros((4, 4), dtype=bool)
    held[0, 3] = True
    held[1, 2] = True
    fit = upper & ~held

    result = fit_train_only_structure_mixture(
        structural,
        functional,
        fit_mask=fit,
        held_out_mask=held,
    )
    assert result.feature_names == ("direct", "two_hop", "signed_direct")
    assert np.allclose(result.coefficients, true_beta)
    assert result.fit_pair_count == 4
    assert result.held_out_pair_count == 2
    assert np.allclose(result.held_out_residuals, 0.0)

    # Mutating held-out observations must not change fitted coefficients. This
    # directly guards the no-held-out-peeking contract.
    changed = observed.copy()
    changed[0, 3] = 999.0
    changed[1, 2] = -999.0
    changed_functional = FunctionalAssociation(regions, changed)
    changed_result = fit_train_only_structure_mixture(
        structural,
        changed_functional,
        fit_mask=fit,
        held_out_mask=held,
    )
    assert np.allclose(changed_result.coefficients, result.coefficients)
    assert not np.allclose(changed_result.held_out_residuals, result.held_out_residuals)


def test_train_only_mixture_omits_signed_feature_when_unavailable():
    regions = ("A", "B", "C")
    direct = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    two_hop = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 1.5], [1.0, 1.5, 0.0]])
    structural = RegionStructuralFeatures(regions, direct, two_hop, None)
    observed = 0.4 * direct + 0.6 * two_hop
    functional = FunctionalAssociation(regions, observed)

    fit = np.zeros((3, 3), dtype=bool)
    fit[0, 1] = True
    fit[0, 2] = True
    held = np.zeros((3, 3), dtype=bool)
    held[1, 2] = True

    result = fit_train_only_structure_mixture(
        structural,
        functional,
        fit_mask=fit,
        held_out_mask=held,
    )
    assert result.feature_names == ("direct", "two_hop")
    assert result.coefficients.shape == (2,)
    assert result.held_out_pair_count == 1
