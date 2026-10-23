"""Finite toy receipts only: no NSD, Brain-IT checkpoints or fly BOLD data."""
import pytest

from dashi.analysis.consumer_retention import (
    audit_commutation,
    audit_consumer_retention,
)


def coarse_regional(state):
    sensory, association, planning = state
    return sensory + association, planning


def laplacian_variation(state):
    sensory, association, planning = state
    return abs(sensory - association) + abs(association - planning)


def test_neural_collision_matches_dashi_agda_witness():
    states = [(1, 2, 3), (2, 1, 3)]
    report = audit_consumer_retention(states, coarse_regional, laplacian_variation)
    assert report.fibre_count == 1
    assert report.ambiguous_fibre_count == 1
    assert not report.factorises_on_sample
    assert report.collision is not None
    assert (report.collision.left_target, report.collision.right_target) == (2, 3)


def test_observed_consumer_factors_on_same_quotient():
    states = [(1, 2, 3), (2, 1, 3)]
    report = audit_consumer_retention(states, coarse_regional, coarse_regional)
    assert report.factorises_on_sample
    assert report.collision is None


def test_finite_consistency_is_not_global_proof():
    # One sample hides the collision revealed by adding the second sample.
    one = audit_consumer_retention([(1, 2, 3)], coarse_regional, laplacian_variation)
    two = audit_consumer_retention([(1, 2, 3), (2, 1, 3)],
                                    coarse_regional, laplacian_variation)
    assert one.factorises_on_sample
    assert not two.factorises_on_sample


def test_commutation_good_and_bad_coarse_dynamics():
    states = [(1, 2, 3), (2, 1, 3)]
    fine = lambda state: (state[0] + 1, state[1], state[2])
    good = audit_commutation(states, fine, coarse_regional,
                             lambda b: (b[0] + 1, b[1]))
    bad = audit_commutation(states, fine, coarse_regional, lambda b: b)
    assert good.commutes_on_sample and good.mismatches == 0
    assert not bad.commutes_on_sample and bad.mismatches == 2


def test_empty_ensemble_rejected():
    with pytest.raises(ValueError):
        audit_consumer_retention([], coarse_regional, laplacian_variation)
    with pytest.raises(ValueError):
        audit_commutation([], lambda x: x, coarse_regional, lambda x: x)
