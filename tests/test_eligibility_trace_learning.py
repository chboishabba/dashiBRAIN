import numpy as np
import pytest

from dashi.analysis.eligibility_trace_learning import (
    archived_part_e_eligibility_step,
    archived_part_e_weight_update,
    compose_information_energy_signal,
    factorized_eligibility_delta,
)


def test_compose_information_energy_signal_matches_declared_objective_gradient():
    information = np.array([[3.0, 5.0], [7.0, 11.0]])
    energy = np.array([[1.0, 2.0], [3.0, 4.0]])
    signal = compose_information_energy_signal(information, energy, energy_lambda=0.5)
    assert np.allclose(signal, information - 0.5 * energy)


def test_factorized_eligibility_delta_sums_learning_signal_times_local_trace():
    eligibility = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    signal = np.array([[2.0, 3.0], [5.0, 7.0]])
    delta = factorized_eligibility_delta(
        eligibility,
        signal,
        learning_rate=0.1,
    )
    expected = 0.1 * np.einsum("ti,tij->ij", signal, eligibility)
    assert np.allclose(delta, expected)


def test_structural_mask_blocks_updates_outside_scaffold():
    eligibility = np.ones((2, 2, 2), dtype=float)
    signal = np.ones((2, 2), dtype=float)
    mask = np.array([[True, False], [False, True]])
    delta = factorized_eligibility_delta(
        eligibility,
        signal,
        learning_rate=1.0,
        structural_mask=mask,
    )
    assert np.array_equal(delta, np.array([[2.0, 0.0], [0.0, 2.0]]))


def test_factorized_update_rejects_shape_mismatch():
    eligibility = np.ones((2, 3, 3), dtype=float)
    signal = np.ones((2, 2), dtype=float)
    with pytest.raises(ValueError, match="learning_signal"):
        factorized_eligibility_delta(eligibility, signal, learning_rate=1.0)


def test_factorized_update_rejects_negative_learning_rate():
    eligibility = np.ones((1, 1, 1), dtype=float)
    signal = np.ones((1, 1), dtype=float)
    with pytest.raises(ValueError, match="learning_rate"):
        factorized_eligibility_delta(eligibility, signal, learning_rate=-1.0)


def test_signal_composition_rejects_negative_energy_lambda():
    x = np.ones((1, 1), dtype=float)
    with pytest.raises(ValueError, match="energy_lambda"):
        compose_information_energy_signal(x, x, energy_lambda=-0.1)


def test_archived_part_e_trace_step_matches_released_recurrence():
    previous = np.array([1.0, -2.0, 4.9])
    hebbian = np.array([0.5, 0.25, 1.0])
    got = archived_part_e_eligibility_step(
        previous,
        hebbian,
        alpha_trace=0.9,
        trace_clip=5.0,
    )
    expected = np.clip(0.9 * previous + hebbian, -5.0, 5.0)
    assert np.allclose(got, expected)


def test_archived_part_e_weight_update_matches_reward_baseline_and_clips():
    weights = np.array([0.2, 0.05, 0.0])
    trace = np.array([1.0, -2.0, 5.0])
    new_weights, new_baseline, update = archived_part_e_weight_update(
        weights,
        trace,
        objective=3.0,
        running_baseline=1.0,
        learning_rate=0.1,
        update_clip=0.1,
        baseline_decay=0.9,
    )
    expected_update = np.clip(0.1 * (3.0 - 1.0) * trace, -0.1, 0.1)
    assert np.allclose(update, expected_update)
    assert np.allclose(new_weights, np.maximum(weights + expected_update, 0.0))
    assert new_baseline == pytest.approx(0.9 * 1.0 + 0.1 * 3.0)


def test_archived_rule_rejects_invalid_decay_and_shape():
    with pytest.raises(ValueError, match="alpha_trace"):
        archived_part_e_eligibility_step(
            np.ones(2), np.ones(2), alpha_trace=1.1
        )
    with pytest.raises(ValueError, match="shape"):
        archived_part_e_eligibility_step(
            np.ones(2), np.ones(3), alpha_trace=0.9
        )
    with pytest.raises(ValueError, match="baseline_decay"):
        archived_part_e_weight_update(
            np.ones(2),
            np.ones(2),
            objective=1.0,
            running_baseline=0.0,
            learning_rate=0.1,
            baseline_decay=-0.1,
        )
