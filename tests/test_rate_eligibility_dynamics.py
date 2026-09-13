import numpy as np
import pytest

from dashi.analysis.rate_eligibility_dynamics import (
    RateDynamicsParams,
    rate_relu,
    rate_relu_local_slope,
    rate_step,
    run_rate_dynamics,
)


def test_rate_step_matches_published_euler_relu_recurrence():
    W = np.array([[0.0, 0.5], [0.25, 0.0]])
    x = np.array([2.0, -1.0])
    stimulus = np.array([0.1, 0.2])
    params = RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001)

    got = rate_step(W, x, stimulus, params)
    r = np.maximum(x, 0.0)
    expected = x + (params.dt_seconds / params.tau_seconds) * (
        -x + W @ r + stimulus
    )
    assert np.allclose(got, expected)


def test_relu_and_local_slope_are_explicit_and_zero_at_kink():
    x = np.array([-2.0, 0.0, 3.0])
    assert np.array_equal(rate_relu(x), np.array([0.0, 0.0, 3.0]))
    assert np.array_equal(rate_relu_local_slope(x), np.array([0.0, 0.0, 1.0]))


def test_run_returns_source_aligned_state_and_rate_histories():
    W = np.zeros((2, 2))
    stimulus = np.array([[1.0, 0.0], [0.0, 1.0]])
    params = RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001)

    result = run_rate_dynamics(W, stimulus, params)

    assert result.state_history.shape == (2, 2)
    assert result.rate_history.shape == (2, 2)
    assert np.array_equal(result.rate_history, np.maximum(result.state_history, 0.0))
    assert result.receipt["receipt_scope"] == "finite_run_only"
    assert result.receipt["global_stability_claimed"] is False
    assert result.receipt["global_contraction_claimed"] is False


def test_finite_run_receipt_reports_observed_bounds_not_global_theorem():
    W = np.zeros((1, 1))
    stimulus = np.ones((4, 1))
    params = RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001)

    result = run_rate_dynamics(W, stimulus, params, state_bound=2.0)

    assert result.receipt["observed_state_within_bound"] is True
    assert result.receipt["state_bound"] == 2.0
    assert result.receipt["max_abs_state"] <= 2.0
    assert result.receipt["global_stability_claimed"] is False


def test_invalid_time_constants_fail_closed():
    with pytest.raises(ValueError, match="tau_seconds"):
        RateDynamicsParams(tau_seconds=0.0, dt_seconds=0.001)
    with pytest.raises(ValueError, match="dt_seconds"):
        RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.0)


def test_shape_mismatch_fails_closed():
    params = RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001)
    with pytest.raises(ValueError, match="square"):
        rate_step(np.ones((2, 3)), np.ones(2), np.ones(2), params)
    with pytest.raises(ValueError, match="stimulus"):
        run_rate_dynamics(np.eye(2), np.ones((3, 3)), params)
