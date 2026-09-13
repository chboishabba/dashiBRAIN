import numpy as np
import scipy.sparse as sp

from dashi.analysis.rate_eligibility_dynamics import RateDynamicsParams
from dashi.analysis.sparse_rate_training import run_sparse_rate_training_epoch


def fixture_weights() -> sp.csr_matrix:
    # RateRNN convention: W[post, pre].
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 0.2, 0.0],
                [0.1, 0.0, 0.3],
                [0.0, 0.4, 0.0],
            ],
            dtype=float,
        )
    )


def fixture_stimulus() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_training_epoch_preserves_sparse_structural_scaffold():
    W = fixture_weights()
    result = run_sparse_rate_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001),
        objective=2.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=1.0,
        learning_rate=0.1,
    )

    assert np.array_equal(result.weights.indptr, W.indptr)
    assert np.array_equal(result.weights.indices, W.indices)
    assert result.weights.nnz == W.nnz


def test_training_epoch_updates_baseline_by_archived_part_e_rule():
    W = fixture_weights()
    result = run_sparse_rate_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        objective=2.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=1.0,
        learning_rate=0.1,
    )
    assert result.running_baseline == 1.1


def test_zero_learning_rate_is_executable_no_learning_control():
    W = fixture_weights()
    result = run_sparse_rate_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        objective=2.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=1.0,
        learning_rate=0.0,
    )
    assert np.array_equal(result.weights.indptr, W.indptr)
    assert np.array_equal(result.weights.indices, W.indices)
    assert np.allclose(result.weights.data, W.data)
    assert np.allclose(result.update, 0.0)
    assert result.receipt["learning_update_applied"] is False


def test_training_epoch_receipt_is_finite_run_and_names_exact_rule():
    W = fixture_weights()
    result = run_sparse_rate_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        objective=2.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=1.0,
        learning_rate=0.1,
        state_bound=10.0,
    )
    assert result.receipt["learning_rule"] == "archived_part_e_supplemental_specialization"
    assert result.receipt["dynamics_rule"] == "dhiman_panwar_relu_rate_rnn"
    assert result.receipt["receipt_scope"] == "finite_run_only"
    assert result.receipt["global_stability_claimed"] is False
    assert result.receipt["global_contraction_claimed"] is False
    assert result.receipt["paper_level_eprop_constructor_claimed"] is False
    assert result.receipt["viral_demo_training_rule_claimed"] is False


def test_training_epoch_is_deterministic_on_fixed_inputs():
    W = fixture_weights()
    kwargs = dict(
        stimulus_history=fixture_stimulus(),
        params=RateDynamicsParams(),
        objective=2.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=1.0,
        learning_rate=0.1,
    )
    left = run_sparse_rate_training_epoch(W, **kwargs)
    right = run_sparse_rate_training_epoch(W, **kwargs)

    assert np.allclose(left.state_history, right.state_history)
    assert np.allclose(left.rate_history, right.rate_history)
    assert np.allclose(left.eligibility_trace, right.eligibility_trace)
    assert np.allclose(left.weights.data, right.weights.data)
