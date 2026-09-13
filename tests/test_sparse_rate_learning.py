import numpy as np
import scipy.sparse as sp

from dashi.analysis.rate_eligibility_dynamics import RateDynamicsParams, rate_step
from dashi.analysis.sparse_rate_learning import (
    edge_hebbian_drive,
    malecns_adjacency_to_rate_weights,
    sparse_rate_step,
    update_sparse_archived_trace,
)


def test_malecns_source_target_adjacency_is_transposed_to_post_pre_weights():
    # MaleCNS loader convention: adjacency[pre, post].
    source_target = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [4.0, 0.0, 0.0],
            ]
        )
    )
    W = malecns_adjacency_to_rate_weights(source_target)
    assert W[1, 0] == 2.0
    assert W[2, 1] == 3.0
    assert W[0, 2] == 4.0


def test_sparse_rate_step_matches_dense_source_exact_recurrence():
    W_dense = np.array([[0.0, 0.5], [0.25, 0.0]])
    W_sparse = sp.csr_matrix(W_dense)
    x = np.array([2.0, -1.0])
    stimulus = np.array([0.1, 0.2])
    params = RateDynamicsParams(tau_seconds=0.02, dt_seconds=0.001)
    assert np.allclose(
        sparse_rate_step(W_sparse, x, stimulus, params),
        rate_step(W_dense, x, stimulus, params),
    )


def test_edge_hebbian_drive_is_computed_only_on_existing_post_pre_edges():
    W = sp.csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    rates = np.array([[1.0, 2.0], [3.0, 4.0]])
    drive = edge_hebbian_drive(W, rates)

    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    cols = W.indices
    expected = np.mean(rates[:, rows] * rates[:, cols], axis=0)
    assert np.allclose(drive, expected)
    assert drive.shape == W.data.shape


def test_sparse_archived_update_preserves_structural_scaffold():
    W = sp.csr_matrix(np.array([[0.0, 0.2], [0.3, 0.0]]))
    previous_trace = np.zeros_like(W.data)
    hebbian = np.array([1.0, 2.0])

    result = update_sparse_archived_trace(
        W,
        previous_trace,
        hebbian,
        objective=2.0,
        running_baseline=1.0,
        learning_rate=0.1,
        alpha_trace=0.9,
    )

    assert np.array_equal(result.weights.indptr, W.indptr)
    assert np.array_equal(result.weights.indices, W.indices)
    assert result.eligibility_trace.shape == W.data.shape
    assert result.update.shape == W.data.shape
    assert result.running_baseline == 1.1
