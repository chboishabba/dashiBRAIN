"""Sparse continuous-rate learning adapters for MaleCNS-scale experiments.

MaleCNS loader adjacency uses the source/target convention A[pre, post].  The
released Dhiman & Panwar RateRNN uses W[post, pre].  The conversion is therefore
an explicit transpose, not an implicit reinterpretation.

All Hebbian/eligibility work here is edge-local: no dense N x N matrix is
materialised for the MaleCNS carrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from dashi.analysis.eligibility_trace_learning import (
    archived_part_e_eligibility_step,
    archived_part_e_weight_update,
)
from dashi.analysis.rate_eligibility_dynamics import RateDynamicsParams, rate_relu


@dataclass(frozen=True)
class SparseArchivedTraceUpdate:
    weights: sp.csr_matrix
    eligibility_trace: np.ndarray
    running_baseline: float
    update: np.ndarray


def malecns_adjacency_to_rate_weights(
    source_target_adjacency: sp.spmatrix,
) -> sp.csr_matrix:
    """Convert A[pre,post] into RateRNN W[post,pre]."""
    A = source_target_adjacency.tocsr(copy=True)
    if A.shape[0] != A.shape[1]:
        raise ValueError("MaleCNS adjacency must be square")
    W = A.transpose().tocsr()
    W.sum_duplicates()
    W.sort_indices()
    return W


def sparse_rate_step(
    rate_weights: sp.spmatrix,
    state: np.ndarray,
    stimulus: np.ndarray,
    params: RateDynamicsParams,
) -> np.ndarray:
    """Sparse execution of the source-exact Euler/ReLU RateRNN recurrence."""
    W = rate_weights.tocsr()
    if W.shape[0] != W.shape[1]:
        raise ValueError("rate_weights must be square")
    n = W.shape[0]
    x = np.asarray(state, dtype=float)
    I = np.asarray(stimulus, dtype=float)
    if x.shape != (n,):
        raise ValueError("state must have shape (neurons,)")
    if I.shape != (n,):
        raise ValueError("stimulus must have shape (neurons,)")

    recurrent_input = np.asarray(W @ rate_relu(x)).reshape(-1)
    derivative = (-x + recurrent_input + I) / params.tau_seconds
    return x + params.dt_seconds * derivative


def edge_hebbian_drive(
    rate_weights: sp.csr_matrix,
    rate_history: np.ndarray,
) -> np.ndarray:
    """Mean post/pre activity product for each existing W[post,pre] edge."""
    W = rate_weights.tocsr()
    rates = np.asarray(rate_history, dtype=float)
    n = W.shape[0]
    if W.shape[1] != n:
        raise ValueError("rate_weights must be square")
    if rates.ndim != 2 or rates.shape[1] != n:
        raise ValueError("rate_history must have shape (time, neurons)")
    if rates.shape[0] == 0:
        raise ValueError("rate_history must contain at least one time step")

    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(W.indptr))
    cols = W.indices
    accumulator = np.zeros(W.data.shape, dtype=float)
    for t in range(rates.shape[0]):
        accumulator += rates[t, rows] * rates[t, cols]
    return accumulator / float(rates.shape[0])


def update_sparse_archived_trace(
    rate_weights: sp.csr_matrix,
    previous_trace: np.ndarray,
    hebbian_drive: np.ndarray,
    *,
    objective: float,
    running_baseline: float,
    learning_rate: float,
    alpha_trace: float = 0.9,
    trace_clip: float = 5.0,
    update_clip: float = 0.1,
    baseline_decay: float = 0.9,
) -> SparseArchivedTraceUpdate:
    """Apply the archived Part-E trace/update only to existing sparse edges."""
    W = rate_weights.tocsr(copy=True)
    W.sum_duplicates()
    W.sort_indices()
    previous = np.asarray(previous_trace, dtype=float)
    hebbian = np.asarray(hebbian_drive, dtype=float)
    if previous.shape != W.data.shape or hebbian.shape != W.data.shape:
        raise ValueError("trace and Hebbian drive shape must match sparse edge data")

    trace = archived_part_e_eligibility_step(
        previous,
        hebbian,
        alpha_trace=alpha_trace,
        trace_clip=trace_clip,
    )
    new_data, new_baseline, update = archived_part_e_weight_update(
        W.data,
        trace,
        objective=objective,
        running_baseline=running_baseline,
        learning_rate=learning_rate,
        update_clip=update_clip,
        baseline_decay=baseline_decay,
    )
    W.data = new_data
    return SparseArchivedTraceUpdate(
        weights=W,
        eligibility_trace=trace,
        running_baseline=new_baseline,
        update=update,
    )
