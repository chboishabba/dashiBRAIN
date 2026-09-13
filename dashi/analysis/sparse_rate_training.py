"""One sparse continuous-rate learning epoch for MaleCNS-scale experiments.

This composes already-separated source objects:

* Dhiman & Panwar's released continuous ReLU RateRNN recurrence;
* the archived Part-E supplemental eligibility/baseline update;
* DASHI's explicit sparse MaleCNS orientation adapter.

The composition is a clean-room specialization.  It is not the paper-level
information/energy E-prop constructor and is not evidence about the viral
Python/FizzBuzz training rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from dashi.analysis.rate_eligibility_dynamics import RateDynamicsParams, rate_relu
from dashi.analysis.sparse_rate_learning import (
    edge_hebbian_drive,
    sparse_rate_step,
    update_sparse_archived_trace,
)


@dataclass(frozen=True)
class SparseRateTrainingEpoch:
    weights: sp.csr_matrix
    state_history: np.ndarray
    rate_history: np.ndarray
    eligibility_trace: np.ndarray
    running_baseline: float
    update: np.ndarray
    receipt: dict[str, object]


def run_sparse_rate_training_epoch(
    rate_weights: sp.csr_matrix,
    stimulus_history: np.ndarray,
    params: RateDynamicsParams,
    *,
    objective: float,
    previous_trace: np.ndarray,
    running_baseline: float,
    learning_rate: float,
    initial_state: np.ndarray | None = None,
    state_bound: float | None = None,
    alpha_trace: float = 0.9,
    trace_clip: float = 5.0,
    update_clip: float = 0.1,
    baseline_decay: float = 0.9,
) -> SparseRateTrainingEpoch:
    """Run one deterministic sparse rate epoch and apply one archived update."""
    W = rate_weights.tocsr(copy=True)
    W.sum_duplicates()
    W.sort_indices()
    if W.shape[0] != W.shape[1]:
        raise ValueError("rate_weights must be square")
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if state_bound is not None and state_bound < 0:
        raise ValueError("state_bound must be non-negative")

    stimulus = np.asarray(stimulus_history, dtype=float)
    n = W.shape[0]
    if stimulus.ndim != 2 or stimulus.shape[1] != n:
        raise ValueError("stimulus_history must have shape (time, neurons)")
    if stimulus.shape[0] == 0:
        raise ValueError("stimulus_history must contain at least one time step")

    if initial_state is None:
        state = np.zeros(n, dtype=float)
    else:
        state = np.asarray(initial_state, dtype=float).copy()
        if state.shape != (n,):
            raise ValueError("initial_state must have shape (neurons,)")

    states = np.zeros((stimulus.shape[0], n), dtype=float)
    rates = np.zeros_like(states)
    for t in range(stimulus.shape[0]):
        state = sparse_rate_step(W, state, stimulus[t], params)
        states[t] = state
        rates[t] = rate_relu(state)

    hebbian = edge_hebbian_drive(W, rates)
    learned = update_sparse_archived_trace(
        W,
        previous_trace,
        hebbian,
        objective=objective,
        running_baseline=running_baseline,
        learning_rate=learning_rate,
        alpha_trace=alpha_trace,
        trace_clip=trace_clip,
        update_clip=update_clip,
        baseline_decay=baseline_decay,
    )

    max_abs_state = float(np.max(np.abs(states))) if states.size else 0.0
    receipt: dict[str, object] = {
        "receipt_scope": "finite_run_only",
        "dynamics_rule": "dhiman_panwar_relu_rate_rnn",
        "learning_rule": "archived_part_e_supplemental_specialization",
        "steps": int(stimulus.shape[0]),
        "tau_seconds": float(params.tau_seconds),
        "dt_seconds": float(params.dt_seconds),
        "objective": float(objective),
        "learning_rate": float(learning_rate),
        "learning_update_applied": bool(
            learning_rate > 0.0 and np.any(np.abs(learned.update) > 0.0)
        ),
        "max_abs_state": max_abs_state,
        "state_bound": None if state_bound is None else float(state_bound),
        "observed_state_within_bound": (
            None if state_bound is None else bool(max_abs_state <= state_bound)
        ),
        "global_stability_claimed": False,
        "global_contraction_claimed": False,
        "global_convergence_claimed": False,
        "paper_level_eprop_constructor_claimed": False,
        "viral_demo_training_rule_claimed": False,
        "sparse_edge_local_update": True,
        "structural_scaffold_preserved": True,
    }

    return SparseRateTrainingEpoch(
        weights=learned.weights,
        state_history=states,
        rate_history=rates,
        eligibility_trace=learned.eligibility_trace,
        running_baseline=learned.running_baseline,
        update=learned.update,
        receipt=receipt,
    )
