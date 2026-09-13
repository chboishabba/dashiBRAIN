"""Source-exact continuous rate dynamics for clean-room learning experiments.

The recurrence follows the RateRNN released with Dhiman & Panwar (2026):

    tau * dx/dt = -x + W ReLU(x) + I

integrated by explicit Euler.  This module is a continuous-dynamics
specialization beside DASHI's ternary kernel; it does not replace the ternary
kernel or imply that the viral Python/FizzBuzz demo used these dynamics.

Run receipts report only finite-run observations.  They do not establish a
global stability, contraction, or convergence theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DHIMAN_PANWAR_DOI = "10.1038/s41598-026-52140-3"
RATE_RNN_SOURCE_REVISION = "NSSIL/Energy-efficient-information-processing-and-eligibility@19bcd7d83a044f90ad691bff5bfe1df636b7150a"


@dataclass(frozen=True)
class RateDynamicsParams:
    tau_seconds: float = 0.02
    dt_seconds: float = 0.001

    def __post_init__(self) -> None:
        if self.tau_seconds <= 0:
            raise ValueError("tau_seconds must be positive")
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")


@dataclass(frozen=True)
class RateDynamicsRun:
    state_history: np.ndarray
    rate_history: np.ndarray
    receipt: dict[str, object]


def rate_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def rate_relu_local_slope(x: np.ndarray) -> np.ndarray:
    """Return the declared local ReLU slope; choose zero at the kink."""
    values = np.asarray(x, dtype=float)
    return (values > 0.0).astype(float)


def _validated_weights(weights: np.ndarray) -> np.ndarray:
    W = np.asarray(weights, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights must be a square matrix")
    return W


def rate_step(
    weights: np.ndarray,
    state: np.ndarray,
    stimulus: np.ndarray,
    params: RateDynamicsParams,
) -> np.ndarray:
    """Apply one explicit-Euler step of the published ReLU rate recurrence."""
    W = _validated_weights(weights)
    x = np.asarray(state, dtype=float)
    I = np.asarray(stimulus, dtype=float)
    n = W.shape[0]
    if x.shape != (n,):
        raise ValueError("state must have shape (neurons,)")
    if I.shape != (n,):
        raise ValueError("stimulus must have shape (neurons,)")

    recurrent_input = W @ rate_relu(x)
    derivative = (-x + recurrent_input + I) / params.tau_seconds
    return x + params.dt_seconds * derivative


def run_rate_dynamics(
    weights: np.ndarray,
    stimulus_history: np.ndarray,
    params: RateDynamicsParams,
    *,
    initial_state: np.ndarray | None = None,
    state_bound: float | None = None,
) -> RateDynamicsRun:
    """Run the continuous recurrence and emit finite-run-only stability receipts."""
    W = _validated_weights(weights)
    stimulus = np.asarray(stimulus_history, dtype=float)
    n = W.shape[0]
    if stimulus.ndim != 2 or stimulus.shape[1] != n:
        raise ValueError("stimulus must have shape (time, neurons)")
    if state_bound is not None and state_bound < 0:
        raise ValueError("state_bound must be non-negative")

    if initial_state is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(initial_state, dtype=float).copy()
        if x.shape != (n,):
            raise ValueError("initial_state must have shape (neurons,)")

    states = np.zeros((stimulus.shape[0], n), dtype=float)
    rates = np.zeros_like(states)
    for t in range(stimulus.shape[0]):
        x = rate_step(W, x, stimulus[t], params)
        states[t] = x
        rates[t] = rate_relu(x)

    max_abs_state = float(np.max(np.abs(states))) if states.size else 0.0
    receipt: dict[str, object] = {
        "receipt_scope": "finite_run_only",
        "steps": int(stimulus.shape[0]),
        "tau_seconds": float(params.tau_seconds),
        "dt_seconds": float(params.dt_seconds),
        "max_abs_state": max_abs_state,
        "state_bound": None if state_bound is None else float(state_bound),
        "observed_state_within_bound": (
            None if state_bound is None else bool(max_abs_state <= state_bound)
        ),
        "global_stability_claimed": False,
        "global_contraction_claimed": False,
        "global_convergence_claimed": False,
    }
    return RateDynamicsRun(states, rates, receipt)
