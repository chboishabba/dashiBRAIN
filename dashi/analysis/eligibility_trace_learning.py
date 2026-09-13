"""Attributed eligibility-trace primitives for clean-room experiments.

Two source objects are kept separate here:

1. The paper-level Dhiman & Panwar factorization combines a neuron-level
   information/energy learning signal with local synaptic eligibility traces.
2. The archived publication-supplement rerun script labels an executable arm
   ``EProp`` and uses a clipped exponentially accumulated Hebbian trace with a
   reward/objective baseline.

The archived executable rule is not silently identified with the paper-level
factorization, and neither is evidence for the hidden learning rule of the
viral fruit-fly Python/FizzBuzz demonstration.
"""

from __future__ import annotations

import numpy as np


DHIMAN_PANWAR_DOI = "10.1038/s41598-026-52140-3"
PART_E_SOURCE_REVISION = (
    "NSSIL/Energy-efficient-information-processing-and-eligibility"
    "@19bcd7d83a044f90ad691bff5bfe1df636b7150a"
)


def compose_information_energy_signal(
    information_gradient: np.ndarray,
    energy_gradient: np.ndarray,
    *,
    energy_lambda: float,
) -> np.ndarray:
    """Compose L_i(t) = dI/dx_i - lambda * dE/dx_i."""
    if energy_lambda < 0:
        raise ValueError("energy_lambda must be non-negative")

    information = np.asarray(information_gradient, dtype=float)
    energy = np.asarray(energy_gradient, dtype=float)
    if information.shape != energy.shape:
        raise ValueError(
            "information_gradient and energy_gradient must have identical shape"
        )
    if information.ndim != 2:
        raise ValueError("gradient arrays must have shape (time, post_neuron)")
    return information - energy_lambda * energy


def factorized_eligibility_delta(
    eligibility_trace: np.ndarray,
    learning_signal: np.ndarray,
    *,
    learning_rate: float,
    structural_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute delta_w[i,j] = eta * sum_t L[t,i] * e[t,i,j]."""
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")

    eligibility = np.asarray(eligibility_trace, dtype=float)
    signal = np.asarray(learning_signal, dtype=float)
    if eligibility.ndim != 3:
        raise ValueError(
            "eligibility_trace must have shape (time, post_neuron, pre_neuron)"
        )
    if signal.ndim != 2:
        raise ValueError("learning_signal must have shape (time, post_neuron)")
    if signal.shape != eligibility.shape[:2]:
        raise ValueError(
            "learning_signal shape must match eligibility_trace time/post axes"
        )

    delta = learning_rate * np.einsum("ti,tij->ij", signal, eligibility)
    if structural_mask is not None:
        mask = np.asarray(structural_mask, dtype=bool)
        if mask.shape != eligibility.shape[1:]:
            raise ValueError(
                "structural_mask shape must match eligibility post/pre axes"
            )
        delta = np.where(mask, delta, 0.0)
    return delta


def archived_part_e_eligibility_step(
    previous_trace: np.ndarray,
    hebbian_drive: np.ndarray,
    *,
    alpha_trace: float = 0.9,
    trace_clip: float = 5.0,
) -> np.ndarray:
    """Reproduce the archived Part-E ``EProp`` trace recurrence.

    The supplementary rerun script computes

        e <- clip(alpha_trace * e + hebbian, -trace_clip, trace_clip)

    where ``hebbian`` is a per-edge mean post/pre activity product.
    """
    if not 0.0 <= alpha_trace <= 1.0:
        raise ValueError("alpha_trace must lie in [0, 1]")
    if trace_clip <= 0:
        raise ValueError("trace_clip must be positive")

    previous = np.asarray(previous_trace, dtype=float)
    hebbian = np.asarray(hebbian_drive, dtype=float)
    if previous.shape != hebbian.shape:
        raise ValueError("previous_trace and hebbian_drive must have identical shape")
    return np.clip(
        alpha_trace * previous + hebbian,
        -trace_clip,
        trace_clip,
    )


def archived_part_e_weight_update(
    weights: np.ndarray,
    eligibility_trace: np.ndarray,
    *,
    objective: float,
    running_baseline: float,
    learning_rate: float,
    update_clip: float = 0.1,
    baseline_decay: float = 0.9,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Apply the archived Part-E objective-baseline eligibility update.

    This reproduces the released supplementary recurrence:

        update = clip(lr * (J - R_avg) * e, -clip, clip)
        W_new  = max(W + update, 0)
        R_avg  = decay * R_avg + (1-decay) * J

    It does not add the script's separate upper weight clip or construct the
    Hebbian drive; those remain explicit surrounding experiment coordinates.
    """
    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    if update_clip <= 0:
        raise ValueError("update_clip must be positive")
    if not 0.0 <= baseline_decay <= 1.0:
        raise ValueError("baseline_decay must lie in [0, 1]")

    W = np.asarray(weights, dtype=float)
    trace = np.asarray(eligibility_trace, dtype=float)
    if W.shape != trace.shape:
        raise ValueError("weights and eligibility_trace must have identical shape")

    update = np.clip(
        learning_rate * (float(objective) - float(running_baseline)) * trace,
        -update_clip,
        update_clip,
    )
    new_weights = np.maximum(W + update, 0.0)
    new_baseline = (
        baseline_decay * float(running_baseline)
        + (1.0 - baseline_decay) * float(objective)
    )
    return new_weights, new_baseline, update
