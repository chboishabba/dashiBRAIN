"""Attributed eligibility-trace update primitive for clean-room experiments.

This module implements only the factorization used in the learning-rule
specialization discussed by Dhiman & Panwar (Scientific Reports, 2026,
DOI: 10.1038/s41598-026-52140-3): a neuron-level learning signal combining
information and energy terms, multiplied by local synaptic eligibility traces.

It is a clean-room specialization for future experiments. It is not evidence
that the viral fruit-fly Python/FizzBuzz demo used this rule, and it does not
construct eligibility traces or objective gradients from DASHI kernel states.
Those adapters remain separate implementation debt.
"""

from __future__ import annotations

import numpy as np


DHIMAN_PANWAR_DOI = "10.1038/s41598-026-52140-3"


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
