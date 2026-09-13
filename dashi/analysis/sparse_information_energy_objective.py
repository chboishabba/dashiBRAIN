"""Sparse information/energy objective for continuous-rate clean-room runs.

The energy decomposition follows the released Dhiman & Panwar Part-D code:

* metabolic cost: sum of rates;
* synaptic cost: pre-neuron activity weighted by total outgoing absolute weight;
* optional active wiring cost: |edge weight| * edge distance * presynaptic activity.

The information term is supplied by the caller because symbolic-task evidence
must come from its own governed held-out evaluator rather than reusing the
optic-flow decoder by fiat.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def sparse_energy_components(
    rate_history: np.ndarray,
    rate_weights: sp.spmatrix,
    *,
    distance_coordinates: np.ndarray | None = None,
    wiring_penalty_alpha: float = 0.01,
) -> dict[str, float]:
    rates = np.asarray(rate_history, dtype=float)
    W = rate_weights.tocsr()
    if W.shape[0] != W.shape[1]:
        raise ValueError("rate_weights must be square")
    n = W.shape[0]
    if rates.ndim != 2 or rates.shape[1] != n:
        raise ValueError("rate_history must have shape (time, neurons)")
    if wiring_penalty_alpha < 0:
        raise ValueError("wiring_penalty_alpha must be non-negative")

    E_met = float(np.sum(rates))
    out_strength_by_pre = np.asarray(np.abs(W).sum(axis=0)).reshape(-1)
    E_syn = float(np.sum(rates @ out_strength_by_pre))

    E_wire = 0.0
    if distance_coordinates is not None:
        coords = np.asarray(distance_coordinates, dtype=float)
        if coords.ndim != 2 or coords.shape[0] != n:
            raise ValueError("distance_coordinates must have shape (neurons, dimensions)")
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(W.indptr))
        cols = W.indices
        distances = np.linalg.norm(coords[rows] - coords[cols], axis=1)
        total_pre_activity = np.sum(rates, axis=0)
        E_wire = float(
            np.sum(np.abs(W.data) * distances * total_pre_activity[cols])
            * wiring_penalty_alpha
        )

    return {
        "E_met": E_met,
        "E_syn": E_syn,
        "E_wire": E_wire,
        "E_total": E_met + E_syn + E_wire,
    }


def information_energy_objective(
    *,
    information_lower_bound: float,
    total_energy: float,
    energy_lambda: float,
) -> float:
    """Compute J = I_lb - lambda * E_total."""
    if energy_lambda < 0:
        raise ValueError("energy_lambda must be non-negative")
    return float(information_lower_bound) - float(energy_lambda) * float(total_energy)
