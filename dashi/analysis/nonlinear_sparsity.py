from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ExplodedState:
    """Lossless ternary support decomposition plus graph-component labels."""

    positive: np.ndarray
    neutral: np.ndarray
    negative: np.ndarray
    positive_component_labels: np.ndarray
    negative_component_labels: np.ndarray
    changed: np.ndarray | None = None


@dataclass(frozen=True)
class ConstraintDiagnostics:
    """Fixed-point/CSP diagnostics for q_epsilon(A @ state + bias)."""

    field: np.ndarray
    projected: np.ndarray
    violated: np.ndarray
    margins: np.ndarray

    @property
    def defect(self) -> int:
        return int(np.count_nonzero(self.violated))


def ternary_project(field: np.ndarray, deadzone: float = 0.0) -> np.ndarray:
    """Project a real field to {-1, 0, +1} with a symmetric deadzone."""
    if deadzone < 0:
        raise ValueError("deadzone must be non-negative")
    values = np.asarray(field, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.int8)
    out[values > deadzone] = 1
    out[values < -deadzone] = -1
    return out


def local_field(
    adjacency: sp.spmatrix,
    state: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the affine pre-projection field A @ state + bias."""
    flat = np.asarray(state).reshape(-1)
    if adjacency.shape != (flat.size, flat.size):
        raise ValueError("adjacency shape must match the number of state atoms")
    field = np.asarray(adjacency @ flat, dtype=np.float64).reshape(-1)
    if bias is not None:
        bias_flat = np.asarray(bias, dtype=np.float64).reshape(-1)
        if bias_flat.size != flat.size:
            raise ValueError("bias length must match the number of state atoms")
        field = field + bias_flat
    return field


def signed_margins(state: np.ndarray, field: np.ndarray, deadzone: float = 0.0) -> np.ndarray:
    """Return threshold margins from Theorem 2 in docs/nonlinear-sparsity.md."""
    if deadzone < 0:
        raise ValueError("deadzone must be non-negative")
    s = np.asarray(state).reshape(-1)
    h = np.asarray(field, dtype=np.float64).reshape(-1)
    if s.size != h.size:
        raise ValueError("state and field must have the same length")
    if not np.isin(s, (-1, 0, 1)).all():
        raise ValueError("state must be ternary")

    margins = np.empty(h.shape, dtype=np.float64)
    pos = s == 1
    zero = s == 0
    neg = s == -1
    margins[pos] = h[pos] - deadzone
    margins[zero] = deadzone - np.abs(h[zero])
    margins[neg] = -h[neg] - deadzone
    return margins


def constraint_diagnostics(
    adjacency: sp.spmatrix,
    state: np.ndarray,
    *,
    deadzone: float = 0.0,
    bias: np.ndarray | None = None,
) -> ConstraintDiagnostics:
    """Evaluate the weighted threshold CSP induced by the DASHI kernel."""
    flat = np.asarray(state).reshape(-1).astype(np.int8, copy=False)
    if not np.isin(flat, (-1, 0, 1)).all():
        raise ValueError("state must be ternary")
    field = local_field(adjacency, flat, bias)
    projected = ternary_project(field, deadzone)
    violated = projected != flat
    margins = signed_margins(flat, field, deadzone)
    return ConstraintDiagnostics(field, projected, violated, margins)


def _component_labels(adjacency: sp.spmatrix, mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    labels = np.full(mask.shape, -1, dtype=np.int64)
    if indices.size == 0:
        return labels
    sub = adjacency.tocsr()[indices][:, indices]
    undirected = (sub + sub.T).tocsr()
    _, sub_labels = connected_components(undirected, directed=False, return_labels=True)
    labels[indices] = sub_labels
    return labels


def explode_state(
    adjacency: sp.spmatrix,
    state: np.ndarray,
    previous_state: np.ndarray | None = None,
) -> ExplodedState:
    """Expose support, sign components, and optional transition defects."""
    flat = np.asarray(state).reshape(-1).astype(np.int8, copy=False)
    if adjacency.shape != (flat.size, flat.size):
        raise ValueError("adjacency shape must match the number of state atoms")
    if not np.isin(flat, (-1, 0, 1)).all():
        raise ValueError("state must be ternary")

    positive = flat == 1
    neutral = flat == 0
    negative = flat == -1
    changed = None
    if previous_state is not None:
        previous = np.asarray(previous_state).reshape(-1)
        if previous.size != flat.size:
            raise ValueError("previous_state length must match state")
        changed = previous != flat

    return ExplodedState(
        positive=positive,
        neutral=neutral,
        negative=negative,
        positive_component_labels=_component_labels(adjacency, positive),
        negative_component_labels=_component_labels(adjacency, negative),
        changed=changed,
    )


def low_margin_mask(margins: np.ndarray, eta: float) -> np.ndarray:
    """Return B_eta = {i : margin_i <= eta} for perturbation-bound tests."""
    if eta < 0:
        raise ValueError("eta must be non-negative")
    return np.asarray(margins, dtype=np.float64).reshape(-1) <= eta
