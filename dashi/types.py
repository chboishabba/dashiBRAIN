from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

# ternary carrier dtype
T = np.int8


def sgn_deadzone(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Ternary sign with symmetric deadzone."""
    out = np.zeros_like(x, dtype=T)
    out[x > eps] = 1
    out[x < -eps] = -1
    return out


@dataclass
class GraphCarrier:
    """Sparse adjacency carrier for kernel operations."""
    adjacency: sp.csr_matrix  # shape (N, N), weights >=0 (or residual weights)
    channels: int = 1


@dataclass
class KernelParams:
    hops: int = 1  # neighborhood depth
    channel_coupling: np.ndarray | None = None  # shape (C, C)
    deadzone: float = 0.0
