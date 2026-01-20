from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from dashi.types import GraphCarrier, KernelParams, T, sgn_deadzone


def k_hop_adj(A: sp.csr_matrix, hops: int) -> sp.csr_matrix:
    """Accumulate adjacency up to k hops (non-negative weights)."""
    if hops <= 1:
        return A
    M = A.copy().tocsr()
    cur = A
    for _ in range(1, hops):
        cur = cur @ A
        cur.data = np.clip(cur.data, 0, None)
        M = M + cur
    M.eliminate_zeros()
    return M


def kernel_step(carrier: GraphCarrier, s: np.ndarray, params: KernelParams) -> np.ndarray:
    """
    Apply one DASHI kernel step.

    s: shape (N, C) in {-1,0,+1}
    returns: shape (N, C) ternary
    """
    N, C = s.shape
    if C != carrier.channels:
        raise ValueError(f"s has {C} channels; carrier expects {carrier.channels}")

    Nh = k_hop_adj(carrier.adjacency, params.hops)

    if C == 1:
        agg = Nh @ s[:, 0].astype(np.float32)
        return sgn_deadzone(agg, params.deadzone)[:, None]

    agg = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        agg[:, c] = Nh @ s[:, c].astype(np.float32)

    if params.channel_coupling is not None:
        agg = agg @ params.channel_coupling.T

    return sgn_deadzone(agg, params.deadzone).astype(T)
