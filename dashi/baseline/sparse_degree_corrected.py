from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from dashi.baseline.residuals import residual_sign_balance


def _strengths(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute out/in strengths and total weight from sparse adjacency."""
    out_strength = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64)
    in_strength = np.asarray(A.sum(axis=0)).reshape(-1).astype(np.float64)
    total_weight = float(out_strength.sum())
    return out_strength, in_strength, total_weight


def compute_sparse_dc_residual(
    A: sp.csr_matrix,
    *,
    deadzone: float = 1e-9,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """
    Degree-corrected residual computed only on observed edges (sparse).

    Residual for edge (u,v):
        R(u,v) = W(u,v) - (k_out(u) * k_in(v) / sum_w)

    No dense N x N expectation is materialised.
    """
    out_strength, in_strength, total_weight = _strengths(A)
    if total_weight <= 0:
        residual = sp.csr_matrix(A.shape, dtype=np.float32)
        return residual, {"-1": 0, "0": int(A.nnz), "+1": 0}

    coo = A.tocoo()
    expected = (out_strength[coo.row] * in_strength[coo.col]) / total_weight
    data = coo.data.astype(np.float32) - expected.astype(np.float32)
    residual = sp.csr_matrix((data, (coo.row, coo.col)), shape=A.shape)
    balance = residual_sign_balance(residual, deadzone=deadzone)
    return residual, balance
