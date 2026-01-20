from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from dashi.types import T, sgn_deadzone


def degree_corrected_expectation(A: sp.csr_matrix) -> np.ndarray:
    """Degree-corrected expectation matrix (dense)."""
    out_deg = np.asarray(A.sum(axis=1)).reshape(-1)
    in_deg = np.asarray(A.sum(axis=0)).reshape(-1)
    total = float(out_deg.sum())
    if total == 0:
        return np.zeros((A.shape[0], A.shape[1]), dtype=np.float32)
    exp = np.outer(out_deg, in_deg) / total
    return exp.astype(np.float32)


def residual_csr(A: sp.csr_matrix, expectation: np.ndarray) -> sp.csr_matrix:
    """
    Residual adjacency on existing edges: res(u,v) = A(u,v) - E(u,v).
    Only nonzero edges are stored; absent edges remain zero.
    """
    coo = A.tocoo()
    exp_vals = expectation[coo.row, coo.col]
    data = coo.data.astype(np.float32) - exp_vals.astype(np.float32)
    return sp.csr_matrix((data, (coo.row, coo.col)), shape=A.shape)


def residual_sign_balance(residual: sp.csr_matrix, deadzone: float = 1e-9) -> dict[str, int]:
    """Count positive/zero/negative residual entries with a deadzone."""
    signs = sgn_deadzone(residual.data, eps=deadzone)
    return {
        "+1": int((signs == 1).sum()),
        "0": int((signs == 0).sum()),
        "-1": int((signs == -1).sum()),
    }


def compute_residuals(
    A: sp.csr_matrix,
    *,
    expectation: np.ndarray | None = None,
    deadzone: float = 1e-9,
) -> tuple[sp.csr_matrix, dict[str, int]]:
    """
    Compute residual adjacency and sign balance diagnostics.

    If `expectation` is None, uses degree-corrected baseline.
    Returns residual CSR matrix and sign counts.
    """
    exp = expectation if expectation is not None else degree_corrected_expectation(A)
    residual = residual_csr(A, exp)
    balance = residual_sign_balance(residual, deadzone=deadzone)
    return residual, balance
