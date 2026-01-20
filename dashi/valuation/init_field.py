from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from dashi.types import T, sgn_deadzone


def init_field_from_residual(residual: sp.csr_matrix, deadzone: float = 1e-9) -> np.ndarray:
    """
    Build initial ternary field s0 from residual adjacency by row-summing residuals.
    Returns array of shape (N, 1) with dtype T.
    """
    node_res = np.asarray(residual.sum(axis=1)).reshape(-1)
    return sgn_deadzone(node_res, eps=deadzone)[:, None].astype(T)
