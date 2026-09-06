"""Null controls for Drosophila structure/function benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class NullResidualSummary:
    observed_mean: float
    null_means: np.ndarray

    @property
    def empirical_p_value(self) -> float:
        if self.null_means.size == 0:
            return 1.0
        return float((1 + np.sum(self.null_means <= self.observed_mean)) / (1 + self.null_means.size))


def permutation_null(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = rng.permutation(matrix.shape[0])
    return matrix[np.ix_(p, p)]


def degree_preserving_edge_swap(
    adjacency: sp.spmatrix,
    *,
    n_swaps: int,
    seed: int = 0,
) -> sp.csr_matrix:
    """Directed binary-topology double-edge swaps retaining in/out degree.

    Edge weights are carried with their source edge. Self loops and duplicate
    edges are rejected. This is deliberately conservative and bounded.
    """
    rng = np.random.default_rng(seed)
    coo = adjacency.tocoo()
    edges = [(int(i), int(j), float(w)) for i, j, w in zip(coo.row, coo.col, coo.data) if i != j]
    edge_pairs = {(i, j) for i, j, _ in edges}
    if len(edges) < 2:
        return adjacency.tocsr().copy()

    for _ in range(n_swaps):
        a, b = rng.choice(len(edges), size=2, replace=False)
        i, j, wi = edges[a]
        k, l, wk = edges[b]
        if len({i, j, k, l}) < 4:
            continue
        proposed = ((i, l), (k, j))
        if i == l or k == j or proposed[0] in edge_pairs or proposed[1] in edge_pairs:
            continue
        edge_pairs.remove((i, j))
        edge_pairs.remove((k, l))
        edge_pairs.add(proposed[0])
        edge_pairs.add(proposed[1])
        edges[a] = (i, l, wi)
        edges[b] = (k, j, wk)

    if not edges:
        return sp.csr_matrix(adjacency.shape, dtype=float)
    rows, cols, vals = zip(*edges)
    return sp.coo_matrix((vals, (rows, cols)), shape=adjacency.shape).tocsr()


def region_label_permutation(labels: Sequence[str], seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    out = list(labels)
    rng.shuffle(out)
    return out


def residual_against_permutation_nulls(
    prediction: np.ndarray,
    observed: np.ndarray,
    *,
    n_null: int = 100,
    seed: int = 0,
) -> NullResidualSummary:
    rng = np.random.default_rng(seed)
    mask = np.triu(np.ones(observed.shape, dtype=bool), k=1)
    obs_mean = float(np.mean(np.abs(prediction[mask] - observed[mask])))
    nulls = []
    for _ in range(n_null):
        perm = permutation_null(observed, rng)
        nulls.append(float(np.mean(np.abs(prediction[mask] - perm[mask]))))
    return NullResidualSummary(obs_mean, np.asarray(nulls, dtype=float))
