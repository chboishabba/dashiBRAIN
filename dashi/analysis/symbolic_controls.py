"""Matched control producers for symbolic MaleCNS experiments.

These controls keep distinct what is being changed:
- topology rewiring changes edges while preserving directed binary degree and
  weighted in/out strength by swapping only equal-weight edges;
- identity shuffling changes neuron-identity assignment while leaving topology
  untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np
import scipy.sparse as sp


IdentityT = TypeVar("IdentityT")


@dataclass(frozen=True)
class RewireResult:
    adjacency: sp.csr_matrix
    swaps_requested: int
    swaps_completed: int
    preserves_binary_in_degree: bool = True
    preserves_binary_out_degree: bool = True
    preserves_weighted_in_strength: bool = True
    preserves_weighted_out_strength: bool = True


def degree_and_strength_preserving_rewire(
    adjacency: sp.csr_matrix,
    *,
    swaps: int,
    seed: int,
    allow_self_loops: bool = False,
    max_attempts: int | None = None,
) -> RewireResult:
    """Directed equal-weight double-edge swap.

    For edges ``a->b`` and ``c->d`` with the same weight ``w``, propose
    ``a->d`` and ``c->b``.  Because each source and each destination retains
    one edge of the same weight, directed binary in/out degree and weighted
    in/out strength are all preserved exactly.

    Duplicate edges and, by default, self-loops are rejected.  The function is
    deterministic for a fixed seed and fails closed if it cannot complete the
    requested number of swaps.
    """

    A = adjacency.tocsr(copy=True)
    if A.shape[0] != A.shape[1]:
        raise ValueError("topology adjacency must be square")
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    if swaps == 0:
        return RewireResult(A, swaps_requested=0, swaps_completed=0)

    coo = A.tocoo()
    edges: list[tuple[int, int, float]] = [
        (int(r), int(c), float(w))
        for r, c, w in zip(coo.row, coo.col, coo.data)
    ]
    edge_set = {(r, c) for r, c, _ in edges}

    buckets: dict[float, list[int]] = {}
    for i, (_, _, weight) in enumerate(edges):
        buckets.setdefault(weight, []).append(i)
    eligible = [indices for indices in buckets.values() if len(indices) >= 2]
    if not eligible:
        raise ValueError("rewire requires at least two equal-weight edges")

    rng = np.random.default_rng(seed)
    attempt_limit = max_attempts if max_attempts is not None else max(100, swaps * 100)
    completed = 0

    for _ in range(attempt_limit):
        if completed >= swaps:
            break
        bucket = eligible[int(rng.integers(0, len(eligible)))]
        i, j = rng.choice(bucket, size=2, replace=False)
        i = int(i)
        j = int(j)
        a, b, w1 = edges[i]
        c, d, w2 = edges[j]

        if w1 != w2 or a == c or b == d:
            continue

        new1 = (a, d)
        new2 = (c, b)
        if new1 == new2:
            continue
        if not allow_self_loops and (a == d or c == b):
            continue

        existing_without_selected = edge_set - {(a, b), (c, d)}
        if new1 in existing_without_selected or new2 in existing_without_selected:
            continue

        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(new1)
        edge_set.add(new2)
        edges[i] = (a, d, w1)
        edges[j] = (c, b, w2)
        completed += 1

    if completed != swaps:
        raise RuntimeError(
            f"could complete only {completed} of {swaps} requested preserving swaps"
        )

    rows = np.fromiter((r for r, _, _ in edges), dtype=np.int64, count=len(edges))
    cols = np.fromiter((c for _, c, _ in edges), dtype=np.int64, count=len(edges))
    data = np.fromiter((w for _, _, w in edges), dtype=A.dtype, count=len(edges))
    rewired = sp.csr_matrix((data, (rows, cols)), shape=A.shape, dtype=A.dtype)
    rewired.eliminate_zeros()
    return RewireResult(
        adjacency=rewired,
        swaps_requested=swaps,
        swaps_completed=completed,
    )


def shuffle_identity_assignment(
    identities: Sequence[IdentityT],
    *,
    seed: int,
) -> list[IdentityT]:
    """Return a deterministic shuffled identity assignment.

    The graph/topology is not touched.  For more than one identity the result
    is guaranteed not to be identical to the original ordering, while the
    identity multiset is preserved.
    """

    original = list(identities)
    if len(original) <= 1:
        return original.copy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(original))
    shuffled = [original[int(i)] for i in order]
    if shuffled == original:
        shuffled = original[1:] + original[:1]
    return shuffled
