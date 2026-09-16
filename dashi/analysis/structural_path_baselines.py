"""Literature-anchored directed path baselines for structure/function scoring.

Turner, Mann & Clandinin (Curr Biol 2021) convert directed structural edge
strength to edge distance by inverse weight and use directed shortest paths as
an indirect-connectivity predictor.  This module adapts that *distance
convention* to the current MaleCNS region carrier while retaining the repo's
forward/reverse chart discipline for a symmetric functional consumer.

This is a comparator, not a reproduction of Turner's experiment: the atlas,
animals, functional target, structural carrier, controls, and evaluation split
are different.  The returned family therefore has an explicit local name rather
than claiming literature-equivalent performance.
"""

from __future__ import annotations

import numpy as np

from dashi.analysis.ndim_structure_function import StructuralFibreFamily


def directed_inverse_weight_shortest_paths(direct: np.ndarray) -> np.ndarray:
    """All-pairs directed shortest distance with edge cost 1 / positive weight.

    Zero structural weight means no directed edge. Negative weights are rejected
    because the current MaleCNS ``direct`` carrier is unsigned coupling and an
    inverse-distance interpretation for signed weights would be ill-defined.
    Floyd-Warshall is used because the region carrier is tiny (26 nodes) and it
    avoids an additional graph dependency.
    """
    weight = np.asarray(direct, dtype=float)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("direct structural carrier must be a square matrix")
    if not np.all(np.isfinite(weight)):
        raise ValueError("direct structural carrier contains non-finite weights")
    if np.any(weight < 0):
        raise ValueError("inverse-weight path distance requires non-negative weights")

    n = weight.shape[0]
    distance = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(distance, 0.0)
    positive = weight > 0
    distance[positive] = 1.0 / weight[positive]
    np.fill_diagonal(distance, 0.0)

    for k in range(n):
        distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
    return distance


def turner_style_path_family(
    regions: tuple[str, ...],
    direct: np.ndarray,
) -> StructuralFibreFamily:
    """Bidirectional direct + inverse-weight shortest-path comparator family.

    The current 26-region MaleCNS aggregate is complete/nonzero off diagonal, so
    every directed pair has a finite path. Generic disconnected inputs are
    rejected rather than silently replacing infinity with an arbitrary sentinel.
    """
    matrix = np.asarray(direct, dtype=float)
    if matrix.shape != (len(regions), len(regions)):
        raise ValueError("direct matrix must align with declared regions")
    shortest = directed_inverse_weight_shortest_paths(matrix)
    off_diagonal = ~np.eye(len(regions), dtype=bool)
    if not np.all(np.isfinite(shortest[off_diagonal])):
        raise ValueError(
            "Turner-style baseline requires a finite directed shortest path for every off-diagonal pair"
        )

    return StructuralFibreFamily(
        tuple(regions),
        {
            "direct_forward": matrix,
            "direct_reverse": matrix.T,
            "shortest_path_distance_forward": shortest,
            "shortest_path_distance_reverse": shortest.T,
        },
    )
