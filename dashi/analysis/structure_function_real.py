"""Real-data structure/function benchmark kernels for MaleCNS.

This module replaces hash-derived mock features with sparse-connectome features,
region aggregation, measured functional correlations, and explicit train/held-out
separation. Fitting/scaling is performed only on training region pairs; held-out
pairs are reserved exclusively for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class RegionStructuralFeatures:
    regions: tuple[str, ...]
    direct: np.ndarray
    two_hop: np.ndarray
    signed_direct: np.ndarray | None = None


@dataclass(frozen=True)
class FunctionalAssociation:
    unit_ids: tuple[str, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class StructureFunctionResult:
    direct_residuals: np.ndarray
    path_residuals: np.ndarray
    dashi_residuals: np.ndarray
    observed: np.ndarray
    direct_prediction: np.ndarray
    path_prediction: np.ndarray
    dashi_prediction: np.ndarray
    fit_pair_count: int
    held_out_pair_count: int

    @property
    def mean_direct(self) -> float:
        return float(np.mean(self.direct_residuals))

    @property
    def mean_path(self) -> float:
        return float(np.mean(self.path_residuals))

    @property
    def mean_dashi(self) -> float:
        return float(np.mean(self.dashi_residuals))


def row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    m = matrix.tocsr().astype(float)
    row_sum = np.asarray(np.abs(m).sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=float)
    nz = row_sum > 0
    inv[nz] = 1.0 / row_sum[nz]
    return sp.diags(inv) @ m


def aggregate_connectome_by_membership(
    adjacency: sp.spmatrix,
    membership: sp.spmatrix,
    regions: Sequence[str],
    *,
    signed_adjacency: sp.spmatrix | None = None,
) -> RegionStructuralFeatures:
    """Aggregate neuron adjacency through a sparse region-membership operator.

    ``membership`` is region x neuron.  It may be one-hot or fractional.  The
    latter is required for central-brain neurons whose synapses occupy multiple
    neuropils; forcing those neurons to one label would discard real anatomy.
    """
    m = membership.tocsr().astype(float)
    if m.shape[1] != adjacency.shape[0]:
        raise ValueError("membership columns must align with adjacency rows")
    if m.shape[0] != len(regions):
        raise ValueError("regions must align with membership rows")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")

    region_sizes = np.asarray(m.sum(axis=1)).ravel()
    if np.any(region_sizes <= 0):
        raise ValueError("every retained region must have positive membership mass")
    denom = np.outer(region_sizes, region_sizes)
    denom[denom == 0] = 1.0

    direct_sparse = m @ adjacency.tocsr() @ m.T
    direct = np.asarray(direct_sparse.toarray(), dtype=float) / denom

    norm = row_normalize(sp.csr_matrix(direct))
    two_hop = np.asarray((norm @ norm).toarray(), dtype=float)

    signed_direct = None
    if signed_adjacency is not None:
        s = m @ signed_adjacency.tocsr() @ m.T
        signed_direct = np.asarray(s.toarray(), dtype=float) / denom

    return RegionStructuralFeatures(tuple(str(r) for r in regions), direct, two_hop, signed_direct)


def aggregate_connectome_by_region(
    adjacency: sp.spmatrix,
    node_regions: Sequence[str],
    *,
    signed_adjacency: sp.spmatrix | None = None,
    min_nodes_per_region: int = 1,
) -> RegionStructuralFeatures:
    if adjacency.shape[0] != len(node_regions):
        raise ValueError("node_regions must align with adjacency rows")

    counts: dict[str, int] = {}
    for r in node_regions:
        if r:
            counts[r] = counts.get(r, 0) + 1
    regions = tuple(sorted(r for r, n in counts.items() if n >= min_nodes_per_region))
    if not regions:
        raise ValueError("no regions survive min_nodes_per_region")

    r_index = {r: i for i, r in enumerate(regions)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for node_i, region in enumerate(node_regions):
        if region in r_index:
            rows.append(r_index[region])
            cols.append(node_i)
            vals.append(1.0)
    membership = sp.csr_matrix((vals, (rows, cols)), shape=(len(regions), adjacency.shape[0]))
    return aggregate_connectome_by_membership(
        adjacency,
        membership,
        regions,
        signed_adjacency=signed_adjacency,
    )


def functional_correlation(traces: np.ndarray, unit_ids: Sequence[str]) -> FunctionalAssociation:
    x = np.asarray(traces, dtype=float)
    if x.ndim != 2:
        raise ValueError("traces must be 2-D [time, unit] or [unit, time]")
    if x.shape[1] != len(unit_ids) and x.shape[0] == len(unit_ids):
        x = x.T
    if x.shape[1] != len(unit_ids):
        raise ValueError("unit_ids must align with trace columns")
    if x.shape[0] < 3:
        raise ValueError("at least three time samples are required")
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return FunctionalAssociation(tuple(unit_ids), corr)


def _scale_to_observed(feature: np.ndarray, observed: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    f = feature[fit_mask]
    y = observed[fit_mask]
    denom = float(np.dot(f, f))
    beta = float(np.dot(f, y) / denom) if denom > 0 else 0.0
    return beta * feature


def pairwise_train_holdout_masks(n: int, *, modulus: int = 3, held_out_residue: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic disjoint upper-triangle train/held-out pair masks."""
    if n < 2:
        raise ValueError("at least two units are required")
    if modulus < 2:
        raise ValueError("modulus must be >= 2")
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    ii, jj = np.indices((n, n))
    held = upper & (((ii + jj) % modulus) == held_out_residue)
    train = upper & ~held
    if not np.any(train) or not np.any(held):
        raise ValueError("split must select at least one train and one held-out pair")
    return train, held


def evaluate_region_structure_function(
    structural: RegionStructuralFeatures,
    functional: FunctionalAssociation,
    *,
    fit_mask: np.ndarray | None = None,
    held_out_mask: np.ndarray | None = None,
    dashi_direct_weight: float = 0.5,
    dashi_path_weight: float = 0.4,
    dashi_signed_weight: float = 0.1,
) -> StructureFunctionResult:
    if structural.regions != functional.unit_ids:
        raise ValueError("functional units must be ordered exactly like structural regions")

    observed = np.asarray(functional.matrix, dtype=float)
    n = observed.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    if fit_mask is None and held_out_mask is None:
        fit, held = pairwise_train_holdout_masks(n)
    elif fit_mask is None:
        held = upper & np.asarray(held_out_mask, dtype=bool)
        fit = upper & ~held
    elif held_out_mask is None:
        fit = upper & np.asarray(fit_mask, dtype=bool)
        held = upper & ~fit
    else:
        fit = upper & np.asarray(fit_mask, dtype=bool)
        held = upper & np.asarray(held_out_mask, dtype=bool)

    if np.any(fit & held):
        raise ValueError("fit and held-out masks must be disjoint")
    if not np.any(fit):
        raise ValueError("fit mask selects no region pairs")
    if not np.any(held):
        raise ValueError("held-out mask selects no region pairs")

    direct = _scale_to_observed(structural.direct, observed, fit)
    path_raw = structural.direct + structural.two_hop
    path = _scale_to_observed(path_raw, observed, fit)

    signed = structural.signed_direct if structural.signed_direct is not None else np.zeros_like(structural.direct)
    dashi_raw = (
        dashi_direct_weight * structural.direct
        + dashi_path_weight * structural.two_hop
        + dashi_signed_weight * signed
    )
    dashi = _scale_to_observed(dashi_raw, observed, fit)

    return StructureFunctionResult(
        direct_residuals=np.abs(direct[held] - observed[held]),
        path_residuals=np.abs(path[held] - observed[held]),
        dashi_residuals=np.abs(dashi[held] - observed[held]),
        observed=observed[held],
        direct_prediction=direct[held],
        path_prediction=path[held],
        dashi_prediction=dashi[held],
        fit_pair_count=int(np.sum(fit)),
        held_out_pair_count=int(np.sum(held)),
    )
