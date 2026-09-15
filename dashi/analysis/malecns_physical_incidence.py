"""MaleCNS 26-region incidence quotient for the structure/function benchmark.

The source-bound physical connectome is segment/synapse level.  This module does
not redefine that raw graph.  It extracts nonzero directed support from the
membership-aggregated 26-region structural carrier and treats the result as a
region-level incidence quotient used to regenerate the historical NDim chart.

Derived chart coordinates (two-hop, common-input/output, signed reverse, etc.)
are regenerated from that regional quotient rather than copied from the chart.
Exact regeneration therefore checks whether the chart is a derived presentation
of the regional aggregation; it does not prove that the 26-region carrier is the
raw MaleCNS synapse hypergraph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dashi.analysis.ndim_structure_function import (
    StructuralFibreFamily,
    build_ndim_structural_fibres,
)
from dashi.analysis.structure_function_real import RegionStructuralFeatures


@dataclass(frozen=True)
class PhysicalRegionIncidence:
    source: str
    target: str
    direct_weight: float
    signed_weight: float | None


@dataclass(frozen=True)
class MaleCNSPhysicalRegionFabric:
    regions: tuple[str, ...]
    incidences: tuple[PhysicalRegionIncidence, ...]
    source: str = "MaleCNS membership-aggregated nonzero regional coupling"
    carrier_level: str = "region_aggregation"
    raw_segment_incidence: bool = False
    upstream_source_key: str = "full_connection_graph"

    @property
    def incidence_count(self) -> int:
        return len(self.incidences)

    @property
    def possible_pair_count(self) -> int:
        return len(self.regions) ** 2

    @property
    def density(self) -> float:
        if self.possible_pair_count == 0:
            return float("nan")
        return float(self.incidence_count / self.possible_pair_count)


def physical_region_fabric_from_structural(
    structural: RegionStructuralFeatures,
    *,
    atol: float = 0.0,
) -> MaleCNSPhysicalRegionFabric:
    """Extract nonzero directed support from the region-aggregated direct carrier.

    ``atol`` is intentionally explicit. The default keeps every numerically
    nonzero aggregated regional relation. A signed value is metadata on that
    already-present regional relation; it does not create one when the unsigned
    direct carrier is zero.
    """
    if atol < 0:
        raise ValueError("atol must be >= 0")
    regions = tuple(structural.regions)
    direct = np.asarray(structural.direct, dtype=float)
    n = len(regions)
    if direct.shape != (n, n):
        raise ValueError("direct must align with regions")
    signed = None
    if structural.signed_direct is not None:
        signed = np.asarray(structural.signed_direct, dtype=float)
        if signed.shape != direct.shape:
            raise ValueError("signed_direct must align with direct")

    edges: list[PhysicalRegionIncidence] = []
    for i, source in enumerate(regions):
        for j, target in enumerate(regions):
            weight = float(direct[i, j])
            if not np.isfinite(weight):
                raise ValueError("direct contains non-finite values")
            if abs(weight) <= atol:
                continue
            signed_weight = None if signed is None else float(signed[i, j])
            if signed_weight is not None and not np.isfinite(signed_weight):
                raise ValueError("signed_direct contains non-finite values")
            edges.append(
                PhysicalRegionIncidence(source, target, weight, signed_weight)
            )
    return MaleCNSPhysicalRegionFabric(regions, tuple(edges))


def structural_from_physical_region_fabric(
    fabric: MaleCNSPhysicalRegionFabric,
) -> RegionStructuralFeatures:
    """Regenerate canonical structural matrices from the regional quotient.

    Direct and signed-direct matrices are reconstructed from regional support.
    Two-hop is recomputed using the canonical production rule from
    ``aggregate_connectome_by_membership``: row-normalise direct by absolute
    row mass, then square the resulting transition matrix.
    """
    regions = tuple(fabric.regions)
    n = len(regions)
    index = {r: i for i, r in enumerate(regions)}
    if len(index) != n:
        raise ValueError("regions must be unique")

    direct = np.zeros((n, n), dtype=float)
    has_signed = any(edge.signed_weight is not None for edge in fabric.incidences)
    signed = np.zeros((n, n), dtype=float) if has_signed else None
    seen: set[tuple[str, str]] = set()
    for edge in fabric.incidences:
        if edge.source not in index or edge.target not in index:
            raise KeyError("regional incidence references unknown region")
        key = (edge.source, edge.target)
        if key in seen:
            raise ValueError(f"duplicate regional incidence: {key}")
        seen.add(key)
        i, j = index[edge.source], index[edge.target]
        direct[i, j] = float(edge.direct_weight)
        if signed is not None:
            signed[i, j] = 0.0 if edge.signed_weight is None else float(edge.signed_weight)

    denom = np.sum(np.abs(direct), axis=1)
    p = np.zeros_like(direct)
    nz = denom > 0
    p[nz] = direct[nz] / denom[nz, None]
    two_hop = p @ p
    return RegionStructuralFeatures(regions, direct, two_hop, signed)


def ndim_chart_from_physical_region_fabric(
    fabric: MaleCNSPhysicalRegionFabric,
) -> StructuralFibreFamily:
    """Generate the historical NDim chart from the regional incidence quotient."""
    return build_ndim_structural_fibres(structural_from_physical_region_fabric(fabric))


def physical_incidence_chart_round_trip_exact(
    structural: RegionStructuralFeatures,
) -> bool:
    """Check regional-support extraction -> chart generation against canonical chart."""
    expected = build_ndim_structural_fibres(structural)
    regional = physical_region_fabric_from_structural(structural)
    regenerated = ndim_chart_from_physical_region_fabric(regional)
    if tuple(expected.fibres) != tuple(regenerated.fibres):
        return False
    return all(
        np.array_equal(
            np.asarray(expected.fibres[name]),
            np.asarray(regenerated.fibres[name]),
        )
        for name in expected.fibres
    )
