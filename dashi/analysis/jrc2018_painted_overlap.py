"""Assign transformed selected LBM supervoxels to VFB JRC2018 painted domains.

The VFB atlas is represented as 46 separate binary painted domains rather than
one mutually exclusive label image.  This preserves legitimate overlap between
anatomical domains/subdomains.  Each selected supervoxel is scored against every
domain by fractional voxel overlap; a unique maximum above threshold is assigned,
while exact ties are retained as ambiguous and are not silently promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dashi.io.functional_imaging_loader import FunctionalTraceTable


@dataclass(frozen=True)
class PaintedDomainAssignment:
    selected_row: int
    region: str
    overlap_voxel_count: int
    selected_voxel_count: int
    overlap_fraction: float


@dataclass(frozen=True)
class PaintedDomainAmbiguity:
    selected_row: int
    regions: tuple[str, ...]
    overlap_fraction: float


@dataclass(frozen=True)
class PaintedDomainCompilation:
    region_traces: FunctionalTraceTable
    assignments: tuple[PaintedDomainAssignment, ...]
    ambiguities: tuple[PaintedDomainAmbiguity, ...]
    assigned_selected_count: int
    ambiguous_selected_count: int
    unassigned_selected_count: int
    minimum_overlap_fraction: float


def compile_selected_labels_against_painted_domains(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_jrc: np.ndarray,
    painted_domains: Mapping[str, np.ndarray],
    *,
    minimum_overlap_fraction: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> PaintedDomainCompilation:
    rows = np.asarray(selected_rows, dtype=np.int64)
    traces = np.asarray(traces_roi_by_time, dtype=float)
    selected = np.asarray(selected_labels_jrc)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be one-dimensional")
    if traces.ndim != 2 or traces.shape[0] != rows.size:
        raise ValueError("traces_roi_by_time must be [selected_roi,time]")
    if selected.ndim != 3:
        raise ValueError("selected_labels_jrc must be a 3-D label image")
    if not painted_domains:
        raise ValueError("painted_domains is empty")
    if not (0.0 < minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in (0,1]")

    domains: dict[str, np.ndarray] = {}
    for region, image in painted_domains.items():
        arr = np.asarray(image)
        if arr.shape != selected.shape:
            raise ValueError(f"painted domain {region!r} shape {arr.shape} != selected shape {selected.shape}")
        domains[str(region)] = arr != 0

    assignments: list[PaintedDomainAssignment] = []
    ambiguities: list[PaintedDomainAmbiguity] = []
    by_region_trace_indices: dict[str, list[int]] = {}
    accounted: set[int] = set()

    for trace_index, selected_row in enumerate(rows):
        mask = selected == (int(selected_row) + 1)
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            continue
        scores: list[tuple[str, int, float]] = []
        for region, domain in domains.items():
            overlap = int(np.count_nonzero(mask & domain))
            if overlap:
                scores.append((region, overlap, overlap / voxel_count))
        if not scores:
            continue
        best_fraction = max(score[2] for score in scores)
        if best_fraction < minimum_overlap_fraction:
            continue
        winners = sorted(
            score for score in scores if abs(score[2] - best_fraction) <= tie_tolerance
        )
        accounted.add(int(selected_row))
        if len(winners) != 1:
            ambiguities.append(
                PaintedDomainAmbiguity(
                    selected_row=int(selected_row),
                    regions=tuple(w[0] for w in winners),
                    overlap_fraction=float(best_fraction),
                )
            )
            continue
        region, overlap, fraction = winners[0]
        assignments.append(
            PaintedDomainAssignment(
                selected_row=int(selected_row),
                region=region,
                overlap_voxel_count=overlap,
                selected_voxel_count=voxel_count,
                overlap_fraction=float(fraction),
            )
        )
        by_region_trace_indices.setdefault(region, []).append(trace_index)

    if not by_region_trace_indices:
        raise ValueError("no selected ROIs uniquely satisfy painted-domain overlap threshold")
    regions = tuple(sorted(by_region_trace_indices))
    time_by_region = np.column_stack(
        [np.mean(traces[by_region_trace_indices[r], :], axis=0) for r in regions]
    )
    return PaintedDomainCompilation(
        region_traces=FunctionalTraceTable(
            regions,
            time_by_region,
            None,
            identity_kind="jrc2018_vfb_painted_domain_from_exact_selected_supervoxel_overlap",
        ),
        assignments=tuple(assignments),
        ambiguities=tuple(ambiguities),
        assigned_selected_count=len(assignments),
        ambiguous_selected_count=len(ambiguities),
        unassigned_selected_count=int(rows.size - len(accounted)),
        minimum_overlap_fraction=float(minimum_overlap_fraction),
    )
