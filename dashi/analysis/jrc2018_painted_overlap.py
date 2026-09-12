"""Assign transformed selected LBM supervoxels to VFB JRC2018 painted domains.

The VFB atlas is represented as 46 separate binary painted domains rather than
one mutually exclusive label image. This preserves legitimate overlap between
anatomical domains/subdomains. Each selected supervoxel is scored against every
domain by fractional voxel overlap; a unique maximum above threshold is assigned,
while exact ties are retained as ambiguous and are not silently promoted.

The production path is deliberately streaming: one painted domain is consumed at
a time and all selected-label overlaps are counted together with ``np.bincount``.
This avoids retaining the 46 full JRC2018 rasters (or allocating one full-raster
mask for every selected-ROI/domain pair) in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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


def compile_selected_labels_against_painted_domain_stream(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_jrc: np.ndarray,
    painted_domains: Iterable[tuple[str, np.ndarray]],
    *,
    minimum_overlap_fraction: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> PaintedDomainCompilation:
    """Compile selected labels against a one-domain-at-a-time atlas stream.

    ``selected_labels_jrc`` stores ``selected_row + 1`` as its non-zero label
    values. For each painted domain, one vectorized ``np.bincount`` over the
    selected labels inside that domain computes overlap counts for *all* selected
    ROIs simultaneously. Only the current domain array and O(number-of-labels)
    score state are retained.
    """
    rows = np.asarray(selected_rows, dtype=np.int64)
    traces = np.asarray(traces_roi_by_time, dtype=float)
    selected = np.asarray(selected_labels_jrc)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be one-dimensional")
    if traces.ndim != 2 or traces.shape[0] != rows.size:
        raise ValueError("traces_roi_by_time must be [selected_roi,time]")
    if selected.ndim != 3:
        raise ValueError("selected_labels_jrc must be a 3-D label image")
    if not (0.0 < minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in (0,1]")
    if tie_tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")

    selected_values = rows + 1
    if np.any(selected_values <= 0):
        raise ValueError("selected_rows must map to positive selected-label values")

    max_label = int(max(int(selected.max(initial=0)), int(selected_values.max(initial=0))))
    voxel_counts = np.bincount(
        selected.astype(np.int64, copy=False).ravel(), minlength=max_label + 1
    )

    best_fraction = np.full(rows.size, -1.0, dtype=float)
    best_overlap = np.zeros(rows.size, dtype=np.int64)
    best_regions: list[list[str]] = [[] for _ in range(rows.size)]
    domain_count = 0

    for region_raw, image in painted_domains:
        region = str(region_raw)
        arr = np.asarray(image)
        if arr.shape != selected.shape:
            raise ValueError(
                f"painted domain {region!r} shape {arr.shape} != selected shape {selected.shape}"
            )
        domain_count += 1

        # One domain-sized boolean is the only full-raster temporary. The fancy
        # indexing result contains only selected-label values inside that domain.
        domain_mask = arr != 0
        inside = selected[domain_mask].astype(np.int64, copy=False)
        overlap_counts = np.bincount(inside, minlength=max_label + 1)

        for trace_index, label_value in enumerate(selected_values):
            label = int(label_value)
            selected_voxels = int(voxel_counts[label]) if label < voxel_counts.size else 0
            if selected_voxels == 0:
                continue
            overlap = int(overlap_counts[label]) if label < overlap_counts.size else 0
            if overlap == 0:
                continue
            fraction = overlap / selected_voxels
            current = float(best_fraction[trace_index])
            if fraction > current + tie_tolerance:
                best_fraction[trace_index] = fraction
                best_overlap[trace_index] = overlap
                best_regions[trace_index] = [region]
            elif abs(fraction - current) <= tie_tolerance:
                best_regions[trace_index].append(region)

    if domain_count == 0:
        raise ValueError("painted_domains is empty")

    assignments: list[PaintedDomainAssignment] = []
    ambiguities: list[PaintedDomainAmbiguity] = []
    by_region_trace_indices: dict[str, list[int]] = {}
    accounted: set[int] = set()

    for trace_index, selected_row in enumerate(rows):
        fraction = float(best_fraction[trace_index])
        if fraction < minimum_overlap_fraction:
            continue
        winners = tuple(sorted(set(best_regions[trace_index])))
        if not winners:
            continue
        accounted.add(int(selected_row))
        if len(winners) != 1:
            ambiguities.append(
                PaintedDomainAmbiguity(
                    selected_row=int(selected_row),
                    regions=winners,
                    overlap_fraction=fraction,
                )
            )
            continue

        region = winners[0]
        label = int(selected_row) + 1
        assignment = PaintedDomainAssignment(
            selected_row=int(selected_row),
            region=region,
            overlap_voxel_count=int(best_overlap[trace_index]),
            selected_voxel_count=int(voxel_counts[label]),
            overlap_fraction=fraction,
        )
        assignments.append(assignment)
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


def compile_selected_labels_against_painted_domains(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_jrc: np.ndarray,
    painted_domains: Mapping[str, np.ndarray],
    *,
    minimum_overlap_fraction: float = 0.5,
    tie_tolerance: float = 1e-12,
) -> PaintedDomainCompilation:
    """Compatibility wrapper for callers that already hold domain arrays.

    The overlap algorithm itself is the same streaming/vectorized implementation
    used by the production file-backed path.
    """
    return compile_selected_labels_against_painted_domain_stream(
        selected_rows,
        traces_roi_by_time,
        selected_labels_jrc,
        painted_domains.items(),
        minimum_overlap_fraction=minimum_overlap_fraction,
        tie_tolerance=tie_tolerance,
    )
