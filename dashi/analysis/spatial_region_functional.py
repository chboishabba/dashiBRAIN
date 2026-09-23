"""Compile plane-local supervoxel activity into atlas-region time series.

This is an experiment-facing bridge for neuropil-scale functional imaging.
It deliberately does not infer neuron identity. Instead it assigns each
plane-local supervoxel to an atlas region by voxel overlap, then aggregates
measured traces onto the shared region carrier used by the MaleCNS benchmark.

The expected source row order matches the Gauthey preprocessing pipeline:
plane-major, then cluster-major within each plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dashi.io.functional_imaging_loader import FunctionalTraceTable


@dataclass(frozen=True)
class SupervoxelRegionAssignment:
    plane_index: int
    cluster_index: int
    trace_row: int
    atlas_region_id: int
    atlas_region: str
    voxel_count: int
    overlap_voxel_count: int
    overlap_fraction: float


@dataclass(frozen=True)
class SpatialRegionCompilation:
    region_traces: FunctionalTraceTable
    assignments: tuple[SupervoxelRegionAssignment, ...]
    assigned_trace_count: int
    unassigned_trace_count: int
    minimum_overlap_fraction: float


def _normalize_plane_axis(labels: np.ndarray, n_planes: int, plane_axis: int | None) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 3:
        raise ValueError("segmentation and atlas labels must be 3-D")
    if plane_axis is None:
        candidates = [i for i, size in enumerate(arr.shape) if size == n_planes]
        if len(candidates) != 1:
            raise ValueError("plane axis is ambiguous; supply plane_axis explicitly")
        plane_axis = candidates[0]
    if arr.shape[plane_axis] != n_planes:
        raise ValueError("plane_axis does not have n_planes entries")
    return np.moveaxis(arr, plane_axis, 0)


def compile_plane_local_supervoxels_to_regions(
    traces_time_by_roi: np.ndarray,
    segmentation_labels: np.ndarray,
    atlas_labels: np.ndarray,
    atlas_region_names: Mapping[int, str],
    *,
    n_planes: int,
    clusters_per_plane: int,
    plane_axis: int | None = None,
    minimum_overlap_fraction: float = 0.5,
    include_cluster_zero: bool = True,
) -> SpatialRegionCompilation:
    """Aggregate measured plane-local supervoxel traces onto atlas regions.

    ``traces_time_by_roi`` must contain exactly ``n_planes * clusters_per_plane``
    columns in plane-major/cluster-major order. Segmentation cluster integers are
    interpreted locally within each plane, which prevents accidental merging of
    equal cluster IDs across z-planes.

    A supervoxel is assigned only when one atlas region occupies at least
    ``minimum_overlap_fraction`` of its voxels. Atlas label 0 is treated as
    unassigned unless explicitly named in ``atlas_region_names``.
    """
    traces = np.asarray(traces_time_by_roi, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces_time_by_roi must be 2-D [time, roi]")
    expected = n_planes * clusters_per_plane
    if traces.shape[1] != expected:
        raise ValueError(f"expected {expected} plane-local trace columns, got {traces.shape[1]}")
    if not (0.0 < minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in (0, 1]")

    seg = _normalize_plane_axis(segmentation_labels, n_planes, plane_axis)
    atlas = _normalize_plane_axis(atlas_labels, n_planes, plane_axis)
    if seg.shape != atlas.shape:
        raise ValueError("segmentation_labels and atlas_labels must have identical spatial shape")

    assignments: list[SupervoxelRegionAssignment] = []
    by_region_rows: dict[str, list[int]] = {}
    assigned_rows: set[int] = set()

    cluster_start = 0 if include_cluster_zero else 1
    for plane in range(n_planes):
        seg_plane = seg[plane]
        atlas_plane = atlas[plane]
        for cluster in range(cluster_start, clusters_per_plane):
            row = plane * clusters_per_plane + cluster
            mask = seg_plane == cluster
            voxel_count = int(np.count_nonzero(mask))
            if voxel_count == 0:
                continue
            vals = atlas_plane[mask].astype(int, copy=False)
            labels, counts = np.unique(vals, return_counts=True)
            valid = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) in atlas_region_names]
            if not valid:
                continue
            region_id, overlap = max(valid, key=lambda pair: pair[1])
            frac = overlap / voxel_count
            if frac < minimum_overlap_fraction:
                continue
            region = str(atlas_region_names[region_id])
            assignments.append(
                SupervoxelRegionAssignment(
                    plane_index=plane,
                    cluster_index=cluster,
                    trace_row=row,
                    atlas_region_id=region_id,
                    atlas_region=region,
                    voxel_count=voxel_count,
                    overlap_voxel_count=overlap,
                    overlap_fraction=float(frac),
                )
            )
            by_region_rows.setdefault(region, []).append(row)
            assigned_rows.add(row)

    if not by_region_rows:
        raise ValueError("no supervoxels satisfy atlas-overlap assignment threshold")

    regions = tuple(sorted(by_region_rows))
    region_traces = np.column_stack(
        [np.mean(traces[:, by_region_rows[region]], axis=1) for region in regions]
    )
    table = FunctionalTraceTable(
        regions,
        region_traces,
        None,
        identity_kind="atlas_region_from_supervoxel_overlap",
    )
    return SpatialRegionCompilation(
        region_traces=table,
        assignments=tuple(assignments),
        assigned_trace_count=len(assigned_rows),
        unassigned_trace_count=expected - len(assigned_rows),
        minimum_overlap_fraction=float(minimum_overlap_fraction),
    )


def auditory_region_calibration(
    region_traces: FunctionalTraceTable,
    stimulus: Sequence[float],
    *,
    expected_regions: Sequence[str] = ("AMMC", "WED"),
) -> dict[str, object]:
    """Rank atlas regions by correlation with an auditory stimulus.

    The Gauthey paper reports that the highest stimulus-correlated subgroup is
    primarily concentrated in AMMC and WED. This function turns that report into
    a calibration check for a reconstructed region-functional carrier; it does
    not force those regions into the reconstruction.
    """
    stim = np.asarray(stimulus, dtype=float)
    traces = np.asarray(region_traces.traces, dtype=float)
    if stim.ndim != 1 or stim.shape[0] != traces.shape[0]:
        raise ValueError("stimulus must be 1-D and align with trace time samples")
    if np.std(stim) == 0:
        raise ValueError("stimulus has zero variance")

    scores: list[tuple[str, float]] = []
    for i, region in enumerate(region_traces.unit_ids):
        trace = traces[:, i]
        if np.std(trace) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(trace, stim)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        scores.append((region, corr))
    scores.sort(key=lambda item: item[1], reverse=True)
    rank = {region: i + 1 for i, (region, _) in enumerate(scores)}
    expected = tuple(str(r) for r in expected_regions)
    present_expected = tuple(r for r in expected if r in rank)
    return {
        "ranked_regions": scores,
        "expected_regions": expected,
        "expected_regions_present": present_expected,
        "expected_region_ranks": {r: rank[r] for r in present_expected},
        "expected_regions_in_top_quartile": bool(
            present_expected
            and all(rank[r] <= max(1, int(np.ceil(len(scores) / 4))) for r in present_expected)
        ),
    }
