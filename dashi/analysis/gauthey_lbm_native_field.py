"""Compile recovered Gauthey LBM selected ROIs into a compact native-space field.

This module is experiment-facing. It combines exact recovered selected-row
identities with the same-trial plane-local segmentation and the deposited
selected trace matrix. The result is a compact spatial carrier containing the
measured traces, one static voxel mask per recovered supervoxel (encoded as a
selected-row label volume), and native pixel/plane centroids.

It deliberately does not claim common-atlas coordinates or neuron identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_EXPECTED_SELECTED,
    LBM_MIN_TIMEPOINTS,
    load_deposited_lbm_selected,
    pooled_lbm_row_to_trial_plane_cluster,
)
from dashi.io.functional_imaging_loader import FunctionalTraceTable

LBM_NATIVE_HEIGHT = 226
LBM_NATIVE_WIDTH = 512
LBM_N_PLANES = 27
LBM_PIXELS_PER_PLANE = LBM_NATIVE_HEIGHT * LBM_NATIVE_WIDTH


@dataclass(frozen=True)
class RecoveredSelectedROI:
    selected_row: int
    pooled_source_row: int
    trial_id: str
    plane_index: int
    cluster_index: int
    correlation: float


@dataclass(frozen=True)
class NativeSelectedROI:
    selected_row: int
    pooled_source_row: int
    trial_id: str
    plane_index: int
    cluster_index: int
    correlation: float
    voxel_count: int
    centroid_y: float
    centroid_x: float


@dataclass(frozen=True)
class NativeFunctionalField:
    trial_id: str
    selected_rows: np.ndarray
    traces_roi_by_time: np.ndarray
    selected_label_volume: np.ndarray  # [plane, y, x], value = selected_row + 1
    rois: tuple[NativeSelectedROI, ...]


@dataclass(frozen=True)
class NativeAtlasAssignment:
    selected_row: int
    atlas_region_id: int
    atlas_region: str
    voxel_count: int
    overlap_voxel_count: int
    overlap_fraction: float


@dataclass(frozen=True)
class NativeAtlasCompilation:
    region_traces: FunctionalTraceTable
    assignments: tuple[NativeAtlasAssignment, ...]
    assigned_selected_count: int
    unassigned_selected_count: int
    minimum_overlap_fraction: float


def load_recovered_identity_csv(path: str | Path) -> tuple[RecoveredSelectedROI, ...]:
    rows: list[RecoveredSelectedROI] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("identity CSV lacks required reconstruction columns")
        for row in reader:
            identity = RecoveredSelectedROI(
                selected_row=int(row["selected_row"]),
                pooled_source_row=int(row["pooled_source_row"]),
                trial_id=str(row["trial_id"]),
                plane_index=int(row["plane_index"]),
                cluster_index=int(row["cluster_index"]),
                correlation=float(row["correlation"]),
            )
            decoded = pooled_lbm_row_to_trial_plane_cluster(identity.pooled_source_row)
            declared = (identity.trial_id, identity.plane_index, identity.cluster_index)
            if decoded != declared:
                raise ValueError(
                    "recovered identity geometry disagrees with pooled_source_row: "
                    f"selected_row={identity.selected_row} decoded={decoded!r} declared={declared!r}"
                )
            rows.append(identity)
    if not rows:
        raise ValueError("identity CSV contains no recovered rows")
    selected = [r.selected_row for r in rows]
    if len(selected) != len(set(selected)):
        raise ValueError("identity CSV contains duplicate selected_row values")
    return tuple(rows)


def _source_oriented_label_volume(arr: np.ndarray) -> np.ndarray:
    """Reproduce ``loadmat_h5(...)[\"labels\"].reshape((226,512,27))``.

    Gauthey's published ``loadmat_h5`` transposes every numerical HDF5 dataset.
    Therefore a deposited raw HDF5 label matrix shaped ``(27, 115712)`` becomes
    ``(115712, 27)`` before the source code reshapes it to ``(226, 512, 27)``.
    We reproduce that sequence exactly, then move the plane axis first.
    """
    raw = np.asarray(arr)
    if raw.shape == (LBM_N_PLANES, LBM_PIXELS_PER_PLANE):
        source_loaded = raw.T
    elif raw.shape == (LBM_PIXELS_PER_PLANE, LBM_N_PLANES):
        source_loaded = raw
    elif raw.shape == (LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH, LBM_N_PLANES):
        return np.moveaxis(raw, 2, 0)
    elif raw.shape == (LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH):
        return raw
    else:
        raise ValueError(f"unexpected Gauthey LBM segmentation shape {raw.shape}")

    y_x_plane = source_loaded.reshape(LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH, LBM_N_PLANES)
    return np.moveaxis(y_x_plane, 2, 0)


def load_gauthey_lbm_segmentation(path: str | Path) -> np.ndarray:
    """Load Gauthey n2000 segmentation as source-faithful ``[plane,y,x]``."""
    with h5py.File(path, "r") as handle:
        if "labels" in handle:
            arr = np.asarray(handle["labels"])
        else:
            datasets: list[np.ndarray] = []
            handle.visititems(
                lambda _name, obj: datasets.append(np.asarray(obj))
                if isinstance(obj, h5py.Dataset)
                else None
            )
            if len(datasets) != 1:
                raise ValueError("segmentation HDF5 must contain a unique labels dataset")
            arr = datasets[0]
    return _source_oriented_label_volume(arr)


def compile_native_selected_field(
    deposited_selected: np.ndarray,
    identities: Sequence[RecoveredSelectedROI],
    segmentation: np.ndarray,
    *,
    trial_id: str,
) -> NativeFunctionalField:
    traces = np.asarray(deposited_selected)
    if traces.shape != (LBM_EXPECTED_SELECTED, LBM_MIN_TIMEPOINTS):
        raise ValueError(
            "deposited selected matrix must have shape "
            f"({LBM_EXPECTED_SELECTED}, {LBM_MIN_TIMEPOINTS})"
        )
    if not np.issubdtype(traces.dtype, np.number):
        raise ValueError("deposited selected matrix must be numeric")
    seg = np.asarray(segmentation)
    if seg.shape != (LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH):
        raise ValueError("segmentation must be normalized to [27, 226, 512]")

    same_trial = sorted(
        (identity for identity in identities if identity.trial_id == trial_id),
        key=lambda identity: identity.selected_row,
    )
    if not same_trial:
        raise ValueError(f"no recovered identities for trial {trial_id}")

    label_volume = np.zeros(seg.shape, dtype=np.int32)
    native_rois: list[NativeSelectedROI] = []
    selected_rows: list[int] = []

    for identity in same_trial:
        if not (0 <= identity.selected_row < LBM_EXPECTED_SELECTED):
            raise ValueError("selected_row outside deposited selected matrix")
        if not (0 <= identity.plane_index < LBM_N_PLANES):
            raise ValueError("plane_index outside LBM carrier")
        plane = seg[identity.plane_index]
        mask = plane == identity.cluster_index
        yy, xx = np.nonzero(mask)
        if yy.size == 0:
            raise ValueError(
                "recovered cluster absent from segmentation: "
                f"plane={identity.plane_index} cluster={identity.cluster_index}"
            )
        occupied = label_volume[identity.plane_index][mask]
        if np.any(occupied != 0):
            raise ValueError("recovered selected supervoxel masks overlap unexpectedly")
        label_volume[identity.plane_index][mask] = identity.selected_row + 1
        selected_rows.append(identity.selected_row)
        native_rois.append(
            NativeSelectedROI(
                selected_row=identity.selected_row,
                pooled_source_row=identity.pooled_source_row,
                trial_id=identity.trial_id,
                plane_index=identity.plane_index,
                cluster_index=identity.cluster_index,
                correlation=identity.correlation,
                voxel_count=int(yy.size),
                centroid_y=float(np.mean(yy)),
                centroid_x=float(np.mean(xx)),
            )
        )

    selected_array = np.asarray(selected_rows, dtype=np.int64)
    return NativeFunctionalField(
        trial_id=trial_id,
        selected_rows=selected_array,
        traces_roi_by_time=np.asarray(traces[selected_array, :], dtype=float).copy(),
        selected_label_volume=label_volume,
        rois=tuple(native_rois),
    )


def aggregate_native_selected_field_to_regions(
    field: NativeFunctionalField,
    atlas_labels_on_native_grid: np.ndarray,
    atlas_region_names: Mapping[int, str],
    *,
    minimum_overlap_fraction: float = 0.5,
) -> NativeAtlasCompilation:
    """Aggregate recovered selected traces after atlas labels are on the native grid.

    ``atlas_labels_on_native_grid`` must already be transformed/resampled onto
    exactly the same ``[plane,y,x]`` grid as ``field.selected_label_volume``.
    This function does not perform or infer registration.
    """
    atlas = np.asarray(atlas_labels_on_native_grid)
    if atlas.shape != field.selected_label_volume.shape:
        raise ValueError("atlas labels must already be on the native [plane,y,x] grid")
    if not (0.0 < minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in (0,1]")

    assignments: list[NativeAtlasAssignment] = []
    by_region_trace_indices: dict[str, list[int]] = {}

    for trace_index, roi in enumerate(field.rois):
        mask = field.selected_label_volume == (roi.selected_row + 1)
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            raise ValueError(f"native field lost selected-row mask {roi.selected_row}")
        vals = atlas[mask].astype(int, copy=False)
        labels, counts = np.unique(vals, return_counts=True)
        valid = [
            (int(label), int(count))
            for label, count in zip(labels, counts)
            if int(label) in atlas_region_names
        ]
        if not valid:
            continue
        region_id, overlap = max(valid, key=lambda pair: pair[1])
        fraction = overlap / voxel_count
        if fraction < minimum_overlap_fraction:
            continue
        region = str(atlas_region_names[region_id])
        assignments.append(
            NativeAtlasAssignment(
                selected_row=roi.selected_row,
                atlas_region_id=region_id,
                atlas_region=region,
                voxel_count=voxel_count,
                overlap_voxel_count=overlap,
                overlap_fraction=float(fraction),
            )
        )
        by_region_trace_indices.setdefault(region, []).append(trace_index)

    if not by_region_trace_indices:
        raise ValueError("no recovered selected ROIs satisfy atlas-overlap threshold")

    regions = tuple(sorted(by_region_trace_indices))
    region_traces = np.column_stack(
        [
            np.mean(field.traces_roi_by_time[by_region_trace_indices[region], :], axis=0)
            for region in regions
        ]
    ).T
    return NativeAtlasCompilation(
        region_traces=FunctionalTraceTable(
            regions,
            region_traces,
            None,
            identity_kind="atlas_region_from_exact_selected_supervoxel_overlap",
        ),
        assignments=tuple(assignments),
        assigned_selected_count=len(assignments),
        unassigned_selected_count=len(field.rois) - len(assignments),
        minimum_overlap_fraction=float(minimum_overlap_fraction),
    )


def compile_native_selected_field_from_files(
    deposited_selected_path: str | Path,
    identity_csv_path: str | Path,
    segmentation_path: str | Path,
    *,
    trial_id: str,
) -> NativeFunctionalField:
    return compile_native_selected_field(
        load_deposited_lbm_selected(deposited_selected_path),
        load_recovered_identity_csv(identity_csv_path),
        load_gauthey_lbm_segmentation(segmentation_path),
        trial_id=trial_id,
    )
