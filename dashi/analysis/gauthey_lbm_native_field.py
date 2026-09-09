"""Compile recovered Gauthey LBM selected ROIs into a compact native-space field.

This module is experiment-facing.  It combines exact recovered selected-row
identities with the same-trial plane-local segmentation and the deposited
selected trace matrix.  The result is a compact spatial carrier containing the
measured traces, one static voxel mask per recovered supervoxel (encoded as a
selected-row label volume), and native pixel/plane centroids.

It deliberately does not claim common-atlas coordinates or neuron identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_EXPECTED_SELECTED,
    LBM_MIN_TIMEPOINTS,
    load_deposited_lbm_selected,
)

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
            rows.append(
                RecoveredSelectedROI(
                    selected_row=int(row["selected_row"]),
                    pooled_source_row=int(row["pooled_source_row"]),
                    trial_id=str(row["trial_id"]),
                    plane_index=int(row["plane_index"]),
                    cluster_index=int(row["cluster_index"]),
                    correlation=float(row["correlation"]),
                )
            )
    if not rows:
        raise ValueError("identity CSV contains no recovered rows")
    selected = [r.selected_row for r in rows]
    if len(selected) != len(set(selected)):
        raise ValueError("identity CSV contains duplicate selected_row values")
    return tuple(rows)


def load_gauthey_lbm_segmentation(path: str | Path) -> np.ndarray:
    """Load Gauthey n2000 segmentation as [plane, y, x].

    The published Fig. 3 preprocessing reshapes the label carrier to
    ``(226, 512, 27)``.  Deposited files commonly expose ``(27, 115712)``;
    both layouts are accepted and normalized here.
    """
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

    arr = np.asarray(arr)
    if arr.shape == (LBM_N_PLANES, LBM_PIXELS_PER_PLANE):
        return arr.reshape(LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH)
    if arr.shape == (LBM_PIXELS_PER_PLANE, LBM_N_PLANES):
        return arr.T.reshape(LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH)
    if arr.shape == (LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH, LBM_N_PLANES):
        return np.moveaxis(arr, 2, 0)
    if arr.shape == (LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH):
        return arr
    raise ValueError(f"unexpected Gauthey LBM segmentation shape {arr.shape}")


def compile_native_selected_field(
    deposited_selected: np.ndarray,
    identities: Sequence[RecoveredSelectedROI],
    segmentation: np.ndarray,
    *,
    trial_id: str,
) -> NativeFunctionalField:
    traces = np.asarray(deposited_selected, dtype=float)
    if traces.shape != (LBM_EXPECTED_SELECTED, LBM_MIN_TIMEPOINTS):
        raise ValueError(
            f"deposited selected matrix must have shape "
            f"({LBM_EXPECTED_SELECTED}, {LBM_MIN_TIMEPOINTS})"
        )
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
                f"recovered cluster absent from segmentation: "
                f"plane={identity.plane_index} cluster={identity.cluster_index}"
            )
        # Segmentation is a partition within a plane.  A nonzero collision would
        # therefore indicate inconsistent selected identities or source geometry.
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
        traces_roi_by_time=traces[selected_array, :].copy(),
        selected_label_volume=label_volume,
        rois=tuple(native_rois),
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
