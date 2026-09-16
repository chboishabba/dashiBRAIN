"""Plane-local Gauthey supervoxel geometry.

The published extraction pipeline clusters each z-slice independently, so local
cluster integers are reused across planes. The anatomical identity carrier is
``(plane_index, cluster_index)``, not a bare integer label.

Scientific source: Wayan Gauthey et al., "High-speed whole-brain imaging in
Drosophila", DOI 10.1038/s41467-026-72437-1; analysis code
``murthylab/lightbead-analysis/dellaserver_processing/signal_to_supervoxel_roi.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


@dataclass(frozen=True)
class PlaneLocalSupervoxelCentroid:
    plane_index: int
    cluster_index: int
    voxel_count: int
    voxel_x: float
    voxel_y: float
    voxel_z: float
    world_x: float
    world_y: float
    world_z: float
    coordinate_space: str = "gauthey_trial_mean_brain"
    evidence_kind: str = "plane_local_supervoxel_centroid"

    @property
    def source_key(self) -> str:
        return f"plane_{self.plane_index:02d}:cluster_{self.cluster_index:04d}"


def _resolve_plane_axis(labels: np.ndarray, spatial_shape: tuple[int, int, int]) -> tuple[int, tuple[int, int]]:
    if labels.ndim != 2:
        raise ValueError("Gauthey supervoxel labels must be planes x flattened-in-plane pixels")
    planes, flat = labels.shape
    candidates: list[tuple[int, tuple[int, int]]] = []
    for plane_axis in range(3):
        if spatial_shape[plane_axis] != planes:
            continue
        other = tuple(i for i in range(3) if i != plane_axis)
        if spatial_shape[other[0]] * spatial_shape[other[1]] == flat:
            candidates.append((plane_axis, other))
    if len(candidates) != 1:
        raise ValueError(
            "cannot uniquely align per-plane labels to mean-brain axes; "
            f"labels={labels.shape}, spatial={spatial_shape}, candidates={len(candidates)}"
        )
    return candidates[0]


def plane_local_centroids(
    labels: np.ndarray,
    spatial_shape: tuple[int, int, int],
    affine: np.ndarray,
) -> tuple[PlaneLocalSupervoxelCentroid, ...]:
    lab = np.asarray(labels)
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4):
        raise ValueError("affine must be 4x4")
    plane_axis, other_axes = _resolve_plane_axis(lab, spatial_shape)
    plane_shape = (spatial_shape[other_axes[0]], spatial_shape[other_axes[1]])

    out: list[PlaneLocalSupervoxelCentroid] = []
    for plane in range(lab.shape[0]):
        plane_labels = np.asarray(lab[plane]).reshape(plane_shape).astype(np.int64, copy=False)
        if np.any(plane_labels < 0):
            raise ValueError("negative supervoxel labels are unsupported")
        ids = plane_labels.ravel()
        counts = np.bincount(ids)
        a, b = np.indices(plane_shape)
        sum_a = np.bincount(ids, weights=a.ravel(), minlength=len(counts))
        sum_b = np.bincount(ids, weights=b.ravel(), minlength=len(counts))
        for cluster in np.flatnonzero(counts):
            n = int(counts[cluster])
            voxel = np.zeros(4, dtype=float)
            voxel[3] = 1.0
            voxel[plane_axis] = float(plane)
            voxel[other_axes[0]] = float(sum_a[cluster] / n)
            voxel[other_axes[1]] = float(sum_b[cluster] / n)
            world = aff @ voxel
            out.append(PlaneLocalSupervoxelCentroid(
                plane_index=plane,
                cluster_index=int(cluster),
                voxel_count=n,
                voxel_x=float(voxel[0]),
                voxel_y=float(voxel[1]),
                voxel_z=float(voxel[2]),
                world_x=float(world[0]),
                world_y=float(world[1]),
                world_z=float(world[2]),
            ))
    return tuple(out)


def load_plane_local_centroids(
    labels_h5: str | Path,
    mean_brain_nifti: str | Path,
) -> tuple[PlaneLocalSupervoxelCentroid, ...]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read Gauthey labels") from exc
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to read Gauthey mean-brain NIfTI") from exc

    with h5py.File(labels_h5, "r") as f:
        if "labels" not in f:
            raise ValueError("Gauthey labels HDF5 lacks dataset 'labels'")
        labels = np.asarray(f["labels"])
    img = nib.load(str(mean_brain_nifti))
    shape = tuple(int(x) for x in img.shape[:3])
    return plane_local_centroids(labels, shape, np.asarray(img.affine, dtype=float))


def write_plane_local_centroids_csv(
    centroids: tuple[PlaneLocalSupervoxelCentroid, ...],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(PlaneLocalSupervoxelCentroid.__dataclass_fields__) + ["source_key"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in centroids:
            row = {field: getattr(c, field) for field in PlaneLocalSupervoxelCentroid.__dataclass_fields__}
            row["source_key"] = c.source_key
            writer.writerow(row)
