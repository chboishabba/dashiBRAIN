"""Registration staging for compact Gauthey et al. 2026 functional artifacts.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI 10.5281/zenodo.17618684.

This module derives segmentation-label centroids in the *trial mean-brain*
coordinate system.  It deliberately does not identify those labels with the 668
functional matrix columns and does not apply the published BIFROST FDA->JRC2018
transform directly: a trial-mean-brain -> FDA registration receipt is required
first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np


@dataclass(frozen=True)
class LabelCentroid:
    label_id: int
    voxel_count: int
    voxel_x: float
    voxel_y: float
    voxel_z: float
    world_x: float
    world_y: float
    world_z: float
    coordinate_space: str = "gauthey_trial_mean_brain"
    evidence_kind: str = "segmentation_label_centroid_unregistered_to_functional_trace"


@dataclass(frozen=True)
class RegistrationStageBoundary:
    source_space: str
    target_space: str
    transform_identifier: str | None
    verified: bool
    note: str


GAUTHEY_TRIAL_TO_FDA_BOUNDARY = RegistrationStageBoundary(
    source_space="gauthey_trial_mean_brain",
    target_space="bifrost_fda",
    transform_identifier=None,
    verified=False,
    note="No exact trial-mean-brain -> FDA transform is pinned yet.",
)

BIFROST_FDA_TO_JRC2018_BOUNDARY = RegistrationStageBoundary(
    source_space="bifrost_fda",
    target_space="JRC2018_female",
    transform_identifier=(
        "doi:10.5061/dryad.8pk0p2nx1#thresholded_FDA_to_JRC2018_female.h5"
    ),
    verified=True,
    note=(
        "Published BIFROST source transform; verified as a repository artifact, "
        "not yet materialized locally for this benchmark."
    ),
)


def _labels_to_xyz_volume(labels: np.ndarray, spatial_shape: tuple[int, int, int]) -> np.ndarray:
    """Align a deposited labels array to NIfTI voxel axes [x, y, z].

    The Gauthey labels receipt is 2-D (z-planes x flattened in-plane voxels).
    We accept any spatial-axis permutation whose one axis matches the plane count
    and whose remaining product matches the flattened size. Ambiguity is rejected
    rather than guessed.
    """
    lab = np.asarray(labels)
    if lab.ndim == 3:
        if lab.shape == spatial_shape:
            return lab
        raise ValueError(f"3-D label shape {lab.shape} != mean-brain shape {spatial_shape}")
    if lab.ndim != 2:
        raise ValueError("labels must be 2-D deposited planes or a 3-D volume")

    candidates: list[np.ndarray] = []
    planes, flat = lab.shape
    for z_axis in range(3):
        if spatial_shape[z_axis] != planes:
            continue
        other_axes = [a for a in range(3) if a != z_axis]
        if spatial_shape[other_axes[0]] * spatial_shape[other_axes[1]] != flat:
            continue
        # Deposited rows are planes; reshape each plane to the two remaining
        # axes, then move the plane axis into its NIfTI position.
        plane_shape = (spatial_shape[other_axes[0]], spatial_shape[other_axes[1]])
        arr = lab.reshape((planes,) + plane_shape)
        # Current axes: [plane, other0, other1]. Build transpose to [x,y,z].
        current_for_spatial = [None, None, None]
        current_for_spatial[z_axis] = 0
        current_for_spatial[other_axes[0]] = 1
        current_for_spatial[other_axes[1]] = 2
        candidates.append(np.transpose(arr, axes=tuple(current_for_spatial)))

    if len(candidates) != 1:
        raise ValueError(
            "cannot uniquely align deposited label planes to mean-brain spatial shape; "
            f"labels={lab.shape}, spatial={spatial_shape}, candidates={len(candidates)}"
        )
    return candidates[0]


def label_centroids(
    labels_xyz: np.ndarray,
    affine: np.ndarray,
    *,
    background_label: int = 0,
) -> tuple[LabelCentroid, ...]:
    """Compute voxel/world centroids for integer segmentation labels efficiently."""
    lab = np.asarray(labels_xyz)
    if lab.ndim != 3:
        raise ValueError("labels_xyz must be 3-D")
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4):
        raise ValueError("affine must be 4x4")

    flat_labels = lab.ravel().astype(np.int64, copy=False)
    valid = flat_labels != int(background_label)
    if not np.any(valid):
        return ()

    ids = flat_labels[valid]
    max_id = int(ids.max())
    if max_id < 0:
        raise ValueError("negative segmentation labels are unsupported")

    linear = np.flatnonzero(valid)
    x, y, z = np.unravel_index(linear, lab.shape)
    counts = np.bincount(ids, minlength=max_id + 1)
    sum_x = np.bincount(ids, weights=x, minlength=max_id + 1)
    sum_y = np.bincount(ids, weights=y, minlength=max_id + 1)
    sum_z = np.bincount(ids, weights=z, minlength=max_id + 1)

    out: list[LabelCentroid] = []
    for label_id in np.flatnonzero(counts):
        if label_id == background_label:
            continue
        n = int(counts[label_id])
        voxel = np.array([
            sum_x[label_id] / n,
            sum_y[label_id] / n,
            sum_z[label_id] / n,
            1.0,
        ])
        world = aff @ voxel
        out.append(LabelCentroid(
            label_id=int(label_id),
            voxel_count=n,
            voxel_x=float(voxel[0]),
            voxel_y=float(voxel[1]),
            voxel_z=float(voxel[2]),
            world_x=float(world[0]),
            world_y=float(world[1]),
            world_z=float(world[2]),
        ))
    return tuple(out)


def load_gauthey_label_centroids(
    labels_h5: str | Path,
    mean_brain_nifti: str | Path,
) -> tuple[LabelCentroid, ...]:
    """Load deposited labels + trial mean brain and derive trial-space centroids."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to read Gauthey segmentation labels") from exc
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to read Gauthey mean-brain NIfTI") from exc

    with h5py.File(labels_h5, "r") as f:
        if "labels" not in f:
            raise ValueError("Gauthey labels HDF5 lacks dataset 'labels'")
        labels = np.asarray(f["labels"])

    img = nib.load(str(mean_brain_nifti))
    spatial_shape = tuple(int(x) for x in img.shape[:3])
    labels_xyz = _labels_to_xyz_volume(labels, spatial_shape)
    return label_centroids(labels_xyz, np.asarray(img.affine, dtype=float))


def write_centroids_csv(centroids: tuple[LabelCentroid, ...], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(LabelCentroid.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in centroids:
            writer.writerow({field: getattr(c, field) for field in fields})
