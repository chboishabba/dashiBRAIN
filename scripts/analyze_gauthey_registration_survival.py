#!/usr/bin/env python3
"""Analyze which exact Gauthey a2_r5 selected supervoxels survive registration.

This is experiment-facing QC for the native -> FDA hop. It joins the exact
native ROI geometry with the transformed selected-label image and reports whether
label loss is associated with source plane/depth, native centroid, ROI size, or
native field-of-view boundaries.

The published Fig. 3 preprocessing assigns the second LBM trial
``04032024_6f_a2_r5`` the ``depth_a1`` plane-depth vector. Those depths are
retained here as source coordinates; they are not atlas identities.

Because native LBM pixels and FDA raster voxels have different physical sizes,
raw voxel-count ratios are not treated as retention fractions. We report both
raster support counts and physical support volumes, with the latter derived from
the NIfTI affine/unit metadata for the transformed FDA image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


TRIAL_ID = "04032024_6f_a2_r5"
DEPTH_A1_UM = np.array(
    [
        275.0,
        267.1,
        259.2,
        250.3,
        240.4,
        230.5,
        220.6,
        210.7,
        199.8,
        189.9,
        179.0,
        168.2,
        157.3,
        147.4,
        168.2,
        158.3,
        148.4,
        137.5,
        126.6,
        115.7,
        103.9,
        92.0,
        80.1,
        67.2,
        53.4,
        38.6,
        24.7,
    ],
    dtype=float,
)
NATIVE_Y = 226
NATIVE_X = 512
LBM_INPLANE_UM = 1.3
LBM_AXIAL_UM = 9.0
LBM_NATIVE_PIXEL_SUPPORT_UM3 = LBM_INPLANE_UM * LBM_INPLANE_UM * LBM_AXIAL_UM


def _read_native_rois(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "selected_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "voxel_count",
            "centroid_y",
            "centroid_x",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("native ROI CSV lacks required geometry columns")
        rows = []
        for raw in reader:
            trial_id = str(raw["trial_id"])
            if trial_id != TRIAL_ID:
                raise ValueError(f"unexpected trial in native ROI CSV: {trial_id}")
            plane = int(raw["plane_index"])
            if not (0 <= plane < DEPTH_A1_UM.size):
                raise ValueError(f"plane_index outside published depth vector: {plane}")
            y = float(raw["centroid_y"])
            x = float(raw["centroid_x"])
            voxel_count = int(raw["voxel_count"])
            rows.append(
                {
                    "selected_row": int(raw["selected_row"]),
                    "trial_id": trial_id,
                    "plane_index": plane,
                    "cluster_index": int(raw["cluster_index"]),
                    "published_plane_depth_um": float(DEPTH_A1_UM[plane]),
                    "voxel_count": voxel_count,
                    "native_support_volume_um3": float(voxel_count * LBM_NATIVE_PIXEL_SUPPORT_UM3),
                    "centroid_y": y,
                    "centroid_x": x,
                    "native_edge_distance_px": float(
                        min(y, (NATIVE_Y - 1) - y, x, (NATIVE_X - 1) - x)
                    ),
                }
            )
    if not rows:
        raise ValueError("native ROI CSV contains no rows")
    return rows


def _spatial_unit_to_microns(unit: str) -> float:
    normalized = str(unit).strip().lower()
    if normalized in {"micron", "micrometer", "micrometre", "um"}:
        return 1.0
    if normalized in {"mm", "millimeter", "millimetre"}:
        return 1000.0
    if normalized in {"meter", "metre", "m"}:
        return 1_000_000.0
    raise ValueError(f"unsupported/unknown NIfTI spatial unit for physical QC: {unit!r}")


def _voxel_volume_um3(img: nib.spatialimages.SpatialImage) -> float:
    spatial_unit, _ = img.header.get_xyzt_units()
    scale_um = _spatial_unit_to_microns(spatial_unit)
    volume_native_units = abs(float(np.linalg.det(np.asarray(img.affine, dtype=float)[:3, :3])))
    if not np.isfinite(volume_native_units) or volume_native_units <= 0:
        raise ValueError("FDA selected-label affine has non-positive/invalid voxel volume")
    return volume_native_units * (scale_um ** 3)


def _observed_selected_support(path: Path) -> tuple[dict[int, int], float]:
    img = nib.load(path)
    data = np.asarray(img.dataobj).astype(np.int64, copy=False)
    positive = data[data > 0]
    voxel_volume_um3 = _voxel_volume_um3(img)
    if positive.size == 0:
        return {}, voxel_volume_um3
    labels, counts = np.unique(positive, return_counts=True)
    return {int(label) - 1: int(count) for label, count in zip(labels, counts)}, voxel_volume_um3


def _observed_selected_rows(path: Path) -> set[int]:
    support, _ = _observed_selected_support(path)
    return set(support)


def _group_summary(rows: list[dict], key: str) -> list[dict]:
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    out = []
    for value in sorted(groups):
        members = groups[value]
        survived = sum(1 for row in members if row["survived_fda"])
        out.append(
            {
                key: value,
                "total": len(members),
                "survived": survived,
                "lost": len(members) - survived,
                "survival_fraction": survived / len(members),
            }
        )
    return out


def _distribution(rows: list[dict], field: str, survived: bool) -> dict:
    values = np.asarray(
        [float(row[field]) for row in rows if bool(row["survived_fda"]) is survived],
        dtype=float,
    )
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-rois",
        default="data/gauthey_lbm/native_field_a2_r5/selected_native_rois.csv",
    )
    parser.add_argument(
        "--selected-labels-fda",
        default="data/gauthey_lbm/fda_a2_r5/selected_labels_fda.nii",
    )
    parser.add_argument(
        "--output-dir",
        default="data/gauthey_lbm/registration_survival_a2_r5",
    )
    args = parser.parse_args()

    native_rows = _read_native_rois(Path(args.native_rois))
    fda_support, fda_voxel_volume_um3 = _observed_selected_support(Path(args.selected_labels_fda))
    surviving = set(fda_support)
    declared = {int(row["selected_row"]) for row in native_rows}
    unexpected = sorted(surviving - declared)
    if unexpected:
        raise SystemExit(f"FDA image contains labels absent from native ROI table: {unexpected[:10]}")

    for row in native_rows:
        selected_row = int(row["selected_row"])
        fda_voxels = int(fda_support.get(selected_row, 0))
        row["survived_fda"] = selected_row in surviving
        row["fda_voxel_count"] = fda_voxels
        row["fda_support_volume_um3"] = float(fda_voxels * fda_voxel_volume_um3)
        native_volume = float(row["native_support_volume_um3"])
        row["fda_to_native_support_volume_ratio"] = (
            float(row["fda_support_volume_um3"]) / native_volume if native_volume > 0 else 0.0
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_path = out / "native_to_fda_survival.csv"
    fields = [
        "selected_row",
        "trial_id",
        "plane_index",
        "cluster_index",
        "published_plane_depth_um",
        "voxel_count",
        "native_support_volume_um3",
        "centroid_y",
        "centroid_x",
        "native_edge_distance_px",
        "survived_fda",
        "fda_voxel_count",
        "fda_support_volume_um3",
        "fda_to_native_support_volume_ratio",
    ]
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(native_rows)

    plane_summary = _group_summary(native_rows, "plane_index")
    plane_path = out / "survival_by_plane.csv"
    with plane_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["plane_index", "total", "survived", "lost", "survival_fraction"],
        )
        writer.writeheader()
        writer.writerows(plane_summary)

    total = len(native_rows)
    survived_n = sum(1 for row in native_rows if row["survived_fda"])
    summary = {
        "trial_id": TRIAL_ID,
        "native_selected_roi_count": total,
        "fda_surviving_selected_roi_count": survived_n,
        "fda_lost_selected_roi_count": total - survived_n,
        "survival_fraction": survived_n / total,
        "fda_voxel_volume_um3": fda_voxel_volume_um3,
        "native_support_volume_semantics": "source 2-D supervoxel pixel count multiplied by published 1.3um x 1.3um in-plane sampling and 9.0um plane spacing",
        "support_ratio_semantics": "FDA raster support volume / native source support volume; QC diagnostic only, not a conserved biological volume or registration accuracy score",
        "published_depth_semantics": "Fig3 preprocessing depth_a1 vector assigned to second LBM trial; source coordinate, not atlas identity",
        "survived_distributions": {
            field: _distribution(native_rows, field, True)
            for field in (
                "published_plane_depth_um",
                "voxel_count",
                "native_support_volume_um3",
                "centroid_y",
                "centroid_x",
                "native_edge_distance_px",
                "fda_voxel_count",
                "fda_support_volume_um3",
                "fda_to_native_support_volume_ratio",
            )
        },
        "lost_distributions": {
            field: _distribution(native_rows, field, False)
            for field in (
                "published_plane_depth_um",
                "voxel_count",
                "native_support_volume_um3",
                "centroid_y",
                "centroid_x",
                "native_edge_distance_px",
                "fda_voxel_count",
                "fda_support_volume_um3",
                "fda_to_native_support_volume_ratio",
            )
        },
        "row_output": str(row_path),
        "plane_output": str(plane_path),
    }
    summary_path = out / "registration_survival.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
