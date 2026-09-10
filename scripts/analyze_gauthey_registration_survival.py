#!/usr/bin/env python3
"""Analyze which exact Gauthey a2_r5 selected supervoxels survive registration.

This is experiment-facing QC for the native -> FDA hop.  It joins the exact
native ROI geometry with the transformed selected-label image and reports whether
label loss is associated with source plane/depth, native centroid, ROI size, or
native field-of-view boundaries.

The published Fig. 3 preprocessing assigns the second LBM trial
``04032024_6f_a2_r5`` the ``depth_a1`` plane-depth vector.  Those depths are
retained here as source coordinates; they are not atlas identities.
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
            rows.append(
                {
                    "selected_row": int(raw["selected_row"]),
                    "trial_id": trial_id,
                    "plane_index": plane,
                    "cluster_index": int(raw["cluster_index"]),
                    "published_plane_depth_um": float(DEPTH_A1_UM[plane]),
                    "voxel_count": int(raw["voxel_count"]),
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


def _observed_selected_rows(path: Path) -> set[int]:
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    values = {int(v) for v in np.unique(data) if int(v) != 0}
    return {value - 1 for value in values}


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
    surviving = _observed_selected_rows(Path(args.selected_labels_fda))
    declared = {int(row["selected_row"]) for row in native_rows}
    unexpected = sorted(surviving - declared)
    if unexpected:
        raise SystemExit(f"FDA image contains labels absent from native ROI table: {unexpected[:10]}")

    for row in native_rows:
        row["survived_fda"] = int(row["selected_row"]) in surviving

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
        "centroid_y",
        "centroid_x",
        "native_edge_distance_px",
        "survived_fda",
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
        "published_depth_semantics": "Fig3 preprocessing depth_a1 vector assigned to second LBM trial; source coordinate, not atlas identity",
        "survived_distributions": {
            field: _distribution(native_rows, field, True)
            for field in (
                "published_plane_depth_um",
                "voxel_count",
                "centroid_y",
                "centroid_x",
                "native_edge_distance_px",
            )
        },
        "lost_distributions": {
            field: _distribution(native_rows, field, False)
            for field in (
                "published_plane_depth_um",
                "voxel_count",
                "centroid_y",
                "centroid_x",
                "native_edge_distance_px",
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
