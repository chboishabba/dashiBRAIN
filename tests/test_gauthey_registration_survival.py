from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.analyze_gauthey_registration_survival import (
    DEPTH_A1_UM,
    TRIAL_ID,
    _observed_selected_rows,
    _read_native_rois,
)


def _write_rois(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selected_row",
                "pooled_source_row",
                "trial_id",
                "plane_index",
                "cluster_index",
                "correlation",
                "voxel_count",
                "centroid_y",
                "centroid_x",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selected_row": 4,
                "pooled_source_row": 68633,
                "trial_id": TRIAL_ID,
                "plane_index": 7,
                "cluster_index": 633,
                "correlation": 0.06972,
                "voxel_count": 20,
                "centroid_y": 10.0,
                "centroid_x": 20.0,
            }
        )


def test_native_roi_qc_attaches_published_depth_and_edge_distance(tmp_path):
    path = tmp_path / "native.csv"
    _write_rois(path)
    rows = _read_native_rois(path)
    assert len(rows) == 1
    assert rows[0]["published_plane_depth_um"] == float(DEPTH_A1_UM[7])
    assert rows[0]["native_edge_distance_px"] == 10.0


def test_observed_selected_rows_decode_label_values_as_selected_row_plus_one(tmp_path):
    arr = np.zeros((4, 3, 2), dtype=np.int16)
    arr[0, 0, 0] = 5   # selected row 4
    arr[1, 0, 0] = 22  # selected row 21
    path = tmp_path / "labels.nii"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), path)
    assert _observed_selected_rows(path) == {4, 21}
