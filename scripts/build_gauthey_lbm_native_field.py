#!/usr/bin/env python3
"""Build a compact same-trial Gauthey LBM native functional field."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_native_field import compile_native_selected_field_from_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", default="04032024_6f_a2_r5")
    parser.add_argument(
        "--deposited-selected",
        default="data/gauthey_lbm/dffs_audio_LB_corr_top05_all.pkl",
    )
    parser.add_argument(
        "--identities",
        default="data/gauthey_lbm/reconstruction_a2_r5/gauthey_lbm_selected_identities_partial.csv",
    )
    parser.add_argument(
        "--segmentation",
        default="data/gauthey_lbm/04032024_6f_a2_r5_n2000_labels.h5",
    )
    parser.add_argument("--output-dir", default="data/gauthey_lbm/native_field_a2_r5")
    args = parser.parse_args()

    field = compile_native_selected_field_from_files(
        args.deposited_selected,
        args.identities,
        args.segmentation,
        trial_id=args.trial,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "selected_rows.npy", field.selected_rows)
    np.save(out / "selected_traces_roi_by_time.npy", field.traces_roi_by_time)
    np.save(out / "selected_label_volume_plane_y_x.npy", field.selected_label_volume)

    roi_path = out / "selected_native_rois.csv"
    with roi_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "selected_row",
            "pooled_source_row",
            "trial_id",
            "plane_index",
            "cluster_index",
            "correlation",
            "voxel_count",
            "centroid_y",
            "centroid_x",
        ])
        for roi in field.rois:
            writer.writerow([
                roi.selected_row,
                roi.pooled_source_row,
                roi.trial_id,
                roi.plane_index,
                roi.cluster_index,
                f"{roi.correlation:.17g}",
                roi.voxel_count,
                f"{roi.centroid_y:.17g}",
                f"{roi.centroid_x:.17g}",
            ])

    summary = {
        "trial_id": field.trial_id,
        "recovered_selected_roi_count": int(len(field.rois)),
        "timepoints": int(field.traces_roi_by_time.shape[1]),
        "segmentation_shape_plane_y_x": list(field.selected_label_volume.shape),
        "occupied_voxel_count": int(np.count_nonzero(field.selected_label_volume)),
        "selected_rows_output": str(out / "selected_rows.npy"),
        "selected_traces_output": str(out / "selected_traces_roi_by_time.npy"),
        "selected_label_volume_output": str(out / "selected_label_volume_plane_y_x.npy"),
        "selected_native_roi_output": str(roi_path),
        "coordinate_semantics": "native plane/y/x supervoxel geometry; not common-atlas coordinates",
    }
    summary_path = out / "gauthey_lbm_native_field.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
