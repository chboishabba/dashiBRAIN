#!/usr/bin/env python3
"""Build all trial-native Gauthey LBM functional fields and readiness receipts.

This stops deliberately before common-atlas registration for trials without a
same-trial anatomical receipt.  Native field materialization and atlas
registration are separate gates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import load_deposited_lbm_selected
from dashi.analysis.gauthey_lbm_native_field import (
    compile_native_selected_field,
    load_gauthey_lbm_segmentation,
    load_recovered_identity_csv,
)
from dashi.analysis.gauthey_trial_materialization import (
    canonical_trial_specs,
    default_mean_brain_paths,
    default_segmentation_paths,
    identities_by_trial,
    trial_readiness,
)


def _write_field(out: Path, field) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "selected_rows.npy", field.selected_rows)
    np.save(out / "selected_traces_roi_by_time.npy", field.traces_roi_by_time)
    np.save(out / "selected_label_volume_plane_y_x.npy", field.selected_label_volume)

    roi_csv = out / "selected_native_rois.csv"
    with roi_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selected_row",
                "pooled_source_row",
                "trial_id",
                "plane_index",
                "cluster_index",
                "correlation",
                "voxel_count",
                "centroid_y",
                "centroid_x",
            ]
        )
        for roi in field.rois:
            writer.writerow(
                [
                    roi.selected_row,
                    roi.pooled_source_row,
                    roi.trial_id,
                    roi.plane_index,
                    roi.cluster_index,
                    f"{roi.correlation:.17g}",
                    roi.voxel_count,
                    f"{roi.centroid_y:.17g}",
                    f"{roi.centroid_x:.17g}",
                ]
            )

    summary = {
        "trial_id": field.trial_id,
        "selected_roi_count": int(field.selected_rows.size),
        "timepoints": int(field.traces_roi_by_time.shape[1]),
        "occupied_voxel_count": int(np.count_nonzero(field.selected_label_volume)),
        "selected_rows": str(out / "selected_rows.npy"),
        "selected_traces": str(out / "selected_traces_roi_by_time.npy"),
        "selected_label_volume": str(out / "selected_label_volume_plane_y_x.npy"),
        "selected_native_rois": str(roi_csv),
        "coordinate_semantics": "trial-native plane/y/x supervoxel geometry; no common-atlas identity yet",
    }
    (out / "gauthey_lbm_native_field.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deposited-selected", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--segmentation-dir", required=True)
    parser.add_argument("--mean-brain-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-all-native",
        action="store_true",
        help="fail unless every trial with selected ROIs can build a native field",
    )
    args = parser.parse_args()

    selected = load_deposited_lbm_selected(args.deposited_selected)
    identities = load_recovered_identity_csv(args.identities)
    grouped = identities_by_trial(identities)
    segmentation_paths = default_segmentation_paths(args.segmentation_dir)
    mean_brain_paths = (
        default_mean_brain_paths(args.mean_brain_dir)
        if args.mean_brain_dir is not None
        else {}
    )
    readiness = trial_readiness(
        identities,
        segmentation_paths=segmentation_paths,
        mean_brain_paths=mean_brain_paths,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []

    readiness_by_trial = {item.trial_id: item for item in readiness}
    for spec in canonical_trial_specs():
        state = readiness_by_trial[spec.trial_id]
        trial_out = out_root / spec.trial_id
        receipt: dict[str, object] = {
            "trial_id": spec.trial_id,
            "discovery_recording": spec.discovery_recording,
            "selected_roi_count": state.selected_roi_count,
            "segmentation": str(segmentation_paths[spec.trial_id]),
            "segmentation_present": state.segmentation_present,
            "same_trial_mean_brain": (
                str(mean_brain_paths[spec.trial_id])
                if spec.trial_id in mean_brain_paths
                else None
            ),
            "same_trial_mean_brain_present": state.same_trial_mean_brain_present,
            "native_field_materialized": False,
            "common_atlas_registration_ready": state.common_atlas_registration_ready,
        }

        if state.selected_roi_count == 0:
            receipt["native_field_note"] = "no rows from the global top-0.5% selection came from this trial"
        elif not state.segmentation_present:
            receipt["native_field_note"] = "same-trial segmentation is missing"
        else:
            segmentation = load_gauthey_lbm_segmentation(segmentation_paths[spec.trial_id])
            field = compile_native_selected_field(
                selected,
                grouped[spec.trial_id],
                segmentation,
                trial_id=spec.trial_id,
            )
            receipt["native_field"] = _write_field(trial_out, field)
            receipt["native_field_materialized"] = True
            receipt["native_field_note"] = "trial-native field materialized; common-atlas promotion still requires same-trial anatomical registration"
        receipts.append(receipt)

    payload = {
        "status": "gauthey_six_trial_native_materialization",
        "trial_count": len(receipts),
        "global_selected_roi_count": len(identities),
        "materialized_native_trial_count": sum(bool(r["native_field_materialized"]) for r in receipts),
        "common_atlas_registration_ready_count": sum(bool(r["common_atlas_registration_ready"]) for r in receipts),
        "trials": receipts,
        "firewalls": {
            "native_field_implies_common_atlas_identity": False,
            "other_trial_mean_brain_may_be_reused": False,
            "missing_same_trial_anatomy_means_trial_did_not_exist": False,
            "pooled_selected_carrier_is_independent_replication": False,
        },
    }
    receipt_path = out_root / "gauthey_trial_materialization.json"
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.require_all_native:
        blocked = [
            r["trial_id"]
            for r in receipts
            if int(r["selected_roi_count"]) > 0 and not bool(r["native_field_materialized"])
        ]
        if blocked:
            raise SystemExit("native materialization blocked for: " + ", ".join(blocked))


if __name__ == "__main__":
    main()
