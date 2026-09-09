#!/usr/bin/env python3
"""Register the reconstructed Gauthey a2_r5 native selected field into FDA.

This executable uses the published BIFROST CLI rather than reimplementing image
registration.  It:

1. loads the deposited a2_r5 mean-brain NIfTI;
2. converts the sparse native selected-supervoxel label volume onto that NIfTI's
   voxel-axis order while preserving its affine/header;
3. runs ``bifrost register`` with the mean brain as moving and FDA as fixed;
4. runs ``bifrost transform --label_image`` on the sparse selected labels;
5. overlaps the transformed selected labels against an FDA neuropil label image;
6. emits time x region activity ready for the generic MaleCNS benchmark.

The BIFROST paper/source is:
Bella E. Brezovec, Andrew B. Berger, Yukun A. Hao et al.,
"BIFROST: A method for registering diverse imaging datasets of the Drosophila
brain", PNAS 121(47):e2322687121 (2024), DOI 10.1073/pnas.2322687121.
Dataset DOI 10.5061/dryad.8pk0p2nx1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from dashi.analysis.gauthey_lbm_fda_registration import (
    aggregate_transformed_selected_labels_to_regions,
    load_atlas_region_names,
    reorder_native_labels_to_nifti_shape,
)


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit(
            "nibabel is required for the FDA registration runner; run this from the BIFROST environment"
        ) from exc
    return nib


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-brain", required=True)
    parser.add_argument("--native-label-volume", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--fda-template", required=True)
    parser.add_argument("--fda-atlas-labels", required=True)
    parser.add_argument("--atlas-region-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bifrost-bin", default="bifrost")
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-registration", action="store_true")
    args = parser.parse_args()

    nib = _require_nibabel()
    bifrost_bin = shutil.which(args.bifrost_bin)
    if bifrost_bin is None and not args.skip_registration:
        raise SystemExit(f"BIFROST executable not found: {args.bifrost_bin}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registration_dir = output / "mean_brain_to_fda"
    native_label_nifti = output / "selected_labels_native.nii"
    transformed_label_nifti = output / "selected_labels_fda.nii"

    mean_img = nib.load(args.mean_brain)
    if len(mean_img.shape) != 3:
        raise SystemExit(f"mean brain must be 3-D, got {mean_img.shape}")
    native_labels = np.load(args.native_label_volume)
    nifti_labels, permutation = reorder_native_labels_to_nifti_shape(
        native_labels, mean_img.shape
    )
    label_header = mean_img.header.copy()
    label_header.set_data_dtype(np.int32)
    nib.save(
        nib.Nifti1Image(nifti_labels.astype(np.int32, copy=False), mean_img.affine, label_header),
        native_label_nifti,
    )

    if not args.skip_registration:
        register_cmd = [
            bifrost_bin,
            "register",
            str(Path(args.mean_brain).resolve()),
            str(Path(args.fda_template).resolve()),
            str(registration_dir.resolve()),
            "-v",
        ]
        if args.force:
            register_cmd.append("--force")
        _run(register_cmd)

        transform_cmd = [
            bifrost_bin,
            "transform",
            str(registration_dir.resolve()),
            str(native_label_nifti.resolve()),
            "--label_image",
            "--result_name",
            str(transformed_label_nifti.resolve()),
            "-v",
        ]
        _run(transform_cmd)

    if not transformed_label_nifti.exists():
        raise SystemExit(
            f"transformed selected-label image not found: {transformed_label_nifti}"
        )

    selected_img = nib.load(transformed_label_nifti)
    atlas_img = nib.load(args.fda_atlas_labels)
    if selected_img.shape != atlas_img.shape:
        raise SystemExit(
            f"FDA selected-label shape {selected_img.shape} != atlas shape {atlas_img.shape}"
        )
    selected_affine = np.asarray(selected_img.affine)
    atlas_affine = np.asarray(atlas_img.affine)
    if not np.allclose(selected_affine, atlas_affine, atol=1e-6, rtol=1e-6):
        raise SystemExit("FDA selected labels and atlas labels do not share the same affine")

    selected_rows = np.load(args.selected_rows)
    selected_traces = np.load(args.selected_traces)
    region_names = load_atlas_region_names(args.atlas_region_map)
    compiled = aggregate_transformed_selected_labels_to_regions(
        selected_rows,
        selected_traces,
        np.asarray(selected_img.dataobj),
        np.asarray(atlas_img.dataobj),
        region_names,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )

    assignments_path = output / "selected_roi_fda_regions.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selected_row",
                "atlas_region_id",
                "atlas_region",
                "voxel_count",
                "overlap_voxel_count",
                "overlap_fraction",
            ]
        )
        for item in compiled.assignments:
            writer.writerow(
                [
                    item.selected_row,
                    item.atlas_region_id,
                    item.atlas_region,
                    item.voxel_count,
                    item.overlap_voxel_count,
                    f"{item.overlap_fraction:.17g}",
                ]
            )

    region_trace_path = output / "region_functional.csv"
    with region_trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", *compiled.region_traces.unit_ids])
        for i, values in enumerate(compiled.region_traces.traces):
            writer.writerow([i, *[f"{float(v):.17g}" for v in values]])

    summary = {
        "mean_brain": str(Path(args.mean_brain)),
        "fda_template": str(Path(args.fda_template)),
        "fda_atlas_labels": str(Path(args.fda_atlas_labels)),
        "native_to_nifti_axis_permutation": list(permutation),
        "selected_roi_count": int(selected_rows.size),
        "assigned_selected_roi_count": int(compiled.assigned_selected_count),
        "unassigned_selected_roi_count": int(compiled.unassigned_selected_count),
        "region_count": len(compiled.region_traces.unit_ids),
        "timepoints": int(compiled.region_traces.traces.shape[0]),
        "minimum_overlap_fraction": float(compiled.minimum_overlap_fraction),
        "registration_dir": str(registration_dir),
        "transformed_selected_labels": str(transformed_label_nifti),
        "region_assignments": str(assignments_path),
        "region_functional": str(region_trace_path),
        "identity_semantics": "FDA neuropil overlap of exact recovered selected supervoxels; not neuron identity",
    }
    summary_path = output / "gauthey_lbm_fda_registration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
