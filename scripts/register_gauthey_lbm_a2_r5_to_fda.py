#!/usr/bin/env python3
"""Register the reconstructed Gauthey a2_r5 native selected field into FDA.

This executable uses the published BIFROST CLI rather than reimplementing image
registration. It can stop after the experimentally meaningful native -> FDA
registration/label transform, or continue to neuropil aggregation when a painted
FDA atlas and region-name map are supplied.

Scientific source:
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
    centered_lbm_affine_in_fda_axes,
    load_atlas_region_names,
    physical_extent_microns,
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
    parser.add_argument("--fda-atlas-labels")
    parser.add_argument("--atlas-region-map")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bifrost-bin", default="bifrost")
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-registration", action="store_true")
    parser.add_argument("--skip-syn", action="store_true", help="Skip SyN pre-registration in BIFROST")
    parser.add_argument("--skip-synthmorph", action="store_true", help="Skip SynthMorph inference in BIFROST")
    parser.add_argument("--downsample-to", type=float, default=-1.0, help="Downsample to isotropic resolution in microns before registration")
    args = parser.parse_args()

    if bool(args.fda_atlas_labels) != bool(args.atlas_region_map):
        raise SystemExit(
            "--fda-atlas-labels and --atlas-region-map must either both be supplied or both omitted"
        )

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
    fda_img = nib.load(args.fda_template)
    if len(mean_img.shape) != 3:
        raise SystemExit(f"mean brain must be 3-D, got {mean_img.shape}")
    if len(fda_img.shape) != 3:
        raise SystemExit(f"FDA template must be 3-D, got {fda_img.shape}")

    # Gauthey's deposited mean-brain array is (y,x,z)=(226,512,27), while FDA's
    # world axes are (x,y,z).  Calibrate the published 1.3 x 1.3 x 9.0 um voxel
    # sizes, swap moving x/y into FDA world axes, and centre the physical volumes
    # before BIFROST estimates affine/non-linear terms.  Merely changing zooms on
    # the deposited diagonal affine yields ~294 x 666 um in FDA x/y and caused a
    # corner-only 374 -> 77 label survival artifact.
    calibrated_mean_nifti = output / "mean_brain_calibrated.nii"
    mean_header = mean_img.header.copy()
    mean_header.set_xyzt_units("micron", "unknown")
    mean_header.set_zooms((1.3, 1.3, 9.0))
    calibrated_affine = centered_lbm_affine_in_fda_axes(
        mean_img.shape,
        fda_img.shape,
        np.asarray(fda_img.affine),
        inplane_microns=1.3,
        axial_microns=9.0,
    )
    nib.save(
        nib.Nifti1Image(mean_img.get_fdata(dtype=np.float32), calibrated_affine, mean_header),
        calibrated_mean_nifti,
    )

    native_labels = np.load(args.native_label_volume)
    nifti_labels, permutation = reorder_native_labels_to_nifti_shape(
        native_labels, mean_img.shape
    )
    label_header = mean_header.copy()
    label_header.set_data_dtype(np.int32)
    nib.save(
        nib.Nifti1Image(nifti_labels.astype(np.int32, copy=False), calibrated_affine, label_header),
        native_label_nifti,
    )

    moving_extent = physical_extent_microns(mean_img.shape, calibrated_affine)
    fixed_extent = physical_extent_microns(fda_img.shape, np.asarray(fda_img.affine))
    # Because FDA NIfTI units are micron, nibabel reports its affine in microns.
    # Moving voxel-axis 0 is anatomical Y and axis 1 is X; compare anatomical
    # extents after that permutation rather than raw array-axis order.
    moving_anatomical_extent = (moving_extent[1], moving_extent[0], moving_extent[2])
    fixed_anatomical_extent = fixed_extent
    print(
        "staged physical extents (um): "
        f"moving xyz={moving_anatomical_extent}, fixed xyz={fixed_anatomical_extent}",
        flush=True,
    )

    if not args.skip_registration:
        bifrost_downsample = args.downsample_to
        if bifrost_downsample >= 0.1:
            # Convert microns to millimeters for ITK/ANTs internal spacing.
            bifrost_downsample /= 1000.0

        register_cmd = [
            bifrost_bin,
            "register",
            str(calibrated_mean_nifti.resolve()),
            str(Path(args.fda_template).resolve()),
            str(registration_dir.resolve()),
            "-v",
        ]
        if args.force:
            register_cmd.append("--force")
        if args.skip_syn:
            register_cmd.append("--skip_syn")
        if args.skip_synthmorph:
            register_cmd.append("--skip_synthmorph")
        if bifrost_downsample > 0:
            register_cmd.extend(["--downsample_to", str(bifrost_downsample)])
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
    if not (registration_dir / "transform.h5").exists():
        raise SystemExit(f"BIFROST transform missing: {registration_dir / 'transform.h5'}")

    selected_rows = np.load(args.selected_rows)
    selected_traces = np.load(args.selected_traces)
    if selected_rows.ndim != 1:
        raise SystemExit("selected rows must be one-dimensional")
    if selected_traces.ndim != 2 or selected_traces.shape[0] != selected_rows.size:
        raise SystemExit("selected traces must be [selected_roi,time] and align with selected rows")

    selected_img = nib.load(transformed_label_nifti)
    transformed_values = np.asarray(selected_img.dataobj)
    surviving_selected_labels = np.unique(transformed_values.astype(np.int64, copy=False))
    surviving_selected_labels = surviving_selected_labels[surviving_selected_labels > 0]
    expected_labels = {int(row) + 1 for row in selected_rows}
    surviving_labels = {int(value) for value in surviving_selected_labels}
    if not surviving_labels.issubset(expected_labels):
        bad = sorted(surviving_labels - expected_labels)[:10]
        raise SystemExit(f"label-preserving transform emitted unexpected selected-label values: {bad}")

    summary: dict[str, object] = {
        "mean_brain": str(Path(args.mean_brain)),
        "fda_template": str(Path(args.fda_template)),
        "native_to_nifti_axis_permutation": list(permutation),
        "staging_geometry": "moving array (y,x,z) mapped to FDA world (x,y,z) and physical centres aligned before BIFROST",
        "moving_anatomical_extent_microns_xyz": list(moving_anatomical_extent),
        "fixed_extent_microns_xyz": list(fixed_anatomical_extent),
        "calibrated_moving_affine": np.asarray(calibrated_affine).tolist(),
        "selected_roi_count": int(selected_rows.size),
        "selected_labels_surviving_fda_transform": len(surviving_labels),
        "selected_labels_lost_fda_transform": int(selected_rows.size - len(surviving_labels)),
        "registration_dir": str(registration_dir),
        "transform_h5": str(registration_dir / "transform.h5"),
        "transformed_selected_labels": str(transformed_label_nifti),
        "atlas_aggregation_executed": False,
        "identity_semantics": "BIFROST-transformed exact recovered selected supervoxels; not neuron identity",
    }

    if args.fda_atlas_labels is not None:
        atlas_img = nib.load(args.fda_atlas_labels)
        if selected_img.shape != atlas_img.shape:
            raise SystemExit(
                f"FDA selected-label shape {selected_img.shape} != atlas shape {atlas_img.shape}"
            )
        if not np.allclose(
            np.asarray(selected_img.affine), np.asarray(atlas_img.affine), atol=1e-6, rtol=1e-6
        ):
            raise SystemExit("FDA selected labels and atlas labels do not share the same affine")

        region_names = load_atlas_region_names(args.atlas_region_map)
        compiled = aggregate_transformed_selected_labels_to_regions(
            selected_rows,
            selected_traces,
            transformed_values,
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
            for assignment in compiled.assignments:
                writer.writerow(
                    [
                        assignment.selected_row,
                        assignment.atlas_region_id,
                        assignment.atlas_region,
                        assignment.voxel_count,
                        assignment.overlap_voxel_count,
                        f"{assignment.overlap_fraction:.17g}",
                    ]
                )

        region_csv = output / "region_functional.csv"
        with region_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(compiled.region_traces.unit_ids)
            writer.writerows(compiled.region_traces.traces)

        summary.update(
            {
                "atlas_aggregation_executed": True,
                "atlas_region_count": len(compiled.region_traces.unit_ids),
                "atlas_regions": list(compiled.region_traces.unit_ids),
                "atlas_assigned_selected_count": compiled.assigned_selected_count,
                "atlas_unassigned_selected_count": compiled.unassigned_selected_count,
                "atlas_minimum_overlap_fraction": compiled.minimum_overlap_fraction,
                "atlas_assignments": str(assignments_path),
                "region_functional": str(region_csv),
            }
        )

    summary_path = output / "gauthey_lbm_fda_registration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
