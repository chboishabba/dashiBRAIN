#!/usr/bin/env python3
"""Register FDA to the VFB JRC2018Unisex grid and transform selected labels.

Run inside the pinned BIFROST container.  The fixed image is the VFB-native
JRC2018Unisex NRRD used by the 46 painted-domain volumes, so the transformed
selected labels and all painted domains share one exact raster grid.

The VFB NRRD stores physical coordinates in microns.  We preserve its full
``space directions`` and ``space origin`` when staging a NIfTI explicitly marked
as micron units before calling BIFROST, preventing ITK unit reinterpretation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

VFB_JRC_SHAPE = (1210, 566, 174)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _require_imaging():
    try:
        import nibabel as nib
        import nrrd
    except ImportError as exc:
        raise SystemExit("nibabel and pynrrd are required; run inside the BIFROST container") from exc
    return nib, nrrd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fda-template", required=True)
    parser.add_argument("--selected-labels-fda", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--jrc-template-nrrd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bifrost-bin", default="bifrost")
    parser.add_argument(
        "--downsample-to",
        type=float,
        default=2.0,
        help="isotropic registration resolution in microns",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.downsample_to <= 0:
        raise SystemExit("--downsample-to must be positive microns")
    bifrost = shutil.which(args.bifrost_bin)
    if bifrost is None:
        raise SystemExit(f"BIFROST executable not found: {args.bifrost_bin}")
    nib, nrrd = _require_imaging()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jrc_nifti = out / "VFB_JRC2018Unisex_micron_calibrated.nii"
    registration_dir = out / "fda_to_vfb_jrc2018unisex"
    selected_jrc = out / "selected_labels_vfb_jrc2018unisex.nii"

    jrc_data, jrc_header = nrrd.read(args.jrc_template_nrrd)
    jrc_data = np.asarray(jrc_data)
    if jrc_data.shape != VFB_JRC_SHAPE:
        raise SystemExit(
            f"unexpected VFB JRC2018Unisex template shape: {jrc_data.shape}; "
            f"expected {VFB_JRC_SHAPE}"
        )

    directions = np.asarray(jrc_header.get("space directions"), dtype=float)
    if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
        raise SystemExit("VFB JRC2018Unisex NRRD lacks finite 3x3 space directions")
    zooms = np.linalg.norm(directions, axis=1)
    if np.any(zooms <= 0):
        raise SystemExit(f"invalid VFB JRC2018Unisex voxel spacing: {zooms}")

    origin_raw = jrc_header.get("space origin")
    if origin_raw is None:
        origin = np.zeros(3, dtype=float)
    else:
        origin = np.asarray(origin_raw, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise SystemExit(f"invalid VFB JRC2018Unisex space origin: {origin_raw!r}")

    # VFB viewer raster should cover an adult fly brain, not a millimetre-scale
    # or metre-scale object.  This catches another hidden unit mismatch early.
    extents_um = np.asarray(VFB_JRC_SHAPE, dtype=float) * zooms
    if np.any(extents_um < 100.0) or np.any(extents_um > 1000.0):
        raise SystemExit(
            f"implausible VFB JRC2018Unisex physical extents in microns: {extents_um}"
        )

    affine = np.array(
        [
            [zooms[0], 0.0, 0.0, origin[0]],
            [0.0, 0.0, zooms[2], origin[2]],
            [0.0, zooms[1], 0.0, origin[1]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    jrc_img = nib.Nifti1Image(jrc_data, affine)
    jrc_img.header.set_data_dtype(jrc_data.dtype)
    jrc_img.header.set_xyzt_units("micron", "unknown")
    jrc_img.header.set_zooms((float(zooms[0]), float(zooms[1]), float(zooms[2])))
    nib.save(jrc_img, jrc_nifti)
    del jrc_data, jrc_img

    downsample_mm = args.downsample_to / 1000.0
    register_cmd = [
        bifrost,
        "register",
        str(Path(args.fda_template).resolve()),
        str(jrc_nifti.resolve()),
        str(registration_dir.resolve()),
        "--downsample_to",
        f"{downsample_mm:.9g}",
        "-v",
    ]
    if args.force:
        register_cmd.append("--force")
    _run(register_cmd)

    _run(
        [
            bifrost,
            "transform",
            str(registration_dir.resolve()),
            str(Path(args.selected_labels_fda).resolve()),
            "--label_image",
            "--result_name",
            str(selected_jrc.resolve()),
            "-v",
        ]
    )

    selected_rows = np.load(args.selected_rows).astype(np.int64, copy=False)
    img = nib.load(selected_jrc)
    data = np.asarray(img.dataobj)
    if data.shape != VFB_JRC_SHAPE:
        import ants

        resampled = ants.resample_image_to_target(
            ants.image_read(str(selected_jrc)),
            ants.image_read(str(jrc_nifti)),
            interp_type="genericLabel",
        )
        ants.image_write(resampled, str(selected_jrc))
        img = nib.load(selected_jrc)
        data = np.asarray(img.dataobj)
    if data.shape != VFB_JRC_SHAPE:
        raise SystemExit(
            f"transformed selected labels landed on {data.shape}, expected VFB grid {VFB_JRC_SHAPE}"
        )
    observed = {int(v) for v in np.unique(data) if int(v) != 0}
    allowed = {int(row) + 1 for row in selected_rows}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise SystemExit(f"JRC transform manufactured unexpected selected labels: {unexpected[:10]}")
    surviving = {value - 1 for value in observed}

    summary = {
        "moving_template": str(Path(args.fda_template)),
        "fixed_template": str(Path(args.jrc_template_nrrd)),
        "fixed_template_vfb_id": "VFB_00101567",
        "fixed_grid_shape": list(VFB_JRC_SHAPE),
        "fixed_voxel_spacing_microns": [float(v) for v in zooms],
        "fixed_space_origin_microns": [float(v) for v in origin],
        "jrc_staged_nifti": str(jrc_nifti),
        "downsample_microns": args.downsample_to,
        "downsample_itk_mm": downsample_mm,
        "selected_roi_count_input": int(selected_rows.size),
        "selected_labels_surviving_jrc_transform": len(surviving),
        "selected_labels_lost_jrc_transform": int(selected_rows.size - len(surviving)),
        "registration_dir": str(registration_dir),
        "transform_h5": str(registration_dir / "transform.h5"),
        "selected_labels_jrc2018unisex": str(selected_jrc),
        "identity_semantics": (
            "FDA-space exact selected labels transformed by BIFROST onto the exact "
            "VFB JRC2018Unisex painted-domain grid; not neuron identity"
        ),
    }
    (out / "gauthey_fda_to_jrc2018.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
