#!/usr/bin/env python3
"""Register FDA to JRC2018Unisex and transform the surviving selected labels.

Run inside the pinned BIFROST container.  The JRC2018 source template is an NRRD
whose axis spacings are expressed in microns.  We stage it as a NIfTI explicitly
marked as micron units before calling BIFROST, preventing ITK from interpreting
0.38 as millimetres.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


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
    parser.add_argument("--downsample-to", type=float, default=2.0, help="isotropic registration resolution in microns")
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
    jrc_nifti = out / "JRC2018Unisex_micron_calibrated.nii"
    registration_dir = out / "fda_to_jrc2018unisex"
    selected_jrc = out / "selected_labels_jrc2018unisex.nii"

    jrc_data, jrc_header = nrrd.read(args.jrc_template_nrrd)
    jrc_data = np.asarray(jrc_data)
    if jrc_data.shape != (1652, 773, 456):
        raise SystemExit(f"unexpected JRC2018Unisex template shape: {jrc_data.shape}")
    directions = np.asarray(jrc_header.get("space directions"), dtype=float)
    if directions.shape != (3, 3):
        raise SystemExit("JRC2018Unisex NRRD lacks 3x3 space directions")
    zooms = np.linalg.norm(directions, axis=1)
    if not np.allclose(zooms, (0.38, 0.38, 0.38), atol=1e-9, rtol=1e-9):
        raise SystemExit(f"unexpected JRC2018Unisex voxel spacing: {zooms}")

    affine = np.eye(4, dtype=float)
    affine[:3, :3] = directions.T
    header = nib.Nifti1Header()
    header.set_data_dtype(jrc_data.dtype)
    header.set_xyzt_units("micron", "unknown")
    header.set_zooms(tuple(float(v) for v in zooms))
    nib.save(nib.Nifti1Image(jrc_data, affine, header), jrc_nifti)
    del jrc_data

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

    _run([
        bifrost,
        "transform",
        str(registration_dir.resolve()),
        str(Path(args.selected_labels_fda).resolve()),
        "--label_image",
        "--result_name",
        str(selected_jrc.resolve()),
        "-v",
    ])

    selected_rows = np.load(args.selected_rows).astype(np.int64, copy=False)
    img = nib.load(selected_jrc)
    data = np.asarray(img.dataobj)
    observed = {int(v) for v in np.unique(data) if int(v) != 0}
    allowed = {int(row) + 1 for row in selected_rows}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise SystemExit(f"JRC transform manufactured unexpected selected labels: {unexpected[:10]}")
    surviving = {value - 1 for value in observed}

    summary = {
        "moving_template": str(Path(args.fda_template)),
        "fixed_template": str(Path(args.jrc_template_nrrd)),
        "jrc_staged_nifti": str(jrc_nifti),
        "downsample_microns": args.downsample_to,
        "downsample_itk_mm": downsample_mm,
        "selected_roi_count_input": int(selected_rows.size),
        "selected_labels_surviving_jrc_transform": len(surviving),
        "selected_labels_lost_jrc_transform": int(selected_rows.size - len(surviving)),
        "registration_dir": str(registration_dir),
        "transform_h5": str(registration_dir / "transform.h5"),
        "selected_labels_jrc2018unisex": str(selected_jrc),
        "identity_semantics": "FDA-space exact selected labels transformed by BIFROST into JRC2018Unisex; not neuron identity",
    }
    (out / "gauthey_fda_to_jrc2018.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
