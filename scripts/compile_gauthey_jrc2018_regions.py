#!/usr/bin/env python3
"""Compile JRC2018-space selected labels into region-functional traces.

Requires the selected-supervoxel label image already transformed into
JRC2018Unisex coordinates and the 46 VFB painted-domain NRRDs fetched by
``fetch_vfb_jrc2018_painted_domains.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dashi.analysis.jrc2018_painted_overlap import compile_selected_labels_against_painted_domains


def _require_imaging():
    try:
        import nibabel as nib
        import nrrd
    except ImportError as exc:
        raise SystemExit("nibabel and pynrrd are required; run in the BIFROST container") from exc
    return nib, nrrd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-labels-jrc", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--selected-traces", required=True)
    parser.add_argument("--painted-domain-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.5)
    args = parser.parse_args()

    nib, nrrd = _require_imaging()
    selected_img = nib.load(args.selected_labels_jrc)
    selected_labels = np.asarray(selected_img.dataobj)
    selected_rows = np.load(args.selected_rows)
    selected_traces = np.load(args.selected_traces)

    painted: dict[str, np.ndarray] = {}
    with Path(args.painted_domain_manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            region = str(row["region"])
            data, _header = nrrd.read(str(row["local_path"]))
            arr = np.asarray(data)
            if arr.shape != selected_labels.shape:
                raise SystemExit(
                    f"painted domain {region} shape {arr.shape} != selected label shape {selected_labels.shape}"
                )
            painted[region] = arr

    compiled = compile_selected_labels_against_painted_domains(
        selected_rows,
        selected_traces,
        selected_labels,
        painted,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    assignments = out / "selected_roi_jrc2018_regions.csv"
    with assignments.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_row", "region", "overlap_voxel_count", "selected_voxel_count", "overlap_fraction"])
        for item in compiled.assignments:
            writer.writerow([item.selected_row, item.region, item.overlap_voxel_count, item.selected_voxel_count, f"{item.overlap_fraction:.17g}"])

    ambiguities = out / "selected_roi_jrc2018_ambiguities.csv"
    with ambiguities.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_row", "regions", "overlap_fraction"])
        for item in compiled.ambiguities:
            writer.writerow([item.selected_row, "|".join(item.regions), f"{item.overlap_fraction:.17g}"])

    region_functional = out / "region_functional.csv"
    with region_functional.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", *compiled.region_traces.unit_ids])
        for i, values in enumerate(compiled.region_traces.traces):
            writer.writerow([i, *[f"{float(v):.17g}" for v in values]])

    summary = {
        "selected_roi_count": int(selected_rows.size),
        "assigned_selected_roi_count": compiled.assigned_selected_count,
        "ambiguous_selected_roi_count": compiled.ambiguous_selected_count,
        "unassigned_selected_roi_count": compiled.unassigned_selected_count,
        "region_count": len(compiled.region_traces.unit_ids),
        "regions": list(compiled.region_traces.unit_ids),
        "timepoints": int(compiled.region_traces.traces.shape[0]),
        "minimum_overlap_fraction": compiled.minimum_overlap_fraction,
        "contains_ammc": "AMMC" in compiled.region_traces.unit_ids,
        "contains_wed": "WED" in compiled.region_traces.unit_ids,
        "region_functional": str(region_functional),
        "identity_semantics": "VFB JRC2018Unisex painted-domain overlap of BIFROST-transformed exact selected supervoxels; not neuron identity",
    }
    (out / "gauthey_jrc2018_regions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
