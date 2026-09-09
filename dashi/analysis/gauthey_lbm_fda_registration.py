"""Native Gauthey LBM selected-supervoxel registration helpers.

This module owns only the array-level seam around a published BIFROST
registration.  The BIFROST executable computes the moving native mean-brain ->
fixed FDA transform and applies that same transform to the selected-supervoxel
label image with label-preserving interpolation.  Once both selected labels and
neuropil labels are in FDA space, this module performs literal voxel overlap and
aggregates the measured traces onto atlas regions.

No neuron identity is inferred.

Scientific source:
Bella E. Brezovec, Andrew B. Berger, Yukun A. Hao et al.,
"BIFROST: A method for registering diverse imaging datasets of the Drosophila
brain", PNAS 121(47):e2322687121 (2024), DOI 10.1073/pnas.2322687121.
Dataset DOI 10.5061/dryad.8pk0p2nx1.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Mapping, Sequence
import csv
import json

import numpy as np

from dashi.io.functional_imaging_loader import FunctionalTraceTable


@dataclass(frozen=True)
class FDAAtlasAssignment:
    selected_row: int
    atlas_region_id: int
    atlas_region: str
    voxel_count: int
    overlap_voxel_count: int
    overlap_fraction: float


@dataclass(frozen=True)
class FDAAtlasCompilation:
    region_traces: FunctionalTraceTable
    assignments: tuple[FDAAtlasAssignment, ...]
    assigned_selected_count: int
    unassigned_selected_count: int
    minimum_overlap_fraction: float


def reorder_native_labels_to_nifti_shape(
    labels_plane_y_x: np.ndarray,
    target_shape: Sequence[int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Permute ``[plane,y,x]`` labels onto a NIfTI voxel-axis shape.

    Gauthey's functional label carrier is normalized internally to
    ``[plane,y,x]``.  The deposited mean-brain NIfTI may expose the same three
    physical extents in another axis order.  Because the extents 27, 226 and 512
    are distinct for the LBM carrier, the matching permutation is unique.  Any
    non-permutation mismatch fails closed rather than inventing resampling.
    """
    labels = np.asarray(labels_plane_y_x)
    if labels.ndim != 3:
        raise ValueError("native selected labels must be 3-D")
    target = tuple(int(v) for v in target_shape)
    if len(target) != 3:
        raise ValueError("target NIfTI shape must be three-dimensional")

    matches: list[tuple[int, int, int]] = []
    for perm in permutations(range(3)):
        if tuple(labels.shape[i] for i in perm) == target:
            matches.append(tuple(int(i) for i in perm))
    if len(matches) != 1:
        raise ValueError(
            f"native label shape {labels.shape} does not map uniquely to NIfTI shape {target}"
        )
    perm = matches[0]
    return np.transpose(labels, perm), perm


def load_atlas_region_names(path: str | Path) -> dict[int, str]:
    """Load atlas label names from JSON or a two-column CSV.

    JSON accepts ``{"101": "AMMC", ...}``. CSV accepts a label column named
    ``label_id``, ``label`` or ``id`` and a name column named ``region``, ``name``
    or ``neuropil``.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("atlas-region JSON must be an object mapping IDs to names")
        mapping = {int(k): str(v) for k, v in payload.items() if str(v).strip()}
    else:
        with p.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            id_col = next((c for c in ("label_id", "label", "id") if c in fields), None)
            name_col = next((c for c in ("region", "name", "neuropil") if c in fields), None)
            if id_col is None or name_col is None:
                raise ValueError("atlas-region CSV needs label_id/label/id and region/name/neuropil")
            mapping = {}
            for row in reader:
                name = str(row[name_col]).strip()
                if name:
                    mapping[int(row[id_col])] = name
    if not mapping:
        raise ValueError("atlas-region mapping is empty")
    return mapping


def aggregate_transformed_selected_labels_to_regions(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_fda: np.ndarray,
    atlas_labels_fda: np.ndarray,
    atlas_region_names: Mapping[int, str],
    *,
    minimum_overlap_fraction: float = 0.5,
) -> FDAAtlasCompilation:
    """Aggregate measured selected traces using literal FDA-space voxel overlap.

    ``selected_labels_fda`` is the label-preserving BIFROST transform of the
    sparse native selected-supervoxel label image.  Its values remain
    ``selected_row + 1``.  ``atlas_labels_fda`` must already be on the identical
    FDA voxel grid.
    """
    rows = np.asarray(selected_rows, dtype=np.int64)
    traces = np.asarray(traces_roi_by_time, dtype=float)
    selected_labels = np.asarray(selected_labels_fda)
    atlas_labels = np.asarray(atlas_labels_fda)

    if rows.ndim != 1:
        raise ValueError("selected_rows must be one-dimensional")
    if traces.ndim != 2 or traces.shape[0] != rows.size:
        raise ValueError("traces_roi_by_time must be [selected_roi,time]")
    if selected_labels.shape != atlas_labels.shape or selected_labels.ndim != 3:
        raise ValueError("selected and atlas FDA label images must share one 3-D grid")
    if not (0.0 < minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in (0,1]")

    assignments: list[FDAAtlasAssignment] = []
    by_region_trace_indices: dict[str, list[int]] = {}

    for trace_index, selected_row in enumerate(rows):
        mask = selected_labels == (int(selected_row) + 1)
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            continue
        vals = atlas_labels[mask].astype(int, copy=False)
        labels, counts = np.unique(vals, return_counts=True)
        valid = [
            (int(label), int(count))
            for label, count in zip(labels, counts)
            if int(label) in atlas_region_names
        ]
        if not valid:
            continue
        region_id, overlap = max(valid, key=lambda pair: pair[1])
        fraction = overlap / voxel_count
        if fraction < minimum_overlap_fraction:
            continue
        region = str(atlas_region_names[region_id])
        assignments.append(
            FDAAtlasAssignment(
                selected_row=int(selected_row),
                atlas_region_id=region_id,
                atlas_region=region,
                voxel_count=voxel_count,
                overlap_voxel_count=overlap,
                overlap_fraction=float(fraction),
            )
        )
        by_region_trace_indices.setdefault(region, []).append(trace_index)

    if not by_region_trace_indices:
        raise ValueError("no transformed selected ROIs satisfy FDA atlas-overlap threshold")

    regions = tuple(sorted(by_region_trace_indices))
    time_by_region = np.column_stack(
        [
            np.mean(traces[by_region_trace_indices[region], :], axis=0)
            for region in regions
        ]
    )
    return FDAAtlasCompilation(
        region_traces=FunctionalTraceTable(
            regions,
            time_by_region,
            None,
            identity_kind="fda_region_from_bifrost_selected_supervoxel_overlap",
        ),
        assignments=tuple(assignments),
        assigned_selected_count=len(assignments),
        unassigned_selected_count=int(rows.size - len(assignments)),
        minimum_overlap_fraction=float(minimum_overlap_fraction),
    )
