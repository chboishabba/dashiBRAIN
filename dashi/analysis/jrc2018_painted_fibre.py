"""Soft JRC2018 painted-domain functional carrier.

The VFB JRC2018 source atlas is a family of overlapping painted domains, not a
mutually exclusive label image.  This module therefore keeps every non-zero
ROI->domain overlap as a functional fibre instead of selecting one winner.

For selected ROI k and painted domain r,

    W[r,k] = |ROI_k intersect Domain_r| / |ROI_k|.

Domain traces are then the overlap-weighted average of the exact selected ROI
traces.  No row-wise renormalisation is performed: if one ROI belongs to nested
or overlapping atlas domains, that overlap remains explicit rather than being
forced into a partition.

The implementation is streaming and vectorised.  One full painted-domain raster
is resident at a time; the persistent carrier is only domain x selected-ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from dashi.io.functional_imaging_loader import FunctionalTraceTable


@dataclass(frozen=True)
class PaintedDomainFibreCompilation:
    regions: tuple[str, ...]
    selected_rows: np.ndarray
    overlap_membership: np.ndarray  # region x selected ROI
    region_traces: FunctionalTraceTable
    contributing_selected_count: int
    source_domain_count: int
    minimum_overlap_fraction: float


def compile_selected_labels_to_painted_domain_fibres_stream(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_jrc: np.ndarray,
    painted_domains: Iterable[tuple[str, np.ndarray]],
    *,
    minimum_overlap_fraction: float = 0.0,
) -> PaintedDomainFibreCompilation:
    """Compile every admitted ROI/domain overlap into a soft atlas carrier.

    ``minimum_overlap_fraction`` is a per-edge admission threshold.  A value of
    zero retains every positive overlap.  The threshold does not cause a winner
    selection and does not renormalise the surviving memberships.
    """
    rows = np.asarray(selected_rows, dtype=np.int64)
    traces = np.asarray(traces_roi_by_time, dtype=float)
    selected = np.asarray(selected_labels_jrc)
    if rows.ndim != 1:
        raise ValueError("selected_rows must be one-dimensional")
    if traces.ndim != 2 or traces.shape[0] != rows.size:
        raise ValueError("traces_roi_by_time must be [selected_roi,time]")
    if selected.ndim != 3:
        raise ValueError("selected_labels_jrc must be a 3-D label image")
    if not (0.0 <= minimum_overlap_fraction <= 1.0):
        raise ValueError("minimum_overlap_fraction must be in [0,1]")

    selected_values = rows + 1
    if np.any(selected_values <= 0):
        raise ValueError("selected_rows must map to positive selected-label values")
    max_label = int(max(int(selected.max(initial=0)), int(selected_values.max(initial=0))))
    voxel_counts = np.bincount(
        selected.astype(np.int64, copy=False).ravel(), minlength=max_label + 1
    )

    regions: list[str] = []
    membership_rows: list[np.ndarray] = []
    seen_regions: set[str] = set()

    for region_raw, image in painted_domains:
        region = str(region_raw)
        if region in seen_regions:
            raise ValueError(f"duplicate painted-domain name {region!r}")
        seen_regions.add(region)
        arr = np.asarray(image)
        if arr.shape != selected.shape:
            raise ValueError(
                f"painted domain {region!r} shape {arr.shape} != selected shape {selected.shape}"
            )

        inside = selected[arr != 0].astype(np.int64, copy=False)
        overlap_counts = np.bincount(inside, minlength=max_label + 1)
        weights = np.zeros(rows.size, dtype=np.float64)
        for trace_index, label_value in enumerate(selected_values):
            label = int(label_value)
            total = int(voxel_counts[label]) if label < voxel_counts.size else 0
            if total == 0:
                continue
            overlap = int(overlap_counts[label]) if label < overlap_counts.size else 0
            if overlap == 0:
                continue
            fraction = overlap / total
            if fraction >= minimum_overlap_fraction:
                weights[trace_index] = fraction
        if np.any(weights > 0):
            regions.append(region)
            membership_rows.append(weights)

    if not membership_rows:
        raise ValueError("no positive selected-ROI / painted-domain overlaps survived")

    membership = np.vstack(membership_rows)
    denom = membership.sum(axis=1)
    if np.any(denom <= 0):
        raise AssertionError("retained painted domains must have positive membership mass")
    # traces is ROI x time; weighted domain means become time x domain.
    weighted = membership @ traces
    time_by_region = (weighted / denom[:, None]).T
    contributing = int(np.count_nonzero(np.any(membership > 0, axis=0)))

    return PaintedDomainFibreCompilation(
        regions=tuple(regions),
        selected_rows=rows.copy(),
        overlap_membership=membership,
        region_traces=FunctionalTraceTable(
            tuple(regions),
            time_by_region,
            None,
            identity_kind="jrc2018_vfb_overlapping_painted_domain_soft_membership",
        ),
        contributing_selected_count=contributing,
        source_domain_count=len(seen_regions),
        minimum_overlap_fraction=float(minimum_overlap_fraction),
    )


def compile_selected_labels_to_painted_domain_fibres(
    selected_rows: Sequence[int],
    traces_roi_by_time: np.ndarray,
    selected_labels_jrc: np.ndarray,
    painted_domains: Mapping[str, np.ndarray],
    *,
    minimum_overlap_fraction: float = 0.0,
) -> PaintedDomainFibreCompilation:
    return compile_selected_labels_to_painted_domain_fibres_stream(
        selected_rows,
        traces_roi_by_time,
        selected_labels_jrc,
        painted_domains.items(),
        minimum_overlap_fraction=minimum_overlap_fraction,
    )
