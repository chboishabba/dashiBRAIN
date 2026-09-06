"""Conservative alignment diagnostics for Gauthey functional columns and ROIs.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; data DOI 10.5281/zenodo.17618684.

The deposited functional matrix has 668 archive-local columns.  The segmentation
and responsive-ROI products are separate objects.  Numerical coincidences such
as equal cardinality or contiguous integer labels are diagnostics only; they do
not establish column-to-ROI identity.  Promotion requires an explicit mapping
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Mapping, Sequence

import numpy as np

from dashi.io.gauthey_compact import GautheyFunctionalMatrix
from dashi.io.gauthey_registration_staging import LabelCentroid


@dataclass(frozen=True)
class AlignmentDiagnostic:
    functional_column_count: int
    segmentation_label_count: int
    responsive_roi_count: int | None
    same_cardinality: bool
    segmentation_labels_contiguous_from_one: bool
    responsive_rois_subset_of_segmentation_labels: bool | None
    explicit_mapping_present: bool
    identity_promotable: bool
    evidence_kind: str
    note: str


@dataclass(frozen=True)
class FunctionalToLabelMap:
    """Explicit archive-column -> segmentation-label mapping receipt.

    This type is intentionally constructible only from a mapping table supplied
    by an external producer or a separately justified derivation.  The auditor
    below never manufactures this map from cardinality/order alone.
    """

    column_to_label: Mapping[str, int]
    source_identifier: str
    evidence_kind: str = "explicit_functional_column_to_segmentation_label_map"

    def validate(self, functional: GautheyFunctionalMatrix, centroids: Sequence[LabelCentroid]) -> None:
        labels = {int(c.label_id) for c in centroids}
        expected = set(functional.unit_ids)
        actual = set(self.column_to_label)
        if actual != expected:
            missing = sorted(expected - actual)[:5]
            extra = sorted(actual - expected)[:5]
            raise ValueError(f"mapping domain mismatch; missing={missing}, extra={extra}")
        bad = sorted({int(v) for v in self.column_to_label.values()} - labels)[:5]
        if bad:
            raise ValueError(f"mapping references absent segmentation labels: {bad}")


def _parse_roi_values(path: str | Path) -> tuple[int, ...]:
    """Read a Gauthey responsive-ROI CSV conservatively.

    The published compact file observed by the runner contains columns
    ``Unnamed: 0`` and ``ROI``.  We use the explicit ``ROI`` column only and
    require integer-like values; the row index is never treated as identity.
    """
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "ROI" not in reader.fieldnames:
            raise ValueError("responsive ROI CSV lacks explicit 'ROI' column")
        out: list[int] = []
        for row in reader:
            raw = (row.get("ROI") or "").strip()
            if not raw:
                continue
            value = float(raw)
            if not value.is_integer():
                raise ValueError(f"non-integer ROI value: {raw}")
            out.append(int(value))
    return tuple(out)


def audit_functional_label_alignment(
    functional: GautheyFunctionalMatrix,
    centroids: Sequence[LabelCentroid],
    *,
    responsive_roi_csv: str | Path | None = None,
    explicit_mapping: FunctionalToLabelMap | None = None,
) -> AlignmentDiagnostic:
    labels = sorted({int(c.label_id) for c in centroids})
    responsive: tuple[int, ...] | None = None
    if responsive_roi_csv is not None:
        responsive = _parse_roi_values(responsive_roi_csv)

    contiguous = labels == list(range(1, len(labels) + 1)) if labels else False
    same_cardinality = len(functional.unit_ids) == len(labels)
    subset: bool | None = None
    if responsive is not None:
        subset = set(responsive).issubset(set(labels))

    mapping_present = explicit_mapping is not None
    promotable = False
    note = (
        "Cardinality/order diagnostics do not identify functional columns with segmentation labels."
    )
    if explicit_mapping is not None:
        explicit_mapping.validate(functional, centroids)
        promotable = True
        note = "Explicit mapping domain/codomain validated against deposited functional units and labels."

    return AlignmentDiagnostic(
        functional_column_count=len(functional.unit_ids),
        segmentation_label_count=len(labels),
        responsive_roi_count=None if responsive is None else len(responsive),
        same_cardinality=same_cardinality,
        segmentation_labels_contiguous_from_one=contiguous,
        responsive_rois_subset_of_segmentation_labels=subset,
        explicit_mapping_present=mapping_present,
        identity_promotable=promotable,
        evidence_kind=(
            "explicit_mapping_validated" if promotable
            else "cardinality_and_domain_diagnostic_only"
        ),
        note=note,
    )


def load_explicit_mapping_csv(
    path: str | Path,
    *,
    source_identifier: str,
) -> FunctionalToLabelMap:
    """Load an explicit two-column mapping: functional_id, segmentation_label."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"functional_id", "segmentation_label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("mapping CSV requires functional_id,segmentation_label")
        mapping: dict[str, int] = {}
        for row in reader:
            fid = (row.get("functional_id") or "").strip()
            raw = (row.get("segmentation_label") or "").strip()
            if not fid or not raw:
                raise ValueError("mapping rows require non-empty functional_id and segmentation_label")
            if fid in mapping:
                raise ValueError(f"duplicate functional_id in mapping: {fid}")
            mapping[fid] = int(raw)
    return FunctionalToLabelMap(mapping, source_identifier)


def mapped_centroid_table(
    functional: GautheyFunctionalMatrix,
    centroids: Sequence[LabelCentroid],
    mapping: FunctionalToLabelMap,
) -> tuple[tuple[str, LabelCentroid], ...]:
    mapping.validate(functional, centroids)
    by_label = {int(c.label_id): c for c in centroids}
    return tuple((uid, by_label[int(mapping.column_to_label[uid])]) for uid in functional.unit_ids)
