"""Compile explicit Gauthey ROI mappings into benchmark-ready region registration.

This module does not infer either missing scientific edge.  It composes two
independently supplied receipts:

  functional archive column -> segmentation label
  segmentation label -> atlas region

Only after both are complete does it emit the generic registration CSV consumed
by ``load_registration_map``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Mapping, Sequence

from dashi.io.gauthey_compact import GautheyFunctionalMatrix
from dashi.io.gauthey_registration_staging import LabelCentroid
from dashi.io.gauthey_roi_alignment import FunctionalToLabelMap


@dataclass(frozen=True)
class LabelToRegionMap:
    label_to_region: Mapping[int, str]
    source_identifier: str
    evidence_kind: str = "registered_label_to_atlas_region"
    residual_by_label: Mapping[int, float] | None = None

    def validate(self, centroids: Sequence[LabelCentroid], required_labels: set[int] | None = None) -> None:
        existing = {int(c.label_id) for c in centroids}
        domain = {int(k) for k in self.label_to_region}
        if not domain.issubset(existing):
            bad = sorted(domain - existing)[:5]
            raise ValueError(f"region map references absent segmentation labels: {bad}")
        if required_labels is not None and not required_labels.issubset(domain):
            missing = sorted(required_labels - domain)[:5]
            raise ValueError(f"region map missing labels required by functional mapping: {missing}")
        empty = sorted(k for k, v in self.label_to_region.items() if not str(v).strip())[:5]
        if empty:
            raise ValueError(f"region map contains empty region labels: {empty}")


@dataclass(frozen=True)
class FunctionalRegionRow:
    functional_id: str
    region: str
    evidence_kind: str
    residual: float | None
    segmentation_label: int
    mapping_source: str
    region_source: str


def load_label_to_region_csv(path: str | Path, *, source_identifier: str) -> LabelToRegionMap:
    """Load ``segmentation_label,region[,residual]``."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"segmentation_label", "region"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("region CSV requires segmentation_label,region")
        mapping: dict[int, str] = {}
        residuals: dict[int, float] = {}
        for row in reader:
            label = int((row.get("segmentation_label") or "").strip())
            region = (row.get("region") or "").strip()
            if label in mapping:
                raise ValueError(f"duplicate segmentation_label in region map: {label}")
            mapping[label] = region
            raw_residual = (row.get("residual") or "").strip()
            if raw_residual:
                residuals[label] = float(raw_residual)
    return LabelToRegionMap(mapping, source_identifier, residual_by_label=residuals or None)


def compile_functional_region_registration(
    functional: GautheyFunctionalMatrix,
    centroids: Sequence[LabelCentroid],
    functional_to_label: FunctionalToLabelMap,
    label_to_region: LabelToRegionMap,
) -> tuple[FunctionalRegionRow, ...]:
    functional_to_label.validate(functional, centroids)
    required_labels = {int(v) for v in functional_to_label.column_to_label.values()}
    label_to_region.validate(centroids, required_labels=required_labels)

    rows: list[FunctionalRegionRow] = []
    for fid in functional.unit_ids:
        label = int(functional_to_label.column_to_label[fid])
        residual = None
        if label_to_region.residual_by_label is not None:
            residual = label_to_region.residual_by_label.get(label)
        rows.append(FunctionalRegionRow(
            functional_id=fid,
            region=str(label_to_region.label_to_region[label]),
            evidence_kind=(
                f"{functional_to_label.evidence_kind}+{label_to_region.evidence_kind}"
            ),
            residual=residual,
            segmentation_label=label,
            mapping_source=functional_to_label.source_identifier,
            region_source=label_to_region.source_identifier,
        ))
    return tuple(rows)


def write_registration_csv(rows: Sequence[FunctionalRegionRow], output_path: str | Path) -> None:
    """Emit the generic registration schema consumed by the real benchmark."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "functional_id",
        "region",
        "evidence_kind",
        "residual",
        "segmentation_label",
        "mapping_source",
        "region_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "functional_id": row.functional_id,
                "region": row.region,
                "evidence_kind": row.evidence_kind,
                "residual": "" if row.residual is None else row.residual,
                "segmentation_label": row.segmentation_label,
                "mapping_source": row.mapping_source,
                "region_source": row.region_source,
            })
