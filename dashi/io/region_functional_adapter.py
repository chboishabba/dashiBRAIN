"""Generic region-resolved functional producer for structure/function experiments.

A producer may provide either:
1. unit-level functional traces plus an independently supplied registration map; or
2. traces already indexed by atlas/region identifiers.

This module owns the second route. It deliberately does not infer atlas identity
from column names alone: callers must declare the atlas/region vocabulary that
those columns use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dashi.io.functional_imaging_loader import FunctionalTraceTable, load_functional_traces


@dataclass(frozen=True)
class RegionFunctionalProducer:
    atlas_identifier: str
    traces: FunctionalTraceTable
    source_identifier: str
    evidence_kind: str = "declared_region_resolved_functional_observation"

    def validate(self) -> None:
        if not self.atlas_identifier.strip():
            raise ValueError("atlas_identifier must be non-empty")
        if not self.source_identifier.strip():
            raise ValueError("source_identifier must be non-empty")
        if not self.traces.unit_ids:
            raise ValueError("region-resolved producer has no regions")
        if self.traces.traces.ndim != 2:
            raise ValueError("region-resolved traces must be 2-D [time, region]")
        if self.traces.traces.shape[1] != len(self.traces.unit_ids):
            raise ValueError("region identifiers must align with trace columns")
        if len(set(self.traces.unit_ids)) != len(self.traces.unit_ids):
            raise ValueError("region identifiers must be unique")


def load_region_functional_producer(
    path: str | Path,
    *,
    atlas_identifier: str,
    source_identifier: str,
) -> RegionFunctionalProducer:
    """Load a functional table whose units are already declared atlas regions."""
    traces = load_functional_traces(path)
    producer = RegionFunctionalProducer(
        atlas_identifier=atlas_identifier,
        traces=FunctionalTraceTable(
            unit_ids=traces.unit_ids,
            traces=traces.traces,
            time=traces.time,
            identity_kind="declared_region_identity",
        ),
        source_identifier=source_identifier,
    )
    producer.validate()
    return producer


def restrict_to_regions(
    producer: RegionFunctionalProducer,
    allowed_regions: Iterable[str],
) -> RegionFunctionalProducer:
    """Return the producer restricted to an explicitly supplied region vocabulary."""
    producer.validate()
    allowed = set(str(r) for r in allowed_regions)
    indices = [i for i, r in enumerate(producer.traces.unit_ids) if r in allowed]
    if not indices:
        raise ValueError("no region-resolved functional units overlap the allowed region vocabulary")
    unit_ids = tuple(producer.traces.unit_ids[i] for i in indices)
    traces = producer.traces.traces[:, indices]
    out = RegionFunctionalProducer(
        atlas_identifier=producer.atlas_identifier,
        traces=FunctionalTraceTable(
            unit_ids=unit_ids,
            traces=traces,
            time=producer.traces.time,
            identity_kind=producer.traces.identity_kind,
        ),
        source_identifier=producer.source_identifier,
        evidence_kind=producer.evidence_kind,
    )
    out.validate()
    return out
