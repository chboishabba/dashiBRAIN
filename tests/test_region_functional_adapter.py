from pathlib import Path

import numpy as np
import pytest

from dashi.io.functional_imaging_loader import FunctionalTraceTable
from dashi.io.region_functional_adapter import (
    RegionFunctionalProducer,
    load_region_functional_producer,
    restrict_to_regions,
)


def test_declared_region_producer_loads_wide_table(tmp_path: Path):
    p = tmp_path / "regions.csv"
    p.write_text(
        "time,AL,MB,PB\n0,0.1,0.2,0.3\n1,0.2,0.1,0.4\n2,0.4,0.3,0.2\n",
        encoding="utf-8",
    )
    producer = load_region_functional_producer(
        p,
        atlas_identifier="test-atlas:v1",
        source_identifier="receipt:test-regions",
    )
    assert producer.traces.unit_ids == ("AL", "MB", "PB")
    assert producer.traces.traces.shape == (3, 3)
    assert producer.traces.identity_kind == "declared_region_identity"


def test_region_producer_requires_declared_atlas_and_source():
    traces = FunctionalTraceTable(("AL", "MB"), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="atlas_identifier"):
        RegionFunctionalProducer("", traces, "source").validate()
    with pytest.raises(ValueError, match="source_identifier"):
        RegionFunctionalProducer("atlas", traces, "").validate()


def test_region_restriction_preserves_only_shared_vocabulary():
    producer = RegionFunctionalProducer(
        "atlas:v1",
        FunctionalTraceTable(("AL", "MB", "PB"), np.arange(12, dtype=float).reshape(4, 3)),
        "receipt:test",
    )
    restricted = restrict_to_regions(producer, {"PB", "AL"})
    assert restricted.traces.unit_ids == ("AL", "PB")
    np.testing.assert_array_equal(restricted.traces.traces, producer.traces.traces[:, [0, 2]])


def test_region_restriction_rejects_zero_overlap():
    producer = RegionFunctionalProducer(
        "atlas:v1",
        FunctionalTraceTable(("AL", "MB"), np.zeros((4, 2))),
        "receipt:test",
    )
    with pytest.raises(ValueError, match="no region-resolved functional units overlap"):
        restrict_to_regions(producer, {"NOPE"})
