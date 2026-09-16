from __future__ import annotations

import csv

from dashi.io.region_functional_adapter import load_region_functional_producer


def test_time_index_is_not_admitted_as_region(tmp_path):
    path = tmp_path / "region_functional.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_index", "AL", "AMMC", "SAD"])
        writer.writerow([0, 1.0, 2.0, 3.0])
        writer.writerow([1, 1.5, 2.5, 3.5])
        writer.writerow([2, 2.0, 3.0, 4.0])

    producer = load_region_functional_producer(
        path,
        atlas_identifier="VFB_JRC2018Unisex",
        source_identifier="fixture",
    )
    assert producer.traces.unit_ids == ("AL", "AMMC", "SAD")
    assert producer.traces.traces.shape == (3, 3)
    assert producer.traces.time is not None
