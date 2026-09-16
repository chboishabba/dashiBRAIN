from pathlib import Path

import numpy as np
import pytest

from dashi.io.functional_imaging_loader import load_registration_map
from dashi.io.gauthey_compact import GautheyFunctionalMatrix
from dashi.io.gauthey_registration_staging import LabelCentroid
from dashi.io.gauthey_roi_alignment import FunctionalToLabelMap
from dashi.io.gauthey_region_registration import (
    LabelToRegionMap,
    compile_functional_region_registration,
    write_registration_csv,
)


def _functional():
    return GautheyFunctionalMatrix(
        traces=np.zeros((4, 2)),
        unit_ids=("archive_unit_0000", "archive_unit_0001"),
        identity_kind="archive_array_index_unregistered",
        source_member="test",
    )


def _centroids():
    return (
        LabelCentroid(10, 5, 0, 0, 0, 0, 0, 0),
        LabelCentroid(20, 5, 1, 1, 1, 1, 1, 1),
    )


def test_composes_only_explicit_complete_receipts(tmp_path: Path):
    f = _functional()
    c = _centroids()
    fl = FunctionalToLabelMap(
        {"archive_unit_0000": 20, "archive_unit_0001": 10},
        "receipt:functional-to-label",
    )
    lr = LabelToRegionMap(
        {10: "R1", 20: "R2"},
        "receipt:label-to-region",
        residual_by_label={10: 0.2, 20: 0.1},
    )
    rows = compile_functional_region_registration(f, c, fl, lr)
    assert [(r.functional_id, r.region, r.segmentation_label) for r in rows] == [
        ("archive_unit_0000", "R2", 20),
        ("archive_unit_0001", "R1", 10),
    ]

    out = tmp_path / "registration.csv"
    write_registration_csv(rows, out)
    generic = load_registration_map(out)
    assert generic.functional_to_region["archive_unit_0000"] == "R2"
    assert generic.residual["archive_unit_0001"] == pytest.approx(0.2)


def test_missing_region_payment_is_rejected():
    f = _functional()
    c = _centroids()
    fl = FunctionalToLabelMap(
        {"archive_unit_0000": 10, "archive_unit_0001": 20},
        "receipt:functional-to-label",
    )
    lr = LabelToRegionMap({10: "R1"}, "receipt:partial")
    with pytest.raises(ValueError, match="missing labels required"):
        compile_functional_region_registration(f, c, fl, lr)


def test_region_map_cannot_reference_unknown_segmentation_label():
    lr = LabelToRegionMap({99: "R9"}, "receipt:bad")
    with pytest.raises(ValueError, match="absent segmentation labels"):
        lr.validate(_centroids())
