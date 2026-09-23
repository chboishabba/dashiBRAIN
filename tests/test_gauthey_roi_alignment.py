from pathlib import Path

import numpy as np
import pytest

from dashi.io.gauthey_compact import GautheyFunctionalMatrix
from dashi.io.gauthey_registration_staging import LabelCentroid
from dashi.io.gauthey_roi_alignment import (
    FunctionalToLabelMap,
    audit_functional_label_alignment,
    mapped_centroid_table,
)


def _functional(n: int = 3) -> GautheyFunctionalMatrix:
    return GautheyFunctionalMatrix(
        traces=np.zeros((5, n)),
        unit_ids=tuple(f"archive_unit_{i:04d}" for i in range(n)),
        identity_kind="archive_array_index_unregistered",
        source_member="test",
    )


def _centroids(ids=(1, 2, 3)):
    return tuple(
        LabelCentroid(i, 10, float(i), 0.0, 0.0, float(i), 0.0, 0.0)
        for i in ids
    )


def test_equal_cardinality_does_not_promote_identity():
    d = audit_functional_label_alignment(_functional(3), _centroids((1, 2, 3)))
    assert d.same_cardinality
    assert d.segmentation_labels_contiguous_from_one
    assert not d.explicit_mapping_present
    assert not d.identity_promotable
    assert d.evidence_kind == "cardinality_and_domain_diagnostic_only"


def test_explicit_mapping_can_promote_after_domain_validation():
    f = _functional(3)
    c = _centroids((10, 20, 30))
    m = FunctionalToLabelMap(
        {
            "archive_unit_0000": 20,
            "archive_unit_0001": 10,
            "archive_unit_0002": 30,
        },
        "receipt:test-map",
    )
    d = audit_functional_label_alignment(f, c, explicit_mapping=m)
    assert d.identity_promotable
    table = mapped_centroid_table(f, c, m)
    assert [row[1].label_id for row in table] == [20, 10, 30]


def test_explicit_mapping_rejects_missing_or_unknown_labels():
    f = _functional(2)
    c = _centroids((1, 2))
    incomplete = FunctionalToLabelMap({"archive_unit_0000": 1}, "receipt:bad")
    with pytest.raises(ValueError, match="mapping domain mismatch"):
        incomplete.validate(f, c)

    bad_label = FunctionalToLabelMap(
        {"archive_unit_0000": 1, "archive_unit_0001": 99},
        "receipt:bad",
    )
    with pytest.raises(ValueError, match="absent segmentation labels"):
        bad_label.validate(f, c)


def test_responsive_roi_subset_is_diagnostic_only(tmp_path: Path):
    roi = tmp_path / "roi.csv"
    roi.write_text("Unnamed: 0,ROI\n0,1\n1,3\n", encoding="utf-8")
    d = audit_functional_label_alignment(
        _functional(3),
        _centroids((1, 2, 3)),
        responsive_roi_csv=roi,
    )
    assert d.responsive_roi_count == 2
    assert d.responsive_rois_subset_of_segmentation_labels is True
    assert not d.identity_promotable
