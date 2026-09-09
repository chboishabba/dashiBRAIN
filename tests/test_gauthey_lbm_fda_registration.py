from __future__ import annotations

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_SOURCE_CONTAINER_MEMBERS,
    LBM_SOURCE_PICKLE_NAMES,
)
from dashi.analysis.gauthey_lbm_fda_registration import (
    aggregate_transformed_selected_labels_to_regions,
    reorder_native_labels_to_nifti_shape,
)


def test_source_container_names_match_deposited_namespace():
    assert LBM_SOURCE_CONTAINER_MEMBERS[1] == (
        "Data/Dffs/Aligned/GCaMP6f_04032024_a2_r5.zip"
    )
    assert LBM_SOURCE_PICKLE_NAMES[1] == "GCaMP6f_04032024_a2_r5.pkl"
    assert "_6f_" not in LBM_SOURCE_CONTAINER_MEMBERS[1]


def test_native_label_volume_reorders_to_mean_brain_shape_without_resampling():
    labels = np.arange(27 * 3 * 5, dtype=np.int32).reshape(27, 3, 5)
    reordered, permutation = reorder_native_labels_to_nifti_shape(labels, (5, 3, 27))
    assert permutation == (2, 1, 0)
    assert reordered.shape == (5, 3, 27)
    assert reordered[4, 2, 26] == labels[26, 2, 4]


def test_fda_transformed_labels_aggregate_to_time_by_region():
    selected_rows = np.array([4, 21], dtype=np.int64)
    traces = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    selected = np.zeros((4, 4, 2), dtype=np.int32)
    atlas = np.zeros_like(selected)

    selected[0:2, 0:2, 0] = 5   # selected_row 4
    atlas[0:2, 0:2, 0] = 101
    selected[2:4, 2:4, 1] = 22  # selected_row 21
    atlas[2:4, 2:4, 1] = 202

    compiled = aggregate_transformed_selected_labels_to_regions(
        selected_rows,
        traces,
        selected,
        atlas,
        {101: "AMMC", 202: "WED"},
    )

    assert compiled.region_traces.unit_ids == ("AMMC", "WED")
    assert compiled.region_traces.traces.shape == (4, 2)
    assert np.array_equal(compiled.region_traces.traces[:, 0], traces[0])
    assert np.array_equal(compiled.region_traces.traces[:, 1], traces[1])
    assert compiled.assigned_selected_count == 2
    assert compiled.unassigned_selected_count == 0
