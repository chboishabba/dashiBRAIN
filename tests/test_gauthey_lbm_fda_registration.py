from __future__ import annotations

import numpy as np

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_SOURCE_CONTAINER_MEMBERS,
    LBM_SOURCE_PICKLE_NAMES,
)
from dashi.analysis.gauthey_lbm_fda_registration import (
    aggregate_transformed_selected_labels_to_regions,
    centered_lbm_affine_in_fda_axes,
    physical_extent_microns,
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


def test_centered_lbm_affine_swaps_array_yx_into_fda_world_xy():
    # Synthetic FDA affine uses micron units and simple signed axis directions.
    fixed_shape = (1652, 768, 479)
    fixed_affine = np.array(
        [
            [-0.38, 0.0, 0.0, 10.0],
            [0.0, -0.38, 0.0, 20.0],
            [0.0, 0.0, 0.38, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    moving_shape = (226, 512, 27)  # Gauthey array order is y,x,z.
    moving_affine = centered_lbm_affine_in_fda_axes(
        moving_shape,
        fixed_shape,
        fixed_affine,
    )

    # Moving voxel axis 0 follows FDA world-Y; voxel axis 1 follows FDA world-X.
    assert np.allclose(moving_affine[:3, 0], [0.0, -1.3, 0.0])
    assert np.allclose(moving_affine[:3, 1], [-1.3, 0.0, 0.0])
    assert np.allclose(moving_affine[:3, 2], [0.0, 0.0, 9.0])

    fixed_center_idx = (np.asarray(fixed_shape, dtype=float) - 1.0) / 2.0
    moving_center_idx = (np.asarray(moving_shape, dtype=float) - 1.0) / 2.0
    fixed_center = fixed_affine[:3, :3] @ fixed_center_idx + fixed_affine[:3, 3]
    moving_center = moving_affine[:3, :3] @ moving_center_idx + moving_affine[:3, 3]
    assert np.allclose(moving_center, fixed_center)

    extent = physical_extent_microns(moving_shape, moving_affine)
    assert np.allclose(extent, [226 * 1.3, 512 * 1.3, 27 * 9.0])
    # Anatomically, x comes from moving array axis 1 and y from array axis 0.
    assert np.allclose([extent[1], extent[0], extent[2]], [665.6, 293.8, 243.0])


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
