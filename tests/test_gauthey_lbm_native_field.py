from __future__ import annotations

import numpy as np

from dashi.analysis.gauthey_lbm_native_field import (
    LBM_EXPECTED_SELECTED,
    LBM_MIN_TIMEPOINTS,
    LBM_NATIVE_HEIGHT,
    LBM_NATIVE_WIDTH,
    LBM_N_PLANES,
    LBM_PIXELS_PER_PLANE,
    RecoveredSelectedROI,
    _source_oriented_label_volume,
    compile_native_selected_field,
)


def test_source_orientation_matches_transpose_then_published_reshape():
    raw = np.arange(LBM_N_PLANES * LBM_PIXELS_PER_PLANE, dtype=np.int64).reshape(
        LBM_N_PLANES, LBM_PIXELS_PER_PLANE
    )
    expected = np.moveaxis(
        raw.T.reshape(LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH, LBM_N_PLANES),
        2,
        0,
    )
    actual = _source_oriented_label_volume(raw)
    assert actual.shape == (LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH)
    assert np.array_equal(actual, expected)


def test_exact_recovered_rows_compile_to_native_masks_and_traces():
    seg = np.zeros((LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH), dtype=np.int32)
    seg[3, 10:12, 20:23] = 912
    seg[7, 100:103, 200:202] = 633

    base = np.arange(LBM_MIN_TIMEPOINTS, dtype=float)[None, :]
    traces = np.broadcast_to(base, (LBM_EXPECTED_SELECTED, LBM_MIN_TIMEPOINTS))
    identities = (
        RecoveredSelectedROI(21, 60912, "04032024_6f_a2_r5", 3, 912, 0.09),
        RecoveredSelectedROI(4, 68633, "04032024_6f_a2_r5", 7, 633, 0.07),
    )

    field = compile_native_selected_field(
        traces,
        identities,
        seg,
        trial_id="04032024_6f_a2_r5",
    )

    assert field.selected_rows.tolist() == [4, 21]
    assert field.traces_roi_by_time.shape == (2, LBM_MIN_TIMEPOINTS)
    assert np.array_equal(field.traces_roi_by_time[0], base[0])
    assert np.count_nonzero(field.selected_label_volume == 5) == 6
    assert np.count_nonzero(field.selected_label_volume == 22) == 6
    assert [(roi.plane_index, roi.cluster_index) for roi in field.rois] == [(7, 633), (3, 912)]
    assert field.rois[0].centroid_y == 101.0
    assert field.rois[0].centroid_x == 200.5


def test_native_field_rejects_identity_missing_from_segmentation():
    seg = np.zeros((LBM_N_PLANES, LBM_NATIVE_HEIGHT, LBM_NATIVE_WIDTH), dtype=np.int32)
    traces = np.broadcast_to(
        np.zeros((1, LBM_MIN_TIMEPOINTS), dtype=float),
        (LBM_EXPECTED_SELECTED, LBM_MIN_TIMEPOINTS),
    )
    identity = RecoveredSelectedROI(4, 68633, "04032024_6f_a2_r5", 7, 633, 0.07)

    try:
        compile_native_selected_field(
            traces,
            (identity,),
            seg,
            trial_id="04032024_6f_a2_r5",
        )
    except ValueError as exc:
        assert "cluster absent" in str(exc)
    else:
        raise AssertionError("expected missing-cluster failure")
