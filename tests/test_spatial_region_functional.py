import numpy as np
import pytest

from dashi.analysis.spatial_region_functional import (
    auditory_region_calibration,
    compile_plane_local_supervoxels_to_regions,
)


def test_plane_local_supervoxels_compile_to_region_traces():
    # 2 planes x 3 clusters/plane; same integer cluster IDs are intentionally
    # reused across planes and must remain distinct source rows.
    traces = np.array(
        [
            [1, 2, 3, 10, 20, 30],
            [2, 3, 4, 11, 21, 31],
            [3, 4, 5, 12, 22, 32],
        ],
        dtype=float,
    )
    seg = np.array(
        [
            [[0, 0, 1], [1, 2, 2]],
            [[0, 1, 1], [2, 2, 0]],
        ]
    )
    atlas = np.array(
        [
            [[1, 1, 1], [1, 2, 2]],
            [[2, 2, 2], [1, 1, 2]],
        ]
    )
    compiled = compile_plane_local_supervoxels_to_regions(
        traces,
        seg,
        atlas,
        {1: "AMMC", 2: "WED"},
        n_planes=2,
        clusters_per_plane=3,
        plane_axis=0,
        minimum_overlap_fraction=0.5,
    )
    assert compiled.region_traces.identity_kind == "atlas_region_from_supervoxel_overlap"
    assert compiled.region_traces.unit_ids == ("AMMC", "WED")
    assert compiled.assigned_trace_count > 0
    # Plane 0 cluster 0 and plane 1 cluster 0 correspond to distinct rows.
    zero_rows = [a.trace_row for a in compiled.assignments if a.cluster_index == 0]
    assert 0 in zero_rows and 3 in zero_rows


def test_supervoxel_region_compiler_rejects_wrong_trace_count():
    with pytest.raises(ValueError, match="expected 6"):
        compile_plane_local_supervoxels_to_regions(
            np.zeros((5, 5)),
            np.zeros((2, 2, 2), dtype=int),
            np.ones((2, 2, 2), dtype=int),
            {1: "AMMC"},
            n_planes=2,
            clusters_per_plane=3,
            plane_axis=0,
        )


def test_auditory_calibration_ranks_ammc_and_wed():
    from dashi.io.functional_imaging_loader import FunctionalTraceTable

    stimulus = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    traces = np.column_stack(
        [
            stimulus,
            stimulus * 0.9 + np.array([0, 0.05, 0, -0.05, 0, 0.05]),
            np.array([1, 0, 1, 0, 1, 0], dtype=float),
            np.array([0, 0, 1, 1, 0, 0], dtype=float),
        ]
    )
    table = FunctionalTraceTable(("AMMC", "WED", "PB", "FB"), traces)
    result = auditory_region_calibration(table, stimulus)
    assert result["expected_region_ranks"]["AMMC"] == 1
    assert result["expected_region_ranks"]["WED"] == 2
    assert result["expected_regions_in_top_quartile"] is False  # 4 regions => top quartile is rank 1 only
