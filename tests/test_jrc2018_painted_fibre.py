import numpy as np

from dashi.analysis.jrc2018_painted_fibre import (
    compile_selected_labels_to_painted_domain_fibres,
)


def test_overlapping_domains_are_retained_without_winner_take_all_collapse():
    selected = np.zeros((2, 2, 2), dtype=np.int16)
    selected[0, :, :] = 5   # selected_row 4, 4 voxels
    selected[1, :, :] = 22  # selected_row 21, 4 voxels
    parent = np.zeros_like(selected)
    child = np.zeros_like(selected)
    other = np.zeros_like(selected)

    parent[selected == 5] = 1
    child[0, 0, :] = 1      # 2/4 overlap with same ROI
    other[selected == 22] = 1

    rows = np.array([4, 21], dtype=np.int64)
    traces = np.vstack([
        np.arange(5, dtype=float),
        np.arange(5, dtype=float) + 10,
    ])

    compiled = compile_selected_labels_to_painted_domain_fibres(
        rows,
        traces,
        selected,
        {"PARENT": parent, "CHILD": child, "OTHER": other},
    )

    assert compiled.regions == ("PARENT", "CHILD", "OTHER")
    assert np.allclose(compiled.overlap_membership[:, 0], [1.0, 0.5, 0.0])
    assert np.allclose(compiled.overlap_membership[:, 1], [0.0, 0.0, 1.0])
    assert np.allclose(compiled.region_traces.traces[:, 0], traces[0])
    assert np.allclose(compiled.region_traces.traces[:, 1], traces[0])
    assert np.allclose(compiled.region_traces.traces[:, 2], traces[1])
    # Parent + child overlap is intentionally not row-normalised to one.
    assert compiled.overlap_membership[:, 0].sum() == 1.5


def test_soft_threshold_prunes_edges_not_domains_by_winner_selection():
    selected = np.zeros((1, 2, 4), dtype=np.int16)
    selected[0, :, :] = 5
    a = np.zeros_like(selected)
    b = np.zeros_like(selected)
    a[0, :, :4] = 1          # 1.0 overlap
    b[0, 0, :2] = 1          # 0.25 overlap
    rows = np.array([4], dtype=np.int64)
    traces = np.arange(6, dtype=float)[None, :]

    compiled = compile_selected_labels_to_painted_domain_fibres(
        rows,
        traces,
        selected,
        {"A": a, "B": b},
        minimum_overlap_fraction=0.5,
    )
    assert compiled.regions == ("A",)
    assert np.allclose(compiled.overlap_membership, [[1.0]])
