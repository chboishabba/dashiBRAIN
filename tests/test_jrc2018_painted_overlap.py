from __future__ import annotations

import numpy as np

from dashi.analysis.jrc2018_painted_overlap import compile_selected_labels_against_painted_domains


def test_painted_domains_assign_unique_winners_and_preserve_time_by_region():
    selected = np.zeros((4, 4, 4), dtype=np.int16)
    selected[0, 0:2, 0:2] = 5   # selected_row 4
    selected[1, 0:2, 0:2] = 22  # selected_row 21
    ammc = np.zeros_like(selected)
    wed = np.zeros_like(selected)
    ammc[selected == 5] = 1
    wed[selected == 22] = 1
    rows = np.array([4, 21], dtype=np.int64)
    traces = np.vstack([np.arange(6, dtype=float), np.arange(6, dtype=float) + 10])

    compiled = compile_selected_labels_against_painted_domains(
        rows,
        traces,
        selected,
        {"AMMC": ammc, "WED": wed},
        minimum_overlap_fraction=0.5,
    )

    assert compiled.assigned_selected_count == 2
    assert compiled.ambiguous_selected_count == 0
    assert compiled.unassigned_selected_count == 0
    assert compiled.region_traces.unit_ids == ("AMMC", "WED")
    assert compiled.region_traces.traces.shape == (6, 2)
    assert np.array_equal(compiled.region_traces.traces[:, 0], traces[0])
    assert np.array_equal(compiled.region_traces.traces[:, 1], traces[1])


def test_exact_domain_tie_is_ambiguous_not_order_selected():
    selected = np.zeros((3, 3, 3), dtype=np.int16)
    selected[0, 0:2, 0:2] = 5
    selected[1, 0:2, 0:2] = 22
    a = np.zeros_like(selected)
    b = np.zeros_like(selected)
    c = np.zeros_like(selected)
    a[0, 0, 0:2] = 1
    b[0, 1, 0:2] = 1
    c[selected == 22] = 1
    rows = np.array([4, 21], dtype=np.int64)
    traces = np.vstack([np.arange(4, dtype=float), np.arange(4, dtype=float) + 5])

    compiled = compile_selected_labels_against_painted_domains(
        rows,
        traces,
        selected,
        {"A": a, "B": b, "C": c},
        minimum_overlap_fraction=0.5,
    )

    assert compiled.assigned_selected_count == 1
    assert compiled.ambiguous_selected_count == 1
    assert compiled.ambiguities[0].selected_row == 4
    assert compiled.ambiguities[0].regions == ("A", "B")
    assert compiled.region_traces.unit_ids == ("C",)
