from __future__ import annotations

import numpy as np

from dashi.analysis.jrc2018_painted_overlap import (
    compile_selected_labels_against_painted_domain_stream,
    compile_selected_labels_against_painted_domains,
)


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


def _reference_best(rows, selected, domains, threshold):
    assignments = {}
    ambiguities = {}
    for row in rows:
        label = int(row) + 1
        mask = selected == label
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            continue
        scores = []
        for region, domain in domains.items():
            overlap = int(np.count_nonzero(mask & (domain != 0)))
            if overlap:
                scores.append((region, overlap, overlap / voxel_count))
        if not scores:
            continue
        best = max(s[2] for s in scores)
        if best < threshold:
            continue
        winners = sorted(s for s in scores if abs(s[2] - best) <= 1e-12)
        if len(winners) == 1:
            assignments[int(row)] = winners[0]
        else:
            ambiguities[int(row)] = (tuple(w[0] for w in winners), best)
    return assignments, ambiguities


def test_streamed_bincount_matches_reference_per_roi_semantics():
    selected = np.zeros((5, 5, 4), dtype=np.int16)
    selected[0:2, 0:2, 0] = 5
    selected[2:4, 1:3, 1] = 22
    selected[1:5, 4, 2] = 31
    rows = np.array([4, 21, 30], dtype=np.int64)
    traces = np.vstack([
        np.arange(5, dtype=float),
        np.arange(5, dtype=float) + 10,
        np.arange(5, dtype=float) + 20,
    ])

    domains = {
        "AMMC": np.zeros_like(selected),
        "WED": np.zeros_like(selected),
        "GNG": np.zeros_like(selected),
    }
    domains["AMMC"][selected == 5] = 1
    domains["WED"][2:4, 1:3, 1] = 1
    domains["GNG"][1:3, 4, 2] = 1  # 2/4 of row 30, exactly threshold.

    threshold = 0.5
    expected_assignments, expected_ambiguities = _reference_best(
        rows, selected, domains, threshold
    )
    streamed = compile_selected_labels_against_painted_domain_stream(
        rows,
        traces,
        selected,
        ((name, arr) for name, arr in domains.items()),
        minimum_overlap_fraction=threshold,
    )

    observed = {
        item.selected_row: (item.region, item.overlap_voxel_count, item.overlap_fraction)
        for item in streamed.assignments
    }
    assert set(observed) == set(expected_assignments)
    for row, (region, overlap, fraction) in expected_assignments.items():
        assert observed[row][0] == region
        assert observed[row][1] == overlap
        assert observed[row][2] == fraction

    observed_ambiguities = {
        item.selected_row: (item.regions, item.overlap_fraction)
        for item in streamed.ambiguities
    }
    assert observed_ambiguities == expected_ambiguities
