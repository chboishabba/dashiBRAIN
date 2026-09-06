import numpy as np
import scipy.sparse as sp

from dashi.analysis.benchmark_nulls import degree_preserving_edge_swap
from dashi.analysis.structure_function_real import (
    aggregate_connectome_by_region,
    evaluate_region_structure_function,
    functional_correlation,
)


def test_region_aggregation_and_functional_correlation_align():
    adjacency = sp.csr_matrix(
        np.array([
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 3],
            [1, 0, 0, 0],
        ], dtype=float)
    )
    regions = ["A", "A", "B", "B"]
    structural = aggregate_connectome_by_region(adjacency, regions)
    assert structural.regions == ("A", "B")
    traces = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    functional = functional_correlation(traces, ("A", "B"))
    assert functional.matrix.shape == (2, 2)


def test_evaluate_real_features_not_hash_features():
    adjacency = sp.csr_matrix(np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
    ], dtype=float))
    structural = aggregate_connectome_by_region(adjacency, ["A", "A", "B", "B", "C", "C"])
    t = np.linspace(0, 2 * np.pi, 20)
    traces = np.column_stack([np.sin(t), np.sin(t + 0.2), np.cos(t)])
    functional = functional_correlation(traces, structural.regions)
    result = evaluate_region_structure_function(structural, functional)
    assert result.observed.size == 3
    assert np.isfinite(result.mean_dashi)


def test_degree_preserving_swap_preserves_in_out_degree():
    a = sp.csr_matrix(np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float))
    b = degree_preserving_edge_swap(a, n_swaps=50, seed=4)
    assert np.array_equal(np.diff(a.indptr), np.diff(b.indptr))
    assert np.array_equal(np.asarray((a != 0).sum(axis=0)).ravel(), np.asarray((b != 0).sum(axis=0)).ravel())
