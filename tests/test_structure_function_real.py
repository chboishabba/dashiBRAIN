import numpy as np
import scipy.sparse as sp

from dashi.analysis.benchmark_nulls import (
    degree_preserving_edge_swap,
    train_heldout_permutation_nulls,
)
from dashi.analysis.structure_function_real import (
    aggregate_connectome_by_region,
    evaluate_region_structure_function,
    functional_correlation,
    pairwise_train_holdout_masks,
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


def test_evaluate_real_features_uses_disjoint_fit_and_holdout_pairs():
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
    fit, held = pairwise_train_holdout_masks(3)
    assert not np.any(fit & held)
    result = evaluate_region_structure_function(structural, functional, fit_mask=fit, held_out_mask=held)
    assert result.fit_pair_count + result.held_out_pair_count == 3
    assert result.observed.size == result.held_out_pair_count
    assert np.isfinite(result.mean_dashi)


def test_fit_and_holdout_overlap_is_rejected():
    adjacency = sp.csr_matrix(np.eye(3))
    structural = aggregate_connectome_by_region(adjacency, ["A", "B", "C"])
    functional = functional_correlation(
        np.array([[0, 1, 2], [1, 2, 1], [2, 1, 0], [1, 0, 1]], dtype=float),
        structural.regions,
    )
    mask = np.triu(np.ones((3, 3), dtype=bool), 1)
    try:
        evaluate_region_structure_function(structural, functional, fit_mask=mask, held_out_mask=mask)
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("overlapping fit/held-out masks must fail")


def test_train_heldout_permutation_null_respects_masks():
    feature = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    observed = np.array([[0, .2, .6], [.2, 0, .8], [.6, .8, 0]], dtype=float)
    fit, held = pairwise_train_holdout_masks(3)
    summary = train_heldout_permutation_nulls(feature, observed, fit_mask=fit, held_out_mask=held, n_null=8, seed=3)
    assert summary.null_means.shape == (8,)
    assert 0.0 <= summary.empirical_p_value <= 1.0


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
