import numpy as np

from dashi.analysis.sender_scalar_polarity_null import (
    distinct_sign_assignment_count,
    evaluate_sender_scalar_polarity_null,
)


def test_distinct_sign_assignment_count_with_one_negative():
    signs = np.array([1.0, 1.0, -1.0, 1.0])
    assert distinct_sign_assignment_count(signs) == 4


def test_exact_polarity_null_enumerates_all_single_negative_assignments():
    regions = ("A", "B", "C", "D")
    d = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    tendency = np.array([0.9, 0.8, -0.7, 0.6])
    s = d * tendency[:, None]
    # Build an observation that directly follows the true signed reverse carrier.
    observed = s.T.copy()
    observed = (observed + observed.T) / 2.0
    np.fill_diagonal(observed, 0.0)
    overlap = np.eye(4)

    result = evaluate_sender_scalar_polarity_null(
        regions,
        d,
        s,
        observed,
        overlap,
        max_exact_assignments=100,
        correlation_threshold=1.0,
    )
    assert result.exact_enumeration is True
    assert result.assignment_count == 4
    assert result.null_residuals.shape == (4,)
    assert result.negative_sender_count == 1
    assert np.isfinite(result.empirical_p_value)
    assert 0.0 < result.empirical_p_value <= 1.0


def test_magnitude_only_control_preserves_sender_magnitudes_but_removes_sign():
    regions = ("A", "B", "C", "D")
    d = np.ones((4, 4), dtype=float) - np.eye(4)
    tendency = np.array([0.9, -0.4, 0.2, 0.1])
    s = d * tendency[:, None]
    observed = np.zeros((4, 4), dtype=float)
    overlap = np.eye(4)

    result = evaluate_sender_scalar_polarity_null(
        regions,
        d,
        s,
        observed,
        overlap,
        max_exact_assignments=100,
        correlation_threshold=1.0,
    )
    assert np.allclose(result.sender_magnitudes, np.abs(tendency))
    assert result.negative_sender_count == 1
