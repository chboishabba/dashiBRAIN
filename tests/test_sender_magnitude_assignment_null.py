import numpy as np

from dashi.analysis.sender_magnitude_assignment_null import (
    evaluate_sender_magnitude_assignment_null,
    scale_free_sender_gain_matrix,
    sender_magnitude_fields,
)


def test_magnitude_definitions_diverge_for_mixed_sign_sender():
    d = np.array([[0.0, 2.0, 2.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    s = np.array([[0.0, 2.0, -2.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    net, absolute_ratio = sender_magnitude_fields(d, s)
    assert np.isclose(net[0], 0.0)
    assert np.isclose(absolute_ratio[0], 1.0)
    assert np.all(absolute_ratio >= net - 1e-12)


def test_scale_free_gain_ignores_sender_absolute_scale():
    d = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 4.0, 0.0]])
    magnitude = np.array([0.2, 0.5, 0.8])
    first = scale_free_sender_gain_matrix(d, magnitude)
    scaled = d * np.array([10.0, 0.1, 7.0])[:, None]
    second = scale_free_sender_gain_matrix(scaled, magnitude)
    assert np.allclose(first, second)


def test_assignment_null_preserves_magnitude_multiset():
    n = 5
    ii, jj = np.indices((n, n))
    d = (1.0 + ((ii + 2 * jj) % 5)).astype(float)
    np.fill_diagonal(d, 0.0)
    sender = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    s = d * sender[:, None]
    observed = np.corrcoef(np.arange(n * n).reshape(n, n))
    np.fill_diagonal(observed, 0.0)
    overlap = np.eye(n)
    result = evaluate_sender_magnitude_assignment_null(
        tuple(f"R{i}" for i in range(n)),
        d,
        s,
        observed,
        overlap,
        n_null=7,
        seed=4,
    )
    assert result.net_assignment_null_residuals.shape == (7,)
    assert 0.0 < result.net_assignment_empirical_p_value <= 1.0
    assert np.allclose(np.sort(result.net_magnitudes), np.sort(sender))


def test_net_and_absolute_ratio_match_for_single_sign_rows():
    d = np.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [1.0, 4.0, 0.0]])
    sender = np.array([0.2, 0.5, 0.8])
    s = d * sender[:, None]
    net, absolute_ratio = sender_magnitude_fields(d, s)
    assert np.allclose(net, absolute_ratio)
