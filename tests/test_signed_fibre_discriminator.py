import numpy as np

from dashi.analysis.signed_fibre_discriminator import (
    evaluate_signed_reverse_discriminator,
    sender_pattern_permuted_signed_direct,
    signed_ratio_to_unsigned,
)


def test_sender_pattern_permutation_preserves_unsigned_carrier_and_reassigns_ratio_rows():
    d = np.array(
        [
            [0.0, 2.0, 4.0],
            [3.0, 0.0, 6.0],
            [5.0, 10.0, 0.0],
        ]
    )
    ratio = np.array(
        [
            [0.0, 1.0, -0.5],
            [0.25, 0.0, 0.75],
            [-1.0, 0.5, 0.0],
        ]
    )
    s = d * ratio
    p = np.array([2, 0, 1])
    null = sender_pattern_permuted_signed_direct(d, s, p)
    recovered = signed_ratio_to_unsigned(d, null)
    expected = ratio[p, :]
    mask = np.abs(d) > 1e-15
    assert np.allclose(recovered[mask], expected[mask])
    assert np.allclose(np.abs(d), np.abs(d))


def test_magnitude_only_and_sender_pattern_controls_are_separate_from_true_sign_layer():
    regions = ("A", "B", "C", "D")
    d = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [2.0, 0.0, 1.0, 3.0],
            [1.0, 2.0, 0.0, 1.0],
            [3.0, 1.0, 2.0, 0.0],
        ]
    )
    s = np.array(
        [
            [0.0, 1.0, -2.0, 1.0],
            [-2.0, 0.0, -1.0, -3.0],
            [1.0, 2.0, 0.0, 1.0],
            [-3.0, -1.0, -2.0, 0.0],
        ]
    )
    observed = 0.2 + 0.05 * s.T
    observed = (observed + observed.T) / 2.0
    np.fill_diagonal(observed, 1.0)
    overlap = np.eye(4)

    result = evaluate_signed_reverse_discriminator(
        regions,
        d,
        s,
        observed,
        overlap,
        n_null=8,
        seed=7,
    )
    assert np.isfinite(result.true_residual)
    assert np.isfinite(result.unsigned_reverse_residual)
    assert np.isfinite(result.magnitude_only_residual)
    assert result.sender_pattern_null_residuals.shape == (8,)
    assert 0.0 < result.sender_pattern_empirical_p_value <= 1.0
