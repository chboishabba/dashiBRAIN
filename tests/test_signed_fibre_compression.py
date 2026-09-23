import numpy as np

from dashi.analysis.signed_fibre_compression import (
    low_rank_ratio_signed_direct,
    sender_scalar_signed_direct,
)


def test_sender_scalar_reconstructs_constant_sender_ratio_exactly():
    d = np.array(
        [
            [0.0, 2.0, 4.0],
            [1.0, 0.0, 3.0],
            [5.0, 2.0, 0.0],
        ]
    )
    tendency = np.array([0.5, -0.25, 1.0])
    s = d * tendency[:, None]
    reconstructed, inferred = sender_scalar_signed_direct(d, s)
    assert np.allclose(inferred, tendency)
    assert np.allclose(reconstructed, s)


def test_sender_scalar_does_not_preserve_within_sender_target_pattern():
    d = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    s = np.array(
        [
            [0.0, 2.0, -2.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
        ]
    )
    reconstructed, _ = sender_scalar_signed_direct(d, s)
    assert not np.allclose(reconstructed, s)
    assert np.all(reconstructed[d == 0] == 0)


def test_low_rank_ratio_preserves_unsigned_zero_support():
    d = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 4.0, 0.0],
        ]
    )
    s = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.5],
            [0.0, -2.0, 0.0],
        ]
    )
    compressed = low_rank_ratio_signed_direct(d, s, rank=1)
    assert np.all(compressed[d == 0] == 0)


def test_full_rank_ratio_reconstruction_recovers_signed_matrix():
    rng = np.random.default_rng(7)
    d = rng.uniform(0.1, 2.0, size=(5, 5))
    np.fill_diagonal(d, 0.0)
    ratio = rng.uniform(-1.0, 1.0, size=(5, 5))
    np.fill_diagonal(ratio, 0.0)
    s = d * ratio
    reconstructed = low_rank_ratio_signed_direct(d, s, rank=5)
    assert np.allclose(reconstructed, s)
