import numpy as np

from dashi.analysis.network_scale_shape_discriminator import (
    decompose_network_scale_shape,
    reconstruct_signed_from_scale_shape,
    reconstruct_unsigned_from_scale_shape,
    sender_strength_only_matrix,
)


def test_scale_shape_decomposition_reconstructs_unsigned_and_signed():
    d = np.array(
        [
            [0.0, 2.0, 1.0],
            [4.0, 0.0, 2.0],
            [1.0, 3.0, 0.0],
        ]
    )
    ratio = np.array(
        [
            [0.0, 0.5, -0.25],
            [1.0, 0.0, -0.5],
            [0.2, 0.4, 0.0],
        ]
    )
    s = d * ratio
    dec = decompose_network_scale_shape(d, s)
    assert np.allclose(reconstruct_unsigned_from_scale_shape(dec), d)
    assert np.allclose(reconstruct_signed_from_scale_shape(dec), s)


def test_row_shape_removes_sender_scale_but_preserves_relative_pattern():
    d = np.array(
        [
            [0.0, 2.0, 1.0],
            [8.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
        ]
    )
    dec = decompose_network_scale_shape(d, d)
    assert np.allclose(dec.row_shape[0], [0.0, 2 / 3, 1 / 3])
    assert np.allclose(dec.row_shape[1], [2 / 3, 0.0, 1 / 3])
    assert np.allclose(dec.row_shape[2], 0.0)


def test_strength_only_erases_target_specific_shape():
    d = np.array(
        [
            [0.0, 4.0, 2.0],
            [3.0, 0.0, 1.0],
            [2.0, 6.0, 0.0],
        ]
    )
    strength_only = sender_strength_only_matrix(d)
    assert np.allclose(np.sum(np.abs(strength_only), axis=1), np.sum(np.abs(d), axis=1))
    assert np.allclose(np.diag(strength_only), 0.0)
    assert np.isclose(strength_only[0, 1], strength_only[0, 2])
