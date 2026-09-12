import numpy as np

from dashi.analysis.sender_magnitude_shape_composition import (
    compose_sender_magnitude_shape,
)


def test_joint_composition_matches_declared_formulas():
    d = np.array(
        [
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
            [1.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    # Row tendencies are +0.5, -0.25, +0.8 relative to D row mass.
    a = np.array([0.5, -0.25, 0.8])
    s = d * a[:, None]

    comp = compose_sender_magnitude_shape(d, s)
    strength = np.sum(np.abs(d), axis=1)
    p = d / strength[:, None]

    assert np.allclose(comp.sender_tendency, a)
    assert np.allclose(comp.sender_magnitude, np.abs(a))
    assert np.allclose(comp.relative_shape, p)
    assert np.allclose(comp.signed_sender_density, a[:, None] * d)
    assert np.allclose(comp.magnitude_sender_density, np.abs(a)[:, None] * d)
    assert np.allclose(comp.signed_sender_shape, a[:, None] * p)
    assert np.allclose(comp.magnitude_sender_shape, np.abs(a)[:, None] * p)


def test_joint_quotient_removes_sender_scale_and_binary_polarity():
    d = np.array(
        [
            [0.0, 9.0, 3.0],
            [1.0, 0.0, 1.0],
            [2.0, 6.0, 0.0],
        ],
        dtype=float,
    )
    a = np.array([0.7, -0.4, 0.2])
    s = d * a[:, None]

    comp = compose_sender_magnitude_shape(d, s)
    row_sums = np.sum(comp.relative_shape, axis=1)

    # Relative shape is row-normalized: sender scale has been removed.
    assert np.allclose(row_sums, np.ones(3))
    # Magnitude-shape keeps sender tendency magnitude but erases polarity.
    assert np.all(comp.magnitude_sender_shape >= 0.0)
    assert np.allclose(
        np.sum(comp.magnitude_sender_shape, axis=1),
        np.abs(a),
    )


def test_zero_strength_sender_is_well_defined():
    d = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    s = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    comp = compose_sender_magnitude_shape(d, s)

    assert np.allclose(comp.relative_shape[0], 0.0)
    assert comp.sender_tendency[0] == 0.0
    assert comp.sender_magnitude[0] == 0.0
    assert np.allclose(comp.magnitude_sender_shape[0], 0.0)
