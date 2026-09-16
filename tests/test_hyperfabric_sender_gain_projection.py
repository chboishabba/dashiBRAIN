import numpy as np

from dashi.analysis.hyperfabric_sender_gain_projection import (
    project_sender_gain_from_hyperfabric,
    sender_gain_projection_matches_direct_composition,
)
from dashi.analysis.local_fibre_hyperfabric import hyperfabric_from_structural_family
from dashi.analysis.ndim_structure_function import StructuralFibreFamily
from dashi.analysis.sender_magnitude_shape_composition import compose_sender_magnitude_shape


def _family():
    regions = ("A", "B", "C")
    d = np.array(
        [
            [0.0, 2.0, 1.0],
            [3.0, 0.0, 1.0],
            [1.0, 4.0, 0.0],
        ]
    )
    s = np.array(
        [
            [0.0, 2.0, 1.0],
            [-3.0, 0.0, -1.0],
            [1.0, 4.0, 0.0],
        ]
    )
    return StructuralFibreFamily(
        regions,
        {
            "direct_forward": d,
            "signed_forward": s,
            "direct_reverse": d.T,
            "signed_reverse": s.T,
        },
    ), d, s


def test_hyperfabric_sender_gain_projection_matches_direct_composition_exactly():
    family, d, s = _family()
    fabric = hyperfabric_from_structural_family(family)
    projected = project_sender_gain_from_hyperfabric(fabric)
    direct = compose_sender_magnitude_shape(d, s)

    assert np.array_equal(projected.unsigned_direct, d)
    assert np.array_equal(projected.signed_direct, s)
    assert np.array_equal(projected.composition.relative_shape, direct.relative_shape)
    assert np.array_equal(projected.composition.sender_magnitude, direct.sender_magnitude)
    assert np.array_equal(projected.magnitude_sender_shape, direct.magnitude_sender_shape)
    assert sender_gain_projection_matches_direct_composition(fabric, d, s)


def test_projection_depends_only_on_declared_chart_coordinates():
    family, d, s = _family()
    fabric = hyperfabric_from_structural_family(family)
    projected = project_sender_gain_from_hyperfabric(fabric)

    # Reverse coordinates remain present in the richer fabric but are not read
    # by this quotient adapter.
    assert np.array_equal(projected.unsigned_direct, family.fibres["direct_forward"])
    assert np.array_equal(projected.signed_direct, family.fibres["signed_forward"])


def test_missing_signed_coordinate_is_a_projection_failure_not_silent_fallback():
    family, _, _ = _family()
    incomplete = StructuralFibreFamily(
        family.regions,
        {"direct_forward": family.fibres["direct_forward"]},
    )
    fabric = hyperfabric_from_structural_family(incomplete)

    try:
        project_sender_gain_from_hyperfabric(fabric)
    except KeyError as exc:
        assert "signed_forward" in str(exc)
    else:
        raise AssertionError("missing signed coordinate must fail explicitly")
