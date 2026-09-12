import numpy as np
import pytest

from dashi.analysis.local_fibre_hyperfabric import (
    BaseLocality,
    LocalFibreCoordinate,
    PantsPatch,
    Seam,
    SymmetryAction,
    add_pair_composition_incidences,
    chart_round_trip_exact,
    hyperfabric_from_structural_family,
)
from dashi.analysis.ndim_structure_function import StructuralFibreFamily


def _family():
    regions = ("A", "B")
    return StructuralFibreFamily(
        regions,
        {
            "direct_forward": np.array([[0.0, 1.0], [2.0, 0.0]]),
            "signed_reverse": np.array([[0.0, -2.0], [1.0, 0.0]]),
        },
    )


def test_legacy_chart_round_trips_exactly():
    family = _family()
    assert chart_round_trip_exact(family)
    fabric = hyperfabric_from_structural_family(family)
    projected = fabric.project_chart(("signed_reverse", "direct_forward"))
    assert np.array_equal(projected.fibres["direct_forward"], family.fibres["direct_forward"])
    assert np.array_equal(projected.fibres["signed_reverse"], family.fibres["signed_reverse"])


def test_any_number_of_fibres_can_be_added_at_one_crossing_without_global_cardinality_change():
    fabric = hyperfabric_from_structural_family(_family())
    loc = BaseLocality("A", "B")
    refined = fabric.refine_at_crossing(
        loc,
        (
            LocalFibreCoordinate("delay", 3.0, "hop/time"),
            LocalFibreCoordinate("phase", 0.25, "phase"),
            LocalFibreCoordinate("trial", 7.0, "trial"),
        ),
    )
    assert refined.fibre_names_at(loc)[-3:] == ("delay", "phase", "trial")
    assert "delay" not in refined.fibre_names_at(BaseLocality("B", "A"))
    assert refined.crossings[-1].participating_fibres == ("delay", "phase", "trial")


def test_refinement_rejects_duplicate_local_coordinate_name():
    fabric = hyperfabric_from_structural_family(_family())
    loc = BaseLocality("A", "B")
    with pytest.raises(ValueError):
        fabric.refine_at_crossing(
            loc, (LocalFibreCoordinate("direct_forward", 99.0, "bad"),)
        )


def test_pair_composition_is_base_incidence_not_automatic_fibre_transport():
    fabric = add_pair_composition_incidences(hyperfabric_from_structural_family(_family()))
    assert len(fabric.incidences) == 2
    assert all(edge.relation == "pair-composition" for edge in fabric.incidences)
    assert fabric.pants_patches == ()
    assert fabric.crossings == ()


def test_pants_gluing_requires_interface_receipt_and_does_not_delete_local_fibres():
    fabric = hyperfabric_from_structural_family(_family())
    ab = BaseLocality("A", "B")
    ba = BaseLocality("B", "A")
    patch = PantsPatch(
        inputs=(ab,),
        outputs=(ba,),
        seams=(Seam(ab, ba, "shared-boundary", True, "typed seam receipt"),),
        receipt="one-to-one finite specimen; n-ary tuples are supported",
    )
    glued = fabric.with_pants_patch(patch)
    assert glued.pants_patches[-1].is_gluable
    assert glued.fibre_names_at(ab) == fabric.fibre_names_at(ab)
    assert glued.fibre_names_at(ba) == fabric.fibre_names_at(ba)

    bad = PantsPatch(
        inputs=(ab,), outputs=(ba,),
        seams=(Seam(ab, ba, "shared-boundary", False, "mismatch"),),
        receipt="bad",
    )
    with pytest.raises(ValueError):
        fabric.with_pants_patch(bad)


def test_symmetry_is_not_quotient_authority_without_consumer_invariance():
    fabric = hyperfabric_from_structural_family(_family())
    aa, ab = BaseLocality("A", "A"), BaseLocality("A", "B")
    ba, bb = BaseLocality("B", "A"), BaseLocality("B", "B")
    swap = SymmetryAction(
        "swap A/B",
        {aa: bb, bb: aa, ab: ba, ba: ab},
        {},
    )
    fabric = fabric.with_symmetry(swap)

    invariant_consumer = lambda locality, values: locality.source == locality.target
    assert fabric.symmetry_preserves_consumer(swap, invariant_consumer)

    value_consumer = lambda locality, values: values["direct_forward"]
    assert not fabric.symmetry_preserves_consumer(swap, value_consumer)
