import numpy as np

from dashi.analysis.malecns_physical_incidence import (
    ndim_chart_from_physical_region_fabric,
    physical_incidence_chart_round_trip_exact,
    physical_region_fabric_from_structural,
    structural_from_physical_region_fabric,
)
from dashi.analysis.ndim_structure_function import build_ndim_structural_fibres
from dashi.analysis.structure_function_real import RegionStructuralFeatures


def _structural():
    direct = np.array(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )
    row_mass = np.sum(np.abs(direct), axis=1)
    p = np.zeros_like(direct)
    nz = row_mass > 0
    p[nz] = direct[nz] / row_mass[nz, None]
    two_hop = p @ p
    signed = np.array(
        [
            [0.0, 1.5, 0.0],
            [-0.5, 0.0, 2.5],
            [0.0, -3.0, 0.0],
        ],
        dtype=float,
    )
    return RegionStructuralFeatures(("A", "B", "C"), direct, two_hop, signed)


def test_sparse_physical_incidence_keeps_only_nonzero_direct_support():
    structural = _structural()
    fabric = physical_region_fabric_from_structural(structural)
    assert fabric.possible_pair_count == 9
    assert fabric.incidence_count == 4
    assert {(e.source, e.target) for e in fabric.incidences} == {
        ("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")
    }


def test_sparse_physical_incidence_regenerates_structural_matrices_exactly():
    structural = _structural()
    fabric = physical_region_fabric_from_structural(structural)
    regenerated = structural_from_physical_region_fabric(fabric)
    assert np.array_equal(regenerated.direct, structural.direct)
    assert np.array_equal(regenerated.signed_direct, structural.signed_direct)
    assert np.array_equal(regenerated.two_hop, structural.two_hop)


def test_sparse_physical_incidence_regenerates_full_ndim_chart_exactly():
    structural = _structural()
    expected = build_ndim_structural_fibres(structural)
    regenerated = ndim_chart_from_physical_region_fabric(
        physical_region_fabric_from_structural(structural)
    )
    assert tuple(expected.fibres) == tuple(regenerated.fibres)
    for name in expected.fibres:
        assert np.array_equal(expected.fibres[name], regenerated.fibres[name])
    assert physical_incidence_chart_round_trip_exact(structural)


def test_signed_metadata_does_not_create_physical_edge_without_unsigned_support():
    structural = _structural()
    signed = structural.signed_direct.copy()
    signed[0, 2] = 99.0
    altered = RegionStructuralFeatures(
        structural.regions, structural.direct, structural.two_hop, signed
    )
    fabric = physical_region_fabric_from_structural(altered)
    assert ("A", "C") not in {(e.source, e.target) for e in fabric.incidences}
