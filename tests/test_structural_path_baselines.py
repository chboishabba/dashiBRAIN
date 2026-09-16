import numpy as np
import pytest

from dashi.analysis.structural_path_baselines import (
    directed_inverse_weight_shortest_paths,
    turner_style_path_family,
)


def test_shortest_path_uses_strong_indirect_route_over_weak_direct_edge():
    direct = np.array(
        [
            [0.0, 10.0, 1.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    distances = directed_inverse_weight_shortest_paths(direct)

    assert distances[0, 1] == pytest.approx(0.1)
    assert distances[1, 2] == pytest.approx(0.1)
    assert distances[0, 2] == pytest.approx(0.2)
    assert distances[2, 0] == np.inf


def test_shortest_path_retains_directionality():
    direct = np.array(
        [
            [0.0, 4.0, 0.0],
            [1.0, 0.0, 4.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    distances = directed_inverse_weight_shortest_paths(direct)

    assert distances[0, 2] == pytest.approx(0.5)
    assert distances[2, 0] == pytest.approx(0.125)
    assert distances[0, 2] != distances[2, 0]


def test_turner_style_family_contains_direct_and_path_forward_reverse_coordinates():
    direct = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.5, 0.0, 3.0],
            [4.0, 2.5, 0.0],
        ],
        dtype=float,
    )
    family = turner_style_path_family(("A", "B", "C"), direct)

    assert tuple(family.fibres) == (
        "direct_forward",
        "direct_reverse",
        "shortest_path_distance_forward",
        "shortest_path_distance_reverse",
    )
    assert np.array_equal(family.fibres["direct_forward"], direct)
    assert np.array_equal(family.fibres["direct_reverse"], direct.T)
    assert np.array_equal(
        family.fibres["shortest_path_distance_reverse"],
        family.fibres["shortest_path_distance_forward"].T,
    )


def test_turner_style_family_rejects_disconnected_off_diagonal_path_carrier():
    direct = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="finite directed shortest path"):
        turner_style_path_family(("A", "B", "C"), direct)
