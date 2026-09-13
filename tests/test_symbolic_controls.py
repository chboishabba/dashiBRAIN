import numpy as np
import scipy.sparse as sp

from dashi.analysis.symbolic_controls import (
    degree_and_strength_preserving_rewire,
    shuffle_identity_assignment,
)


def weighted_cycle() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )


def binary_degrees(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    B = A.copy()
    B.data = np.ones_like(B.data)
    return (
        np.asarray(B.sum(axis=1)).reshape(-1),
        np.asarray(B.sum(axis=0)).reshape(-1),
    )


def strengths(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(A.sum(axis=1)).reshape(-1),
        np.asarray(A.sum(axis=0)).reshape(-1),
    )


def test_rewire_preserves_binary_degree_and_weighted_strength():
    A = weighted_cycle()
    result = degree_and_strength_preserving_rewire(A, swaps=1, seed=7)

    out0, in0 = binary_degrees(A)
    out1, in1 = binary_degrees(result.adjacency)
    sout0, sin0 = strengths(A)
    sout1, sin1 = strengths(result.adjacency)

    assert result.swaps_completed == 1
    assert np.array_equal(out0, out1)
    assert np.array_equal(in0, in1)
    assert np.array_equal(sout0, sout1)
    assert np.array_equal(sin0, sin1)
    assert (A != result.adjacency).nnz > 0


def test_rewire_is_deterministic_for_seed():
    A = weighted_cycle()
    first = degree_and_strength_preserving_rewire(A, swaps=1, seed=11)
    second = degree_and_strength_preserving_rewire(A, swaps=1, seed=11)
    assert (first.adjacency != second.adjacency).nnz == 0


def test_rewire_rejects_non_square_carrier():
    A = sp.csr_matrix(np.ones((2, 3), dtype=np.float32))
    try:
        degree_and_strength_preserving_rewire(A, swaps=1, seed=0)
    except ValueError as exc:
        assert "square" in str(exc)
    else:
        raise AssertionError("non-square topology must be rejected")


def test_identity_shuffle_preserves_identity_multiset_and_changes_assignment():
    identities = ["n0", "n1", "n2", "n3"]
    shuffled = shuffle_identity_assignment(identities, seed=7)
    assert sorted(shuffled) == sorted(identities)
    assert shuffled != identities


def test_identity_shuffle_is_deterministic_for_seed():
    identities = ["n0", "n1", "n2", "n3"]
    assert shuffle_identity_assignment(identities, seed=3) == shuffle_identity_assignment(
        identities,
        seed=3,
    )
