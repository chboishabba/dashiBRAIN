import unittest

import numpy as np
import scipy.sparse as sp

from dashi.analysis.nonlinear_sparsity import (
    constraint_diagnostics,
    explode_state,
    low_margin_mask,
    ternary_project,
)


class NonlinearSparsityTests(unittest.TestCase):
    def test_exact_threshold_sparsification(self) -> None:
        field = np.array([-2.0, -0.2, 0.0, 0.2, 2.0])
        projected = ternary_project(field, deadzone=0.25)
        np.testing.assert_array_equal(projected, np.array([-1, 0, 0, 0, 1], dtype=np.int8))

    def test_fixed_point_is_weighted_csp_solution(self) -> None:
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        )
        state = np.array([1, 1, 1], dtype=np.int8)
        diagnostics = constraint_diagnostics(adjacency, state, deadzone=0.0)
        self.assertEqual(diagnostics.defect, 0)
        self.assertTrue((diagnostics.margins > 0).all())

    def test_violated_constraints_equal_kernel_defect(self) -> None:
        adjacency = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
        state = np.array([1, -1], dtype=np.int8)
        diagnostics = constraint_diagnostics(adjacency, state, deadzone=0.0)
        self.assertEqual(diagnostics.defect, 2)
        np.testing.assert_array_equal(diagnostics.projected, np.array([-1, 1], dtype=np.int8))

    def test_exploded_state_tracks_components_and_changes(self) -> None:
        adjacency = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=float,
            )
        )
        state = np.array([1, 1, 0, -1], dtype=np.int8)
        previous = np.array([1, 0, 0, -1], dtype=np.int8)
        exploded = explode_state(adjacency, state, previous)

        self.assertEqual(int(exploded.positive.sum()), 2)
        self.assertEqual(int(exploded.neutral.sum()), 1)
        self.assertEqual(int(exploded.negative.sum()), 1)
        self.assertEqual(int(exploded.changed.sum()), 1)
        self.assertEqual(len(set(exploded.positive_component_labels[exploded.positive])), 1)

    def test_low_margin_mask(self) -> None:
        margins = np.array([0.1, 0.5, 1.0])
        np.testing.assert_array_equal(low_margin_mask(margins, 0.5), np.array([True, True, False]))


if __name__ == "__main__":
    unittest.main()
