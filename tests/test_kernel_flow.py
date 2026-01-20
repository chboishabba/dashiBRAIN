import unittest
import numpy as np
import scipy.sparse as sp

from dashi.kernel.flow import kernel_flow
from dashi.kernel.operator import k_hop_adj
from dashi.types import GraphCarrier, KernelParams


class KernelFlowTests(unittest.TestCase):
    def setUp(self):
        # Simple undirected chain 0-1-2
        data = np.array([1, 1, 1, 1], dtype=np.float32)
        rows = np.array([0, 1, 1, 2])
        cols = np.array([1, 0, 2, 1])
        A = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))
        self.carrier = GraphCarrier(adjacency=A, channels=1)
        self.params = KernelParams(hops=1, deadzone=0.0)

    def test_k_hop_expansion_contains_two_hop_paths(self):
        expanded = k_hop_adj(self.carrier.adjacency, hops=2)
        # Path 0 -> 1 -> 2 should appear in (0,2)
        self.assertGreater(expanded[0, 2], 0)
        # Original edge preserved
        self.assertGreater(expanded[0, 1], 0)

    def test_kernel_flow_converges_on_chain(self):
        s0 = np.ones((3, 1), dtype=np.int8)
        final, history, status = kernel_flow(self.carrier, s0, self.params, steps=5)
        self.assertEqual(status.get("status"), "converged")
        self.assertTrue(np.array_equal(final, s0))
        self.assertEqual(history[-1]["defect"], 0)

    def test_kernel_flow_is_deterministic(self):
        s0 = np.array([[1], [-1], [1]], dtype=np.int8)
        final1, history1, status1 = kernel_flow(self.carrier, s0, self.params, steps=5)
        final2, history2, status2 = kernel_flow(self.carrier, s0, self.params, steps=5)
        self.assertTrue(np.array_equal(final1, final2))
        self.assertEqual([h["defect"] for h in history1], [h["defect"] for h in history2])
        self.assertEqual(status1.get("status"), status2.get("status"))


if __name__ == "__main__":
    unittest.main()
