import unittest
import numpy as np
import scipy.sparse as sp

from dashi.baseline.residuals import compute_residuals
from dashi.baseline.sparse_degree_corrected import compute_sparse_dc_residual


class SparseBaselineTests(unittest.TestCase):
    def test_sparse_matches_dense_on_small_graph(self):
        data = np.array([1, 2, 3, 4], dtype=np.float32)
        rows = np.array([0, 0, 1, 2])
        cols = np.array([1, 2, 2, 0])
        A = sp.csr_matrix((data, (rows, cols)), shape=(3, 3))

        dense_residual, dense_balance = compute_residuals(A, deadzone=0.0)
        sparse_residual, sparse_balance = compute_sparse_dc_residual(A, deadzone=0.0)

        np.testing.assert_allclose(sparse_residual.toarray(), dense_residual.toarray())
        self.assertEqual(dense_balance, sparse_balance)


if __name__ == "__main__":
    unittest.main()
