import csv
import os
import tempfile
import unittest

from dashi.io.hemibrain_loader import load_edge_list, validate_edge_list


class HemibrainLoaderTests(unittest.TestCase):
    def test_validate_edge_list_counts_and_rejects_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            edge_path = os.path.join(tmp, "edges.csv")
            with open(edge_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_id", "target_id", "weight"])
                writer.writerow(["a", "b", "1.0"])
                writer.writerow(["b", "c", "0"])
                writer.writerow(["c", "a", ""])

            summary = validate_edge_list(edge_path)
            self.assertEqual(summary.rows, 3)
            self.assertEqual(summary.nodes, 3)
            self.assertEqual(summary.zero_weight_edges, 1)
            self.assertEqual(summary.missing_weights, 1)

            bad_path = os.path.join(tmp, "edges_bad.csv")
            with open(bad_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_id", "target_id", "weight"])
                writer.writerow(["a", "b", "-1"])
            with self.assertRaises(ValueError):
                validate_edge_list(bad_path)

    def test_load_edge_list_with_partition_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            edge_path = os.path.join(tmp, "edges.csv")
            meta_path = os.path.join(tmp, "meta.csv")

            with open(edge_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source_id", "target_id", "weight"])
                writer.writerow(["n1", "n2", "2"])
                writer.writerow(["n2", "n3", "1"])
                writer.writerow(["n3", "n1", "1"])

            with open(meta_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["neuron_id", "cell_type"])
                writer.writerow(["n1", "T1"])
                writer.writerow(["n2", "T2"])
                writer.writerow(["n3", "T1"])

            hb = load_edge_list(
                edge_path,
                metadata_path=meta_path,
                partition_fields=["cell_type"],
                validate=True,
            )
            self.assertIsNotNone(hb.partitions)
            self.assertEqual(hb.carrier.adjacency.shape[0], 3)
            self.assertEqual(hb.partitions["cell_type"].tolist(), ["T1", "T2", "T1"])


if __name__ == "__main__":
    unittest.main()
