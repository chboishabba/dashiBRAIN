"""Unit tests for MaleCNS connectome feather/CSV loader."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from dashi.io.malecns_loader import infer_transmitter_sign, load_malecns_graph


def test_infer_transmitter_sign():
    assert infer_transmitter_sign("acetylcholine") == 1
    assert infer_transmitter_sign("ACh") == 1
    assert infer_transmitter_sign("gaba") == -1
    assert infer_transmitter_sign("glutamate") == -1
    assert infer_transmitter_sign("Glu") == -1
    assert infer_transmitter_sign("unknown", default_sign=1) == 1
    assert infer_transmitter_sign(None, default_sign=-1) == -1


def test_load_malecns_graph_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        weights_csv = tmp_path / "weights.csv"
        weights_csv.write_text(
            "body_pre,body_post,weight\n"
            "101,102,5.0\n"
            "102,103,3.0\n"
            "103,101,2.0\n",
            encoding="utf-8",
        )

        ann_csv = tmp_path / "annotations.csv"
        ann_csv.write_text(
            "body,class,dimorphism\n"
            "101,motor_neuron,dimorphic\n"
            "102,central_interneuron,unisex\n"
            "103,sensory,none\n",
            encoding="utf-8",
        )

        nt_csv = tmp_path / "neurotransmitters.csv"
        nt_csv.write_text(
            "body,predicted_nt\n"
            "101,acetylcholine\n"
            "102,gaba\n"
            "103,glutamate\n",
            encoding="utf-8",
        )

        graph = load_malecns_graph(
            weights_csv,
            annotations_path=ann_csv,
            neurotransmitters_path=nt_csv,
            signed_by_transmitter=True,
        )

        assert graph.carrier.shape == (3, 3)
        assert len(graph.idx_to_id) == 3
        assert set(graph.id_map.keys()) == {"101", "102", "103"}

        # Check signs: 101->102 is exc (+5.0), 102->103 is inh (-3.0), 103->101 is inh (-2.0)
        idx_101 = graph.id_map["101"]
        idx_102 = graph.id_map["102"]
        idx_103 = graph.id_map["103"]

        assert graph.carrier[idx_101, idx_102] == 5.0
        assert graph.carrier[idx_102, idx_103] == -3.0
        assert graph.carrier[idx_103, idx_101] == -2.0

        # Check metadata alignment
        assert graph.metadata is not None
        assert "class" in graph.metadata
        assert graph.metadata["class"][idx_101] == "motor_neuron"

        # Check partitions
        assert graph.partitions is not None
        assert "class:motor_neuron" in graph.partitions
        assert graph.partitions["class:motor_neuron"][idx_101] is True or graph.partitions["class:motor_neuron"][idx_101] == True
        assert graph.partitions["is_dimorphic"][idx_101] == True
        assert graph.partitions["is_dimorphic"][idx_102] == False


def test_load_malecns_graph_from_feather():
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.feather as feather

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        weights_feather = tmp_path / "weights.feather"

        data = {
            "body_pre": ["201", "202", "203"],
            "body_post": ["202", "203", "201"],
            "weight": [10.0, 20.0, 30.0],
        }
        import pyarrow as pa
        table = pa.Table.from_pydict(data)
        feather.write_feather(table, weights_feather)

        graph = load_malecns_graph(weights_feather)
        assert graph.carrier.shape == (3, 3)
        assert graph.carrier[graph.id_map["201"], graph.id_map["202"]] == 10.0
