"""Transmitters -> row-signed adjacency for an already loaded MaleCNS graph.

This avoids reading the 25M-edge Feather a second time merely to construct the
signed structural feature. The sign remains a coarse benchmark feature rather
than a receptor-resolved physiological claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from dashi.io.hemibrain_loader import HemibrainGraph
from dashi.io.malecns_loader import infer_transmitter_sign


def _read_nt_columns(path: str | Path) -> tuple[list[str], list[str]]:
    p = Path(path)
    if p.suffix == ".feather":
        import pyarrow.feather as feather
        table = feather.read_table(p, memory_map=True)
        names = table.column_names
        body_col = "body" if "body" in names else ("bodyId" if "bodyId" in names else "neuron_id")
        nt_col = (
            "consensus_nt" if "consensus_nt" in names
            else "predicted_nt" if "predicted_nt" in names
            else "neurotransmitter"
        )
        bodies = table[body_col].combine_chunks().to_pylist()
        nts = table[nt_col].combine_chunks().to_pylist()
        return [str(x) for x in bodies], ["" if x is None else str(x) for x in nts]

    import csv
    with open(p, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "," if "," in sample else "\t"
        rows = list(csv.DictReader(f, delimiter=delimiter))
    if not rows:
        return [], []
    names = rows[0].keys()
    body_col = "body" if "body" in names else ("bodyId" if "bodyId" in names else "neuron_id")
    nt_col = "consensus_nt" if "consensus_nt" in names else ("predicted_nt" if "predicted_nt" in names else "neurotransmitter")
    return [str(r[body_col]) for r in rows], [str(r.get(nt_col, "")) for r in rows]


def transmitter_signs_for_graph(
    graph: HemibrainGraph,
    neurotransmitters_path: str | Path,
    *,
    default_sign: int = 1,
) -> np.ndarray:
    signs = np.full(len(graph.idx_to_id), float(default_sign), dtype=np.float32)
    bodies, nts = _read_nt_columns(neurotransmitters_path)
    for body, nt in zip(bodies, nts):
        idx = graph.id_map.get(body)
        if idx is not None and nt:
            signs[idx] = float(infer_transmitter_sign(nt, default_sign))
    return signs


def signed_adjacency_for_graph(
    graph: HemibrainGraph,
    neurotransmitters_path: str | Path,
    *,
    default_sign: int = 1,
) -> sp.csr_matrix:
    signs = transmitter_signs_for_graph(graph, neurotransmitters_path, default_sign=default_sign)
    # Presynaptic transmitter sign multiplies outgoing rows.
    return (sp.diags(signs) @ graph.carrier.tocsr()).tocsr()
