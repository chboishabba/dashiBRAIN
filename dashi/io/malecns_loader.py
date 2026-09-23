"""Loader for MaleCNS connectome weights, annotations, and transmitter signs.

The canonical MaleCNS edge table contains tens of millions of rows. Feather
inputs therefore use a vectorized Arrow/NumPy path and never expand every edge
column into Python objects. CSV/TSV remains available as a small-fixture and
compatibility path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import csv

import numpy as np
import scipy.sparse as sp

from dashi.io.hemibrain_loader import HemibrainGraph


def _read_table(path: str | Path) -> Mapping[str, Sequence]:
    """Read a modest metadata/fixture table into column sequences.

    Do not use this helper for the canonical 25M-row edge Feather; that file is
    handled by ``_load_feather_edges_vectorized`` below.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix == ".feather":
        try:
            import pyarrow.feather as feather
            table = feather.read_table(p, memory_map=True)
            out: dict[str, Sequence] = {}
            for name in table.column_names:
                col = table[name].combine_chunks()
                if getattr(col.type, "is_integer", False) or getattr(col.type, "is_floating", False):
                    out[name] = col.to_numpy(zero_copy_only=False)
                else:
                    out[name] = col.to_pylist()
            return out
        except ImportError:
            import pandas as pd
            df = pd.read_feather(p)
            return {col: df[col].to_numpy(copy=False) for col in df.columns}

    with open(p, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "," if "," in sample else "\t"
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        if not rows:
            return {}
        cols = list(rows[0].keys())
        return {c: [r[c] for r in rows] for c in cols}


def infer_transmitter_sign(predicted_nt: str | None, default_sign: int = 1) -> int:
    """Infer a coarse signed-transmission feature.

    The sign is a modelling feature, not a claim that every transmitter has a
    globally fixed postsynaptic effect. Acetylcholine is assigned +1 and the
    canonical GABA/glutamate inhibitory classes -1 for the present benchmark;
    modulators fall back to +1 unless a later receptor-aware model is supplied.
    """
    if not predicted_nt:
        return default_sign
    nt = str(predicted_nt).lower().strip()
    if "acetylcholine" in nt or nt == "ach":
        return 1
    if "gaba" in nt or "glutamate" in nt or nt == "glu":
        return -1
    if "dopamine" in nt or "serotonin" in nt or "octopamine" in nt:
        return 1
    return default_sign


def _resolve_column(names: Sequence[str], preferred: str, aliases: Sequence[str]) -> str:
    if preferred in names:
        return preferred
    for alias in aliases:
        if alias in names:
            return alias
    raise KeyError(f"Missing required column {preferred!r}; available columns: {list(names)!r}")


def _load_feather_edges_vectorized(
    path: Path,
    *,
    pre_col: str,
    post_col: str,
    weight_col: str,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted ids, row indices, column indices and weights.

    Peak memory is dominated by NumPy integer arrays and the final sparse
    carrier rather than tens of millions of Python ``int`` objects.
    """
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise ImportError("pyarrow is required for large MaleCNS Feather ingestion") from exc

    # Canonical MaleCNS uses these names. If a future version changes them, read
    # only the tiny schema metadata first by opening the table and resolving the
    # aliases rather than converting payload columns to Python lists.
    table = feather.read_table(path, memory_map=True)
    names = table.column_names
    pre = _resolve_column(names, pre_col, ("source_id", "pre", "bodyId_pre", "body_pre"))
    post = _resolve_column(names, post_col, ("target_id", "post", "bodyId_post", "body_post"))
    weight = _resolve_column(names, weight_col, ("weight", "syn_count", "count"))

    pres = table[pre].combine_chunks().to_numpy(zero_copy_only=False)
    posts = table[post].combine_chunks().to_numpy(zero_copy_only=False)
    weights = table[weight].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

    keep = np.asarray(weights >= float(min_weight), dtype=bool)
    if not np.all(keep):
        pres = pres[keep]
        posts = posts[keep]
        weights = weights[keep]

    if pres.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), weights

    ids = np.unique(np.concatenate((pres, posts)))
    rows = np.searchsorted(ids, pres).astype(np.int64, copy=False)
    cols = np.searchsorted(ids, posts).astype(np.int64, copy=False)
    return ids, rows, cols, weights


def _transmitter_sign_vector(
    ids: np.ndarray,
    neurotransmitters_path: str | Path,
    *,
    default_sign: int = 1,
) -> np.ndarray:
    table = _read_table(neurotransmitters_path)
    body_col = "body" if "body" in table else ("bodyId" if "bodyId" in table else "neuron_id")
    if body_col not in table:
        return np.full(ids.size, default_sign, dtype=np.float32)

    # Prefer consensus_nt from the actual MaleCNS schema; retain compatibility
    # with earlier predicted_nt fixtures.
    nt_col = (
        "consensus_nt" if "consensus_nt" in table
        else "predicted_nt" if "predicted_nt" in table
        else "neurotransmitter" if "neurotransmitter" in table
        else None
    )
    if nt_col is None:
        return np.full(ids.size, default_sign, dtype=np.float32)

    signs = np.full(ids.size, default_sign, dtype=np.float32)
    id_lookup = {str(v): i for i, v in enumerate(ids.tolist())}
    for body, nt in zip(table[body_col], table[nt_col]):
        idx = id_lookup.get(str(body))
        if idx is not None and nt not in (None, ""):
            signs[idx] = float(infer_transmitter_sign(str(nt), default_sign))
    return signs


def _aligned_annotations(ids: np.ndarray, annotations_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    table = _read_table(annotations_path)
    body_col = "bodyId" if "bodyId" in table else ("body" if "body" in table else "neuron_id")
    if body_col not in table:
        return {}, {}

    graph_lookup = {str(v): i for i, v in enumerate(ids.tolist())}
    row_positions: list[tuple[int, int]] = []
    for source_i, body in enumerate(table[body_col]):
        target_i = graph_lookup.get(str(body))
        if target_i is not None:
            row_positions.append((source_i, target_i))

    metadata: dict[str, np.ndarray] = {}
    partitions: dict[str, np.ndarray] = {}
    n = ids.size
    for col, source_values in table.items():
        if col == body_col:
            continue
        aligned = np.full(n, "", dtype=object)
        for source_i, target_i in row_positions:
            value = source_values[source_i]
            aligned[target_i] = "" if value is None else str(value)
        metadata[col] = aligned

        lc = col.lower()
        if "class" in lc:
            for cls in set(aligned.tolist()):
                if cls:
                    partitions[f"{col}:{cls}"] = np.asarray(aligned == cls, dtype=bool)
        if "dimorphism" in lc or lc in {"sex", "sexcorrespondence", "sex_correspondence"}:
            partitions["is_dimorphic"] = np.asarray(
                [bool(x and str(x).lower() not in ("none", "false", "0", "unisex", "isomorphic")) for x in aligned],
                dtype=bool,
            )

    # Actual MaleCNS annotations expose somaNeuromere. Surface it as an explicit
    # biological partition family without asserting that a soma neuromere is a
    # functional neuropil or a neuron identity.
    if "somaNeuromere" in metadata:
        for region in set(metadata["somaNeuromere"].tolist()):
            if region:
                partitions[f"somaNeuromere:{region}"] = np.asarray(metadata["somaNeuromere"] == region, dtype=bool)

    return metadata, partitions


def load_malecns_graph(
    weights_path: str | Path,
    *,
    annotations_path: str | Path | None = None,
    neurotransmitters_path: str | Path | None = None,
    pre_col: str = "body_pre",
    post_col: str = "body_post",
    weight_col: str = "weight",
    signed_by_transmitter: bool = False,
    min_weight: float = 1.0,
) -> HemibrainGraph:
    """Load MaleCNS weights and metadata into a CSR ``HemibrainGraph``."""
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix == ".feather":
        ids, row_idx, col_idx, values = _load_feather_edges_vectorized(
            p,
            pre_col=pre_col,
            post_col=post_col,
            weight_col=weight_col,
            min_weight=min_weight,
        )
        if signed_by_transmitter and neurotransmitters_path and ids.size:
            signs = _transmitter_sign_vector(ids, neurotransmitters_path)
            values = values * signs[row_idx]
    else:
        table = _read_table(p)
        names = list(table)
        pre = _resolve_column(names, pre_col, ("source_id", "pre", "bodyId_pre", "body_pre"))
        post = _resolve_column(names, post_col, ("target_id", "post", "bodyId_post", "body_post"))
        weight = weight_col if weight_col in table else next((x for x in ("weight", "syn_count", "count") if x in table), None)
        pres = np.asarray(table[pre], dtype=object)
        posts = np.asarray(table[post], dtype=object)
        raw = np.ones(pres.size, dtype=np.float32) if weight is None else np.asarray(table[weight], dtype=np.float32)
        keep = raw >= float(min_weight)
        pres, posts, values = pres[keep], posts[keep], raw[keep]
        ids = np.unique(np.concatenate((pres.astype(str), posts.astype(str))))
        row_idx = np.searchsorted(ids, pres.astype(str)).astype(np.int64, copy=False)
        col_idx = np.searchsorted(ids, posts.astype(str)).astype(np.int64, copy=False)
        if signed_by_transmitter and neurotransmitters_path and ids.size:
            signs = _transmitter_sign_vector(ids, neurotransmitters_path)
            values = values * signs[row_idx]

    n = int(ids.size)
    csr = sp.coo_matrix((values, (row_idx, col_idx)), shape=(n, n), dtype=np.float32).tocsr() if n else sp.csr_matrix((0, 0), dtype=np.float32)
    csr.sum_duplicates()

    idx_to_id = [str(v) for v in ids.tolist()]
    id_to_idx = {node_id: i for i, node_id in enumerate(idx_to_id)}

    metadata: dict[str, np.ndarray] | None = None
    partitions: dict[str, np.ndarray] | None = None
    if annotations_path and n:
        metadata, partitions = _aligned_annotations(ids, annotations_path)

    return HemibrainGraph(
        carrier=csr,
        id_map=id_to_idx,
        idx_to_id=idx_to_id,
        metadata=metadata,
        partitions=partitions,
    )
