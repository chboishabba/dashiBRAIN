"""Loader for MaleCNS connectome weights, annotations, and transmitter signs.

This module converts Apache Arrow Feather or CSV connectome tables into
standardized HemibrainGraph (CSR sparse carrier) instances with aligned
metadata, partition masks, and sign assignments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
import numpy as np
import scipy.sparse as sp

from dashi.io.hemibrain_loader import HemibrainGraph
from dashi.types import GraphCarrier


def _read_table(path: str | Path) -> Mapping[str, Sequence]:
    """Read Feather or CSV table into a dictionary of column sequences."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix == ".feather":
        try:
            import pyarrow.feather as feather
            table = feather.read_table(p)
            return {name: table[name].to_pylist() for name in table.column_names}
        except ImportError:
            import pandas as pd
            df = pd.read_feather(p)
            return {col: df[col].to_list() for col in df.columns}
    else:
        # Assume CSV or TSV
        import csv
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


def infer_transmitter_sign(
    predicted_nt: str | None,
    default_sign: int = 1,
) -> int:
    """Infer ternary sign (+1 exc, -1 inh, 0 neutral) from predicted neurotransmitter."""
    if not predicted_nt:
        return default_sign
    nt = str(predicted_nt).lower().strip()
    if "acetylcholine" in nt or "ach" in nt:
        return 1
    if "gaba" in nt or "glutamate" in nt or "glu" in nt:
        return -1
    if "dopamine" in nt or "serotonin" in nt or "octopamine" in nt:
        return 1
    return default_sign


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
    """Load MaleCNS weights and metadata into a HemibrainGraph (CSR)."""
    weights_table = _read_table(weights_path)
    if pre_col not in weights_table or post_col not in weights_table:
        # Fallback for common column variants
        candidates = {
            pre_col: ("source_id", "pre", "bodyId_pre", "body_pre"),
            post_col: ("target_id", "post", "bodyId_post", "body_post"),
            weight_col: ("weight", "syn_count", "count"),
        }
        for target, aliases in candidates.items():
            if target not in weights_table:
                for alias in aliases:
                    if alias in weights_table:
                        if target == pre_col:
                            pre_col = alias
                        elif target == post_col:
                            post_col = alias
                        elif target == weight_col:
                            weight_col = alias
                        break

    pres = weights_table[pre_col]
    posts = weights_table[post_col]
    weights_raw = weights_table.get(weight_col, [1.0] * len(pres))

    # Optional transmitter sign mapping
    nt_map: dict[str, str] = {}
    if neurotransmitters_path:
        nt_table = _read_table(neurotransmitters_path)
        body_col = "body" if "body" in nt_table else ("bodyId" if "bodyId" in nt_table else "neuron_id")
        pred_col = "predicted_nt" if "predicted_nt" in nt_table else "neurotransmitter"
        if body_col in nt_table and pred_col in nt_table:
            for b, nt in zip(nt_table[body_col], nt_table[pred_col]):
                nt_map[str(b)] = str(nt)

    id_to_idx: dict[str, int] = {}
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []

    for pre, post, w in zip(pres, posts, weights_raw):
        w_float = float(w) if w is not None and str(w).strip() != "" else 1.0
        if w_float < min_weight:
            continue

        pre_s = str(pre)
        post_s = str(post)

        if pre_s not in id_to_idx:
            id_to_idx[pre_s] = len(id_to_idx)
        if post_s not in id_to_idx:
            id_to_idx[post_s] = len(id_to_idx)

        val = w_float
        if signed_by_transmitter:
            sign = infer_transmitter_sign(nt_map.get(pre_s))
            val *= sign

        row_idx.append(id_to_idx[pre_s])
        col_idx.append(id_to_idx[post_s])
        values.append(val)

    n = len(id_to_idx)
    if n == 0:
        csr = sp.csr_matrix((0, 0), dtype=np.float32)
    else:
        coo = sp.coo_matrix((values, (row_idx, col_idx)), shape=(n, n), dtype=np.float32)
        csr = coo.tocsr()

    idx_to_id = [None] * n
    for node_id, idx in id_to_idx.items():
        idx_to_id[idx] = node_id

    # Align annotations if provided
    metadata: dict[str, np.ndarray] | None = None
    partitions: dict[str, np.ndarray] | None = None

    if annotations_path and n > 0:
        ann_table = _read_table(annotations_path)
        body_col = "body" if "body" in ann_table else ("bodyId" if "bodyId" in ann_table else "neuron_id")
        if body_col in ann_table:
            ann_map: dict[str, dict[str, str]] = {}
            for i, b in enumerate(ann_table[body_col]):
                b_str = str(b)
                ann_map[b_str] = {c: str(ann_table[c][i]) for c in ann_table if c != body_col}

            metadata = {}
            partitions = {}
            for col in ann_table:
                if col == body_col:
                    continue
                aligned_col = [ann_map.get(node_id, {}).get(col, "") for node_id in idx_to_id]
                metadata[col] = np.array(aligned_col, dtype=object)

                # Identify partitions: e.g. class, dimorphism
                if "class" in col.lower():
                    classes = set(aligned_col)
                    for cls in classes:
                        if cls:
                            partitions[f"class:{cls}"] = np.array([x == cls for x in aligned_col], dtype=bool)
                if "dimorphism" in col.lower() or "sex" in col.lower():
                    partitions["is_dimorphic"] = np.array(
                        [bool(x and x.lower() not in ("none", "false", "0", "unisex")) for x in aligned_col],
                        dtype=bool,
                    )

    return HemibrainGraph(
        carrier=csr,
        id_map=id_to_idx,
        idx_to_id=idx_to_id,
        metadata=metadata,
        partitions=partitions,
    )
