from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import scipy.sparse as sp

from dashi.types import GraphCarrier


@dataclass
class HemibrainGraph:
    carrier: GraphCarrier
    id_map: dict[str, int]
    idx_to_id: list[str]
    metadata: dict[str, np.ndarray] | None = None
    partitions: dict[str, np.ndarray] | None = None


@dataclass
class EdgeListSummary:
    rows: int
    nodes: int
    zero_weight_edges: int
    missing_weights: int
    delimiter: str


def _infer_delimiter(path: str, sample_bytes: int = 4096) -> str:
    with open(path, "r", newline="") as f:
        sample = f.read(sample_bytes)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ","


def load_edge_list(
    path: str,
    *,
    source_col: str = "source_id",
    target_col: str = "target_id",
    weight_col: str = "weight",
    delimiter: str | None = None,
    channels: int = 1,
    metadata_path: str | None = None,
    metadata_id_col: str = "neuron_id",
    partition_fields: Iterable[str] | None = None,
    validate: bool = False,
) -> HemibrainGraph:
    """
    Load hemibrain edge list into a GraphCarrier (CSR). IDs are mapped to dense indices.

    Weights must be non-negative. Duplicates are summed during CSR conversion.
    If `metadata_path` is provided, metadata columns are aligned to the same ordering.
    """
    summary = validate_edge_list(
        path,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        delimiter=delimiter,
    ) if validate else None

    delim = delimiter or (summary.delimiter if summary else _infer_delimiter(path))
    id_to_idx: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            try:
                src_id = row[source_col]
                tgt_id = row[target_col]
            except KeyError as exc:
                raise KeyError(f"Missing expected column: {exc.args[0]}") from exc

            if src_id not in id_to_idx:
                id_to_idx[src_id] = len(id_to_idx)
            if tgt_id not in id_to_idx:
                id_to_idx[tgt_id] = len(id_to_idx)

            src_idx = id_to_idx[src_id]
            tgt_idx = id_to_idx[tgt_id]
            weight_val = row.get(weight_col, "")
            weight = float(weight_val) if weight_val not in (None, "") else 1.0
            if weight < 0:
                raise ValueError("Weights must be non-negative")
            rows.append(src_idx)
            cols.append(tgt_idx)
            data.append(weight)

    n = len(id_to_idx)
    coo = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    A = coo.tocsr()
    idx_to_id = [None] * n
    for node_id, idx in id_to_idx.items():
        idx_to_id[idx] = node_id

    metadata = None
    if metadata_path:
        metadata = load_metadata_table(metadata_path, id_to_idx, id_col=metadata_id_col, delimiter=delimiter)

    partitions = None
    if partition_fields:
        if metadata is None:
            raise ValueError("partition_fields provided but no metadata file supplied.")
        missing = [field for field in partition_fields if field not in metadata]
        if missing:
            raise KeyError(f"Partition fields missing from metadata: {missing}")
        partitions = {field: metadata[field] for field in partition_fields}

    carrier = GraphCarrier(adjacency=A, channels=channels)
    return HemibrainGraph(
        carrier=carrier,
        id_map=id_to_idx,
        idx_to_id=idx_to_id,
        metadata=metadata,
        partitions=partitions,
    )


def load_metadata_table(
    path: str,
    id_map: dict[str, int],
    *,
    id_col: str = "neuron_id",
    delimiter: str | None = None,
) -> dict[str, np.ndarray]:
    """Align metadata table to existing id map. Non-numeric values stored as object arrays."""
    delim = delimiter or _infer_delimiter(path)
    field_values: dict[str, list[object]] = {}
    count = len(id_map)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = [col for col in reader.fieldnames or [] if col != id_col]
        for field in fields:
            field_values[field] = [None] * count
        for row in reader:
            node_id = row.get(id_col)
            if node_id is None or node_id not in id_map:
                continue
            idx = id_map[node_id]
            for field in fields:
                raw_val = row.get(field)
                if raw_val is None or raw_val == "":
                    val: object = None
                else:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        val = raw_val
                field_values[field][idx] = val

    return {field: np.array(values, dtype=object) for field, values in field_values.items()}


def validate_edge_list(
    path: str,
    *,
    source_col: str = "source_id",
    target_col: str = "target_id",
    weight_col: str = "weight",
    delimiter: str | None = None,
) -> EdgeListSummary:
    """
    Validate edge list schema and weights before loading.

    Checks for required columns, non-negative numeric weights, and blank IDs.
    Returns simple counts for quick diagnostics.
    """
    delim = delimiter or _infer_delimiter(path)
    rows = 0
    zero_weight = 0
    missing_weights = 0
    id_to_idx: dict[str, int] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise ValueError("Edge list has no header row.")
        for col in (source_col, target_col, weight_col):
            if col not in reader.fieldnames:
                raise KeyError(f"Missing expected column: {col}")

        for row in reader:
            rows += 1
            src_id = row.get(source_col)
            tgt_id = row.get(target_col)
            if src_id in (None, "") or tgt_id in (None, ""):
                raise ValueError(f"Blank source/target ID at row {rows}")

            if src_id not in id_to_idx:
                id_to_idx[src_id] = len(id_to_idx)
            if tgt_id not in id_to_idx:
                id_to_idx[tgt_id] = len(id_to_idx)

            raw_weight = row.get(weight_col, "")
            if raw_weight in ("", None):
                missing_weights += 1
                weight = 1.0
            else:
                try:
                    weight = float(raw_weight)
                except ValueError as exc:
                    raise ValueError(f"Non-numeric weight at row {rows}: {raw_weight}") from exc
            if weight < 0:
                raise ValueError(f"Negative weight at row {rows}")
            if weight == 0:
                zero_weight += 1

    if rows == 0:
        raise ValueError("Edge list is empty.")

    return EdgeListSummary(
        rows=rows,
        nodes=len(id_to_idx),
        zero_weight_edges=zero_weight,
        missing_weights=missing_weights,
        delimiter=delim,
    )
