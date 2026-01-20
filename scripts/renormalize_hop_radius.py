#!/usr/bin/env python3
"""
Coarse-grain hemibrain edge list by hop-radius neighborhoods (locality-preserving aggregate).

Nodes are grouped into deterministic clusters containing all nodes within radius r (undirected hops)
of a seed node. Seeds are processed in index order to keep the assignment stable.
Edges are then aggregated between clusters.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from datetime import datetime
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from dashi.io.hemibrain_loader import load_edge_list


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hop-radius coarse-grain of hemibrain edge list; aggregates nodes within r undirected hops."
    )
    parser.add_argument("edge_path", help="Path to hemibrain edge list (CSV/TSV).")
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Radius (in undirected hops) for neighborhood aggregation (default 1).",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/coarse_radius",
        help="Prefix for outputs; timestamp and suffixes are appended automatically.",
    )
    return parser.parse_args(argv)


def build_undirected_adjacency(A: sp.csr_matrix) -> sp.csr_matrix:
    """Return a boolean CSR adjacency treating the graph as undirected."""
    undirected = (A + A.transpose()).sign().astype(np.int8).tocsr()
    undirected.eliminate_zeros()
    return undirected


def assign_radius_clusters(adj: sp.csr_matrix, radius: int) -> np.ndarray:
    """
    Deterministically assign each node to a cluster containing all nodes within <= radius hops.
    Clusters are seeded by the first unassigned node in index order.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    n = adj.shape[0]
    assigned = np.full(n, -1, dtype=np.int64)
    cluster_id = 0
    neighbors = adj.indices
    indptr = adj.indptr

    for start in range(n):
        if assigned[start] != -1:
            continue
        # BFS to depth=radius.
        dq: deque[tuple[int, int]] = deque()
        dq.append((start, 0))
        assigned[start] = cluster_id
        while dq:
            node, depth = dq.popleft()
            if depth == radius:
                continue
            for nbr in neighbors[indptr[node] : indptr[node + 1]]:
                if assigned[nbr] == -1:
                    assigned[nbr] = cluster_id
                    dq.append((nbr, depth + 1))
        cluster_id += 1
    return assigned


def aggregate_edges(A: sp.csr_matrix, block_ids: np.ndarray, num_blocks: int) -> sp.csr_matrix:
    """Aggregate fine edges into block-to-block weighted edges."""
    coo = A.tocoo(copy=False)
    block_rows = block_ids[coo.row]
    block_cols = block_ids[coo.col]
    aggregated = sp.coo_matrix((coo.data, (block_rows, block_cols)), shape=(num_blocks, num_blocks), dtype=np.float32)
    return aggregated.tocsr()


def save_coarse_edges(path: str, coarse: sp.csr_matrix) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    coo = coarse.tocoo()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "weight"])
        for r, c, w in zip(coo.row, coo.col, coo.data):
            if w == 0:
                continue
            writer.writerow([int(r), int(c), float(w)])


def save_block_map(path: str, block_ids: np.ndarray, idx_to_id: list[str]) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "body_id", "block_id"])
        for idx, (body_id, block_id) in enumerate(zip(idx_to_id, block_ids)):
            writer.writerow([idx, body_id, int(block_id)])


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    hb = load_edge_list(args.edge_path)
    A = hb.carrier.adjacency.tocsr()
    n = A.shape[0]
    undirected = build_undirected_adjacency(A)
    block_ids = assign_radius_clusters(undirected, args.radius)
    num_blocks = int(block_ids.max()) + 1
    coarse = aggregate_edges(A, block_ids, num_blocks)

    ts = timestamp()
    edge_out = f"{args.output_prefix}_r{args.radius}_{ts}_edges.csv"
    block_out = f"{args.output_prefix}_r{args.radius}_{ts}_blocks.csv"

    save_coarse_edges(edge_out, coarse)
    save_block_map(block_out, block_ids, hb.idx_to_id)

    avg_block_size = n / num_blocks if num_blocks else 0
    print(f"Loaded graph: N={n}, edges={A.nnz}")
    print(f"Radius: {args.radius}, clusters: {num_blocks}, avg size {avg_block_size:.2f}")
    print(f"Coarse edges: {coarse.nnz}")
    print(f"Saved coarse edge list to {edge_out}")
    print(f"Saved block assignments to {block_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
