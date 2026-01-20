#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import os

import numpy as np
import scipy.sparse as sp

from dashi.io.hemibrain_loader import load_edge_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Degree-binned coarse-grain of hemibrain edge list; outputs aggregated edge CSV."
    )
    parser.add_argument("edge_path", help="Path to hemibrain edge list (CSV/TSV).")
    parser.add_argument(
        "--block-size",
        type=int,
        default=100,
        help="Target nodes per block (deterministic grouping by degree; default 100).",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/coarse_degree_binned",
        help="Prefix for outputs; timestamp and suffixes are appended automatically.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def assign_blocks_by_degree(A: sp.csr_matrix, block_size: int) -> tuple[np.ndarray, int]:
    """
    Deterministically assign nodes to blocks by sorted total degree (in+out).

    Returns (block_ids_per_node, num_blocks).
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    out_deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.asarray(A.sum(axis=0)).ravel()
    total_deg = out_deg + in_deg
    order = np.lexsort((np.arange(total_deg.size), total_deg))  # sort by degree, tie-break by index
    block_ids = np.empty_like(order, dtype=np.int64)
    block_id = 0
    for start in range(0, A.shape[0], block_size):
        end = min(start + block_size, A.shape[0])
        block_ids[order[start:end]] = block_id
        block_id += 1
    return block_ids, block_id


def aggregate_edges(A: sp.csr_matrix, block_ids: np.ndarray, num_blocks: int) -> sp.csr_matrix:
    """Aggregate fine edges into block-to-block weighted edges."""
    coo = A.tocoo(copy=False)
    block_rows = block_ids[coo.row]
    block_cols = block_ids[coo.col]
    aggregated = sp.coo_matrix((coo.data, (block_rows, block_cols)), shape=(num_blocks, num_blocks), dtype=np.float32)
    return aggregated.tocsr()


def save_coarse_edges(path: str, coarse: sp.csr_matrix) -> None:
    """Write aggregated edges to CSV with columns: source_id, target_id, weight."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    coo = coarse.tocoo()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_id", "target_id", "weight"])
        for r, c, w in zip(coo.row, coo.col, coo.data):
            if w == 0:
                continue
            writer.writerow([int(r), int(c), float(w)])


def save_block_map(path: str, block_ids: np.ndarray, idx_to_id: list[str]) -> None:
    """Persist block assignment for reproducibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "body_id", "block_id"])
        for idx, (body_id, block_id) in enumerate(zip(idx_to_id, block_ids)):
            writer.writerow([idx, body_id, int(block_id)])


def main() -> int:
    args = parse_args()
    hb = load_edge_list(args.edge_path)
    A = hb.carrier.adjacency.tocsr()
    n = A.shape[0]
    block_ids, num_blocks = assign_blocks_by_degree(A, args.block_size)
    coarse = aggregate_edges(A, block_ids, num_blocks)

    ts = timestamp()
    edge_out = f"{args.output_prefix}_{ts}_edges.csv"
    block_out = f"{args.output_prefix}_{ts}_blocks.csv"

    save_coarse_edges(edge_out, coarse)
    save_block_map(block_out, block_ids, hb.idx_to_id)

    avg_block_size = n / num_blocks if num_blocks else 0
    print(f"Loaded graph: N={n}, edges={A.nnz}")
    print(f"Blocks: {num_blocks} (target size {args.block_size}, avg size {avg_block_size:.2f})")
    print(f"Coarse edges: {coarse.nnz}")
    print(f"Saved coarse edge list to {edge_out}")
    print(f"Saved block assignments to {block_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
