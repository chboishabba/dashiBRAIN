#!/usr/bin/env python
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import math
import os
import sys
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from dashi.baseline.residuals import compute_residuals
from dashi.baseline.residuals import residual_sign_balance
from dashi.baseline.sparse_degree_corrected import compute_sparse_dc_residual
from dashi.io.hemibrain_loader import load_edge_list
from dashi.kernel.flow import kernel_flow
from dashi.types import GraphCarrier, KernelParams
from dashi.valuation.init_field import init_field_from_residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute kernel defect curve from hemibrain edge list.")
    parser.add_argument("edge_path", help="Path to hemibrain edge list (CSV/TSV)")
    parser.add_argument("--metadata", help="Optional neuron metadata file", default=None)
    parser.add_argument(
        "--metadata-id-col",
        default="neuron_id",
        help="Column in metadata used as ID to align with edges (default: neuron_id).",
    )
    parser.add_argument(
        "--partition-field",
        action="append",
        help="Metadata field(s) to expose as partitions (requires --metadata). Can be passed multiple times.",
    )
    parser.add_argument("--hops", type=int, default=1, help="Neighborhood depth for kernel")
    parser.add_argument("--deadzone", type=float, default=1e-9, help="Ternary deadzone")
    parser.add_argument("--steps", type=int, default=20, help="Max kernel iterations")
    parser.add_argument(
        "--baseline",
        choices=["sparse_dc", "dense_dc", "raw"],
        default="sparse_dc",
        help="Baseline for residuals: sparse degree-corrected (default), dense degree-corrected (small graphs), or raw weights.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/defect_curve",
        help="Prefix for timestamped outputs (CSV/PNG). Actual files get a _YYYYMMDD-HHMMSS suffix.",
    )
    parser.add_argument("--validate", action="store_true", help="Validate edge schema before loading.")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_defect_csv(path: str, history: Iterable[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "defect"])
        for h in history:
            writer.writerow([h["step"], h["defect"]])


def save_defect_plot(path: str, history: Iterable[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    steps = [h["step"] for h in history]
    defects = [h["defect"] for h in history]
    plt.figure()
    plt.plot(steps, defects, marker="o")
    plt.xlabel("Step")
    plt.ylabel("Defect")
    plt.title("DASHI kernel defect curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_final_state(path: str, state: np.ndarray, ids: list[str]) -> None:
    """Save final ternary state with node and body IDs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_index", "body_id", "value"])
        flat = state[:, 0] if state.ndim == 2 else state
        for idx, val in enumerate(flat):
            writer.writerow([idx, ids[idx], int(val)])


def save_defect_nodes(path: str, s0: np.ndarray, history: list[dict[str, object]], ids: list[str]) -> None:
    """
    Save nodes that changed value at each step.

    Output columns: step, node_index, body_id, prev_value, new_value.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = s0[:, 0] if s0.ndim == 2 else s0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "node_index", "body_id", "prev_value", "new_value"])
        for h in history:
            curr = h["state"]
            curr_flat = curr[:, 0] if curr.ndim == 2 else curr
            changed = np.nonzero(curr_flat != prev)[0]
            for idx in changed:
                writer.writerow([h["step"], idx, ids[idx], int(prev[idx]), int(curr_flat[idx])])
            prev = curr_flat


def component_counts(A: sp.csr_matrix, mask: np.ndarray) -> tuple[int, int]:
    """Return (n_components, largest_component_size) for subgraph induced by mask."""
    sub = A[mask][:, mask]
    undirected = (sub + sub.T).tocsr()
    from scipy.sparse.csgraph import connected_components

    n_comp, labels = connected_components(undirected, directed=False, return_labels=True)
    largest = int(np.bincount(labels).max()) if labels.size else 0
    return int(n_comp), largest


def entropy_and_purity(counts: tuple[int, int, int]) -> tuple[float, float]:
    """Shannon entropy (base 2) and purity for a (+1, 0, -1) distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0
    probs = [c / total for c in counts if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    purity = max(counts) / total
    return float(entropy), float(purity)


def main() -> int:
    args = parse_args()
    hb = load_edge_list(
        args.edge_path,
        metadata_path=args.metadata,
        metadata_id_col=args.metadata_id_col,
        partition_fields=args.partition_field,
        validate=args.validate,
    )

    if args.baseline == "sparse_dc":
        residual, balance = compute_sparse_dc_residual(hb.carrier.adjacency, deadzone=args.deadzone)
    elif args.baseline == "dense_dc":
        N = hb.carrier.adjacency.shape[0]
        if N > 50000:
            raise RuntimeError("Dense baseline disabled for graphs with >50k nodes; use --baseline sparse_dc")
        residual, balance = compute_residuals(hb.carrier.adjacency, deadzone=args.deadzone)
    else:  # raw
        residual = hb.carrier.adjacency.copy()
        balance = residual_sign_balance(residual, deadzone=args.deadzone)

    s0 = init_field_from_residual(residual, deadzone=args.deadzone)

    params = KernelParams(hops=args.hops, deadzone=args.deadzone)
    carrier = GraphCarrier(adjacency=hb.carrier.adjacency, channels=hb.carrier.channels)
    final, history, status = kernel_flow(carrier, s0, params, steps=args.steps)

    print(f"Loaded graph: N={carrier.adjacency.shape[0]}, edges={carrier.adjacency.nnz}")
    print(f"Residual sign balance: {balance}")
    if args.partition_field and hb.partitions:
        for field in args.partition_field:
            part = hb.partitions.get(field)
            if part is None:
                raise KeyError(f"Partition field not found in metadata: {field}")
            values = [v for v in part.tolist() if v is not None]
            unique, counts = (np.unique(values, return_counts=True) if values else ([], []))
            missing = len(part) - len(values)
            print(f"Partition '{field}': {len(unique)} labels, missing={missing}")
            for val, count in zip(unique, counts):
                print(f"  {val}: {count}")
    print(f"Kernel status: {status}")
    print("Defect curve (step: defect):")
    for h in history:
        print(f"  {h['step']}: {h['defect']}")

    ts = timestamp()
    csv_path = f"{args.output_prefix}_{ts}.csv"
    png_path = f"{args.output_prefix}_{ts}.png"
    state_path = f"{args.output_prefix}_{ts}_state.csv"
    defect_nodes_path = f"{args.output_prefix}_{ts}_defect_nodes.csv"

    save_defect_csv(csv_path, history)
    save_defect_plot(png_path, history)
    save_final_state(state_path, final, hb.idx_to_id)
    save_defect_nodes(defect_nodes_path, s0, history, hb.idx_to_id)

    # Diagnostics: flips and component counts
    flat_s0 = s0[:, 0] if s0.ndim == 2 else s0
    flat_final = final[:, 0] if final.ndim == 2 else final
    flips = int(np.sum(flat_s0 != flat_final))
    final_balance = {
        "+1": int(np.sum(flat_final == 1)),
        "0": int(np.sum(flat_final == 0)),
        "-1": int(np.sum(flat_final == -1)),
    }
    print(f"Flips s0 -> s_final: {flips}")
    print(f"Final sign balance: {final_balance}")

    comp_summary = {}
    try:
        pos_mask = flat_final == 1
        neg_mask = flat_final == -1
        pos_comp, pos_largest = component_counts(carrier.adjacency, pos_mask)
        neg_comp, neg_largest = component_counts(carrier.adjacency, neg_mask)
        comp_summary = {
            "+1_components": pos_comp,
            "+1_largest": pos_largest,
            "-1_components": neg_comp,
            "-1_largest": neg_largest,
        }
        print(f"Component counts: {comp_summary}")
    except Exception as exc:  # pragma: no cover - best-effort diagnostic
        print(f"Component count failed: {exc}")

    # Partition-aware summary if metadata partitions were loaded
    if hb.partitions:
        for field, values in hb.partitions.items():
            value_array = np.array(values)
            valid_mask = np.array(
                [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in value_array], dtype=bool
            )
            if not valid_mask.any():
                print(f"Partition '{field}': no non-null labels; skipping summary.")
                continue
            labels, _ = np.unique(value_array[valid_mask], return_counts=True)
            missing = len(value_array) - int(valid_mask.sum())
            print(f"Partition '{field}' labels: {len(labels)}, missing={missing}")
            summary_path = f"{args.output_prefix}_{ts}_partition_{field}.csv"
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["label", "count", "+1", "0", "-1", "entropy", "purity"])
                purity_stats: list[tuple[str, float, int]] = []
                for label in labels:
                    mask = value_array == label
                    count = int(mask.sum())
                    pos = int(np.sum(flat_final[mask] == 1))
                    zero = int(np.sum(flat_final[mask] == 0))
                    neg = int(np.sum(flat_final[mask] == -1))
                    entropy, purity = entropy_and_purity((pos, zero, neg))
                    writer.writerow(
                        [
                            label,
                            count,
                            pos,
                            zero,
                            neg,
                            f"{entropy:.6f}",
                            f"{purity:.6f}",
                        ]
                    )
                    purity_stats.append((label, purity, count))

            if purity_stats:
                top_label, top_purity, top_count = max(purity_stats, key=lambda x: x[1])
                print(
                    f"Partition '{field}' summary saved to {summary_path}; "
                    f"top purity label: {top_label} (purity={top_purity:.3f}, n={top_count})"
                )
            else:
                print(f"Partition '{field}' summary saved to {summary_path}")

    print(f"Saved defect curve CSV to {csv_path}")
    print(f"Saved defect plot PNG to {png_path}")
    print(f"Saved final state CSV to {state_path}")
    print(f"Saved defecting nodes per step to {defect_nodes_path}")

    # Prevent unused variable warning for final state in this CLI.
    _ = final
    return 0


if __name__ == "__main__":
    sys.exit(main())
