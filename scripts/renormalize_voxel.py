#!/usr/bin/env python3
"""
Coarse-grain hemibrain edge list by spatial voxels (locality-preserving aggregate).

Neurons are assigned to cubic bins based on soma_x/y/z from neurons_metadata.csv
(exported by export_neuprint_metadata.py). Edges are aggregated between voxels.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Voxel coarse-grain of hemibrain edge list using soma coordinates."
    )
    parser.add_argument("edge_path", help="Path to hemibrain edge list (CSV).")
    parser.add_argument(
        "metadata_path",
        help="Path to neuron metadata CSV (from export_neuprint_metadata.py).",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=10_000.0,
        help="Voxel edge length (same units as soma coordinates; default 10000).",
    )
    parser.add_argument(
        "--unknown-label",
        default="unknown",
        help="Label for neurons lacking soma coordinates (default: unknown).",
    )
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="Drop edges incident to unknown-labeled neurons instead of keeping them.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/coarse_voxel",
        help="Prefix for outputs; timestamp and suffixes are appended automatically.",
    )
    return parser.parse_args(argv)


def assign_voxel(row: pd.Series, bin_size: float, unknown_label: str) -> Tuple[str, int, int, int]:
    x, y, z = row.get("soma_x"), row.get("soma_y"), row.get("soma_z")
    if any(pd.isna(val) for val in (x, y, z)):
        return unknown_label, 0, 0, 0
    xb = int(np.floor(float(x) / bin_size))
    yb = int(np.floor(float(y) / bin_size))
    zb = int(np.floor(float(z) / bin_size))
    label = f"{xb}_{yb}_{zb}"
    return label, xb, yb, zb


def load_voxel_map(metadata_path: str, bin_size: float, unknown_label: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    for col in ("soma_x", "soma_y", "soma_z", "bodyId"):
        if col not in meta.columns:
            raise KeyError(f"metadata is missing required column '{col}'")

    voxels = meta.apply(lambda row: assign_voxel(row, bin_size, unknown_label), axis=1, result_type="expand")
    voxels.columns = ["voxel_label", "xb", "yb", "zb"]
    meta = pd.concat([meta[["bodyId"]], voxels], axis=1)
    return meta


def aggregate_edges(edge_path: str, voxel_map: pd.DataFrame, unknown_label: str, drop_unknown: bool) -> pd.DataFrame:
    edges = pd.read_csv(edge_path)
    required = {"source_id", "target_id", "weight"}
    if not required.issubset(edges.columns):
        raise KeyError(f"edge list missing required columns: {required - set(edges.columns)}")

    lookup = dict(zip(voxel_map["bodyId"], voxel_map["voxel_label"]))
    edges["src_voxel"] = edges["source_id"].map(lookup).fillna(unknown_label)
    edges["tgt_voxel"] = edges["target_id"].map(lookup).fillna(unknown_label)

    if drop_unknown:
        mask = (edges["src_voxel"] != unknown_label) & (edges["tgt_voxel"] != unknown_label)
        edges = edges.loc[mask]

    grouped = (
        edges.groupby(["src_voxel", "tgt_voxel"], as_index=False)["weight"]
        .sum()
        .sort_values(by=["src_voxel", "tgt_voxel"])
    )
    return grouped


def save_edges(path: str, df: pd.DataFrame) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False, columns=["source_id", "target_id", "weight"])


def save_voxel_table(path: str, voxel_map: pd.DataFrame) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    counts = Counter(voxel_map["voxel_label"])
    rows = []
    for idx, label in enumerate(sorted(counts)):
        rows.append((label, idx, counts[label]))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["voxel_label", "voxel_index", "neuron_count"])
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    voxel_map = load_voxel_map(args.metadata_path, args.bin_size, args.unknown_label)
    aggregated = aggregate_edges(args.edge_path, voxel_map, args.unknown_label, args.drop_unknown)

    ts = timestamp()
    edge_out = f"{args.output_prefix}_{ts}_edges.csv"
    voxel_out = f"{args.output_prefix}_{ts}_voxels.csv"

    save_edges(edge_out, aggregated.rename(columns={"src_voxel": "source_id", "tgt_voxel": "target_id"}))
    save_voxel_table(voxel_out, voxel_map)

    print(f"Loaded edges from {args.edge_path}")
    print(f"Metadata: {len(voxel_map)} neurons mapped to voxels (unknown={args.unknown_label!r})")
    print(f"Aggregated edges: {len(aggregated)} pairs")
    print(f"Saved coarse edge list to {edge_out}")
    print(f"Saved voxel table to {voxel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
