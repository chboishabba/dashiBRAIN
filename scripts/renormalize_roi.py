#!/usr/bin/env python3
"""
Coarse-grain hemibrain edge list by ROI (locality-preserving aggregate).

Reads edges plus neuron metadata (from export_neuprint_metadata.py) and assigns each body
to a primary ROI using roi_info counts (or the first entry in `rois`).
Aggregates edge weights between ROIs and writes an edge list + ROI membership table.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime
from typing import Iterable

import pandas as pd


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROI coarse-grain of hemibrain edge list; outputs aggregated edge CSV."
    )
    parser.add_argument("edge_path", help="Path to hemibrain edge list (CSV).")
    parser.add_argument(
        "metadata_path",
        help="Path to neuron metadata CSV (from export_neuprint_metadata.py).",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/coarse_roi",
        help="Prefix for outputs; timestamp and suffixes are appended automatically.",
    )
    parser.add_argument(
        "--unknown-label",
        default="unknown",
        help="Label for neurons with no ROI info (default: unknown).",
    )
    parser.add_argument(
        "--use-side",
        action="store_true",
        help="Append side (L/R) to ROI label when available in metadata.",
    )
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="Drop edges incident to unknown-labeled neurons instead of keeping them.",
    )
    return parser.parse_args(argv)


def select_primary_roi(row: pd.Series, unknown_label: str) -> str:
    """Choose the highest-count ROI; fallback to first in `rois`; otherwise unknown."""
    roi_json = row.get("roi_info_json")
    if isinstance(roi_json, str) and roi_json:
        try:
            roi_counts = json.loads(roi_json)
            if isinstance(roi_counts, dict) and roi_counts:
                def score(item: tuple[str, object]) -> float:
                    _, val = item
                    if isinstance(val, dict):
                        pre = val.get("pre", 0) if isinstance(val.get("pre", 0), (int, float)) else 0
                        post = val.get("post", 0) if isinstance(val.get("post", 0), (int, float)) else 0
                        return float(pre) + float(post)
                    if isinstance(val, (int, float)):
                        return float(val)
                    return 0.0

                items = sorted(roi_counts.items())
                primary = max(items, key=score)[0]
                return str(primary)
        except json.JSONDecodeError:
            pass

    rois = row.get("rois")
    if isinstance(rois, str) and rois:
        parts = [p for p in rois.split("|") if p]
        if parts:
            return parts[0]

    return unknown_label


def load_roi_mapping(metadata_path: str, unknown_label: str, use_side: bool) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    if "bodyId" not in meta.columns:
        raise KeyError("metadata is missing required column 'bodyId'")

    meta["primary_roi"] = meta.apply(lambda row: select_primary_roi(row, unknown_label), axis=1)
    if use_side and "side" in meta.columns:
        meta["primary_roi"] = meta.apply(
            lambda row: f"{row['primary_roi']}_{row['side']}" if pd.notnull(row.get("side")) else row["primary_roi"],
            axis=1,
        )
    return meta[["bodyId", "primary_roi"]]


def aggregate_edges(edge_path: str, roi_map: pd.DataFrame, unknown_label: str, drop_unknown: bool) -> pd.DataFrame:
    edges = pd.read_csv(edge_path)
    required = {"source_id", "target_id", "weight"}
    if not required.issubset(edges.columns):
        raise KeyError(f"edge list missing required columns: {required - set(edges.columns)}")

    roi_lookup = dict(zip(roi_map["bodyId"], roi_map["primary_roi"]))
    edges["src_roi"] = edges["source_id"].map(roi_lookup).fillna(unknown_label)
    edges["tgt_roi"] = edges["target_id"].map(roi_lookup).fillna(unknown_label)

    if drop_unknown:
        mask_known = (edges["src_roi"] != unknown_label) & (edges["tgt_roi"] != unknown_label)
        edges = edges.loc[mask_known]

    grouped = (
        edges.groupby(["src_roi", "tgt_roi"], as_index=False)["weight"]
        .sum()
        .sort_values(by=["src_roi", "tgt_roi"])
    )
    return grouped


def save_edges(path: str, df: pd.DataFrame) -> None:
    path_parent = os.path.dirname(path)
    if path_parent:
        os.makedirs(path_parent, exist_ok=True)
    df.to_csv(path, index=False, columns=["source_id", "target_id", "weight"])


def save_roi_table(path: str, roi_map: pd.DataFrame) -> None:
    path_parent = os.path.dirname(path)
    if path_parent:
        os.makedirs(path_parent, exist_ok=True)
    # Count how many neurons map to each ROI.
    counts = Counter(roi_map["primary_roi"])
    rows = [(roi, idx, counts[roi]) for idx, roi in enumerate(sorted(counts))]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roi_label", "roi_index", "neuron_count"])
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    roi_map = load_roi_mapping(args.metadata_path, args.unknown_label, args.use_side)
    aggregated = aggregate_edges(args.edge_path, roi_map, args.unknown_label, args.drop_unknown)

    ts = timestamp()
    edge_out = f"{args.output_prefix}_{ts}_edges.csv"
    roi_out = f"{args.output_prefix}_{ts}_rois.csv"

    save_edges(edge_out, aggregated.rename(columns={"src_roi": "source_id", "tgt_roi": "target_id"}))
    save_roi_table(roi_out, roi_map)

    print(f"Loaded edges from {args.edge_path}")
    print(f"Metadata: {len(roi_map)} neurons mapped to ROIs (unknown={args.unknown_label!r})")
    print(f"Aggregated edges: {len(aggregated)} pairs")
    print(f"Saved coarse edge list to {edge_out}")
    print(f"Saved ROI table to {roi_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
