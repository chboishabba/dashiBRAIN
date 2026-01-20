#!/usr/bin/env python
from __future__ import annotations
import argparse
from dashi.io.hemibrain_loader import validate_edge_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hemibrain edge list schema before loading.")
    parser.add_argument("edge_path", help="Path to edge list CSV/TSV")
    parser.add_argument("--source-col", default="source_id", help="Source column name (default: source_id)")
    parser.add_argument("--target-col", default="target_id", help="Target column name (default: target_id)")
    parser.add_argument("--weight-col", default="weight", help="Weight column name (default: weight)")
    parser.add_argument("--delimiter", help="Optional delimiter override (default: auto-sniff)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_edge_list(
        args.edge_path,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
        delimiter=args.delimiter,
    )

    print(f"Validated {args.edge_path}")
    print(f"Delimiter: '{summary.delimiter}'")
    print(f"Rows: {summary.rows}")
    print(f"Unique nodes: {summary.nodes}")
    print(f"Zero-weight edges: {summary.zero_weight_edges}")
    print(f"Edges missing weights (treated as 1.0): {summary.missing_weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
