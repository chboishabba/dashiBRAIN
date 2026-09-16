#!/usr/bin/env python3
"""Range-extract the compact Gauthey LBM experimental working set."""

from __future__ import annotations

import argparse

from dashi.analysis.gauthey_lbm_experiment import extract_compact_lbm_working_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract only the compact Gauthey LBM members needed for region reconstruction"
    )
    parser.add_argument("--output-dir", default="data/gauthey_lbm")
    args = parser.parse_args()

    paths = extract_compact_lbm_working_set(args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
