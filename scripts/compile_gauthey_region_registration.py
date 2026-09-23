#!/usr/bin/env python3
"""Compose explicit Gauthey functional->label and label->region receipts."""

from __future__ import annotations

import argparse
from pathlib import Path

from dashi.io.gauthey_compact import load_audio_correlated_pickle
from dashi.io.gauthey_registration_staging import load_gauthey_label_centroids
from dashi.io.gauthey_roi_alignment import load_explicit_mapping_csv
from dashi.io.gauthey_region_registration import (
    compile_functional_region_registration,
    load_label_to_region_csv,
    write_registration_csv,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--functional", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--mean-brain", required=True)
    p.add_argument("--functional-label-map", required=True)
    p.add_argument("--functional-label-source", required=True)
    p.add_argument("--label-region-map", required=True)
    p.add_argument("--label-region-source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    functional = load_audio_correlated_pickle(args.functional)
    centroids = load_gauthey_label_centroids(args.labels, args.mean_brain)
    functional_to_label = load_explicit_mapping_csv(
        args.functional_label_map,
        source_identifier=args.functional_label_source,
    )
    label_to_region = load_label_to_region_csv(
        args.label_region_map,
        source_identifier=args.label_region_source,
    )
    rows = compile_functional_region_registration(
        functional,
        centroids,
        functional_to_label,
        label_to_region,
    )
    write_registration_csv(rows, args.output)
    print(f"Wrote {len(rows)} complete functional->region registration rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
