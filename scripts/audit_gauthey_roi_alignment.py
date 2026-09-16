#!/usr/bin/env python3
"""Audit Gauthey functional-column vs segmentation-label alignment without promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dashi.io.gauthey_compact import load_audio_correlated_pickle
from dashi.io.gauthey_registration_staging import load_gauthey_label_centroids
from dashi.io.gauthey_roi_alignment import (
    audit_functional_label_alignment,
    load_explicit_mapping_csv,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--functional", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--mean-brain", required=True)
    p.add_argument("--responsive-rois")
    p.add_argument("--explicit-map")
    p.add_argument("--map-source", default="unspecified")
    p.add_argument("--output")
    args = p.parse_args()

    functional = load_audio_correlated_pickle(args.functional)
    centroids = load_gauthey_label_centroids(args.labels, args.mean_brain)
    mapping = None
    if args.explicit_map:
        mapping = load_explicit_mapping_csv(args.explicit_map, source_identifier=args.map_source)

    result = audit_functional_label_alignment(
        functional,
        centroids,
        responsive_roi_csv=args.responsive_rois,
        explicit_mapping=mapping,
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
