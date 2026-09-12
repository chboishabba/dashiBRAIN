#!/usr/bin/env python3
"""Derive Gauthey plane-local supervoxel centroids in trial mean-brain coordinates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dashi.io.gauthey_registration_staging import (
    BIFROST_FDA_TO_JRC2018_BOUNDARY,
    GAUTHEY_TRIAL_TO_FDA_BOUNDARY,
)
from dashi.io.gauthey_supervoxel_geometry import (
    load_plane_local_centroids,
    write_plane_local_centroids_csv,
)


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mean-brain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="outputs/gauthey_supervoxel_centroid_receipt.json")
    args = parser.parse_args()

    labels = Path(args.labels)
    mean_brain = Path(args.mean_brain)
    output = Path(args.output)
    receipt = Path(args.receipt)

    centroids = load_plane_local_centroids(labels, mean_brain)
    write_plane_local_centroids_csv(centroids, output)

    payload = {
        "scientific_source": {
            "author_or_consortium": "Wayan Gauthey; Albert Lin; Osama M. Ahmed; Andrew M. Leifer; Mala Murthy; Stephan Y. Thiberge",
            "title": "High-speed whole-brain imaging in Drosophila",
            "paper_identifier": "doi:10.1038/s41467-026-72437-1",
            "dataset_identifier": "doi:10.5281/zenodo.17618684",
            "code_repository": "github:murthylab/lightbead-analysis",
        },
        "inputs": {
            "labels": {"path": str(labels), "sha256": _sha256(labels)},
            "mean_brain": {"path": str(mean_brain), "sha256": _sha256(mean_brain)},
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "plane_local_supervoxel_count": len(centroids),
            "coordinate_space": "gauthey_trial_mean_brain",
            "identity_carrier": "plane_index,cluster_index",
            "evidence_kind": "plane_local_supervoxel_centroid",
        },
        "registration_stages": {
            "trial_mean_brain_to_fda": vars(GAUTHEY_TRIAL_TO_FDA_BOUNDARY),
            "fda_to_jrc2018": vars(BIFROST_FDA_TO_JRC2018_BOUNDARY),
        },
        "boundaries": [
            "bare cluster integer != globally unique supervoxel identity",
            "plane-local supervoxel != selected pooled functional row until source-row recovery",
            "trial mean-brain coordinates != BIFROST FDA coordinates",
            "FDA->JRC2018 transform != trial->FDA transform",
            "JRC2018 coordinate != MaleCNS neuron identity",
        ],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Derived {len(centroids)} plane-local supervoxel centroids")
    print(f"Output: {output}")
    print(f"Receipt: {receipt}")
    print("Registration frontier: selected-row recovery and trial mean brain -> FDA remain separate payments")


if __name__ == "__main__":
    main()
