#!/usr/bin/env python3
"""Derive Gauthey segmentation-label centroids in trial mean-brain coordinates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dashi.io.gauthey_registration_staging import (
    BIFROST_FDA_TO_JRC2018_BOUNDARY,
    GAUTHEY_TRIAL_TO_FDA_BOUNDARY,
    load_gauthey_label_centroids,
    write_centroids_csv,
)


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        default="data/malecns/functional/gauthey_compact/04032024_6f_a2_r5_n2000_labels.h5",
    )
    parser.add_argument(
        "--mean-brain",
        default="data/malecns/functional/gauthey_compact/04032024_GCamp6f_a2_r5_w3_mean_G.nii",
    )
    parser.add_argument(
        "--output",
        default="data/malecns/registration/gauthey_trial_mean_brain_label_centroids.csv",
    )
    parser.add_argument("--receipt", default="outputs/gauthey_roi_centroid_receipt.json")
    args = parser.parse_args()

    labels = Path(args.labels)
    mean_brain = Path(args.mean_brain)
    output = Path(args.output)
    receipt = Path(args.receipt)

    centroids = load_gauthey_label_centroids(labels, mean_brain)
    write_centroids_csv(centroids, output)

    payload = {
        "scientific_source": {
            "author_or_consortium": "Wayan Gauthey; Albert Lin; Osama M. Ahmed; Andrew M. Leifer; Mala Murthy; Stephan Y. Thiberge",
            "title": "High-speed whole-brain imaging in Drosophila",
            "paper_identifier": "doi:10.1038/s41467-026-72437-1",
            "dataset_identifier": "doi:10.5281/zenodo.17618684",
        },
        "inputs": {
            "labels": {"path": str(labels), "sha256": _sha256(labels)},
            "mean_brain": {"path": str(mean_brain), "sha256": _sha256(mean_brain)},
        },
        "output": {
            "path": str(output),
            "sha256": _sha256(output),
            "label_count": len(centroids),
            "coordinate_space": "gauthey_trial_mean_brain",
            "evidence_kind": "segmentation_label_centroid_unregistered_to_functional_trace",
        },
        "registration_stages": {
            "trial_mean_brain_to_fda": vars(GAUTHEY_TRIAL_TO_FDA_BOUNDARY),
            "fda_to_jrc2018": vars(BIFROST_FDA_TO_JRC2018_BOUNDARY),
        },
        "boundaries": [
            "segmentation label != functional matrix column identity",
            "trial mean-brain coordinates != BIFROST FDA coordinates",
            "FDA->JRC2018 transform != trial->FDA transform",
            "JRC2018 coordinate != MaleCNS neuron identity",
        ],
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Derived {len(centroids)} segmentation-label centroids")
    print(f"Output: {output}")
    print(f"Receipt: {receipt}")
    print("Registration frontier: trial mean brain -> FDA remains UNVERIFIED")


if __name__ == "__main__":
    main()
