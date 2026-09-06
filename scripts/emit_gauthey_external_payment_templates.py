#!/usr/bin/env python3
"""Emit non-promotable templates for external Gauthey scientific payments.

The generated files are schemas only. Empty fields are deliberate and prevent
an absent source receipt from being replaced by inferred identities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dashi.io.gauthey_compact import GAUTHEY_2P_EXPECTED_SELECTED_ROIS
from dashi.io.gauthey_payment_frontier import canonical_public_atlas_registration_frontier


def write_payment_a(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "functional_id",
        "selected_row",
        "trial_id",
        "plane_index",
        "cluster_index",
        "source_artifact_identifier",
        "source_row_identifier",
        "evidence_kind",
        "producer_note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(GAUTHEY_2P_EXPECTED_SELECTED_ROIS):
            writer.writerow(
                {
                    "functional_id": f"selected_roi_{i:04d}",
                    "selected_row": i,
                    "trial_id": "",
                    "plane_index": "",
                    "cluster_index": "",
                    "source_artifact_identifier": "",
                    "source_row_identifier": "",
                    "evidence_kind": "external_scientific_receipt_required",
                    "producer_note": "",
                }
            )


def write_payment_b(path: Path) -> None:
    frontier = canonical_public_atlas_registration_frontier()
    payload = {
        "source_space": "gauthey_trial_or_anatomical_mean_brain",
        "target_space": "registered_atlas_space",
        "transform_artifact_identifier": "",
        "transform_sha256": "",
        "fixed_image_identifier": "",
        "moving_image_identifier": "",
        "software_or_method_identifier": "",
        "registration_residual_metric": "",
        "registration_residual_value": None,
        "atlas_region_assignment_artifact": "",
        "producer": "",
        "evidence_kind": "external_registration_receipt_required",
        "promotable": False,
        "public_source_frontier": {
            "launcher_present": frontier.launcher_present,
            "implementation_present": frontier.implementation_present,
            "ants_dependency_declared": frontier.ants_dependency_declared,
            "route": frontier.payment_b_route.value,
            "note": frontier.note,
        },
        "boundaries": [
            "launcher presence != transform execution receipt",
            "declared ANTsPy dependency != exact transform artifact",
            "atlas coordinate != MaleCNS neuron identity",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--payment-a",
        default="outputs/gauthey_payment_a_selected_roi_identity_template.csv",
    )
    p.add_argument(
        "--payment-b",
        default="outputs/gauthey_payment_b_registration_template.json",
    )
    args = p.parse_args()
    a = Path(args.payment_a)
    b = Path(args.payment_b)
    write_payment_a(a)
    write_payment_b(b)
    print(f"Payment A template: {a} ({GAUTHEY_2P_EXPECTED_SELECTED_ROIS} rows; NOT promotable)")
    print(f"Payment B template: {b} (NOT promotable)")


if __name__ == "__main__":
    main()
