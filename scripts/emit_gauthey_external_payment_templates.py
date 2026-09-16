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
from dashi.io.gauthey_payment_frontier import (
    canonical_public_atlas_registration_frontier,
    canonical_public_raw_acquisition_frontier,
)


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
    atlas_frontier = canonical_public_atlas_registration_frontier()
    raw_frontier = canonical_public_raw_acquisition_frontier()
    payload = {
        "trial_identity": "",
        "source_space": "gauthey_trial_or_anatomical_mean_brain",
        "target_space": "registered_atlas_space",
        "same_trial_anatomy_identifier": "",
        "same_trial_anatomy_sha256": "",
        "same_trial_anatomy_source": "",
        "same_trial_binding_evidence": "",
        "executed_transform_receipt_identifier": "",
        "transform_artifact_identifier": "",
        "transform_sha256": "",
        "fixed_image_identifier": "",
        "moving_image_identifier": "",
        "moving_image_trial_identity": "",
        "software_or_method_identifier": "",
        "registration_residual_metric": "",
        "registration_residual_value": None,
        "atlas_region_assignment_artifact": "",
        "producer": "",
        "evidence_kind": "external_same_trial_registration_receipt_required",
        "promotable": False,
        "public_source_frontier": {
            "representative_raw_public": raw_frontier.representative_raw_public,
            "all_trial_preprocessed_public": raw_frontier.all_trial_preprocessed_public,
            "public_non_discovery_raw_route_found": raw_frontier.public_non_discovery_raw_route_found,
            "public_non_discovery_anatomy_route_found": raw_frontier.public_non_discovery_anatomy_route_found,
            "dataset_specific_public_globus_endpoint_found": raw_frontier.dataset_specific_public_globus_endpoint_found,
            "next_payment_route": raw_frontier.next_payment_route.value,
            "note": raw_frontier.note,
            "launcher_present": atlas_frontier.launcher_present,
            "registration_implementation_present": atlas_frontier.implementation_present,
            "ants_dependency_declared": atlas_frontier.ants_dependency_declared,
            "registration_route": atlas_frontier.payment_b_route.value,
            "registration_note": atlas_frontier.note,
        },
        "acceptance": [
            "trial_identity names one non-discovery recording",
            "same_trial_anatomy_identifier or executed_transform_receipt_identifier is authoritative and source-bound",
            "moving_image_trial_identity equals trial_identity when an executed transform is supplied",
            "same_trial_binding_evidence demonstrates that the anatomical/moving object belongs to the declared trial",
            "transform artifact, method and fixed/target atlas identity are retained when registration has already been executed",
        ],
        "boundaries": [
            "launcher presence != transform execution receipt",
            "declared ANTsPy dependency != exact transform artifact",
            "anatomy/transform from another trial != same-trial registration receipt",
            "ROI-label geometry != atlas identity",
            "same region labels != same structural carrier",
            "atlas coordinate != MaleCNS neuron identity",
            "same-trial registration receipt != independent replication result",
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
    print(f"Payment B template: {b} (same-trial anatomy/transform required; NOT promotable)")


if __name__ == "__main__":
    main()
