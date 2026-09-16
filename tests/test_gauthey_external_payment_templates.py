from __future__ import annotations

import json

from scripts import emit_gauthey_external_payment_templates as templates


def test_registration_payment_requires_same_trial_object_binding(tmp_path):
    path = tmp_path / "payment_b.json"
    templates.write_payment_b(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["trial_identity"] == ""
    assert payload["same_trial_anatomy_identifier"] == ""
    assert payload["same_trial_anatomy_sha256"] == ""
    assert payload["executed_transform_receipt_identifier"] == ""
    assert payload["moving_image_trial_identity"] == ""
    assert payload["same_trial_binding_evidence"] == ""
    assert payload["promotable"] is False
    assert "anatomy/transform from another trial != same-trial registration receipt" in payload["boundaries"]
    assert "ROI-label geometry != atlas identity" in payload["boundaries"]


def test_registration_payment_frontier_stays_external_until_receipt_filled(tmp_path):
    path = tmp_path / "payment_b.json"
    templates.write_payment_b(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    frontier = payload["public_source_frontier"]
    assert frontier["public_non_discovery_anatomy_route_found"] is False
    assert frontier["next_payment_route"] == "external_scientific_receipt"
