"""Unit tests for MaleCNS experiment manifest and receipt generation."""

import tempfile
from pathlib import Path
import pytest

from dashi.analysis.consumer_evidence import (
    EvidenceChannel,
    EvidenceConsumer,
)
from dashi.analysis.provenance_dependence import (
    EvidenceRelation,
    classify_evidence_relation,
)
from dashi.io.malecns_manifest import (
    CANONICAL_ARTIFACT_SPECS,
    MaleCNSManifest,
)


def test_manifest_contains_all_five_tiers():
    manifest = MaleCNSManifest()
    tiers = {spec.tier for spec in manifest.specs.values()}
    assert tiers == {"connectome", "functional", "registration", "effector", "behaviour"}


def test_manifest_expected_bytes_budget():
    manifest = MaleCNSManifest()
    # Ensure tier 1 connectome is under 600 MB
    conn_bytes = manifest.total_expected_bytes(["connectome"])
    assert 500_000_000 < conn_bytes < 600_000_000

    # Ensure total recommended pipeline is under 700 MB
    total_bytes = manifest.total_expected_bytes()
    assert total_bytes < 700_000_000


def test_manifest_provenance_graph_relations():
    manifest = MaleCNSManifest()
    graph = manifest.build_provenance_graph()

    # Shared dataset root alone must NOT be promoted to same-trial corroboration
    # once the functional artifact lost its fabricated animal/trial provenance.
    rel_trial = classify_evidence_relation(
        graph, "functional_trial_calcium", "behaviour_fictrac_kinematics"
    )
    assert rel_trial == EvidenceRelation.INDEPENDENCE_UNDETERMINED

    # Distinct animals/datasets = CROSS_ANIMAL_REPLICATION
    rel_dataset = classify_evidence_relation(
        graph, "connectome_weights_significant", "functional_trial_calcium"
    )
    assert rel_dataset == EvidenceRelation.CROSS_ANIMAL_REPLICATION



def test_manifest_split_receipt_leakage_guard():
    manifest = MaleCNSManifest()
    train = ["body_001", "body_002"]
    held_out = ["body_003", "body_004"]

    receipt = manifest.create_benchmark_split_receipt(train, held_out)
    assert receipt.leakage_check_passed
    assert len(receipt.split_artifact_sha256) == 64

    # Leaking overlap must raise ValueError
    with pytest.raises(ValueError, match="leakage"):
        manifest.create_benchmark_split_receipt(["body_001", "body_002"], ["body_002", "body_003"])


def test_manifest_consumer_evidence_promotion():
    manifest = MaleCNSManifest()

    # Independent replication bundles
    bundles_rep = manifest.build_consumer_bundles(
        dependence_relation=EvidenceRelation.CROSS_DATASET_REPLICATION,
        provenance_adequate=True,
    )
    assert bundles_rep[EvidenceConsumer.STRUCTURE_FUNCTION].promotable
    assert bundles_rep[EvidenceConsumer.EFFECTOR_STATE].promotable
    assert bundles_rep[EvidenceConsumer.BEHAVIOUR].promotable
    assert bundles_rep[EvidenceConsumer.SEMANTIC].promotable

    # Same trial corroboration cannot promote semantic consumer (requires independent replication)
    bundles_same_trial = manifest.build_consumer_bundles(
        dependence_relation=EvidenceRelation.SAME_TRIAL_CORROBORATION,
        provenance_adequate=True,
    )
    assert bundles_same_trial[EvidenceConsumer.STRUCTURE_FUNCTION].promotable
    assert not bundles_same_trial[EvidenceConsumer.SEMANTIC].promotable

    # Missing muscle channels cannot promote effector
    neural_only_channels = frozenset({EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM})
    bundle_no_effector = manifest.build_consumer_bundles()[EvidenceConsumer.EFFECTOR_STATE]
    from dataclasses import replace
    bundle_tampered = replace(bundle_no_effector, channels=neural_only_channels)
    assert not bundle_tampered.promotable


def test_artifact_receipt_generation_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest = MaleCNSManifest(base_dir=tmp_path)

        target = manifest.target_path("body_annotations")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("dummy feather content", encoding="utf-8")

        receipt = manifest.create_artifact_receipt("body_annotations")
        assert len(receipt.sha256) == 64
        assert receipt.dataset_version == "male-cns:v1.0"
        assert receipt.role == "structural_annotations"
