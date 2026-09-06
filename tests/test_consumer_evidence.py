from dashi.analysis.consumer_evidence import (
    BEHAVIOUR_POLICY,
    EFFECTOR_POLICY,
    SEMANTIC_POLICY,
    STRUCTURE_FUNCTION_POLICY,
    ConsumerEvidenceBundle,
    EvidenceChannel,
    EvidenceConsumer,
    transfer_requires_own_receipt,
)
from dashi.analysis.provenance_dependence import EvidenceRelation


def test_structure_function_bundle_can_promote_without_semantic_channels():
    bundle = ConsumerEvidenceBundle(
        STRUCTURE_FUNCTION_POLICY,
        frozenset({EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION}),
        EvidenceRelation.SAME_TRIAL_CORROBORATION,
        provenance_adequate=True,
    )
    assert bundle.promotable


def test_neural_receipt_does_not_transfer_to_effector_consumer():
    assert transfer_requires_own_receipt(EvidenceConsumer.NEURAL_STATE, EvidenceConsumer.EFFECTOR_STATE)


def test_effector_requires_physical_channels():
    bundle = ConsumerEvidenceBundle(
        EFFECTOR_POLICY,
        frozenset({EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION}),
        EvidenceRelation.SAME_TRIAL_CORROBORATION,
        provenance_adequate=True,
    )
    assert not bundle.promotable


def test_behaviour_can_use_kinematics_without_semantic_promotion():
    bundle = ConsumerEvidenceBundle(
        BEHAVIOUR_POLICY,
        frozenset({EvidenceChannel.KINEMATICS}),
        EvidenceRelation.SAME_TRIAL_CORROBORATION,
        provenance_adequate=True,
    )
    assert bundle.promotable
    assert transfer_requires_own_receipt(EvidenceConsumer.BEHAVIOUR, EvidenceConsumer.SEMANTIC)


def test_semantic_policy_rejects_same_trial_only_evidence():
    bundle = ConsumerEvidenceBundle(
        SEMANTIC_POLICY,
        frozenset({EvidenceChannel.KINEMATICS, EvidenceChannel.INTERACTION, EvidenceChannel.ENVIRONMENT}),
        EvidenceRelation.SAME_TRIAL_CORROBORATION,
        provenance_adequate=True,
    )
    assert not bundle.promotable


def test_semantic_policy_accepts_independent_replication_when_other_gates_pass():
    bundle = ConsumerEvidenceBundle(
        SEMANTIC_POLICY,
        frozenset({EvidenceChannel.KINEMATICS, EvidenceChannel.INTERACTION, EvidenceChannel.ENVIRONMENT}),
        EvidenceRelation.CROSS_ANIMAL_REPLICATION,
        provenance_adequate=True,
    )
    assert bundle.promotable
