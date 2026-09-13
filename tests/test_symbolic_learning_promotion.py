from dashi.analysis.benchmark_promotion import BenchmarkRunMode, RunEvidenceStatus
from dashi.analysis.symbolic_learning_promotion import (
    LearningClaimKind,
    SymbolicLearningEvidence,
    learning_claim_promotable,
)


def run_status(mode: BenchmarkRunMode) -> RunEvidenceStatus:
    verified = mode is BenchmarkRunMode.REAL_HASH_VERIFIED
    return RunEvidenceStatus(
        mode=mode,
        input_hashes_verified=verified,
        registration_verified=verified,
        held_out_split_verified=verified,
        output_hash_verified=verified,
        same_trial_only=False,
        synthetic_observations_present=(mode is BenchmarkRunMode.SYNTHETIC),
    )


def evidence(mode: BenchmarkRunMode = BenchmarkRunMode.REAL_HASH_VERIFIED):
    return SymbolicLearningEvidence(
        run=run_status(mode),
        learning_update_applied=True,
        held_out_task_passed=True,
        matched_topology_control_present=False,
        cross_task_transfer_passed=False,
    )


def test_synthetic_training_cannot_promote_empirical_learning():
    assert not learning_claim_promotable(
        evidence(BenchmarkRunMode.SYNTHETIC),
        LearningClaimKind.SYMBOLIC_LEARNING,
    )


def test_artifact_verified_held_out_learning_can_promote_learning_claim():
    assert learning_claim_promotable(
        evidence(),
        LearningClaimKind.SYMBOLIC_LEARNING,
    )


def test_connectome_advantage_additionally_requires_matched_topology_control():
    base = evidence()
    assert not learning_claim_promotable(base, LearningClaimKind.CONNECTOME_ADVANTAGE)
    paid = SymbolicLearningEvidence(
        run=base.run,
        learning_update_applied=True,
        held_out_task_passed=True,
        matched_topology_control_present=True,
        cross_task_transfer_passed=False,
    )
    assert learning_claim_promotable(paid, LearningClaimKind.CONNECTOME_ADVANTAGE)


def test_cross_task_transfer_requires_separate_transfer_receipt():
    base = evidence()
    assert not learning_claim_promotable(base, LearningClaimKind.CROSS_TASK_TRANSFER)
    paid = SymbolicLearningEvidence(
        run=base.run,
        learning_update_applied=True,
        held_out_task_passed=True,
        matched_topology_control_present=True,
        cross_task_transfer_passed=True,
    )
    assert learning_claim_promotable(paid, LearningClaimKind.CROSS_TASK_TRANSFER)


def test_general_programming_competence_is_not_promotable_from_this_carrier():
    paid = SymbolicLearningEvidence(
        run=run_status(BenchmarkRunMode.REAL_HASH_VERIFIED),
        learning_update_applied=True,
        held_out_task_passed=True,
        matched_topology_control_present=True,
        cross_task_transfer_passed=True,
    )
    assert not learning_claim_promotable(
        paid,
        LearningClaimKind.GENERAL_PROGRAMMING_COMPETENCE,
    )
