from dashi.analysis.benchmark_promotion import (
    BenchmarkRunMode,
    PromotionLevel,
    RunEvidenceStatus,
    consumer_promotable_for_run,
)
from dashi.analysis.consumer_evidence import (
    ConsumerEvidenceBundle,
    EvidenceChannel,
    SEMANTIC_POLICY,
    STRUCTURE_FUNCTION_POLICY,
)
from dashi.analysis.provenance_dependence import EvidenceRelation


def _bundle(policy, channels, relation=EvidenceRelation.CROSS_ANIMAL_REPLICATION):
    return ConsumerEvidenceBundle(
        policy=policy,
        channels=frozenset(channels),
        dependence_relation=relation,
        provenance_adequate=True,
    )


def _verified(mode, *, same_trial_only=False, synthetic=False):
    return RunEvidenceStatus(
        mode=mode,
        input_hashes_verified=True,
        registration_verified=True,
        held_out_split_verified=True,
        output_hash_verified=True,
        same_trial_only=same_trial_only,
        synthetic_observations_present=synthetic,
    )


def test_mock_complete_bundle_is_pipeline_only_not_empirical():
    bundle = _bundle(
        STRUCTURE_FUNCTION_POLICY,
        {EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION},
    )
    run = _verified(BenchmarkRunMode.MOCK)
    assert run.promotion_level is PromotionLevel.PIPELINE_ONLY
    assert not consumer_promotable_for_run(bundle, run)


def test_synthetic_complete_bundle_is_diagnostic_only():
    bundle = _bundle(
        STRUCTURE_FUNCTION_POLICY,
        {EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION},
    )
    run = _verified(BenchmarkRunMode.SYNTHETIC)
    assert run.promotion_level is PromotionLevel.DIAGNOSTIC
    assert not consumer_promotable_for_run(bundle, run)


def test_real_hash_verified_structure_function_bundle_can_promote():
    bundle = _bundle(
        STRUCTURE_FUNCTION_POLICY,
        {EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION},
    )
    run = _verified(BenchmarkRunMode.REAL_HASH_VERIFIED)
    assert run.promotion_level is PromotionLevel.EMPIRICALLY_PROMOTABLE
    assert consumer_promotable_for_run(bundle, run)


def test_semantic_same_trial_only_cannot_promote():
    bundle = _bundle(
        SEMANTIC_POLICY,
        {EvidenceChannel.KINEMATICS, EvidenceChannel.INTERACTION, EvidenceChannel.ENVIRONMENT},
        EvidenceRelation.CROSS_TRIAL_REPLICATION,
    )
    run = _verified(BenchmarkRunMode.REAL_HASH_VERIFIED, same_trial_only=True)
    assert not consumer_promotable_for_run(bundle, run)


def test_unverified_real_run_is_candidate_only():
    run = RunEvidenceStatus(
        mode=BenchmarkRunMode.REAL_UNVERIFIED,
        input_hashes_verified=False,
        registration_verified=True,
        held_out_split_verified=True,
        output_hash_verified=False,
    )
    assert run.promotion_level is PromotionLevel.EMPIRICAL_CANDIDATE
