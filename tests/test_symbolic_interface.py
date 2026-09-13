from dataclasses import replace

import pytest

from dashi.analysis.symbolic_interface import (
    AssistanceBudget,
    IdentityAssignmentKind,
    InterventionKind,
    SymbolicRunReceipt,
    TopologyKind,
    assert_matched_identity_control,
    assert_matched_intervention_control,
    assert_matched_topology_control,
    competence_level,
)


def budget() -> AssistanceBudget:
    return AssistanceBudget(
        input_encoding="declared sensory/task encoder",
        dynamics_rule="MaleCNS executable dynamics revision",
        initial_state_or_seed="seed=7",
        decoder_mapping="neural population -> token mapping v1",
        update_rule="declared reinforcement/update rule",
        reward_or_evaluator="FizzBuzz exact evaluator v1",
        prompt_or_scaffold="none",
        parser_runtime="CPython 3.x",
        attempt_budget=1,
        selection_policy="first attempt; no cherry-picking",
    )


def receipt(
    topology: TopologyKind = TopologyKind.MALE_CNS,
    identity: IdentityAssignmentKind = IdentityAssignmentKind.NATIVE,
    intervention: InterventionKind = InterventionKind.BASELINE,
) -> SymbolicRunReceipt:
    topology_hash = "1" * 64 if topology == TopologyKind.MALE_CNS else "6" * 64
    identity_hash = "7" * 64 if identity == IdentityAssignmentKind.NATIVE else "8" * 64
    return SymbolicRunReceipt(
        run_id="run-001",
        topology_kind=topology,
        identity_assignment_kind=identity,
        intervention_kind=intervention,
        assistance=budget(),
        connectome_or_topology_sha256=topology_hash,
        identity_assignment_sha256=identity_hash,
        dynamics_artifact_sha256="2" * 64,
        decoder_artifact_sha256="3" * 64,
        output_artifact_sha256="4" * 64,
        evaluator_artifact_sha256="5" * 64,
        emitted_program="for i in range(1, 101): pass",
        parse_succeeded=True,
        execution_succeeded=True,
        declared_cases_passed=True,
        held_out_cases_passed=False,
        cross_task_transfer_passed=False,
    )


def test_assistance_budget_requires_nonempty_provenance_fields():
    with pytest.raises(ValueError, match="decoder_mapping"):
        replace(budget(), decoder_mapping="")


def test_assistance_budget_rejects_zero_attempt_budget():
    with pytest.raises(ValueError, match="attempt_budget"):
        replace(budget(), attempt_budget=0)


def test_receipt_requires_sha256_artifact_hashes():
    with pytest.raises(ValueError, match="sha256"):
        replace(receipt(), identity_assignment_sha256="bad")


def test_competence_stops_at_declared_cases_without_held_out_receipt():
    assert competence_level(receipt()) == "fizzbuzz_declared_cases"


def test_competence_promotes_to_held_out_only_when_paid():
    paid = replace(receipt(), held_out_cases_passed=True)
    assert competence_level(paid) == "fizzbuzz_held_out_cases"


def test_general_programming_is_never_inferred_from_fizzbuzz():
    paid = replace(
        receipt(),
        held_out_cases_passed=True,
        cross_task_transfer_passed=True,
    )
    assert competence_level(paid) == "cross_task_transfer"


def test_topology_control_requires_identical_assistance_budget():
    candidate = receipt(TopologyKind.MALE_CNS)
    control = replace(
        receipt(TopologyKind.DEGREE_PRESERVING_REWIRE),
        run_id="run-null",
    )
    assert_matched_topology_control(candidate, control)

    mismatched = replace(
        control,
        assistance=replace(control.assistance, attempt_budget=2),
    )
    with pytest.raises(ValueError, match="assistance budget"):
        assert_matched_topology_control(candidate, mismatched)


def test_topology_control_rejects_same_topology_kind():
    candidate = receipt(TopologyKind.MALE_CNS)
    with pytest.raises(ValueError, match="distinct topology"):
        assert_matched_topology_control(candidate, replace(candidate, run_id="run-002"))


def test_topology_control_requires_changed_topology_artifact():
    candidate = receipt(TopologyKind.MALE_CNS)
    control = receipt(TopologyKind.DEGREE_PRESERVING_REWIRE)
    control = replace(
        control,
        connectome_or_topology_sha256=candidate.connectome_or_topology_sha256,
    )
    with pytest.raises(ValueError, match="topology artifact"):
        assert_matched_topology_control(candidate, control)


def test_identity_control_preserves_topology_and_assistance():
    candidate = receipt()
    control = replace(
        receipt(identity=IdentityAssignmentKind.SHUFFLED),
        run_id="identity-control",
    )
    assert_matched_identity_control(candidate, control)

    bad = replace(control, connectome_or_topology_sha256="9" * 64)
    with pytest.raises(ValueError, match="preserve topology artifact"):
        assert_matched_identity_control(candidate, bad)


def test_identity_control_requires_distinct_identity_assignment():
    candidate = receipt()
    with pytest.raises(ValueError, match="distinct identity assignment"):
        assert_matched_identity_control(candidate, replace(candidate, run_id="same-identity"))


def test_identity_control_requires_changed_identity_artifact():
    candidate = receipt()
    control = receipt(identity=IdentityAssignmentKind.SHUFFLED)
    control = replace(
        control,
        identity_assignment_sha256=candidate.identity_assignment_sha256,
    )
    with pytest.raises(ValueError, match="identity-assignment artifact"):
        assert_matched_identity_control(candidate, control)


def test_alternate_initialization_changes_only_seed_coordinate():
    candidate = receipt()
    control = replace(
        candidate,
        run_id="seed-control",
        intervention_kind=InterventionKind.ALTERNATE_INITIALIZATION,
        assistance=replace(candidate.assistance, initial_state_or_seed="seed=8"),
    )
    assert_matched_intervention_control(candidate, control)

    bad = replace(
        control,
        assistance=replace(control.assistance, decoder_mapping="changed decoder"),
    )
    with pytest.raises(ValueError, match="only initial_state_or_seed"):
        assert_matched_intervention_control(candidate, bad)


def test_no_learning_control_changes_only_update_rule():
    candidate = receipt()
    control = replace(
        candidate,
        run_id="no-learning-control",
        intervention_kind=InterventionKind.NO_LEARNING,
        assistance=replace(candidate.assistance, update_rule="no learning/update"),
    )
    assert_matched_intervention_control(candidate, control)

    bad = replace(
        control,
        assistance=replace(control.assistance, attempt_budget=2),
    )
    with pytest.raises(ValueError, match="only update_rule"):
        assert_matched_intervention_control(candidate, bad)
