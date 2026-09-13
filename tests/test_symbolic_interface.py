from dataclasses import replace

import pytest

from dashi.analysis.symbolic_interface import (
    AssistanceBudget,
    NullModelKind,
    SymbolicRunReceipt,
    assert_matched_assistance_budget,
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


def receipt(kind: NullModelKind = NullModelKind.MALE_CNS) -> SymbolicRunReceipt:
    return SymbolicRunReceipt(
        run_id="run-001",
        topology_kind=kind,
        assistance=budget(),
        connectome_or_topology_sha256="1" * 64,
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
        replace(receipt(), output_artifact_sha256="bad")


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


def test_null_comparison_requires_identical_assistance_budget():
    candidate = receipt(NullModelKind.MALE_CNS)
    null = replace(
        receipt(NullModelKind.DEGREE_PRESERVING_REWIRE),
        run_id="run-null",
    )
    assert_matched_assistance_budget(candidate, null)

    mismatched = replace(
        null,
        assistance=replace(null.assistance, attempt_budget=2),
    )
    with pytest.raises(ValueError, match="assistance budget"):
        assert_matched_assistance_budget(candidate, mismatched)


def test_null_comparison_rejects_same_topology_kind():
    candidate = receipt(NullModelKind.MALE_CNS)
    with pytest.raises(ValueError, match="distinct topology"):
        assert_matched_assistance_budget(candidate, replace(candidate, run_id="run-002"))
