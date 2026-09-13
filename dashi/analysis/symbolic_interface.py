"""Clean-room receipt layer for MaleCNS symbolic-output experiments.

This module deliberately separates topology, neuron-identity assignment,
intervention, executable dynamics, interface semantics, emitted source text,
task evaluation, and competence promotion. It does not claim that any viral
Python/FizzBuzz demonstration has been reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class TopologyKind(str, Enum):
    MALE_CNS = "male_cns"
    DEGREE_PRESERVING_REWIRE = "degree_preserving_rewire"
    MATCHED_GENERIC_RECURRENT_NETWORK = "matched_generic_recurrent_network"


class IdentityAssignmentKind(str, Enum):
    NATIVE = "native"
    SHUFFLED = "shuffled"


class InterventionKind(str, Enum):
    BASELINE = "baseline"
    NO_LEARNING = "no_learning"
    ALTERNATE_INITIALIZATION = "alternate_initialization"


_REQUIRED_ASSISTANCE_FIELDS = (
    "input_encoding",
    "dynamics_rule",
    "initial_state_or_seed",
    "decoder_mapping",
    "update_rule",
    "reward_or_evaluator",
    "prompt_or_scaffold",
    "parser_runtime",
    "selection_policy",
)


@dataclass(frozen=True)
class AssistanceBudget:
    input_encoding: str
    dynamics_rule: str
    initial_state_or_seed: str
    decoder_mapping: str
    update_rule: str
    reward_or_evaluator: str
    prompt_or_scaffold: str
    parser_runtime: str
    attempt_budget: int
    selection_policy: str

    def __post_init__(self) -> None:
        for field_name in _REQUIRED_ASSISTANCE_FIELDS:
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} is required")
        if self.attempt_budget < 1:
            raise ValueError("attempt_budget must be at least 1")


_HASH_FIELDS = (
    "connectome_or_topology_sha256",
    "dynamics_artifact_sha256",
    "decoder_artifact_sha256",
    "output_artifact_sha256",
    "evaluator_artifact_sha256",
)


@dataclass(frozen=True)
class SymbolicRunReceipt:
    run_id: str
    topology_kind: TopologyKind
    identity_assignment_kind: IdentityAssignmentKind
    intervention_kind: InterventionKind
    assistance: AssistanceBudget
    connectome_or_topology_sha256: str
    dynamics_artifact_sha256: str
    decoder_artifact_sha256: str
    output_artifact_sha256: str
    evaluator_artifact_sha256: str
    emitted_program: str
    parse_succeeded: bool
    execution_succeeded: bool
    declared_cases_passed: bool
    held_out_cases_passed: bool
    cross_task_transfer_passed: bool

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for field_name in _HASH_FIELDS:
            digest = getattr(self, field_name)
            if len(digest) != 64:
                raise ValueError(f"{field_name} must be sha256")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(f"{field_name} must be sha256") from exc

        if self.execution_succeeded and not self.parse_succeeded:
            raise ValueError("execution success requires parse success")
        if self.declared_cases_passed and not self.execution_succeeded:
            raise ValueError("declared case success requires execution success")
        if self.held_out_cases_passed and not self.declared_cases_passed:
            raise ValueError("held-out success requires declared-case success")
        if self.cross_task_transfer_passed and not self.held_out_cases_passed:
            raise ValueError("cross-task transfer requires held-out task success")


def competence_level(receipt: SymbolicRunReceipt) -> str:
    """Return the highest competence level actually paid by this receipt.

    General programming competence is intentionally absent: no single
    FizzBuzz-family receipt can promote to that claim.
    """

    if receipt.cross_task_transfer_passed:
        return "cross_task_transfer"
    if receipt.held_out_cases_passed:
        return "fizzbuzz_held_out_cases"
    if receipt.declared_cases_passed:
        return "fizzbuzz_declared_cases"
    if receipt.execution_succeeded:
        return "python_executes"
    if receipt.parse_succeeded:
        return "python_parses"
    if receipt.emitted_program:
        return "nonempty_program_text"
    return "mapped_token_emission"


def _budget_equal_except(
    left: AssistanceBudget,
    right: AssistanceBudget,
    allowed_difference: str,
) -> bool:
    for field in fields(AssistanceBudget):
        if field.name == allowed_difference:
            continue
        if getattr(left, field.name) != getattr(right, field.name):
            return False
    return True


def assert_matched_topology_control(
    candidate: SymbolicRunReceipt,
    control: SymbolicRunReceipt,
) -> None:
    """Require a fair topology comparison with all other axes held fixed."""

    if candidate.topology_kind == control.topology_kind:
        raise ValueError("topology control requires a distinct topology kind")
    if candidate.identity_assignment_kind != control.identity_assignment_kind:
        raise ValueError("topology control must preserve identity assignment")
    if candidate.intervention_kind != control.intervention_kind:
        raise ValueError("topology control must preserve intervention kind")
    if candidate.assistance != control.assistance:
        raise ValueError("topology control requires an identical assistance budget")


def assert_matched_identity_control(
    candidate: SymbolicRunReceipt,
    control: SymbolicRunReceipt,
) -> None:
    """Require an identity-assignment control with topology/dynamics fixed."""

    if candidate.topology_kind != control.topology_kind:
        raise ValueError("identity control must preserve topology kind")
    if candidate.intervention_kind != control.intervention_kind:
        raise ValueError("identity control must preserve intervention kind")
    if candidate.identity_assignment_kind == control.identity_assignment_kind:
        raise ValueError("identity control requires a distinct identity assignment")
    if candidate.assistance != control.assistance:
        raise ValueError("identity control requires an identical assistance budget")


def assert_matched_intervention_control(
    candidate: SymbolicRunReceipt,
    control: SymbolicRunReceipt,
) -> None:
    """Require that an intervention changes only its declared coordinate."""

    if candidate.topology_kind != control.topology_kind:
        raise ValueError("intervention control must preserve topology kind")
    if candidate.identity_assignment_kind != control.identity_assignment_kind:
        raise ValueError("intervention control must preserve identity assignment")
    if candidate.intervention_kind != InterventionKind.BASELINE:
        raise ValueError("candidate intervention must be baseline")

    if control.intervention_kind == InterventionKind.ALTERNATE_INITIALIZATION:
        if candidate.assistance.initial_state_or_seed == control.assistance.initial_state_or_seed:
            raise ValueError("alternate initialization must change initial_state_or_seed")
        if not _budget_equal_except(
            candidate.assistance,
            control.assistance,
            "initial_state_or_seed",
        ):
            raise ValueError("alternate initialization may change only initial_state_or_seed")
        return

    if control.intervention_kind == InterventionKind.NO_LEARNING:
        if candidate.assistance.update_rule == control.assistance.update_rule:
            raise ValueError("no-learning control must change update_rule")
        if not _budget_equal_except(candidate.assistance, control.assistance, "update_rule"):
            raise ValueError("no-learning control may change only update_rule")
        return

    raise ValueError("control intervention must declare a non-baseline intervention")
