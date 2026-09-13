import json

import numpy as np
import scipy.sparse as sp

from dashi.analysis.symbolic_experiment import (
    decoder_from_identity_assignment,
    initial_state_from_active_ids,
    run_symbolic_arm,
    sha256_sparse_csr,
)
from dashi.analysis.symbolic_interface import IdentityAssignmentKind, TopologyKind
from dashi.types import GraphCarrier, KernelParams


def test_initial_state_uses_declared_identity_assignment():
    identities = ["a", "b", "c"]
    state = initial_state_from_active_ids(identities, ["b"])
    assert state.tolist() == [[0], [1], [0]]


def test_initial_state_rejects_unknown_identity():
    try:
        initial_state_from_active_ids(["a", "b"], ["c"])
    except KeyError as exc:
        assert "c" in str(exc)
    else:
        raise AssertionError("unknown active identity must be rejected")


def test_decoder_is_derived_from_current_identity_assignment():
    decoder = decoder_from_identity_assignment(
        ["b", "a"],
        {"a": "A", "b": "B"},
    )
    assert dict(decoder.token_by_neuron) == {0: "B", 1: "A"}


def test_sparse_hash_is_deterministic_and_content_sensitive():
    first = sp.csr_matrix(np.eye(2, dtype=np.float32))
    second = sp.csr_matrix(np.eye(2, dtype=np.float32))
    changed = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    assert sha256_sparse_csr(first) == sha256_sparse_csr(second)
    assert sha256_sparse_csr(first) != sha256_sparse_csr(changed)


def test_run_symbolic_arm_emits_hashed_nonexecuting_receipt():
    adjacency = sp.csr_matrix(np.eye(2, dtype=np.float32))
    carrier = GraphCarrier(adjacency=adjacency, channels=1)
    receipt = run_symbolic_arm(
        run_id="baseline",
        carrier=carrier,
        identity_assignment=["a", "b"],
        active_ids=["a"],
        token_by_identity={"a": "x = 1\n"},
        topology_kind=TopologyKind.MALE_CNS,
        identity_assignment_kind=IdentityAssignmentKind.NATIVE,
        params=KernelParams(hops=1, deadzone=0.0),
        steps=2,
        assistance_overrides={
            "input_encoding": "active body IDs",
            "dynamics_rule": "DASHI kernel_flow",
            "initial_state_or_seed": "active=a;seed=7",
            "decoder_mapping": "declared JSON body-id to token mapping",
            "update_rule": "no learning in this clean-room runner",
            "reward_or_evaluator": "ast.parse only",
            "prompt_or_scaffold": "none",
            "parser_runtime": "Python ast.parse",
            "attempt_budget": 1,
            "selection_policy": "first run; no selection",
        },
    )

    assert receipt.emitted_program == "x = 1\n"
    assert receipt.parse_succeeded is True
    assert receipt.execution_succeeded is False
    assert receipt.declared_cases_passed is False
    assert len(receipt.output_artifact_sha256) == 64
    assert len(receipt.connectome_or_topology_sha256) == 64


def test_run_symbolic_arm_marks_invalid_python_without_execution():
    adjacency = sp.csr_matrix(np.eye(1, dtype=np.float32))
    carrier = GraphCarrier(adjacency=adjacency, channels=1)
    receipt = run_symbolic_arm(
        run_id="invalid",
        carrier=carrier,
        identity_assignment=["a"],
        active_ids=["a"],
        token_by_identity={"a": "for"},
        topology_kind=TopologyKind.MALE_CNS,
        identity_assignment_kind=IdentityAssignmentKind.NATIVE,
        params=KernelParams(),
        steps=1,
        assistance_overrides={
            "input_encoding": "active body IDs",
            "dynamics_rule": "DASHI kernel_flow",
            "initial_state_or_seed": "active=a;seed=7",
            "decoder_mapping": "declared mapping",
            "update_rule": "no learning in this clean-room runner",
            "reward_or_evaluator": "ast.parse only",
            "prompt_or_scaffold": "none",
            "parser_runtime": "Python ast.parse",
            "attempt_budget": 1,
            "selection_policy": "first run; no selection",
        },
    )
    assert receipt.parse_succeeded is False
    assert receipt.execution_succeeded is False
