"""Build auditable symbolic clean-room run receipts over an existing graph.

This module intentionally stops at Python parsing.  It never executes emitted
source code.  Task execution/correctness requires a separately governed
evaluator receipt in a later tranche.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from dashi.analysis.symbolic_interface import (
    AssistanceBudget,
    IdentityAssignmentKind,
    InterventionKind,
    SymbolicRunReceipt,
    TopologyKind,
)
from dashi.analysis.symbolic_runner import SymbolDecoder, run_kernel_symbolic_trace
from dashi.types import GraphCarrier, KernelParams, T


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def sha256_json(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_text(canonical)


def sha256_sparse_csr(adjacency: sp.csr_matrix) -> str:
    """Hash canonical CSR content, including shape and dtype."""

    A = adjacency.tocsr(copy=True)
    A.sum_duplicates()
    A.sort_indices()
    h = sha256()
    h.update(json.dumps([int(A.shape[0]), int(A.shape[1])]).encode("ascii"))
    h.update(str(A.dtype).encode("ascii"))
    h.update(A.indptr.tobytes(order="C"))
    h.update(A.indices.tobytes(order="C"))
    h.update(A.data.tobytes(order="C"))
    return h.hexdigest()


def initial_state_from_active_ids(
    identity_assignment: Sequence[str],
    active_ids: Sequence[str],
) -> np.ndarray:
    """Create a one-channel ternary state from declared active identities."""

    index_by_identity: dict[str, int] = {}
    for index, identity in enumerate(identity_assignment):
        if identity in index_by_identity:
            raise ValueError(f"duplicate identity assignment: {identity}")
        index_by_identity[identity] = index

    state = np.zeros((len(identity_assignment), 1), dtype=T)
    for identity in active_ids:
        if identity not in index_by_identity:
            raise KeyError(f"active identity not present in assignment: {identity}")
        state[index_by_identity[identity], 0] = 1
    return state


def decoder_from_identity_assignment(
    identity_assignment: Sequence[str],
    token_by_identity: Mapping[str, str],
) -> SymbolDecoder:
    """Resolve a declared identity->token map against the current assignment."""

    token_by_neuron: dict[int, str] = {}
    for index, identity in enumerate(identity_assignment):
        token = token_by_identity.get(identity)
        if token is not None:
            token_by_neuron[index] = token
    return SymbolDecoder(token_by_neuron)


def python_parse_succeeds(program: str) -> bool:
    if not program:
        return False
    try:
        ast.parse(program)
    except SyntaxError:
        return False
    return True


def _kernel_params_payload(params: KernelParams, *, steps: int) -> dict[str, object]:
    coupling = params.channel_coupling
    return {
        "runner": "dashi.kernel.flow.kernel_flow",
        "hops": int(params.hops),
        "deadzone": float(params.deadzone),
        "steps": int(steps),
        "channel_coupling": None if coupling is None else np.asarray(coupling).tolist(),
    }


def run_symbolic_arm(
    *,
    run_id: str,
    carrier: GraphCarrier,
    identity_assignment: Sequence[str],
    active_ids: Sequence[str],
    token_by_identity: Mapping[str, str],
    topology_kind: TopologyKind,
    identity_assignment_kind: IdentityAssignmentKind,
    params: KernelParams,
    steps: int,
    assistance_overrides: Mapping[str, object],
) -> SymbolicRunReceipt:
    """Run one clean-room arm and return a hashed non-executing receipt."""

    if carrier.channels != 1:
        raise ValueError("symbolic clean-room runner currently requires one channel")
    if len(identity_assignment) != carrier.adjacency.shape[0]:
        raise ValueError("identity assignment must align with graph carrier")

    assistance = AssistanceBudget(**dict(assistance_overrides))
    initial = initial_state_from_active_ids(identity_assignment, active_ids)
    decoder = decoder_from_identity_assignment(identity_assignment, token_by_identity)
    trace = run_kernel_symbolic_trace(
        carrier,
        initial,
        params,
        decoder,
        steps=steps,
    )
    emitted = trace.emitted_program

    decoder_payload = {
        str(index): token
        for index, token in sorted(decoder.token_by_neuron.items())
    }
    evaluator_payload = {
        "evaluator": "python.ast.parse",
        "executes_code": False,
        "version": 1,
    }

    return SymbolicRunReceipt(
        run_id=run_id,
        topology_kind=topology_kind,
        identity_assignment_kind=identity_assignment_kind,
        intervention_kind=InterventionKind.BASELINE,
        assistance=assistance,
        connectome_or_topology_sha256=sha256_sparse_csr(carrier.adjacency),
        dynamics_artifact_sha256=sha256_json(_kernel_params_payload(params, steps=steps)),
        decoder_artifact_sha256=sha256_json(decoder_payload),
        output_artifact_sha256=sha256_text(emitted),
        evaluator_artifact_sha256=sha256_json(evaluator_payload),
        emitted_program=emitted,
        parse_succeeded=python_parse_succeeds(emitted),
        execution_succeeded=False,
        declared_cases_passed=False,
        held_out_cases_passed=False,
        cross_task_transfer_passed=False,
    )
