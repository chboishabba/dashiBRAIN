"""Animalexic governance cross-pollination for Drosophila experiments.

This module reuses the core Animalexic distinction between substrate observations,
candidates, promoted state, abstention/rejection, provenance, and behavioural
motifs.  It does not transfer species-specific semantics into Drosophila.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Iterable, Mapping, Sequence, TypeVar

from dashi.analysis.embodied import ScientificSource


class Decision(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    ABSTAIN = "abstain"
    REJECT = "reject"


class Modality(str, Enum):
    OPTICAL_CALCIUM = "optical_calcium"
    OPTICAL_VOLTAGE = "optical_voltage"
    EPHYS = "electrophysiology"
    POSTURE = "posture"
    MOTION = "motion"
    CONTACT = "contact"
    ENVIRONMENT = "environment"


ObservationT = TypeVar("ObservationT")
CandidateT = TypeVar("CandidateT")
StateT = TypeVar("StateT")
ReceiptT = TypeVar("ReceiptT")
ResidualT = TypeVar("ResidualT")
MotifT = TypeVar("MotifT")
IntervalT = TypeVar("IntervalT")


@dataclass(frozen=True)
class GovernedCandidate(Generic[ObservationT, CandidateT, StateT, ReceiptT]):
    observation: ObservationT
    candidate: CandidateT
    decision: Decision
    receipt: ReceiptT | None
    materialise: Callable[[CandidateT], StateT]
    admissible_receipt: Callable[[ObservationT, CandidateT, ReceiptT], bool]

    def promoted_state(self) -> StateT:
        if self.decision is not Decision.PROMOTED:
            raise ValueError("canonical state is available only for promoted candidates")
        if self.receipt is None:
            raise ValueError("promotion requires an explicit receipt")
        if not self.admissible_receipt(self.observation, self.candidate, self.receipt):
            raise ValueError("promotion receipt is not admissible")
        return self.materialise(self.candidate)


@dataclass(frozen=True)
class ProvenanceNode:
    identifier: str
    upstream: tuple[str, ...] = ()


def upstream_closure(nodes: Mapping[str, ProvenanceNode], start: str) -> frozenset[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is not None:
            stack.extend(node.upstream)
    return frozenset(seen)


def independent_by_upstream_closure(
    nodes: Mapping[str, ProvenanceNode], left: str, right: str
) -> bool:
    return upstream_closure(nodes, left).isdisjoint(upstream_closure(nodes, right))


@dataclass(frozen=True)
class MultimodalTrace:
    modalities: frozenset[Modality]
    provenance_ids: tuple[str, ...]

    @property
    def is_single_channel(self) -> bool:
        return len(self.modalities) <= 1


@dataclass(frozen=True)
class BehaviourMotif(Generic[MotifT, IntervalT]):
    motif: MotifT
    interval: IntervalT
    source: ScientificSource
    semantic_label: str | None = None

    @property
    def semantics_promoted(self) -> bool:
        return self.semantic_label is not None


@dataclass(frozen=True)
class ResidualGate(Generic[ResidualT]):
    residuals: Sequence[ResidualT]
    admissible: Callable[[ResidualT], bool]

    @property
    def all_admissible(self) -> bool:
        return all(self.admissible(value) for value in self.residuals)


def can_promote_multimodal_state(
    trace: MultimodalTrace,
    residual_gate: ResidualGate[ResidualT],
    provenance_nodes: Mapping[str, ProvenanceNode],
    *,
    require_independent_roots: bool = False,
) -> bool:
    if trace.is_single_channel or not residual_gate.all_admissible:
        return False
    if require_independent_roots:
        ids = trace.provenance_ids
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                if not independent_by_upstream_closure(provenance_nodes, left, right):
                    return False
    return True


def recurrent_motif_does_not_imply_semantics(motif: BehaviourMotif[object, object]) -> bool:
    """Regression firewall: recurrence itself carries no semantic promotion."""

    return not motif.semantics_promoted
