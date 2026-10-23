"""Finite consumer-retention and coarse-dynamics audits.

A runnable diagnostic companion to dashi_agda's
DASHI.Biology.BrainITConsumerRelativeExact and generic
CoarseObservableFactorisationBidiExact.

No BOLD, connectome or Brain-IT data is loaded here. A finite exact
factorisation test on supplied states is NOT proof of population-level
generalisation, nor is it a numerical fMRI reconstruction experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, TypeVar

Fine = TypeVar("Fine")
Coarse = TypeVar("Coarse", bound=Hashable)
Target = TypeVar("Target")


@dataclass(frozen=True)
class ConsumerCollision(Generic[Fine, Coarse, Target]):
    """Two fine states with the same measured value but different consumer."""

    left: Fine
    right: Fine
    observation: Coarse
    left_target: Target
    right_target: Target


@dataclass(frozen=True)
class RetentionAudit(Generic[Fine, Coarse, Target]):
    """Exact relative to the finite supplied state ensemble only."""

    state_count: int
    fibre_count: int
    ambiguous_fibre_count: int
    factorises_on_sample: bool
    collision: ConsumerCollision[Fine, Coarse, Target] | None


def audit_consumer_retention(
    states: Iterable[Fine],
    observation: Callable[[Fine], Coarse],
    consumer: Callable[[Fine], Target],
) -> RetentionAudit[Fine, Coarse, Target]:
    """Test whether `consumer = decoder(observation)` on *all supplied states*.

    Observations must be hashable. Targets need well-defined equality.
    A collision is a mathematical certificate of no exact decoder on even
    this finite subset; absence of a collision is only finite-sample evidence.
    """
    samples = list(states)
    if not samples:
        raise ValueError("at least one fine state must be supplied")
    representatives: dict[Coarse, tuple[Fine, Target]] = {}
    ambiguous: set[Coarse] = set()
    first_collision: ConsumerCollision[Fine, Coarse, Target] | None = None
    for state in samples:
        q = observation(state)
        y = consumer(state)
        if q not in representatives:
            representatives[q] = (state, y)
            continue
        previous_state, previous_value = representatives[q]
        if previous_value != y:
            ambiguous.add(q)
            if first_collision is None:
                first_collision = ConsumerCollision(
                    previous_state, state, q, previous_value, y
                )
    return RetentionAudit(
        state_count=len(samples),
        fibre_count=len(representatives),
        ambiguous_fibre_count=len(ambiguous),
        factorises_on_sample=not ambiguous,
        collision=first_collision,
    )


@dataclass(frozen=True)
class CommutationAudit(Generic[Fine]):
    """Equality-based `Q(K(x)) == K_coarse(Q(x))` on a finite sample."""

    state_count: int
    mismatches: int
    first_mismatch_state: Fine | None

    @property
    def commutes_on_sample(self) -> bool:
        return self.mismatches == 0


def audit_commutation(
    states: Iterable[Fine],
    fine_step: Callable[[Fine], Fine],
    observation: Callable[[Fine], Coarse],
    coarse_step: Callable[[Coarse], Coarse],
) -> CommutationAudit[Fine]:
    """Compare declared fine and coarse dynamics, without inventing a norm.

    Use only where fine/coarse state operations have the same physical,
    temporal and preprocessing identity. Before running real data, define
    an appropriate numerical discrepancy metric in a *separate* protocol.
    """
    samples = list(states)
    if not samples:
        raise ValueError("at least one fine state must be supplied")
    mismatches = 0
    first = None
    for state in samples:
        if observation(fine_step(state)) != coarse_step(observation(state)):
            mismatches += 1
            if first is None:
                first = state
    return CommutationAudit(len(samples), mismatches, first)
