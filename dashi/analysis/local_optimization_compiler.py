"""Least-privilege optimization certificates for streaming computations.

The production algorithms in dashiBRAIN are free to use Arrow, NumPy, sparse
matrices, chunking, or other implementation tactics.  This module captures the
smaller facts a consumer actually needs:

1. a local update law;
2. an observation of state relevant to the consumer;
3. a proof/check that spec and optimized updates have the same observation; and
4. a residency budget that is independent of semantic correctness.

Whole-fold observational equivalence is compiler output from the local law.
Resource bounds are carried separately; empirical RSS measurements may validate
an execution, but are not premises of semantic correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Sequence, TypeVar

S = TypeVar("S")
X = TypeVar("X")
Q = TypeVar("Q")


@dataclass(frozen=True)
class LocalSemanticCertificate(Generic[S, X, Q]):
    """Local spec/optimized update agreement at the consumer observation."""

    spec_update: Callable[[S, X], S]
    optimized_update: Callable[[S, X], S]
    observe: Callable[[S], Q]

    def step_observation_equal(self, state: S, item: X) -> bool:
        return self.observe(self.spec_update(state, item)) == self.observe(
            self.optimized_update(state, item)
        )


@dataclass(frozen=True)
class FoldEquivalenceResult(Generic[Q]):
    item_count: int
    spec_observation: Q
    optimized_observation: Q
    equal: bool


def compile_fold_observational_equivalence(
    initial_spec: S,
    initial_optimized: S,
    items: Iterable[X],
    certificate: LocalSemanticCertificate[S, X, Q],
    *,
    require_local_step_checks: bool = True,
) -> FoldEquivalenceResult[Q]:
    """Compile global observed equivalence from local observed update laws.

    This checker is intentionally for finite fixtures/certificates, not for
    replaying giant production datasets.  It compares only ``observe(state)``;
    complete internal-state equality is neither assumed nor required.
    """

    spec = initial_spec
    optimized = initial_optimized
    count = 0
    for item in items:
        if require_local_step_checks:
            # Evaluate the same local item from each current implementation
            # state and require agreement of the consumer-visible projection.
            spec_next = certificate.spec_update(spec, item)
            optimized_next = certificate.optimized_update(optimized, item)
            if certificate.observe(spec_next) != certificate.observe(optimized_next):
                raise AssertionError(f"local observational law failed at item {count}")
            spec, optimized = spec_next, optimized_next
        else:
            spec = certificate.spec_update(spec, item)
            optimized = certificate.optimized_update(optimized, item)
        count += 1

    spec_q = certificate.observe(spec)
    optimized_q = certificate.observe(optimized)
    return FoldEquivalenceResult(
        item_count=count,
        spec_observation=spec_q,
        optimized_observation=optimized_q,
        equal=spec_q == optimized_q,
    )


@dataclass(frozen=True)
class LocalResidencyBound:
    """One-step live-residency bound, separate from semantic correctness.

    ``stream_item_bytes`` is one live chunk/domain, not the complete source
    collection. ``persistent_state_bytes`` may depend on fixed consumer
    dimensions (for example regions x neurons).  The theorem-shaped bound is
    therefore constant in the number of streamed source chunks while allowing
    the consumer's fixed output carrier to have its own size.
    """

    persistent_state_bytes: int
    stream_item_bytes: int
    scratch_bytes: int = 0
    fixed_runtime_overhead_bytes: int = 0

    def __post_init__(self) -> None:
        if min(
            self.persistent_state_bytes,
            self.stream_item_bytes,
            self.scratch_bytes,
            self.fixed_runtime_overhead_bytes,
        ) < 0:
            raise ValueError("residency components must be non-negative")

    @property
    def peak_bound_bytes(self) -> int:
        return (
            self.persistent_state_bytes
            + self.stream_item_bytes
            + self.scratch_bytes
            + self.fixed_runtime_overhead_bytes
        )


def source_count_does_not_scale_live_stream_item(
    source_item_count: int,
    bound: LocalResidencyBound,
) -> bool:
    """Record the streaming distinction without pretending work is O(1).

    The number of source items may affect total work.  It does not multiply the
    one-item live residency term represented by ``LocalResidencyBound``.
    """

    if source_item_count < 0:
        raise ValueError("source_item_count must be non-negative")
    return bound.peak_bound_bytes == (
        bound.persistent_state_bytes
        + bound.stream_item_bytes
        + bound.scratch_bytes
        + bound.fixed_runtime_overhead_bytes
    )


@dataclass(frozen=True)
class ExecutionResourceWitness:
    """Empirical execution observation; never a semantic proof premise."""

    completed: bool
    peak_rss_bytes: int | None = None
    elapsed_seconds: float | None = None
    resource_exhausted: bool = False

    def fits(self, bound: LocalResidencyBound) -> bool | None:
        if self.peak_rss_bytes is None:
            return None
        return self.peak_rss_bytes <= bound.peak_bound_bytes
