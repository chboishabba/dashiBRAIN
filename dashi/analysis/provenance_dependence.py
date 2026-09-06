"""Same-trial provenance and dependence-aware evidence aggregation.

Distinct modalities can corroborate one another without constituting independent
replication when they share an animal, trial, registration, preprocessing, or
predictor root.  This module makes that dependence explicit for the Drosophila
connectome -> function -> effector benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Hashable, Iterable, Mapping, TypeVar


class RootKind(str, Enum):
    DATASET = "dataset"
    ANIMAL = "animal"
    TRIAL = "trial"
    ACQUISITION = "acquisition"
    REGISTRATION = "registration"
    PREPROCESSING = "preprocessing"
    PREDICTOR = "predictor"
    OBSERVER_PROTOCOL = "observer_protocol"


class EvidenceRelation(str, Enum):
    SAME_TRIAL_CORROBORATION = "same_trial_corroboration"
    SHARED_PIPELINE_CORROBORATION = "shared_pipeline_corroboration"
    CROSS_TRIAL_REPLICATION = "cross_trial_replication"
    CROSS_ANIMAL_REPLICATION = "cross_animal_replication"
    CROSS_DATASET_REPLICATION = "cross_dataset_replication"
    INDEPENDENCE_UNDETERMINED = "independence_undetermined"


@dataclass(frozen=True)
class ProvenanceRoot:
    root_id: str
    kind: RootKind
    stable_reference: str


ArtifactT = TypeVar("ArtifactT", bound=Hashable)
ResidualT = TypeVar("ResidualT")


@dataclass(frozen=True)
class ProvenanceGraph(Generic[ArtifactT]):
    roots_by_artifact: Mapping[ArtifactT, frozenset[ProvenanceRoot]]

    def roots(self, artifact: ArtifactT) -> frozenset[ProvenanceRoot]:
        return self.roots_by_artifact.get(artifact, frozenset())

    def shared_roots(self, left: ArtifactT, right: ArtifactT) -> frozenset[ProvenanceRoot]:
        return self.roots(left) & self.roots(right)

    def independent_by_upstream_closure(self, left: ArtifactT, right: ArtifactT) -> bool:
        left_roots = self.roots(left)
        right_roots = self.roots(right)
        # Unknown provenance cannot be promoted to independence.
        return bool(left_roots and right_roots) and left_roots.isdisjoint(right_roots)

    def shares_kind(self, left: ArtifactT, right: ArtifactT, kind: RootKind) -> bool:
        return any(root.kind is kind for root in self.shared_roots(left, right))

    def relation(self, left: ArtifactT, right: ArtifactT) -> EvidenceRelation:
        if self.shares_kind(left, right, RootKind.TRIAL):
            return EvidenceRelation.SAME_TRIAL_CORROBORATION
        if any(
            self.shares_kind(left, right, kind)
            for kind in (RootKind.REGISTRATION, RootKind.PREPROCESSING, RootKind.PREDICTOR)
        ):
            return EvidenceRelation.SHARED_PIPELINE_CORROBORATION
        if self.independent_by_upstream_closure(left, right):
            left_kinds = {r.kind for r in self.roots(left)}
            right_kinds = {r.kind for r in self.roots(right)}
            if RootKind.ANIMAL in left_kinds | right_kinds:
                return EvidenceRelation.CROSS_ANIMAL_REPLICATION
            if RootKind.DATASET in left_kinds | right_kinds:
                return EvidenceRelation.CROSS_DATASET_REPLICATION
            return EvidenceRelation.CROSS_TRIAL_REPLICATION
        return EvidenceRelation.INDEPENDENCE_UNDETERMINED


@dataclass(frozen=True)
class SameTrialBundle(Generic[ArtifactT]):
    structural: ArtifactT
    functional: ArtifactT
    body: ArtifactT
    behaviour: ArtifactT
    registration: ArtifactT
    animal_id: str
    trial_id: str

    def artifacts(self) -> tuple[ArtifactT, ...]:
        return (
            self.structural,
            self.functional,
            self.body,
            self.behaviour,
            self.registration,
        )


@dataclass(frozen=True)
class DependenceAwareAggregator(Generic[ArtifactT, ResidualT]):
    graph: ProvenanceGraph[ArtifactT]
    combine_within_dependence_class: Callable[[tuple[ResidualT, ...]], ResidualT]
    combine_across_independent_classes: Callable[[tuple[ResidualT, ...]], ResidualT]

    def dependency_classes(self, artifacts: Iterable[ArtifactT]) -> tuple[frozenset[ArtifactT], ...]:
        remaining = set(artifacts)
        classes: list[frozenset[ArtifactT]] = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            changed = True
            while changed:
                changed = False
                for candidate in tuple(remaining):
                    if any(self.graph.shared_roots(candidate, member) for member in component):
                        component.add(candidate)
                        remaining.remove(candidate)
                        changed = True
            classes.append(frozenset(component))
        return tuple(classes)

    def aggregate(self, residuals: Mapping[ArtifactT, ResidualT]) -> ResidualT:
        class_values = []
        for dependency_class in self.dependency_classes(residuals):
            class_values.append(
                self.combine_within_dependence_class(
                    tuple(residuals[a] for a in dependency_class)
                )
            )
        return self.combine_across_independent_classes(tuple(class_values))


def distinct_modality_does_not_imply_independence(
    graph: ProvenanceGraph[ArtifactT], left: ArtifactT, right: ArtifactT
) -> bool:
    """Regression predicate for the Animalexic provenance-closure boundary."""
    return not graph.shared_roots(left, right) or not graph.independent_by_upstream_closure(
        left, right
    )
