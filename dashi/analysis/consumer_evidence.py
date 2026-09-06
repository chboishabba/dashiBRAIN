"""Consumer-indexed evidence sufficiency for the fly benchmark.

A neural or structure/function receipt does not automatically authorize effector,
behavioural, or semantic promotion.  Each consumer declares required channels
and whether independent replication is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet

from dashi.analysis.provenance_dependence import EvidenceRelation


class EvidenceConsumer(str, Enum):
    STRUCTURE_FUNCTION = "structure_function"
    NEURAL_STATE = "neural_state"
    EFFECTOR_STATE = "effector_state"
    BEHAVIOUR = "behaviour"
    SEMANTIC = "semantic"


class EvidenceChannel(str, Enum):
    CONNECTOME = "connectome"
    CALCIUM = "calcium"
    VOLTAGE = "voltage"
    EPHYS = "ephys"
    REGISTRATION = "registration"
    MUSCLE_ACTIVATION = "muscle_activation"
    KINEMATICS = "kinematics"
    CONTACT_FORCE = "contact_force"
    ENVIRONMENT = "environment"
    INTERACTION = "interaction"


@dataclass(frozen=True)
class ConsumerPolicy:
    consumer: EvidenceConsumer
    required: FrozenSet[EvidenceChannel]
    optional: FrozenSet[EvidenceChannel] = frozenset()
    independent_replication_required: bool = False


@dataclass(frozen=True)
class ConsumerEvidenceBundle:
    policy: ConsumerPolicy
    channels: FrozenSet[EvidenceChannel]
    dependence_relation: EvidenceRelation
    provenance_adequate: bool

    @property
    def has_required_channels(self) -> bool:
        return self.policy.required.issubset(self.channels)

    @property
    def independence_adequate(self) -> bool:
        if not self.policy.independent_replication_required:
            return True
        return self.dependence_relation in {
            EvidenceRelation.CROSS_TRIAL_REPLICATION,
            EvidenceRelation.CROSS_ANIMAL_REPLICATION,
            EvidenceRelation.CROSS_DATASET_REPLICATION,
        }

    @property
    def promotable(self) -> bool:
        return self.has_required_channels and self.provenance_adequate and self.independence_adequate


STRUCTURE_FUNCTION_POLICY = ConsumerPolicy(
    consumer=EvidenceConsumer.STRUCTURE_FUNCTION,
    required=frozenset({EvidenceChannel.CONNECTOME, EvidenceChannel.CALCIUM, EvidenceChannel.REGISTRATION}),
)

EFFECTOR_POLICY = ConsumerPolicy(
    consumer=EvidenceConsumer.EFFECTOR_STATE,
    required=frozenset({EvidenceChannel.MUSCLE_ACTIVATION, EvidenceChannel.KINEMATICS}),
)

BEHAVIOUR_POLICY = ConsumerPolicy(
    consumer=EvidenceConsumer.BEHAVIOUR,
    required=frozenset({EvidenceChannel.KINEMATICS}),
    optional=frozenset({EvidenceChannel.CONTACT_FORCE, EvidenceChannel.ENVIRONMENT}),
)

SEMANTIC_POLICY = ConsumerPolicy(
    consumer=EvidenceConsumer.SEMANTIC,
    required=frozenset({EvidenceChannel.KINEMATICS, EvidenceChannel.INTERACTION, EvidenceChannel.ENVIRONMENT}),
    independent_replication_required=True,
)


def transfer_requires_own_receipt(source: EvidenceConsumer, target: EvidenceConsumer) -> bool:
    return source is not target
