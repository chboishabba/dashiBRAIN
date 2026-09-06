"""Embodied MaleCNS analysis carriers.

This module extends the original internal-connectome prototype across the motor
boundary without conflating structural connectivity with physical effector
state.  External scientific sources used by this lane must be cited by
(author/consortium, title, DOI or another stable identifier).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, Mapping, Protocol, TypeVar


class CNSRole(str, Enum):
    SENSORY = "sensory"
    CENTRAL = "central_interneuron"
    DESCENDING = "descending"
    ASCENDING = "ascending"
    VNC = "vnc_interneuron"
    MOTOR = "motor_neuron"
    ENDOCRINE = "endocrine_effector_neuron"
    VISCERAL = "visceral_efferent_neuron"


class ObservationModality(str, Enum):
    STRUCTURAL_CONNECTOME = "structural_connectome"
    OPTICAL_CALCIUM = "optical_calcium"
    OPTICAL_VOLTAGE = "optical_voltage"
    ELECTROPHYSIOLOGY = "electrophysiology"
    BOLD_FMRI = "bold_fmri"
    BEHAVIOURAL_KINEMATICS = "behavioural_kinematics"
    FORCE_OR_CONTACT = "force_or_contact"


@dataclass(frozen=True)
class ScientificSource:
    author_or_consortium: str
    title: str
    stable_identifier: str

    def __post_init__(self) -> None:
        if not self.author_or_consortium.strip():
            raise ValueError("source author/consortium is required")
        if not self.title.strip():
            raise ValueError("source title is required")
        if not self.stable_identifier.strip():
            raise ValueError("DOI or another stable identifier is required")


MALE_CNS_SOURCE = ScientificSource(
    "Berg et al.",
    "Sexual dimorphism in the complete Drosophila male central nervous system connectome",
    "doi:10.1016/j.cell.2026.08.015",
)

BANC_EMBODIED_SOURCE = ScientificSource(
    "Bates et al.; BANC-FlyWire Consortium",
    "Distributed control circuits across a brain-and-cord connectome",
    "doi:10.1038/s41586-026-10735-w",
)

DROSOPHILA_CALCIUM_SOURCE = ScientificSource(
    "Gauthey, Lin, Ahmed, Leifer, Murthy, Thiberge et al.",
    "High-speed whole-brain imaging in Drosophila",
    "doi:10.1038/s41467-026-72437-1",
)


@dataclass(frozen=True)
class NodeAnnotation:
    role: CNSRole
    cell_type: str | None = None
    region: str | None = None
    hemilineage: str | None = None
    neurotransmitter: str | None = None
    sex_correspondence: str | None = None


@dataclass(frozen=True)
class EmbodiedConnectome:
    """Structural graph plus held-out biological annotations.

    Annotations are carried with the graph but need not be inputs to a kernel;
    this permits them to remain independent validation labels.
    """

    annotations: Mapping[str, NodeAnnotation]
    source: ScientificSource = MALE_CNS_SOURCE


EffectorStateT = TypeVar("EffectorStateT")
CoefficientT = TypeVar("CoefficientT")
ResidualT = TypeVar("ResidualT")


@dataclass(frozen=True)
class EffectorROM(Generic[EffectorStateT, CoefficientT, ResidualT]):
    """Reduced-order effector state with an explicit reconstruction residual."""

    project: Callable[[EffectorStateT], CoefficientT]
    reconstruct: Callable[[CoefficientT], EffectorStateT]
    residual: Callable[[EffectorStateT], ResidualT]
    residual_admissible: Callable[[ResidualT], bool]
    receipt: str
    exact_inverse_claim: bool = False

    def __post_init__(self) -> None:
        if self.exact_inverse_claim:
            raise ValueError(
                "Effector ROM exactness requires a separate exact-reconstruction receipt"
            )


NeuralStateT = TypeVar("NeuralStateT")
MotorCommandT = TypeVar("MotorCommandT")
MotorDriveT = TypeVar("MotorDriveT")
BodyStateT = TypeVar("BodyStateT")
BehaviourT = TypeVar("BehaviourT")
SensoryReturnT = TypeVar("SensoryReturnT")


class NeuralToEffector(Protocol[
    NeuralStateT,
    MotorCommandT,
    MotorDriveT,
    EffectorStateT,
    BodyStateT,
    BehaviourT,
    SensoryReturnT,
]):
    """Typed continuation from neural state into body and sensory return."""

    def motor_command(self, state: NeuralStateT) -> MotorCommandT: ...

    def motor_drive(self, command: MotorCommandT) -> MotorDriveT: ...

    def effector_state(self, drive: MotorDriveT) -> EffectorStateT: ...

    def body_state(self, effector: EffectorStateT) -> BodyStateT: ...

    def observe_behaviour(self, body: BodyStateT) -> BehaviourT: ...

    def sensory_return(self, body: BodyStateT) -> SensoryReturnT: ...


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class ObservationQuotient(Generic[NeuralStateT, ObservationT]):
    modality: ObservationModality
    observe: Callable[[NeuralStateT], ObservationT]
    source: ScientificSource
    lossy: bool = True


def fruit_fly_bold_fmri_receipted(sources: tuple[ScientificSource, ...]) -> bool:
    """Bounded evidence check, not a claim that no fly fMRI exists anywhere.

    Returns True only when the supplied source set explicitly contains a BOLD
    fMRI Drosophila receipt.  Optical calcium/voltage imaging does not count.
    """

    tokens = ("bold", "fmri", "functional magnetic resonance")
    return any(
        any(token in (s.title + " " + s.stable_identifier).lower() for token in tokens)
        for s in sources
    )
