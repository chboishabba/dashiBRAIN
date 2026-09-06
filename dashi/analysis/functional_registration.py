"""Registration seam between Drosophila functional imaging and connectomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, TypeVar

from dashi.analysis.embodied import ScientificSource

BIFROST_SOURCE = ScientificSource(
    "Brezovec, Berger, Hao, Lin, Ahmed, Pacheco, Thiberge, Murthy, Clandinin",
    "BIFROST: A method for registering diverse imaging datasets of the Drosophila brain",
    "doi:10.1073/pnas.2322687121",
)

MANN_WHOLE_BRAIN_SOURCE = ScientificSource(
    "Mann, Gallen, Clandinin",
    "Whole-Brain Calcium Imaging Reveals an Intrinsic Functional Network in Drosophila",
    "doi:10.1016/j.cub.2017.06.076",
)

TURNER_STRUCTURE_FUNCTION_SOURCE = ScientificSource(
    "Turner, Mann, Clandinin",
    "The connectome predicts resting-state functional connectivity across the Drosophila brain",
    "doi:10.1016/j.cub.2021.03.004",
)


class RegistrationEvidence(str, Enum):
    ATLAS_REGION = "atlas_region_overlap"
    COORDINATE_PROXIMITY = "coordinate_proximity"
    MORPHOLOGY = "morphology_agreement"
    GENETIC_CELL_TYPE = "genetic_cell_type_agreement"
    DIRECT_IDENTITY = "direct_neuron_identity"


FunctionalUnitT = TypeVar("FunctionalUnitT")
ConnectomeUnitT = TypeVar("ConnectomeUnitT")
ResidualT = TypeVar("ResidualT")
ConfidenceT = TypeVar("ConfidenceT")


@dataclass(frozen=True)
class FunctionalConnectomeRegistration(Generic[
    FunctionalUnitT, ConnectomeUnitT, ResidualT, ConfidenceT
]):
    candidate: Callable[[FunctionalUnitT], ConnectomeUnitT]
    residual: Callable[[FunctionalUnitT], ResidualT]
    confidence: Callable[[FunctionalUnitT], ConfidenceT]
    evidence: Callable[[FunctionalUnitT], tuple[RegistrationEvidence, ...]]
    residual_admissible: Callable[[ResidualT], bool]
    source: ScientificSource = BIFROST_SOURCE


StructuralT = TypeVar("StructuralT")
FunctionalT = TypeVar("FunctionalT")
ComparisonResidualT = TypeVar("ComparisonResidualT")


@dataclass(frozen=True)
class StructuralFunctionalComparison(Generic[
    ConnectomeUnitT, FunctionalUnitT, StructuralT, FunctionalT, ComparisonResidualT
]):
    structural: Callable[[ConnectomeUnitT, ConnectomeUnitT], StructuralT]
    functional: Callable[[FunctionalUnitT, FunctionalUnitT], FunctionalT]
    residual: Callable[[StructuralT, FunctionalT], ComparisonResidualT]
    source: ScientificSource = TURNER_STRUCTURE_FUNCTION_SOURCE
    structure_equals_function_claim: bool = False

    def __post_init__(self) -> None:
        if self.structure_equals_function_claim:
            raise ValueError("structural connectivity and functional association are not identical objects")


def identity_strength(evidence: tuple[RegistrationEvidence, ...]) -> int:
    """Ordinal diagnostic only; does not turn weak evidence into identity."""

    ranking = {
        RegistrationEvidence.ATLAS_REGION: 1,
        RegistrationEvidence.COORDINATE_PROXIMITY: 2,
        RegistrationEvidence.MORPHOLOGY: 3,
        RegistrationEvidence.GENETIC_CELL_TYPE: 4,
        RegistrationEvidence.DIRECT_IDENTITY: 5,
    }
    return max((ranking[item] for item in evidence), default=0)
