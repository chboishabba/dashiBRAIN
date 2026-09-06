"""Cross-modality functional observation bridge.

Calcium, voltage, electrophysiology and BOLD are distinct lossy observations of
latent neural/vascular state.  Calibration may relate them, but does not identify
them.  Mammalian neurovascular receipts do not create a Drosophila BOLD receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from dashi.analysis.embodied import (
    DROSOPHILA_CALCIUM_SOURCE,
    MALE_CNS_SOURCE,
    ScientificSource,
)

LOGOTHETIS_BOLD_SOURCE = ScientificSource(
    "Logothetis, Pauls, Augath, Trinath, Oeltermann",
    "Neurophysiological investigation of the basis of the fMRI signal",
    "doi:10.1038/35084005",
)

UBAGHS_CALCIUM_BOLD_SOURCE = ScientificSource(
    "Ubaghs et al.",
    "Simultaneous single-cell calcium imaging of neuronal population activity and brain-wide BOLD fMRI",
    "doi:10.1038/s41592-026-03154-2",
)

NeuralT = TypeVar("NeuralT")
VascularT = TypeVar("VascularT")
MetabolicT = TypeVar("MetabolicT")
CalciumT = TypeVar("CalciumT")
VoltageT = TypeVar("VoltageT")
EphysT = TypeVar("EphysT")
BoldT = TypeVar("BoldT")
ResidualT = TypeVar("ResidualT")


@dataclass(frozen=True)
class CrossModalityModel(Generic[
    NeuralT, VascularT, MetabolicT, CalciumT, VoltageT, EphysT, BoldT
]):
    calcium_readout: Callable[[NeuralT], CalciumT]
    voltage_readout: Callable[[NeuralT], VoltageT]
    ephys_readout: Callable[[NeuralT], EphysT]
    neurovascular_coupling: Callable[[NeuralT, VascularT, MetabolicT], VascularT]
    bold_readout: Callable[[NeuralT, VascularT, MetabolicT], BoldT]
    coupling_source: ScientificSource
    calibrated: bool = False


@dataclass(frozen=True)
class CrossModalityResiduals(Generic[CalciumT, VoltageT, EphysT, BoldT, ResidualT]):
    calcium_bold: Callable[[CalciumT, BoldT], ResidualT]
    voltage_bold: Callable[[VoltageT, BoldT], ResidualT]
    ephys_bold: Callable[[EphysT, BoldT], ResidualT]
    admissible: Callable[[ResidualT], bool]
    modalities_identified: bool = False

    def __post_init__(self) -> None:
        if self.modalities_identified:
            raise ValueError("cross-modality calibration does not establish modality identity")


@dataclass(frozen=True)
class FlyConnectomeFunctionalAdapter(Generic[NeuralT, CalciumT]):
    connectome_to_latent: Callable[[object], NeuralT]
    optical_readout: Callable[[NeuralT], CalciumT]
    connectome_source: ScientificSource = MALE_CNS_SOURCE
    optical_source: ScientificSource = DROSOPHILA_CALCIUM_SOURCE
    fly_bold_receipt_present: bool = False

    def __post_init__(self) -> None:
        if self.fly_bold_receipt_present:
            raise ValueError(
                "set fly_bold_receipt_present only when an explicit Drosophila BOLD source is supplied"
            )


def mammalian_bridge_does_not_create_fly_bold_receipt() -> bool:
    """Regression firewall for cross-species x-pollination."""

    sources = (LOGOTHETIS_BOLD_SOURCE, UBAGHS_CALCIUM_BOLD_SOURCE)
    return all("drosophila" not in s.title.lower() for s in sources)
