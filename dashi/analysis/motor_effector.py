"""Concrete Drosophila motor-neuron -> effector -> body feedback producers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from dashi.analysis.embodied import ScientificSource

AZEVEDO_MOTOR_ATLAS_SOURCE = ScientificSource(
    "Azevedo et al.",
    "Connectomic reconstruction of a female Drosophila ventral nerve cord",
    "doi:10.1038/s41586-024-07389-x",
)

LESSER_PREMOTOR_SOURCE = ScientificSource(
    "Lesser, Azevedo, Phelps et al.",
    "Synaptic architecture of leg and wing premotor control networks in Drosophila",
    "doi:10.1038/s41586-024-07600-z",
)

SCHNELL_WING_MUSCLE_SOURCE = ScientificSource(
    "Schnell, Weir, Roth, Fairhall, Dickinson",
    "The Function and Organization of the Motor System Controlling Flight Maneuvers in Flies",
    "doi:10.1016/j.cub.2016.12.018",
)

LEHMANN_ACTIVATION_SOURCE = ScientificSource(
    "Lehmann, Gotz",
    "Activation phase ensures kinematic efficacy in flight-steering muscles of Drosophila melanogaster",
    "doi:10.1007/BF00194985",
)

PROPRIOCEPTION_BIOMECHANICS_SOURCE = ScientificSource(
    "Mamiya et al.",
    "Biomechanical origins of proprioceptor feature selectivity and topographic maps in the Drosophila leg",
    "doi:10.1016/j.neuron.2023.07.009",
)

MotorNeuronT = TypeVar("MotorNeuronT")
MuscleT = TypeVar("MuscleT")
SpikeTrainT = TypeVar("SpikeTrainT")
ActivationT = TypeVar("ActivationT")
ForceT = TypeVar("ForceT")
KinematicsT = TypeVar("KinematicsT")
ExternalForceT = TypeVar("ExternalForceT")
SensoryReturnT = TypeVar("SensoryReturnT")


@dataclass(frozen=True)
class MotorEffectorPhysiology(Generic[
    MotorNeuronT,
    MuscleT,
    SpikeTrainT,
    ActivationT,
    ForceT,
    KinematicsT,
    ExternalForceT,
    SensoryReturnT,
]):
    muscle_target: Callable[[MotorNeuronT], MuscleT]
    activation_from_spikes: Callable[[SpikeTrainT], ActivationT]
    force_from_activation: Callable[[ActivationT], ForceT]
    kinematics_from_force: Callable[[ForceT], KinematicsT]
    external_force_from_kinematics: Callable[[KinematicsT], ExternalForceT]
    sensory_return_from_kinematics: Callable[[KinematicsT], SensoryReturnT]
    motor_atlas_source: ScientificSource = AZEVEDO_MOTOR_ATLAS_SOURCE
    activation_source: ScientificSource = SCHNELL_WING_MUSCLE_SOURCE
    biomechanics_source: ScientificSource = LEHMANN_ACTIVATION_SOURCE
    sensory_return_source: ScientificSource = PROPRIOCEPTION_BIOMECHANICS_SOURCE

    def propagate(self, spikes: SpikeTrainT) -> tuple[ActivationT, ForceT, KinematicsT, ExternalForceT, SensoryReturnT]:
        activation = self.activation_from_spikes(spikes)
        force = self.force_from_activation(activation)
        kinematics = self.kinematics_from_force(force)
        external_force = self.external_force_from_kinematics(kinematics)
        sensory_return = self.sensory_return_from_kinematics(kinematics)
        return activation, force, kinematics, external_force, sensory_return
