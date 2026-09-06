from dashi.analysis.motor_effector import (
    AZEVEDO_MOTOR_ATLAS_SOURCE,
    LEHMANN_ACTIVATION_SOURCE,
    MotorEffectorPhysiology,
    PROPRIOCEPTION_BIOMECHANICS_SOURCE,
    SCHNELL_WING_MUSCLE_SOURCE,
)


def test_motor_effector_sources_have_stable_dois():
    for source in (
        AZEVEDO_MOTOR_ATLAS_SOURCE,
        SCHNELL_WING_MUSCLE_SOURCE,
        LEHMANN_ACTIVATION_SOURCE,
        PROPRIOCEPTION_BIOMECHANICS_SOURCE,
    ):
        assert source.stable_identifier.startswith("doi:")


def test_motor_effector_propagates_to_sensory_return():
    model = MotorEffectorPhysiology(
        muscle_target=lambda mn: f"muscle:{mn}",
        activation_from_spikes=lambda spikes: spikes + 1,
        force_from_activation=lambda activation: activation * 2,
        kinematics_from_force=lambda force: force + 3,
        external_force_from_kinematics=lambda kin: kin * 4,
        sensory_return_from_kinematics=lambda kin: kin - 1,
    )
    activation, force, kin, ext, sensory = model.propagate(2)
    assert (activation, force, kin, ext, sensory) == (3, 6, 9, 36, 8)
