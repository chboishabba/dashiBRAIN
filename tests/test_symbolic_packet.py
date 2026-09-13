import numpy as np
import scipy.sparse as sp

from dashi.analysis.symbolic_interface import (
    IdentityAssignmentKind,
    TopologyKind,
)
from dashi.analysis.symbolic_packet import build_cleanroom_packet, packet_to_dict
from dashi.types import GraphCarrier, KernelParams


def carrier() -> GraphCarrier:
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    return GraphCarrier(adjacency=adjacency, channels=1)


def assistance() -> dict[str, object]:
    return {
        "input_encoding": "active body IDs",
        "dynamics_rule": "DASHI kernel_flow",
        "initial_state_or_seed": "active=n0;seed=7",
        "decoder_mapping": "declared body-id to token mapping",
        "update_rule": "no learning in this clean-room runner",
        "reward_or_evaluator": "ast.parse only",
        "prompt_or_scaffold": "none",
        "parser_runtime": "Python ast.parse",
        "attempt_budget": 1,
        "selection_policy": "first run; no selection",
    }


def test_packet_builds_baseline_topology_and_identity_arms():
    packet = build_cleanroom_packet(
        carrier=carrier(),
        identity_assignment=["n0", "n1", "n2", "n3"],
        active_ids=["n0"],
        token_by_identity={"n0": "x=1\n"},
        params=KernelParams(hops=1, deadzone=0.0),
        steps=2,
        assistance_overrides=assistance(),
        rewire_swaps=1,
        seed=7,
    )

    assert packet.baseline.topology_kind == TopologyKind.MALE_CNS
    assert packet.baseline.identity_assignment_kind == IdentityAssignmentKind.NATIVE
    assert packet.topology_control.topology_kind == TopologyKind.DEGREE_PRESERVING_REWIRE
    assert packet.topology_control.identity_assignment_kind == IdentityAssignmentKind.NATIVE
    assert packet.identity_control.topology_kind == TopologyKind.MALE_CNS
    assert packet.identity_control.identity_assignment_kind == IdentityAssignmentKind.SHUFFLED
    assert packet.rewire.swaps_completed == 1


def test_packet_preserves_assistance_budget_across_matched_arms():
    packet = build_cleanroom_packet(
        carrier=carrier(),
        identity_assignment=["n0", "n1", "n2", "n3"],
        active_ids=["n0"],
        token_by_identity={"n0": "x=1\n"},
        params=KernelParams(),
        steps=2,
        assistance_overrides=assistance(),
        rewire_swaps=1,
        seed=11,
    )
    assert packet.baseline.assistance == packet.topology_control.assistance
    assert packet.baseline.assistance == packet.identity_control.assistance


def test_packet_serialization_uses_enum_values_and_receipts():
    packet = build_cleanroom_packet(
        carrier=carrier(),
        identity_assignment=["n0", "n1", "n2", "n3"],
        active_ids=["n0"],
        token_by_identity={"n0": "x=1\n"},
        params=KernelParams(),
        steps=2,
        assistance_overrides=assistance(),
        rewire_swaps=1,
        seed=5,
    )
    payload = packet_to_dict(packet)
    assert payload["baseline"]["topology_kind"] == "male_cns"
    assert payload["identity_control"]["identity_assignment_kind"] == "shuffled"
    assert payload["rewire"]["swaps_completed"] == 1
    assert payload["viral_demo_reproduction_claimed"] is False
