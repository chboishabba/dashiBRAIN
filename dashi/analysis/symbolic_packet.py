"""Three-arm clean-room experiment packet for symbolic MaleCNS studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from dashi.analysis.symbolic_controls import (
    RewireResult,
    degree_and_strength_preserving_rewire,
    shuffle_identity_assignment,
)
from dashi.analysis.symbolic_experiment import run_symbolic_arm
from dashi.analysis.symbolic_interface import (
    IdentityAssignmentKind,
    SymbolicRunReceipt,
    TopologyKind,
    assert_matched_identity_control,
    assert_matched_topology_control,
)
from dashi.types import GraphCarrier, KernelParams


@dataclass(frozen=True)
class CleanRoomPacket:
    baseline: SymbolicRunReceipt
    topology_control: SymbolicRunReceipt
    identity_control: SymbolicRunReceipt
    rewire: RewireResult
    native_identity_assignment: tuple[str, ...]
    shuffled_identity_assignment: tuple[str, ...]
    seed: int
    viral_demo_reproduction_claimed: bool = False


def build_cleanroom_packet(
    *,
    carrier: GraphCarrier,
    identity_assignment: Sequence[str],
    active_ids: Sequence[str],
    token_by_identity: Mapping[str, str],
    params: KernelParams,
    steps: int,
    assistance_overrides: Mapping[str, object],
    rewire_swaps: int,
    seed: int,
) -> CleanRoomPacket:
    """Run baseline, topology-null, and identity-assignment control arms.

    All arms use the same declared semantic interface/assistance budget.  The
    topology arm changes only adjacency via invariant-preserving rewiring.  The
    identity arm changes only the assignment of neuron identities to graph
    indices, leaving adjacency untouched.
    """

    native_ids = tuple(str(x) for x in identity_assignment)
    if len(native_ids) != carrier.adjacency.shape[0]:
        raise ValueError("identity assignment must align with graph carrier")

    baseline = run_symbolic_arm(
        run_id="male-cns-baseline",
        carrier=carrier,
        identity_assignment=native_ids,
        active_ids=active_ids,
        token_by_identity=token_by_identity,
        topology_kind=TopologyKind.MALE_CNS,
        identity_assignment_kind=IdentityAssignmentKind.NATIVE,
        params=params,
        steps=steps,
        assistance_overrides=assistance_overrides,
    )

    rewire = degree_and_strength_preserving_rewire(
        carrier.adjacency,
        swaps=rewire_swaps,
        seed=seed,
    )
    rewired_carrier = GraphCarrier(
        adjacency=rewire.adjacency,
        channels=carrier.channels,
    )
    topology_control = run_symbolic_arm(
        run_id="degree-strength-preserving-rewire",
        carrier=rewired_carrier,
        identity_assignment=native_ids,
        active_ids=active_ids,
        token_by_identity=token_by_identity,
        topology_kind=TopologyKind.DEGREE_PRESERVING_REWIRE,
        identity_assignment_kind=IdentityAssignmentKind.NATIVE,
        params=params,
        steps=steps,
        assistance_overrides=assistance_overrides,
    )

    shuffled_ids = tuple(shuffle_identity_assignment(native_ids, seed=seed))
    identity_control = run_symbolic_arm(
        run_id="shuffled-identity-assignment",
        carrier=carrier,
        identity_assignment=shuffled_ids,
        active_ids=active_ids,
        token_by_identity=token_by_identity,
        topology_kind=TopologyKind.MALE_CNS,
        identity_assignment_kind=IdentityAssignmentKind.SHUFFLED,
        params=params,
        steps=steps,
        assistance_overrides=assistance_overrides,
    )

    assert_matched_topology_control(baseline, topology_control)
    assert_matched_identity_control(baseline, identity_control)

    return CleanRoomPacket(
        baseline=baseline,
        topology_control=topology_control,
        identity_control=identity_control,
        rewire=rewire,
        native_identity_assignment=native_ids,
        shuffled_identity_assignment=shuffled_ids,
        seed=seed,
    )


def _receipt_to_dict(receipt: SymbolicRunReceipt) -> dict[str, object]:
    payload = asdict(receipt)
    payload["topology_kind"] = receipt.topology_kind.value
    payload["identity_assignment_kind"] = receipt.identity_assignment_kind.value
    payload["intervention_kind"] = receipt.intervention_kind.value
    return payload


def packet_to_dict(packet: CleanRoomPacket) -> dict[str, object]:
    """Return a JSON-safe packet without embedding the sparse matrix twice."""

    return {
        "baseline": _receipt_to_dict(packet.baseline),
        "topology_control": _receipt_to_dict(packet.topology_control),
        "identity_control": _receipt_to_dict(packet.identity_control),
        "rewire": {
            "swaps_requested": packet.rewire.swaps_requested,
            "swaps_completed": packet.rewire.swaps_completed,
            "preserves_binary_in_degree": packet.rewire.preserves_binary_in_degree,
            "preserves_binary_out_degree": packet.rewire.preserves_binary_out_degree,
            "preserves_weighted_in_strength": packet.rewire.preserves_weighted_in_strength,
            "preserves_weighted_out_strength": packet.rewire.preserves_weighted_out_strength,
        },
        "native_identity_assignment": list(packet.native_identity_assignment),
        "shuffled_identity_assignment": list(packet.shuffled_identity_assignment),
        "seed": packet.seed,
        "viral_demo_reproduction_claimed": packet.viral_demo_reproduction_claimed,
    }
