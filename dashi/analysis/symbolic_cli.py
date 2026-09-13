"""CLI core for producing a three-arm MaleCNS symbolic clean-room packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from dashi.analysis.symbolic_packet import build_cleanroom_packet, packet_to_dict
from dashi.io.malecns_loader import load_malecns_graph
from dashi.io.malecns_protocol import sha256_file
from dashi.types import GraphCarrier, KernelParams


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean-room MaleCNS symbolic baseline plus matched topology "
            "and identity-assignment controls. Emitted Python is parsed only; "
            "it is never executed."
        )
    )
    parser.add_argument("weights_path", help="MaleCNS weights Feather/CSV path")
    parser.add_argument("--decoder-json", required=True, help="JSON object: body ID -> emitted token")
    parser.add_argument(
        "--active-body-id",
        action="append",
        required=True,
        help="Body ID initially set to +1; repeat for multiple IDs",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--deadzone", type=float, default=0.0)
    parser.add_argument("--rewire-swaps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-version", default="male-cns:v1.0")
    parser.add_argument("--output", required=True, help="Output JSON packet path")
    return parser


def load_token_mapping(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decoder JSON must be a JSON object")
    mapping: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not value:
            raise ValueError("decoder tokens must be non-empty strings")
        mapping[str(key)] = value
    return mapping


def _normalize_carrier(raw_carrier: object) -> GraphCarrier:
    if isinstance(raw_carrier, GraphCarrier):
        return raw_carrier
    return GraphCarrier(adjacency=raw_carrier, channels=1)  # type: ignore[arg-type]


def _assistance_budget(
    *,
    active_ids: list[str],
    decoder_path: str,
    seed: int,
) -> Mapping[str, object]:
    active_reading = ",".join(active_ids)
    return {
        "input_encoding": "declared active MaleCNS body IDs",
        "dynamics_rule": "dashi.kernel.flow.kernel_flow clean-room dynamics; not viral implementation",
        "initial_state_or_seed": f"active_body_ids={active_reading};control_seed={seed}",
        "decoder_mapping": f"external body-id to token JSON:{decoder_path}",
        "update_rule": "no learning/update implemented in this clean-room kernel runner",
        "reward_or_evaluator": "Python ast.parse only; no program execution",
        "prompt_or_scaffold": "none in clean-room runner",
        "parser_runtime": "Python ast.parse; execution disabled",
        "attempt_budget": 1,
        "selection_policy": "first run; no output selection or cherry-picking",
    }


def run_from_args(args: argparse.Namespace) -> dict[str, object]:
    weights_path = Path(args.weights_path)
    decoder_path = Path(args.decoder_json)
    token_by_identity = load_token_mapping(decoder_path)

    graph = load_malecns_graph(weights_path)
    carrier = _normalize_carrier(graph.carrier)
    identity_assignment = [str(identity) for identity in graph.idx_to_id]
    active_ids = [str(identity) for identity in args.active_body_id]

    packet = build_cleanroom_packet(
        carrier=carrier,
        identity_assignment=identity_assignment,
        active_ids=active_ids,
        token_by_identity=token_by_identity,
        params=KernelParams(hops=args.hops, deadzone=args.deadzone),
        steps=args.steps,
        assistance_overrides=_assistance_budget(
            active_ids=active_ids,
            decoder_path=str(decoder_path),
            seed=args.seed,
        ),
        rewire_swaps=args.rewire_swaps,
        seed=args.seed,
    )

    payload: dict[str, object] = {
        "input_artifact": {
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "dataset_version": str(args.dataset_version),
        },
        "decoder_artifact": {
            "path": str(decoder_path),
            "sha256": sha256_file(decoder_path),
            "semantics": "external interface mapping; not biological motor semantics",
        },
        "packet": packet_to_dict(packet),
        "execution_boundary": {
            "python_parsed": True,
            "python_executed": False,
            "viral_demo_reproduction_claimed": False,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_from_args(args)
    packet = payload["packet"]
    assert isinstance(packet, dict)
    print(f"wrote clean-room packet: {args.output}")
    print(f"baseline output: {packet['baseline']['emitted_program']!r}")  # type: ignore[index]
    print("Python execution disabled by design.")
    return 0
