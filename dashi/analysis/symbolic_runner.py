"""Run the existing DASHI kernel flow through an explicit symbolic decoder.

The decoder is intentionally external to the neural dynamics. Changing token
labels must not change the underlying state trajectory; this keeps interface
semantics distinct from connectome/kernel dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dashi.kernel.flow import kernel_flow
from dashi.types import GraphCarrier, KernelParams


@dataclass(frozen=True)
class SymbolDecoder:
    token_by_neuron: Mapping[int, str]

    def __post_init__(self) -> None:
        for neuron_index, token in self.token_by_neuron.items():
            if neuron_index < 0:
                raise ValueError("decoder neuron indices must be non-negative")
            if not token:
                raise ValueError("decoder token must be non-empty")


@dataclass(frozen=True)
class SymbolicTraceResult:
    final_state: np.ndarray
    emitted_program: str
    steps_observed: int
    status: dict[str, int | str]


def decode_trace(states: Sequence[np.ndarray], decoder: SymbolDecoder) -> str:
    """Decode positive activity from a state trace into deterministic tokens.

    Within each state, mapped neuron indices are visited in sorted index order.
    Only strictly positive activity emits a token.  The function does not infer
    that the resulting text has biological or semantic meaning.
    """

    emitted: list[str] = []
    ordered = sorted(decoder.token_by_neuron.items())
    for state in states:
        if state.ndim != 2:
            raise ValueError("symbolic decoder expects state shape (N, C)")
        if state.shape[1] != 1:
            raise ValueError("symbolic decoder currently requires one channel")
        for neuron_index, token in ordered:
            if neuron_index >= state.shape[0]:
                raise ValueError("decoder neuron index exceeds state carrier")
            if state[neuron_index, 0] > 0:
                emitted.append(token)
    return "".join(emitted)


def run_kernel_symbolic_trace(
    carrier: GraphCarrier,
    s0: np.ndarray,
    params: KernelParams,
    decoder: SymbolDecoder,
    *,
    steps: int = 20,
    cycle_check: bool = True,
) -> SymbolicTraceResult:
    """Execute ``kernel_flow`` and decode its observed trajectory.

    This is a clean-room experimental harness, not a reconstruction of the
    viral Python/FizzBuzz implementation.  The dynamics are exactly the repo's
    existing kernel flow; the neural-to-token mapping is supplied separately.
    """

    final_state, history, status = kernel_flow(
        carrier,
        s0,
        params,
        steps=steps,
        cycle_check=cycle_check,
    )
    states = [entry["state"] for entry in history]
    emitted_program = decode_trace(states, decoder)
    return SymbolicTraceResult(
        final_state=final_state,
        emitted_program=emitted_program,
        steps_observed=len(history),
        status=status,
    )
