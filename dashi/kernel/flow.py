from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from dashi.kernel.operator import kernel_step
from dashi.types import GraphCarrier, KernelParams, T


def kernel_flow(
    carrier: GraphCarrier,
    s0: np.ndarray,
    params: KernelParams,
    *,
    steps: int = 20,
    cycle_check: bool = True,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int | str]]:
    """
    Run kernel iterations until convergence, detected cycle, or max steps.

    Returns final state, history of steps, and a status dict.
    """
    s = s0.astype(T, copy=True)
    history: list[dict[str, object]] = []

    seen: Dict[bytes, int] = {}
    if cycle_check:
        seen[s.tobytes()] = 0

    for t in range(1, steps + 1):
        s_next = kernel_step(carrier, s, params)
        defect = int(np.sum(s_next != s))
        history.append({"step": t, "defect": defect, "state": s_next.copy()})

        if defect == 0:
            return s_next, history, {"status": "converged", "step": t}

        if cycle_check:
            key = s_next.tobytes()
            if key in seen:
                return s_next, history, {"status": "cycle", "start": seen[key], "step": t}
            seen[key] = t

        s = s_next

    return s, history, {"status": "max_steps", "step": steps}
