from __future__ import annotations
from typing import Dict
import numpy as np

from dashi.kernel.operator import kernel_step
from dashi.types import GraphCarrier, KernelParams, T


def _run_receipts(
    carrier: GraphCarrier,
    state: np.ndarray,
    params: KernelParams,
    history: list[dict[str, object]],
) -> dict[str, object]:
    """Finite-run receipts aligned with the Agda BIDI contract.

    These fields report only what was observed for this concrete run.  They do
    not promote idempotence, contraction, or defect monotonicity to global
    properties of the kernel family.
    """
    defects = [int(item["defect"]) for item in history]
    observed_nonincreasing = all(
        later <= earlier for earlier, later in zip(defects, defects[1:])
    )
    next_state = kernel_step(carrier, state, params)
    observed_idempotent_at_final = bool(np.array_equal(next_state, state))
    return {
        "receipt_scope": "finite_run_only",
        "observed_defect_nonincreasing": observed_nonincreasing,
        "observed_idempotent_at_final": observed_idempotent_at_final,
        "global_idempotence_claimed": False,
        "global_defect_monotonicity_claimed": False,
        "global_contraction_claimed": False,
    }


def kernel_flow(
    carrier: GraphCarrier,
    s0: np.ndarray,
    params: KernelParams,
    *,
    steps: int = 20,
    cycle_check: bool = True,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    """
    Run kernel iterations until fixed-point convergence, detected cycle, or max steps.

    Returns final state, history of steps, and a status/receipt dict.  A zero
    defect is a literal fixed-point receipt for the observed state.  Stronger
    properties such as idempotence or monotone defect are reported only as
    finite-run observations unless separately proved.
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
            status: dict[str, object] = {
                "status": "converged",
                "step": t,
                "fixed_point_receipt": True,
            }
            status.update(_run_receipts(carrier, s_next, params, history))
            return s_next, history, status

        if cycle_check:
            key = s_next.tobytes()
            if key in seen:
                status = {
                    "status": "cycle",
                    "start": seen[key],
                    "step": t,
                    "fixed_point_receipt": False,
                }
                status.update(_run_receipts(carrier, s_next, params, history))
                return s_next, history, status
            seen[key] = t

        s = s_next

    status = {
        "status": "max_steps",
        "step": steps,
        "fixed_point_receipt": False,
    }
    status.update(_run_receipts(carrier, s, params, history))
    return s, history, status
