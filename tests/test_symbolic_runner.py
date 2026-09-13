import numpy as np
import scipy.sparse as sp

from dashi.analysis.symbolic_runner import SymbolDecoder, decode_trace, run_kernel_symbolic_trace
from dashi.types import GraphCarrier, KernelParams


def test_decode_trace_is_deterministic_and_index_ordered():
    decoder = SymbolDecoder({0: "F", 2: "z", 1: "i"})
    trace = [
        np.array([[1], [0], [1]], dtype=np.int8),
        np.array([[0], [1], [0]], dtype=np.int8),
    ]
    assert decode_trace(trace, decoder) == "Fzi"


def test_decode_trace_ignores_nonpositive_activity():
    decoder = SymbolDecoder({0: "A", 1: "B"})
    trace = [np.array([[-1], [0]], dtype=np.int8)]
    assert decode_trace(trace, decoder) == ""


def test_decoder_rejects_empty_tokens():
    try:
        SymbolDecoder({0: ""})
    except ValueError as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("empty decoder token must be rejected")


def test_kernel_symbolic_trace_runs_existing_kernel_flow():
    adjacency = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    carrier = GraphCarrier(adjacency=adjacency, channels=1)
    s0 = np.array([[1], [0]], dtype=np.int8)
    decoder = SymbolDecoder({0: "F", 1: "B"})

    result = run_kernel_symbolic_trace(
        carrier,
        s0,
        KernelParams(hops=1, deadzone=0.0),
        decoder,
        steps=3,
    )

    assert result.status["status"] == "converged"
    assert result.emitted_program == "F"
    assert result.steps_observed == 1


def test_runner_keeps_decoder_external_to_dynamics():
    adjacency = sp.csr_matrix(np.eye(2, dtype=np.float32))
    carrier = GraphCarrier(adjacency=adjacency, channels=1)
    s0 = np.array([[1], [0]], dtype=np.int8)
    params = KernelParams(hops=1)

    first = run_kernel_symbolic_trace(carrier, s0, params, SymbolDecoder({0: "X"}), steps=1)
    second = run_kernel_symbolic_trace(carrier, s0, params, SymbolDecoder({0: "Y"}), steps=1)

    assert np.array_equal(first.final_state, second.final_state)
    assert first.emitted_program != second.emitted_program
