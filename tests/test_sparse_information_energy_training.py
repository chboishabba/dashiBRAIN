import numpy as np
import scipy.sparse as sp

from dashi.analysis.rate_eligibility_dynamics import RateDynamicsParams
from dashi.analysis.sparse_information_energy_training import (
    run_sparse_information_energy_training_epoch,
)


def fixture_weights() -> sp.csr_matrix:
    return sp.csr_matrix(
        np.array(
            [
                [0.0, 0.2, 0.0],
                [0.1, 0.0, 0.3],
                [0.0, 0.4, 0.0],
            ],
            dtype=float,
        )
    )


def fixture_stimulus() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def test_objective_is_derived_from_same_epoch_energy_and_supplied_information():
    W = fixture_weights()
    result = run_sparse_information_energy_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        information_lower_bound=2.5,
        energy_lambda=0.01,
        previous_trace=np.zeros_like(W.data),
        running_baseline=0.0,
        learning_rate=0.1,
    )

    expected = result.information_lower_bound - 0.01 * result.energy["E_total"]
    assert result.objective == expected
    assert result.receipt["objective"] == expected
    assert result.receipt["information_lower_bound"] == 2.5
    assert result.receipt["energy_lambda"] == 0.01


def test_energy_receipt_is_from_preupdate_rate_trajectory():
    W = fixture_weights()
    result = run_sparse_information_energy_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        information_lower_bound=1.0,
        energy_lambda=0.02,
        previous_trace=np.zeros_like(W.data),
        running_baseline=0.0,
        learning_rate=0.1,
    )

    assert result.receipt["energy_measured_before_weight_update"] is True
    assert set(result.energy) == {"E_met", "E_syn", "E_wire", "E_total"}


def test_symbolic_information_score_is_explicitly_external_and_held_out_bound():
    W = fixture_weights()
    result = run_sparse_information_energy_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        information_lower_bound=0.75,
        energy_lambda=0.0,
        previous_trace=np.zeros_like(W.data),
        running_baseline=0.0,
        learning_rate=0.0,
        information_source="symbolic-heldout-evaluator:v1",
        information_is_held_out=True,
    )

    assert result.receipt["information_source"] == "symbolic-heldout-evaluator:v1"
    assert result.receipt["information_is_held_out"] is True
    assert result.receipt["optic_flow_decoder_reused"] is False


def test_missing_information_source_fails_closed():
    W = fixture_weights()
    try:
        run_sparse_information_energy_training_epoch(
            W,
            fixture_stimulus(),
            RateDynamicsParams(),
            information_lower_bound=0.75,
            energy_lambda=0.0,
            previous_trace=np.zeros_like(W.data),
            running_baseline=0.0,
            learning_rate=0.0,
            information_source="",
            information_is_held_out=True,
        )
    except ValueError as exc:
        assert "information_source" in str(exc)
    else:
        raise AssertionError("empty information_source must fail closed")


def test_objective_wrapper_preserves_no_learning_control():
    W = fixture_weights()
    result = run_sparse_information_energy_training_epoch(
        W,
        fixture_stimulus(),
        RateDynamicsParams(),
        information_lower_bound=1.0,
        energy_lambda=0.01,
        previous_trace=np.zeros_like(W.data),
        running_baseline=0.0,
        learning_rate=0.0,
    )

    assert np.allclose(result.training.weights.data, W.data)
    assert result.training.receipt["learning_update_applied"] is False
