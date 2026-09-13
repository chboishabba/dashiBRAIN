import numpy as np
import scipy.sparse as sp

from dashi.analysis.sparse_information_energy_objective import (
    information_energy_objective,
    sparse_energy_components,
)


def test_sparse_energy_matches_released_metabolic_and_synaptic_definitions():
    rates = np.array([[1.0, 2.0], [3.0, 4.0]])
    # W[post, pre]
    W = sp.csr_matrix(np.array([[0.0, 0.5], [0.25, 0.0]]))

    energy = sparse_energy_components(rates, W)

    expected_met = np.sum(rates)
    out_strength_by_pre = np.asarray(np.abs(W).sum(axis=0)).reshape(-1)
    expected_syn = np.sum(rates @ out_strength_by_pre)
    assert energy["E_met"] == expected_met
    assert energy["E_syn"] == expected_syn
    assert energy["E_wire"] == 0.0
    assert energy["E_total"] == expected_met + expected_syn


def test_sparse_active_wiring_cost_is_edge_local_and_pre_activity_weighted():
    rates = np.array([[1.0, 2.0], [3.0, 4.0]])
    W = sp.csr_matrix(np.array([[0.0, 2.0], [3.0, 0.0]]))
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])

    energy = sparse_energy_components(
        rates,
        W,
        distance_coordinates=coords,
        wiring_penalty_alpha=0.01,
    )

    total_pre_activity = np.sum(rates, axis=0)
    expected_wire = 0.01 * (
        2.0 * 5.0 * total_pre_activity[1]
        + 3.0 * 5.0 * total_pre_activity[0]
    )
    assert energy["E_wire"] == expected_wire


def test_information_energy_objective_matches_declared_J():
    assert information_energy_objective(
        information_lower_bound=3.5,
        total_energy=10.0,
        energy_lambda=0.02,
    ) == 3.3


def test_zero_lambda_separates_information_from_energy_penalty():
    assert information_energy_objective(
        information_lower_bound=1.25,
        total_energy=999.0,
        energy_lambda=0.0,
    ) == 1.25
