from __future__ import annotations

import json
from pathlib import Path

from dashi.analysis.hemibrain_experiment_receipt import (
    build_recorded_sprint3_sprint4_receipt,
    compare_neutral_defect_persistence,
    FineGeometryReceipt,
    CoarseGeometryReceipt,
)


def test_recorded_degree_binned_quotient_erases_neutral_defect_consumer() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt = build_recorded_sprint3_sprint4_receipt(root, quotient="degree_binned_k100")

    assert receipt.fine.neutral_nodes > 0
    assert receipt.fine.defect_associated_nodes > 0
    assert receipt.coarse.all_positive
    assert receipt.coarse.terminal_defect == 0
    assert receipt.quotient_erases_declared_consumer is True
    assert receipt.coarse_persistence_observed is False
    assert receipt.empirical_scope_only is True
    assert receipt.measurement_closes_prediction_claimed is False
    assert receipt.biological_interpretation_claimed is False


def test_recorded_random_block_quotient_erases_neutral_defect_consumer() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt = build_recorded_sprint3_sprint4_receipt(
        root,
        quotient="random_blocks_k100_seed0",
    )
    assert receipt.quotient_erases_declared_consumer is True
    assert receipt.coarse_persistence_observed is False


def test_persistence_can_succeed_when_coarse_consumer_survives() -> None:
    fine = FineGeometryReceipt(
        source="fine.json",
        nodes=4,
        positive_nodes=2,
        neutral_nodes=2,
        negative_nodes=0,
        component_count=1,
        defect_associated_nodes=1,
    )
    coarse = CoarseGeometryReceipt(
        quotient_name="locality_preserving_candidate",
        state_source="coarse.csv",
        defect_curve_source=None,
        nodes=2,
        positive_nodes=1,
        neutral_nodes=1,
        negative_nodes=0,
        terminal_defect=0,
    )
    receipt = compare_neutral_defect_persistence(fine, coarse)
    assert receipt.coarse_persistence_observed is True
    assert receipt.quotient_erases_declared_consumer is False
