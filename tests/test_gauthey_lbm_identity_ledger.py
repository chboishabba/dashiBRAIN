import csv
from pathlib import Path

import pytest

from dashi.io.gauthey_lbm_identity_ledger import (
    LBMIdentityLedger,
    LBMSourceIdentity,
    merge_identity_ledgers,
    read_identity_csv,
)


def _row(selected, source, trial="04032024_6f_a2_r5"):
    return LBMSourceIdentity(selected, source, trial, 0, source % 2000, 0.5)


def test_disjoint_trial_receipts_merge_without_rewriting_paid_rows():
    left = LBMIdentityLedger((_row(0, 10),))
    right = LBMIdentityLedger((_row(1, 54010, "04032024_6f_a2_r1"),))
    merged = merge_identity_ledgers((left, right))
    assert merged.resolved_selected_rows == (0, 1)
    assert merged.counts_by_trial["04032024_6f_a2_r5"] == 1
    assert merged.counts_by_trial["04032024_6f_a2_r1"] == 1


def test_duplicate_identical_receipt_is_idempotent():
    row = _row(0, 10)
    merged = merge_identity_ledgers((LBMIdentityLedger((row,)), LBMIdentityLedger((row,))))
    assert merged.rows == (row,)


def test_conflicting_selected_row_identity_is_rejected():
    with pytest.raises(ValueError, match="conflicting exact identities"):
        merge_identity_ledgers((
            LBMIdentityLedger((_row(0, 10),)),
            LBMIdentityLedger((_row(0, 11),)),
        ))


def test_one_source_row_cannot_pay_two_selected_rows():
    with pytest.raises(ValueError, match="one pooled source row"):
        merge_identity_ledgers((LBMIdentityLedger((_row(0, 10), _row(1, 10))),))


def test_read_partial_csv_preserves_unresolved_frontier(tmp_path: Path):
    path = tmp_path / "partial.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_row", "pooled_source_row", "trial_id", "plane_index", "cluster_index", "correlation"])
        writer.writerow([0, 10, "04032024_6f_a2_r5", 0, 10, 0.5])
    ledger = read_identity_csv(path)
    assert ledger.complete is False
    assert ledger.resolved_selected_rows == (0,)
    assert 1 in ledger.unresolved_selected_rows
