from __future__ import annotations

import pytest

from dashi.analysis.gauthey_lbm_experiment import LBM_EXPECTED_SELECTED, LBM_TRIALS
from dashi.io.gauthey_lbm_identity_receipts import (
    LBMExactIdentity,
    merge_exact_identities,
)


def _id(selected_row: int, pooled_source_row: int, *, correlation: float = 0.5) -> LBMExactIdentity:
    # Trial 0 rows decode directly through the canonical LBM geometry.
    return LBMExactIdentity(
        selected_row=selected_row,
        pooled_source_row=pooled_source_row,
        trial_id=LBM_TRIALS[0],
        plane_index=pooled_source_row // 2000,
        cluster_index=pooled_source_row % 2000,
        correlation=correlation,
    )


def test_unsearched_trials_are_not_interpreted_as_zero_contribution():
    acc = merge_exact_identities(
        [(_id(0, 0), _id(1, 1))],
        searched_trials=(LBM_TRIALS[0],),
    )
    assert acc.coverage.resolved_count_by_trial[LBM_TRIALS[0]] == 2
    assert acc.coverage.resolved_count_by_trial[LBM_TRIALS[1]] == 0
    assert LBM_TRIALS[1] not in acc.coverage.searched_trials
    assert len(acc.coverage.unresolved_selected_rows) == LBM_EXPECTED_SELECTED - 2


def test_exact_duplicate_receipt_is_idempotent():
    identity = _id(0, 0)
    acc = merge_exact_identities(
        [(identity,), (identity,)],
        searched_trials=(LBM_TRIALS[0],),
    )
    assert acc.identities == (identity,)


def test_conflicting_identity_for_same_selected_row_is_rejected():
    a = _id(0, 0, correlation=0.5)
    b = _id(0, 1, correlation=0.6)
    with pytest.raises(ValueError, match="conflicting exact identities"):
        merge_exact_identities([(a,), (b,)], searched_trials=(LBM_TRIALS[0],))


def test_trial_cannot_be_searched_and_missing():
    with pytest.raises(ValueError, match="both searched and source-missing"):
        merge_exact_identities(
            [()],
            searched_trials=(LBM_TRIALS[0],),
            missing_source_trials=(LBM_TRIALS[0],),
        )
