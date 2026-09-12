from pathlib import Path

from dashi.analysis.gauthey_lbm_experiment import LBM_TRIALS, pooled_lbm_row_to_trial_plane_cluster
from dashi.io.gauthey_lbm_identity_receipts import (
    LBMExactIdentity,
    checkpoint_trial_identity_receipt,
    load_identity_csv,
)


def _identity(selected_row: int, pooled_source_row: int) -> LBMExactIdentity:
    trial_id, plane_index, cluster_index = pooled_lbm_row_to_trial_plane_cluster(pooled_source_row)
    return LBMExactIdentity(
        selected_row=selected_row,
        pooled_source_row=pooled_source_row,
        trial_id=trial_id,
        plane_index=plane_index,
        cluster_index=cluster_index,
        correlation=0.75,
    )


def test_checkpoint_writes_standalone_trial_receipt_immediately(tmp_path: Path):
    trial = LBM_TRIALS[0]
    identities = (_identity(0, 0), _identity(1, 1))

    receipt = checkpoint_trial_identity_receipt(
        identities,
        trial_id=trial,
        output_dir=tmp_path,
        remote_archive_accessed=False,
    )

    assert receipt.identity_path.exists()
    assert receipt.summary_path.exists()
    assert load_identity_csv(receipt.identity_path) == identities

    import json

    payload = json.loads(receipt.summary_path.read_text(encoding="utf-8"))
    assert payload["trial_id"] == trial
    assert payload["exact_source_identities_recovered"] == 2
    assert payload["remote_archive_accessed"] is False
    assert payload["standalone_trial_receipt"] is True


def test_checkpoint_rejects_cross_trial_identity_mix(tmp_path: Path):
    first = _identity(0, 0)
    per_trial = 27 * 2000
    second = _identity(1, per_trial)

    import pytest

    with pytest.raises(ValueError, match="different trial"):
        checkpoint_trial_identity_receipt(
            (first, second),
            trial_id=first.trial_id,
            output_dir=tmp_path,
            remote_archive_accessed=True,
        )
