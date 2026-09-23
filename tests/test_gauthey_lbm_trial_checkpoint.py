from pathlib import Path
import importlib.util

from dashi.analysis.gauthey_lbm_experiment import LBM_TRIALS, pooled_lbm_row_to_trial_plane_cluster
from dashi.io.gauthey_lbm_identity_receipts import (
    LBMExactIdentity,
    checkpoint_trial_identity_receipt,
    load_identity_csv,
)


def _recovery_runner_module():
    path = Path("scripts/recover_gauthey_lbm_remaining_identities.py")
    spec = importlib.util.spec_from_file_location("remaining_identity_recovery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_searched_zero_exit_is_accepted_only_with_durable_checkpoint(tmp_path: Path):
    identity = tmp_path / "identities.csv"
    summary = tmp_path / "receipt.json"
    identity.write_text("selected_row,pooled_source_row,trial_id,plane_index,cluster_index,correlation\n")
    summary.write_text("{}")

    runner = _recovery_runner_module()
    runner._require_durable_trial_checkpoint(
        returncode=1,
        trial_id=LBM_TRIALS[0],
        identity_path=identity,
        summary_path=summary,
    )

    import pytest

    with pytest.raises(SystemExit, match="without durable standalone receipt"):
        runner._require_durable_trial_checkpoint(
            returncode=1,
            trial_id=LBM_TRIALS[0],
            identity_path=tmp_path / "missing.csv",
            summary_path=summary,
        )
