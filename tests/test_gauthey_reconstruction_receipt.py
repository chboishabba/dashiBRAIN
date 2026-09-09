from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashi.io.gauthey_reconstruction_receipt import build_gauthey_reconstruction_receipt
from dashi.io.remote_zip import RemoteZipMember


def _member() -> RemoteZipMember:
    return RemoteZipMember(
        name="Data/Dffs/Aligned/GCaMP6f_04032024_a2_r5.zip",
        compressed_size=123,
        uncompressed_size=456,
        compression_method=8,
        local_header_offset=789,
        crc32=0xA1B2C3D4,
    )


def _summary(tmp_path: Path, *, processed=True, recovered=3, unresolved=2, complete=False) -> Path:
    identities = tmp_path / "gauthey_lbm_selected_identities_partial.csv"
    unresolved_rows = tmp_path / "gauthey_lbm_selected_unresolved_rows.csv"
    identities.write_text("selected_row\n1\n", encoding="utf-8")
    unresolved_rows.write_text("selected_row\n2\n", encoding="utf-8")
    summary = {
        "processed_trials": ["04032024_6f_a2_r5"] if processed else [],
        "exact_source_identities_recovered": recovered,
        "unresolved_selected_rows": unresolved,
        "complete_identity_recovery": complete,
        "identity_output": str(identities),
        "unresolved_output": str(unresolved_rows),
        "identity_semantics": "exact trace equality to deposited selected row; not neuron identity",
    }
    path = tmp_path / "gauthey_lbm_reconstruction.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_build_receipt_binds_remote_member_and_hashes_outputs(tmp_path):
    deposited = tmp_path / "selected.pkl"
    deposited.write_bytes(b"published-selection")
    summary = _summary(tmp_path)

    receipt = build_gauthey_reconstruction_receipt(
        trial_id="04032024_6f_a2_r5",
        remote_url="https://example.test/Data.zip",
        remote_member=_member(),
        deposited_selected_path=deposited,
        reconstruction_summary_path=summary,
    )

    assert receipt.processed_trial_confirmed
    assert receipt.fixture_admissible
    assert receipt.remote_member["crc32"] == "a1b2c3d4"
    assert receipt.deposited_selected.size_bytes == len(b"published-selection")
    assert len(receipt.deposited_selected.sha256) == 64
    assert receipt.exact_source_identities_recovered == 3
    assert receipt.unresolved_selected_rows == 2
    assert not receipt.complete_identity_recovery


def test_receipt_rejects_unprocessed_trial(tmp_path):
    deposited = tmp_path / "selected.pkl"
    deposited.write_bytes(b"x")
    summary = _summary(tmp_path, processed=False)

    with pytest.raises(ValueError, match="not confirmed"):
        build_gauthey_reconstruction_receipt(
            trial_id="04032024_6f_a2_r5",
            remote_url="https://example.test/Data.zip",
            remote_member=_member(),
            deposited_selected_path=deposited,
            reconstruction_summary_path=summary,
        )


def test_receipt_rejects_inconsistent_completion_flag(tmp_path):
    deposited = tmp_path / "selected.pkl"
    deposited.write_bytes(b"x")
    summary = _summary(tmp_path, recovered=5, unresolved=0, complete=False)

    with pytest.raises(ValueError, match="disagrees"):
        build_gauthey_reconstruction_receipt(
            trial_id="04032024_6f_a2_r5",
            remote_url="https://example.test/Data.zip",
            remote_member=_member(),
            deposited_selected_path=deposited,
            reconstruction_summary_path=summary,
        )


def test_zero_recovered_rows_is_not_fixture_admissible(tmp_path):
    deposited = tmp_path / "selected.pkl"
    deposited.write_bytes(b"x")
    summary = _summary(tmp_path, recovered=0, unresolved=5, complete=False)

    receipt = build_gauthey_reconstruction_receipt(
        trial_id="04032024_6f_a2_r5",
        remote_url="https://example.test/Data.zip",
        remote_member=_member(),
        deposited_selected_path=deposited,
        reconstruction_summary_path=summary,
    )
    assert not receipt.fixture_admissible
