from __future__ import annotations

import json
from pathlib import Path

from scripts.recover_gauthey_lbm_remaining_identities import _resolve_resume_seed


def test_existing_accumulation_receipt_preserves_searched_zero_trial(tmp_path: Path):
    out = tmp_path / "reconstruction_all_available"
    out.mkdir()
    accumulated = out / "gauthey_lbm_selected_identities_accumulated.csv"
    accumulated.write_text("selected_row,pooled_source_row,trial_id,plane_index,cluster_index,correlation\n", encoding="utf-8")
    summary = out / "gauthey_lbm_remaining_identity_recovery.json"
    summary.write_text(
        json.dumps(
            {
                "identity_output": str(accumulated),
                "searched_trials": [
                    "04032024_6f_a2_r5",
                    "04192024_6f_a1_r9",
                    "04032024_6f_a2_r1",
                ],
            }
        ),
        encoding="utf-8",
    )

    seed_path, searched = _resolve_resume_seed(
        out=out,
        cli_seed_identity="historical_seed.csv",
        cli_seed_searched=("04032024_6f_a2_r5",),
    )

    assert seed_path == accumulated
    assert searched == (
        "04032024_6f_a2_r5",
        "04192024_6f_a1_r9",
        "04032024_6f_a2_r1",
    )


def test_missing_accumulation_receipt_falls_back_to_cli_seed(tmp_path: Path):
    seed_path, searched = _resolve_resume_seed(
        out=tmp_path,
        cli_seed_identity="historical_seed.csv",
        cli_seed_searched=("04032024_6f_a2_r5",),
    )

    assert seed_path == Path("historical_seed.csv")
    assert searched == ("04032024_6f_a2_r5",)
