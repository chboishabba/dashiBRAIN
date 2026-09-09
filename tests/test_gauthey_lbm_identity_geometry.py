from __future__ import annotations

import csv

import pytest

from dashi.analysis.gauthey_lbm_native_field import load_recovered_identity_csv


def _write_identity(path, *, pooled_source_row: int, trial_id: str, plane: int, cluster: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selected_row",
                "pooled_source_row",
                "trial_id",
                "plane_index",
                "cluster_index",
                "correlation",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selected_row": 4,
                "pooled_source_row": pooled_source_row,
                "trial_id": trial_id,
                "plane_index": plane,
                "cluster_index": cluster,
                "correlation": 0.06972,
            }
        )


def test_recovered_identity_accepts_geometry_decoded_from_pooled_row(tmp_path):
    path = tmp_path / "identities.csv"
    # Trial a2_r5 is the second 54,000-row fibre.  68,633 therefore decodes to
    # within-trial row 14,633 = plane 7 * 2,000 + cluster 633.
    _write_identity(
        path,
        pooled_source_row=68633,
        trial_id="04032024_6f_a2_r5",
        plane=7,
        cluster=633,
    )
    rows = load_recovered_identity_csv(path)
    assert len(rows) == 1
    assert (rows[0].trial_id, rows[0].plane_index, rows[0].cluster_index) == (
        "04032024_6f_a2_r5",
        7,
        633,
    )


def test_recovered_identity_rejects_geometry_inconsistent_with_pooled_row(tmp_path):
    path = tmp_path / "identities.csv"
    _write_identity(
        path,
        pooled_source_row=68633,
        trial_id="04032024_6f_a2_r5",
        plane=7,
        cluster=634,
    )
    with pytest.raises(ValueError, match="disagrees with pooled_source_row"):
        load_recovered_identity_csv(path)
