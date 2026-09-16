"""Selective materialization of compact Gauthey et al. 2026 functional artifacts.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI 10.5281/zenodo.17618684.

The full Zenodo Data.zip is ~48.75 GB. This module uses HTTP Range + Zip64
metadata to fetch only selected members.

Important source-code receipt: the published ``fig3_preprocessing.py`` 2-photon
branch sets ``min_dim = 668``, slices each ROI matrix as
``dffs_corrected[:, :668]``, vertically stacks ROI rows across four trials, then
exports ``dffs_all[audio_correlated, :]``. Therefore the deposited
``audio_correlated`` array shape 940 x 668 means 940 selected ROI rows x 668
time samples. The prior interpretation of 668 functional units was incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np

from dashi.io.remote_zip import fetch_named_member


GAUTHEY_DATA_ZIP_URL = "https://zenodo.org/api/records/17618684/files/Data.zip/content"
GAUTHEY_DATA_DOI = "doi:10.5281/zenodo.17618684"
GAUTHEY_PAPER_DOI = "doi:10.1038/s41467-026-72437-1"
GAUTHEY_CODE_REPOSITORY = "github:murthylab/lightbead-analysis"

# Compact members already resolved in the deposit. The LB-specific ROI/label/
# mean-brain companions remain useful as their own lane, but they are NOT
# candidate identity maps for the pooled conventional-2p functional matrix.
GAUTHEY_COMPACT_MEMBERS: Mapping[str, str] = {
    "functional_audio_correlated_2p": "Data/Dffs/Audio correlated/dffs_audio_2p_corr_top05_all.pkl",
    "lb_responsive_roi_list": "Data/Mean brain/audio_roi_04032024_6f_a2_r5_pval.csv",
    "lb_segmentation_labels": "Data/Labels/04032024_6f_a2_r5_n2000_labels.h5",
    "lb_mean_brain": "Data/Mean brain/04032024_GCamp6f_a2_r5_w3_mean_G.nii",
}

# Source-code-declared conventional-2p inputs used to construct the deposited
# pooled top-0.5% matrix. Presence inside Data.zip is a separate repository
# resolution question and is not asserted by these names alone.
GAUTHEY_2P_TRIALS: tuple[tuple[str, str], ...] = (
    ("GCaMP6f_12132024_a2_r2.pkl", "12132024_6f_a2_r2_n1000_labels.h5"),
    ("GCaMP6f_12132024_a2_r3.pkl", "12132024_6f_a2_r3_n1000_labels.h5"),
    ("GCaMP6f_12132024_a2_r4.pkl", "12132024_6f_a2_r4_n1000_labels.h5"),
    ("GCaMP6f_12202024_a1_r2.pkl", "12202024_6f_a1_r2_n1000_labels.h5"),
)
GAUTHEY_2P_PLANES_PER_TRIAL = 47
GAUTHEY_2P_CLUSTERS_PER_PLANE = 1000
GAUTHEY_2P_TIME_SAMPLES = 668
GAUTHEY_2P_SELECTION_PERCENT = 0.5
GAUTHEY_2P_CANDIDATE_ROIS = (
    len(GAUTHEY_2P_TRIALS) * GAUTHEY_2P_PLANES_PER_TRIAL * GAUTHEY_2P_CLUSTERS_PER_PLANE
)
GAUTHEY_2P_EXPECTED_SELECTED_ROIS = int(
    GAUTHEY_2P_CANDIDATE_ROIS * (GAUTHEY_2P_SELECTION_PERCENT / 100.0)
)


@dataclass(frozen=True)
class CompactMemberReceipt:
    role: str
    archive_member: str
    path: str
    sha256: str
    compressed_size: int
    uncompressed_size: int
    crc32: int


@dataclass(frozen=True)
class GautheyFunctionalMatrix:
    """Functional traces normalized to the repository-wide time x unit convention."""

    traces: np.ndarray  # time x selected ROI
    unit_ids: tuple[str, ...]
    identity_kind: str
    source_member: str
    source_array_shape: tuple[int, int] | None = None
    source_axis_semantics: str = "time_x_unit"


def materialize_compact_bundle(
    output_dir: str | Path,
    *,
    url: str = GAUTHEY_DATA_ZIP_URL,
    members: Mapping[str, str] = GAUTHEY_COMPACT_MEMBERS,
) -> dict[str, CompactMemberReceipt]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, CompactMemberReceipt] = {}
    for role, member_name in members.items():
        member, raw = fetch_named_member(url, member_name)
        target = out / Path(member_name).name
        target.write_bytes(raw)
        receipts[role] = CompactMemberReceipt(
            role=role,
            archive_member=member_name,
            path=str(target),
            sha256=sha256(raw).hexdigest(),
            compressed_size=member.compressed_size,
            uncompressed_size=member.uncompressed_size,
            crc32=member.crc32,
        )
    return receipts


def write_bundle_receipt(
    receipts: Mapping[str, CompactMemberReceipt],
    output_path: str | Path,
) -> None:
    payload = {
        "scientific_source": {
            "author_or_consortium": "Wayan Gauthey; Albert Lin; Osama M. Ahmed; Andrew M. Leifer; Mala Murthy; Stephan Y. Thiberge",
            "title": "High-speed whole-brain imaging in Drosophila",
            "paper_identifier": GAUTHEY_PAPER_DOI,
            "dataset_identifier": GAUTHEY_DATA_DOI,
            "code_repository": GAUTHEY_CODE_REPOSITORY,
        },
        "members": {k: vars(v) for k, v in receipts.items()},
        "2p_source_semantics": {
            "candidate_roi_rows": GAUTHEY_2P_CANDIDATE_ROIS,
            "selected_roi_rows": GAUTHEY_2P_EXPECTED_SELECTED_ROIS,
            "time_samples": GAUTHEY_2P_TIME_SAMPLES,
            "selection_percent": GAUTHEY_2P_SELECTION_PERCENT,
            "deposited_matrix_semantics": "selected_roi_x_time",
        },
        "boundary": (
            "selected ROI row index is not trial/slice/supervoxel identity without "
            "recovering the source selection indices"
        ),
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_audio_correlated_pickle(path: str | Path) -> GautheyFunctionalMatrix:
    """Load the deposited pooled conventional-2p top-0.5% ROI matrix.

    The source export is ``dffs_all[audio_correlated, :]``. Hence deposited rows
    are selected ROI traces and deposited columns are time samples. Internally we
    transpose to the repository convention time x unit.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "audio_correlated" not in obj:
        raise ValueError("expected Gauthey pickle dict with 'audio_correlated'")
    arr = np.asarray(obj["audio_correlated"], dtype=float)
    if arr.ndim != 2:
        raise ValueError("audio_correlated must be a 2-D array")
    if arr.shape != (GAUTHEY_2P_EXPECTED_SELECTED_ROIS, GAUTHEY_2P_TIME_SAMPLES):
        raise ValueError(
            "unexpected Gauthey 2p audio_correlated shape: "
            f"{arr.shape}; expected "
            f"({GAUTHEY_2P_EXPECTED_SELECTED_ROIS}, {GAUTHEY_2P_TIME_SAMPLES})"
        )

    traces = arr.T  # repository convention: time x unit
    unit_ids = tuple(f"selected_roi_{i:04d}" for i in range(arr.shape[0]))
    return GautheyFunctionalMatrix(
        traces=traces,
        unit_ids=unit_ids,
        identity_kind="pooled_selected_roi_row_unmapped_to_source_roi",
        source_member=GAUTHEY_COMPACT_MEMBERS["functional_audio_correlated_2p"],
        source_array_shape=tuple(int(x) for x in arr.shape),
        source_axis_semantics="selected_roi_x_time",
    )
