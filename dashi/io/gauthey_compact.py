"""Selective materialization of compact Gauthey et al. 2026 functional artifacts.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI 10.5281/zenodo.17618684.

The full Zenodo Data.zip is ~48.75 GB.  This module uses HTTP Range + Zip64
metadata to fetch only selected members.  Archive-array column indices are not
promoted to anatomical ROI or neuron identities without a separate registration
receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np

from dashi.io.remote_zip import RemoteZipMember, fetch_named_member


GAUTHEY_DATA_ZIP_URL = "https://zenodo.org/api/records/17618684/files/Data.zip/content"
GAUTHEY_DATA_DOI = "doi:10.5281/zenodo.17618684"
GAUTHEY_PAPER_DOI = "doi:10.1038/s41467-026-72437-1"

GAUTHEY_COMPACT_MEMBERS: Mapping[str, str] = {
    "functional_audio_correlated": "Data/Dffs/Audio correlated/dffs_audio_2p_corr_top05_all.pkl",
    "responsive_roi_list": "Data/Mean brain/audio_roi_04032024_6f_a2_r5_pval.csv",
    "segmentation_labels": "Data/Labels/04032024_6f_a2_r5_n2000_labels.h5",
    "mean_brain": "Data/Mean brain/04032024_GCamp6f_a2_r5_w3_mean_G.nii",
}


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
    traces: np.ndarray
    unit_ids: tuple[str, ...]
    identity_kind: str
    source_member: str


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
        },
        "members": {k: vars(v) for k, v in receipts.items()},
        "boundary": "archive array indices are not anatomical ROI/neuron identities without registration",
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_audio_correlated_pickle(path: str | Path) -> GautheyFunctionalMatrix:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "audio_correlated" not in obj:
        raise ValueError("expected Gauthey pickle dict with 'audio_correlated'")
    arr = np.asarray(obj["audio_correlated"], dtype=float)
    if arr.ndim != 2:
        raise ValueError("audio_correlated must be a 2-D array")
    # Runner inspection established the deposited object as 940 x 668.  We do
    # not assume which axis is time solely from that receipt; prefer the larger
    # first dimension as time for this exact known artifact and retain only
    # archive-local unit identities until registration is supplied.
    if arr.shape == (940, 668):
        traces = arr
    elif arr.shape == (668, 940):
        traces = arr.T
    else:
        raise ValueError(f"unexpected Gauthey audio_correlated shape: {arr.shape}")
    unit_ids = tuple(f"archive_unit_{i:04d}" for i in range(traces.shape[1]))
    return GautheyFunctionalMatrix(
        traces=traces,
        unit_ids=unit_ids,
        identity_kind="archive_array_index_unregistered",
        source_member=GAUTHEY_COMPACT_MEMBERS["functional_audio_correlated"],
    )
