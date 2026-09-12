"""Recover source identities for Gauthey pooled conventional-2p selected ROIs.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; code repository
https://github.com/murthylab/lightbead-analysis.

Source-code receipts:
- ``Fig3_aligment.py`` builds each conventional-2p trial matrix by iterating
  slices first and supervoxel traces second, producing 47 * 1000 = 47,000 ROI
  rows per trial.
- ``fig3_preprocessing.py`` takes ``dffs_corrected[:, :668]`` from four trials,
  vertically stacks those ROI rows, selects the top 0.5% auditory-correlated
  rows, and exports only ``dffs_all[audio_correlated, :]``.

The exported pickle therefore omits the selected source indices, but each output
row is literally copied from the pooled source matrix. If the four source
pickles are available, the indices can be recovered by exact row-byte matching
(or an explicitly requested tolerance fallback), then decoded deterministically
as (trial, plane, cluster).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import pickle
from typing import Mapping, Sequence

import numpy as np

from dashi.io.gauthey_compact import (
    GAUTHEY_2P_CLUSTERS_PER_PLANE,
    GAUTHEY_2P_EXPECTED_SELECTED_ROIS,
    GAUTHEY_2P_PLANES_PER_TRIAL,
    GAUTHEY_2P_TIME_SAMPLES,
    GAUTHEY_2P_TRIALS,
)


@dataclass(frozen=True)
class TwoPTrialSpec:
    ordinal: int
    data_filename: str
    label_filename: str

    @property
    def trial_id(self) -> str:
        return Path(self.data_filename).stem.removeprefix("GCaMP6f_")

    @property
    def candidate_roi_count(self) -> int:
        return GAUTHEY_2P_PLANES_PER_TRIAL * GAUTHEY_2P_CLUSTERS_PER_PLANE


TWO_P_TRIAL_SPECS: tuple[TwoPTrialSpec, ...] = tuple(
    TwoPTrialSpec(i, data_name, label_name)
    for i, (data_name, label_name) in enumerate(GAUTHEY_2P_TRIALS)
)


@dataclass(frozen=True)
class RecoveredSelectedROI:
    selected_row: int
    functional_id: str
    global_source_row: int
    trial_ordinal: int
    trial_id: str
    within_trial_row: int
    plane_index: int
    cluster_index: int
    source_data_filename: str
    source_label_filename: str
    evidence_kind: str = "exact_trace_row_recovery_from_published_preprocessing_inputs"


@dataclass(frozen=True)
class SourceRecoveryReceipt:
    selected_roi_count: int
    candidate_roi_count: int
    time_sample_count: int
    exact_match_count: int
    ambiguous_match_count: int
    unmatched_count: int
    complete: bool
    rows: tuple[RecoveredSelectedROI, ...]


def decode_global_source_row(global_row: int) -> RecoveredSelectedROI:
    per_trial = GAUTHEY_2P_PLANES_PER_TRIAL * GAUTHEY_2P_CLUSTERS_PER_PLANE
    total = per_trial * len(TWO_P_TRIAL_SPECS)
    if global_row < 0 or global_row >= total:
        raise ValueError(f"global source row outside 0..{total - 1}: {global_row}")
    trial_ordinal, within = divmod(int(global_row), per_trial)
    plane, cluster = divmod(within, GAUTHEY_2P_CLUSTERS_PER_PLANE)
    spec = TWO_P_TRIAL_SPECS[trial_ordinal]
    return RecoveredSelectedROI(
        selected_row=-1,
        functional_id="",
        global_source_row=int(global_row),
        trial_ordinal=trial_ordinal,
        trial_id=spec.trial_id,
        within_trial_row=within,
        plane_index=plane,
        cluster_index=cluster,
        source_data_filename=spec.data_filename,
        source_label_filename=spec.label_filename,
    )


def _load_trial_matrix(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "dffs_corrected" not in obj:
        raise ValueError(f"{path}: expected pickle dict containing dffs_corrected")
    arr = np.asarray(obj["dffs_corrected"], dtype=float)
    expected_rows = GAUTHEY_2P_PLANES_PER_TRIAL * GAUTHEY_2P_CLUSTERS_PER_PLANE
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} ROI rows, got shape {arr.shape}"
        )
    if arr.shape[1] < GAUTHEY_2P_TIME_SAMPLES:
        raise ValueError(
            f"{path}: fewer than {GAUTHEY_2P_TIME_SAMPLES} aligned time samples"
        )
    return np.ascontiguousarray(arr[:, :GAUTHEY_2P_TIME_SAMPLES])


def load_pooled_source_matrix(trial_paths: Mapping[str, str | Path]) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for spec in TWO_P_TRIAL_SPECS:
        path = trial_paths.get(spec.data_filename)
        if path is None:
            raise ValueError(f"missing source trial pickle: {spec.data_filename}")
        matrices.append(_load_trial_matrix(path))
    return np.vstack(matrices)


def _row_key(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row).view(np.uint8).tobytes()


def recover_selected_source_rows(
    selected_roi_x_time: np.ndarray,
    pooled_source_roi_x_time: np.ndarray,
    *,
    atol: float | None = None,
) -> SourceRecoveryReceipt:
    selected = np.asarray(selected_roi_x_time, dtype=float)
    pooled = np.asarray(pooled_source_roi_x_time, dtype=float)
    if selected.shape != (GAUTHEY_2P_EXPECTED_SELECTED_ROIS, GAUTHEY_2P_TIME_SAMPLES):
        raise ValueError(f"unexpected selected matrix shape: {selected.shape}")
    expected_candidates = (
        len(TWO_P_TRIAL_SPECS)
        * GAUTHEY_2P_PLANES_PER_TRIAL
        * GAUTHEY_2P_CLUSTERS_PER_PLANE
    )
    if pooled.shape != (expected_candidates, GAUTHEY_2P_TIME_SAMPLES):
        raise ValueError(f"unexpected pooled source matrix shape: {pooled.shape}")

    by_bytes: dict[bytes, list[int]] = {}
    for i in range(pooled.shape[0]):
        by_bytes.setdefault(_row_key(pooled[i]), []).append(i)

    recovered: list[RecoveredSelectedROI] = []
    ambiguous = 0
    unmatched = 0
    for selected_row in range(selected.shape[0]):
        matches = list(by_bytes.get(_row_key(selected[selected_row]), ()))
        if not matches and atol is not None:
            # Expensive fallback is intentionally opt-in and used only after an
            # exact-match miss. It still rejects ambiguity.
            mask = np.all(np.isclose(pooled, selected[selected_row], rtol=0.0, atol=atol), axis=1)
            matches = np.flatnonzero(mask).tolist()
        if len(matches) == 0:
            unmatched += 1
            continue
        if len(matches) != 1:
            ambiguous += 1
            continue
        decoded = decode_global_source_row(matches[0])
        recovered.append(RecoveredSelectedROI(
            selected_row=selected_row,
            functional_id=f"selected_roi_{selected_row:04d}",
            global_source_row=decoded.global_source_row,
            trial_ordinal=decoded.trial_ordinal,
            trial_id=decoded.trial_id,
            within_trial_row=decoded.within_trial_row,
            plane_index=decoded.plane_index,
            cluster_index=decoded.cluster_index,
            source_data_filename=decoded.source_data_filename,
            source_label_filename=decoded.source_label_filename,
        ))

    complete = len(recovered) == selected.shape[0] and ambiguous == 0 and unmatched == 0
    return SourceRecoveryReceipt(
        selected_roi_count=selected.shape[0],
        candidate_roi_count=pooled.shape[0],
        time_sample_count=selected.shape[1],
        exact_match_count=len(recovered),
        ambiguous_match_count=ambiguous,
        unmatched_count=unmatched,
        complete=complete,
        rows=tuple(recovered),
    )


def write_source_recovery_csv(receipt: SourceRecoveryReceipt, output_path: str | Path) -> None:
    if not receipt.complete:
        raise ValueError("refusing to write promotable source map from incomplete recovery")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RecoveredSelectedROI.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in receipt.rows:
            writer.writerow({field: getattr(row, field) for field in fields})
