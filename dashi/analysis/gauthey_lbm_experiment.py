"""Experiment-facing reconstruction helpers for Gauthey et al. 2026 LBM data.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI
10.5281/zenodo.17618684; analysis code github:murthylab/lightbead-analysis.

This module reproduces the source paper's no-lag top-correlation selection on a
full LBM trace carrier. The returned selected indices are the missing bridge
between the pooled 1,620-trace matrix and plane-local supervoxel geometry when
full per-trial traces are available. It does not infer neuron identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from dashi.io.remote_zip import fetch_named_member

GAUTHEY_DATA_ZIP_URL = "https://zenodo.org/api/records/17618684/files/Data.zip/content"
LBM_FRAME_RATE_HZ = 28.2893
LBM_N_PLANES = 27
LBM_CLUSTERS_PER_PLANE = 2000
LBM_MIN_TIMEPOINTS = 7977
LBM_TOP_PERCENT = 0.5

LBM_TRIALS = (
    "04032024_6f_a2_r1",
    "04032024_6f_a2_r5",
    "04162024_6f_a1_r1",
    "04192024_6f_a1_r2",
    "04192024_6f_a1_r6",
    "04192024_6f_a1_r9",
)

LBM_LABEL_MEMBERS = tuple(
    f"Data/Labels/{trial}_n2000_labels.h5" for trial in LBM_TRIALS
)
LBM_SELECTED_MEMBER = "Data/Dffs/Audio correlated/dffs_audio_LB_corr_top05_all.pkl"
LBM_MEAN_BRAIN_MEMBER = "Data/Mean brain/04032024_GCamp6f_a2_r5_w3_mean_G.nii"
LBM_STIMULUS_MEMBER = "Data/Stimulus/3min_pulse_train.mat"


@dataclass(frozen=True)
class CorrelationSelection:
    selected_rows: np.ndarray
    selected_correlations: np.ndarray
    all_correlations: np.ndarray
    cutoff_percent: float


def source_faithful_top_correlation_selection(
    traces_roi_by_time: np.ndarray,
    stimulus: Sequence[float],
    *,
    cutoff_percent: float = LBM_TOP_PERCENT,
) -> CorrelationSelection:
    """Reproduce Gauthey ``crosscorr_sort(..., max_lag=0)`` selection.

    The paper code z-normalizes every ROI and the stimulus independently,
    computes the no-lag dot-product correlation, drops NaN rows, sorts ascending,
    and retains ``int(n_roi * cutoff_percent / 100)`` rows from the top.
    """
    dffs = np.asarray(traces_roi_by_time, dtype=float)
    stim = np.asarray(stimulus, dtype=float)
    if dffs.ndim != 2:
        raise ValueError("traces_roi_by_time must be 2-D [roi, time]")
    if stim.ndim != 1 or stim.shape[0] != dffs.shape[1]:
        raise ValueError("stimulus must be 1-D and align with trace time samples")
    if not (0.0 < cutoff_percent <= 100.0):
        raise ValueError("cutoff_percent must be in (0, 100]")
    if np.std(stim) == 0:
        raise ValueError("stimulus has zero variance")

    means = dffs.mean(axis=1, keepdims=True)
    stds = dffs.std(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (dffs - means) / stds
    stim_normalized = (stim - stim.mean()) / stim.std()
    correlations = np.dot(normalized, stim_normalized.T) / normalized.shape[1]
    correlations = np.asarray(correlations).reshape(-1)

    roi = np.arange(dffs.shape[0])
    finite = ~np.isnan(correlations)
    finite_roi = roi[finite]
    finite_corr = correlations[finite]
    n_select = int(dffs.shape[0] * (cutoff_percent / 100.0))
    if n_select <= 0:
        raise ValueError("cutoff selects zero ROIs")
    if n_select > finite_roi.size:
        raise ValueError("cutoff requests more finite ROIs than available")

    order = np.argsort(finite_corr)
    selected_order = order[-n_select:]
    return CorrelationSelection(
        selected_rows=finite_roi[selected_order],
        selected_correlations=finite_corr[selected_order],
        all_correlations=correlations,
        cutoff_percent=float(cutoff_percent),
    )


def pooled_lbm_row_to_trial_plane_cluster(row: int) -> tuple[str, int, int]:
    """Decode a full pooled LBM row under the source trial/plane/cluster order."""
    rows_per_trial = LBM_N_PLANES * LBM_CLUSTERS_PER_PLANE
    total = len(LBM_TRIALS) * rows_per_trial
    if row < 0 or row >= total:
        raise ValueError(f"pooled LBM row must be in [0, {total})")
    trial_index, within = divmod(int(row), rows_per_trial)
    plane, cluster = divmod(within, LBM_CLUSTERS_PER_PLANE)
    return LBM_TRIALS[trial_index], plane, cluster


def extract_compact_lbm_working_set(
    output_dir: str | Path,
    *,
    url: str = GAUTHEY_DATA_ZIP_URL,
) -> tuple[Path, ...]:
    """Range-extract the compact published LBM experiment working set.

    This downloads only named ZIP members, verifies each member through the
    existing remote ZIP CRC/size checks, and never downloads the full ~48 GB
    archive.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    members = (
        LBM_SELECTED_MEMBER,
        *LBM_LABEL_MEMBERS,
        LBM_MEAN_BRAIN_MEMBER,
        LBM_STIMULUS_MEMBER,
    )
    written: list[Path] = []
    for member_name in members:
        _, raw = fetch_named_member(url, member_name)
        target = out / Path(member_name).name
        target.write_bytes(raw)
        written.append(target)
    return tuple(written)
