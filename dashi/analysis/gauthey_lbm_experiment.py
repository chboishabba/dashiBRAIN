"""Experiment-facing reconstruction helpers for Gauthey et al. 2026 LBM data.

Scientific source:
Wayan Gauthey, Albert Lin, Osama M. Ahmed, Andrew M. Leifer, Mala Murthy,
Stephan Y. Thiberge, "High-speed whole-brain imaging in Drosophila",
DOI 10.1038/s41467-026-72437-1; preprocessed data DOI
10.5281/zenodo.17618684; analysis code github:murthylab/lightbead-analysis.

This module reproduces the source paper's no-lag top-correlation selection on the
full six-trial LBM supervoxel carrier and decodes selected rows back to the
source (trial, plane, cluster) geometry. It does not infer neuron identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import pickle
from typing import Sequence

import numpy as np
from scipy.signal import convolve

from dashi.io.remote_zip import fetch_named_member

GAUTHEY_DATA_ZIP_URL = "https://zenodo.org/api/records/17618684/files/Data.zip/content"
LBM_FRAME_RATE_HZ = 28.2893
LBM_N_PLANES = 27
LBM_CLUSTERS_PER_PLANE = 2000
LBM_ROWS_PER_TRIAL = LBM_N_PLANES * LBM_CLUSTERS_PER_PLANE
LBM_MIN_TIMEPOINTS = 7977
LBM_TOP_PERCENT = 0.5
LBM_EXPECTED_SELECTED = 1620

LBM_TRIALS = (
    "04032024_6f_a2_r1",
    "04032024_6f_a2_r5",
    "04162024_6f_a1_r1",
    "04192024_6f_a1_r2",
    "04192024_6f_a1_r6",
    "04192024_6f_a1_r9",
)

# Segmentation IDs include the channel marker ``_6f_`` while the source aligned
# dictionary/container names encode it in the ``GCaMP6f_`` prefix and therefore
# omit the redundant middle token. Keep both namespaces explicit.
LBM_SOURCE_STEMS = tuple(
    f"GCaMP6f_{trial.replace('_6f_', '_', 1)}" for trial in LBM_TRIALS
)
LBM_SOURCE_PICKLE_NAMES = tuple(f"{stem}.pkl" for stem in LBM_SOURCE_STEMS)
LBM_SOURCE_CONTAINER_MEMBERS = tuple(
    f"Data/Dffs/Aligned/{stem}.zip" for stem in LBM_SOURCE_STEMS
)
LBM_LABEL_MEMBERS = tuple(f"Data/Labels/{trial}_n2000_labels.h5" for trial in LBM_TRIALS)
LBM_SELECTED_MEMBER = "Data/Dffs/Audio correlated/dffs_audio_LB_corr_top05_all.pkl"
LBM_MEAN_BRAIN_MEMBER = "Data/Mean brain/04032024_GCamp6f_a2_r5_w3_mean_G.nii"
LBM_STIMULUS_MEMBER = "Data/Stimulus/3min_pulse_train.mat"

LBM_START_BLOCK_SECONDS = np.array(
    [5, 25, 45, 65, 84, 103, 123, 143, 163, 183, 202.99894, 222.99788, 242.99788],
    dtype=float,
)
LBM_END_BLOCK_SECONDS = np.array(
    [15, 35, 55, 75, 94, 113, 133, 153, 173, 192.99894, 212.99788, 232.99788, 252.99788],
    dtype=float,
)


@dataclass(frozen=True)
class CorrelationSelection:
    selected_rows: np.ndarray
    selected_correlations: np.ndarray
    all_correlations: np.ndarray
    cutoff_percent: float


@dataclass(frozen=True)
class SelectedLBMIdentity:
    selected_row: int
    pooled_source_row: int
    trial_id: str
    plane_index: int
    cluster_index: int
    correlation: float


@dataclass(frozen=True)
class LBMSelectionReconstruction:
    selected_traces: np.ndarray  # selected ROI x time
    identities: tuple[SelectedLBMIdentity, ...]
    selected_correlations: np.ndarray
    deposited_match: bool | None
    max_abs_difference: float | None


def source_faithful_top_correlation_selection(
    traces_roi_by_time: np.ndarray,
    stimulus: Sequence[float],
    *,
    cutoff_percent: float = LBM_TOP_PERCENT,
) -> CorrelationSelection:
    """Reproduce Gauthey ``crosscorr_sort(..., max_lag=0)`` selection."""
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


def published_lbm_stimulus_regressor(n_timepoints: int = LBM_MIN_TIMEPOINTS) -> np.ndarray:
    """Recreate the exact binary-block + GCaMP6f kernel regressor in Fig. 3 preprocessing."""
    binary = np.zeros(int(n_timepoints), dtype=float)
    starts = (LBM_START_BLOCK_SECONDS * LBM_FRAME_RATE_HZ).astype(int)
    ends = (LBM_END_BLOCK_SECONDS * LBM_FRAME_RATE_HZ).astype(int)
    for start, end in zip(starts, ends):
        # Source ``create_stim`` uses strict inequalities s < ii < e.
        lo = max(0, int(start) + 1)
        hi = min(binary.size, int(end))
        if lo < hi:
            binary[lo:hi] = 1.0

    dt = 1.0 / LBM_FRAME_RATE_HZ
    kernel_t = np.arange(0.0, 1.0, dt)
    kernel = (1.0 - np.exp(-kernel_t / 0.050)) * np.exp(-kernel_t / 0.140)
    kernel /= np.max(kernel)
    return convolve(binary, kernel, mode="full")[: binary.size]


def pooled_lbm_row_to_trial_plane_cluster(row: int) -> tuple[str, int, int]:
    """Decode a full pooled LBM row under source trial/plane/cluster order."""
    total = len(LBM_TRIALS) * LBM_ROWS_PER_TRIAL
    if row < 0 or row >= total:
        raise ValueError(f"pooled LBM row must be in [0, {total})")
    trial_index, within = divmod(int(row), LBM_ROWS_PER_TRIAL)
    plane, cluster = divmod(within, LBM_CLUSTERS_PER_PLANE)
    return LBM_TRIALS[trial_index], plane, cluster


def _load_trial_aligned(path: str | Path) -> np.ndarray:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "dffs_aligned" not in payload:
        raise ValueError(f"{path} lacks dict key 'dffs_aligned'")
    traces = np.asarray(payload["dffs_aligned"], dtype=float)
    if traces.ndim != 2 or traces.shape[0] != LBM_ROWS_PER_TRIAL:
        raise ValueError(
            f"{path} has dffs_aligned shape {traces.shape}; expected "
            f"({LBM_ROWS_PER_TRIAL}, >= {LBM_MIN_TIMEPOINTS})"
        )
    if traces.shape[1] < LBM_MIN_TIMEPOINTS:
        raise ValueError(f"{path} has fewer than {LBM_MIN_TIMEPOINTS} aligned timepoints")
    return traces[:, :LBM_MIN_TIMEPOINTS]


def load_deposited_lbm_selected(path: str | Path) -> np.ndarray:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "audio_correlated" not in payload:
        raise ValueError("deposited LBM pickle lacks 'audio_correlated'")
    arr = np.asarray(payload["audio_correlated"], dtype=float)
    if arr.shape != (LBM_EXPECTED_SELECTED, LBM_MIN_TIMEPOINTS):
        raise ValueError(
            f"unexpected deposited LBM selected shape {arr.shape}; expected "
            f"({LBM_EXPECTED_SELECTED}, {LBM_MIN_TIMEPOINTS})"
        )
    return arr


def reconstruct_lbm_selected_from_trials(
    trial_pickle_paths: Sequence[str | Path],
    *,
    deposited_selected_path: str | Path | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> LBMSelectionReconstruction:
    """Regenerate the exact global top-0.5% LBM selection without a 20+ GB stack.

    Each trial has 54,000 rows. A row ranked below 1,620 within its own trial
    cannot enter the global top 1,620, because at least 1,620 rows from that same
    trial already outrank it. Therefore retaining each trial's local top 1,620
    is mathematically sufficient for the exact global selection.
    """
    if len(trial_pickle_paths) != len(LBM_TRIALS):
        raise ValueError(f"expected {len(LBM_TRIALS)} trial pickle paths")

    stimulus = published_lbm_stimulus_regressor()
    candidate_rows: list[np.ndarray] = []
    candidate_corrs: list[np.ndarray] = []
    candidate_traces: list[np.ndarray] = []

    for trial_index, path in enumerate(trial_pickle_paths):
        traces = _load_trial_aligned(path)
        means = traces.mean(axis=1, keepdims=True)
        stds = traces.std(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (traces - means) / stds
        stim_norm = (stimulus - stimulus.mean()) / stimulus.std()
        corr = np.dot(normalized, stim_norm.T) / normalized.shape[1]
        finite_rows = np.arange(LBM_ROWS_PER_TRIAL)[~np.isnan(corr)]
        finite_corr = corr[~np.isnan(corr)]
        if finite_rows.size < LBM_EXPECTED_SELECTED:
            raise ValueError(f"trial {LBM_TRIALS[trial_index]} has too few finite traces")
        local_order = np.argsort(finite_corr)[-LBM_EXPECTED_SELECTED:]
        local_rows = finite_rows[local_order]
        pooled = trial_index * LBM_ROWS_PER_TRIAL + local_rows
        candidate_rows.append(pooled.astype(np.int64, copy=False))
        candidate_corrs.append(finite_corr[local_order].astype(float, copy=False))
        candidate_traces.append(np.asarray(traces[local_rows, :], dtype=float).copy())
        del normalized, traces, corr
        gc.collect()

    rows = np.concatenate(candidate_rows)
    corrs = np.concatenate(candidate_corrs)
    traces = np.vstack(candidate_traces)
    global_order = np.argsort(corrs)[-LBM_EXPECTED_SELECTED:]
    selected_rows = rows[global_order]
    selected_corrs = corrs[global_order]
    selected_traces = traces[global_order, :]

    identities: list[SelectedLBMIdentity] = []
    for selected_row, (pooled_row, corr) in enumerate(zip(selected_rows, selected_corrs)):
        trial, plane, cluster = pooled_lbm_row_to_trial_plane_cluster(int(pooled_row))
        identities.append(
            SelectedLBMIdentity(
                selected_row=selected_row,
                pooled_source_row=int(pooled_row),
                trial_id=trial,
                plane_index=plane,
                cluster_index=cluster,
                correlation=float(corr),
            )
        )

    deposited_match: bool | None = None
    max_abs_difference: float | None = None
    if deposited_selected_path is not None:
        deposited = load_deposited_lbm_selected(deposited_selected_path)
        diff = np.abs(selected_traces - deposited)
        max_abs_difference = float(np.max(diff))
        deposited_match = bool(np.allclose(selected_traces, deposited, atol=atol, rtol=rtol, equal_nan=True))

    return LBMSelectionReconstruction(
        selected_traces=selected_traces,
        identities=tuple(identities),
        selected_correlations=selected_corrs,
        deposited_match=deposited_match,
        max_abs_difference=max_abs_difference,
    )


def extract_compact_lbm_working_set(
    output_dir: str | Path,
    *,
    url: str = GAUTHEY_DATA_ZIP_URL,
) -> tuple[Path, ...]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    members = (LBM_SELECTED_MEMBER, *LBM_LABEL_MEMBERS, LBM_MEAN_BRAIN_MEMBER, LBM_STIMULUS_MEMBER)
    written: list[Path] = []
    for member_name in members:
        _, raw = fetch_named_member(url, member_name)
        target = out / Path(member_name).name
        target.write_bytes(raw)
        written.append(target)
    return tuple(written)
