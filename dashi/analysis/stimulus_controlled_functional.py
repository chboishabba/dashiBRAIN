"""Exogenous stimulus nuisance control for Gauthey region-functional traces.

The recovered Gauthey selected ROIs were chosen for high correlation with the
published audio stimulus regressor. Shared stimulus drive can therefore create
region-region covariance even when atlas-overlap covariance is controlled.

This module removes only the fixed experimental regressor from each region trace
independently:

    x_r(t) = a_r + b_r s(t) + epsilon_r(t).

No structural feature, region-pair outcome, or held-out relationship participates
in the nuisance fit. The residual traces can then be converted to a functional
correlation target and passed to the existing overlap-controlled NDim consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StimulusResidualization:
    residual_traces: np.ndarray  # time x region
    intercepts: np.ndarray
    coefficients: np.ndarray
    variance_fraction_removed: np.ndarray


def residualize_region_traces_against_stimulus(
    traces_time_by_region: np.ndarray,
    stimulus: np.ndarray,
) -> StimulusResidualization:
    """Regress a fixed stimulus and intercept from each region independently."""
    traces = np.asarray(traces_time_by_region, dtype=float)
    stim = np.asarray(stimulus, dtype=float)
    if traces.ndim != 2:
        raise ValueError("traces_time_by_region must be 2-D [time, region]")
    if stim.ndim != 1 or stim.shape[0] != traces.shape[0]:
        raise ValueError("stimulus must be 1-D and align with trace time samples")
    if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(stim)):
        raise ValueError("traces and stimulus must be finite")
    if np.std(stim) == 0:
        raise ValueError("stimulus has zero variance")

    design = np.column_stack([np.ones(stim.size, dtype=float), stim])
    beta, *_ = np.linalg.lstsq(design, traces, rcond=None)
    fitted = design @ beta
    residual = traces - fitted

    centered = traces - np.mean(traces, axis=0, keepdims=True)
    total_var = np.sum(centered * centered, axis=0)
    residual_var = np.sum(residual * residual, axis=0)
    removed = np.zeros(traces.shape[1], dtype=float)
    nz = total_var > 0
    removed[nz] = 1.0 - residual_var[nz] / total_var[nz]
    removed = np.clip(removed, 0.0, 1.0)

    return StimulusResidualization(
        residual_traces=residual,
        intercepts=np.asarray(beta[0], dtype=float),
        coefficients=np.asarray(beta[1], dtype=float),
        variance_fraction_removed=removed,
    )
