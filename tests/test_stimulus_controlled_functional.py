from __future__ import annotations

import numpy as np

from dashi.analysis.stimulus_controlled_functional import (
    residualize_region_traces_against_stimulus,
)


def _orthogonal_residual(stimulus: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=stimulus.size)
    design = np.column_stack([np.ones(stimulus.size), stimulus])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


def test_stimulus_residualization_recovers_independent_residuals_exactly():
    stim = np.linspace(-1.0, 1.0, 101)
    r0 = _orthogonal_residual(stim, 1)
    r1 = _orthogonal_residual(stim, 2)
    traces = np.column_stack([
        2.5 + 3.0 * stim + r0,
        -1.0 - 0.75 * stim + r1,
    ])

    out = residualize_region_traces_against_stimulus(traces, stim)

    assert np.allclose(out.intercepts, [2.5, -1.0], atol=1e-12)
    assert np.allclose(out.coefficients, [3.0, -0.75], atol=1e-12)
    assert np.allclose(out.residual_traces[:, 0], r0, atol=1e-12)
    assert np.allclose(out.residual_traces[:, 1], r1, atol=1e-12)
    assert np.all(out.variance_fraction_removed > 0)


def test_stimulus_residualization_removes_shared_stimulus_correlation():
    stim = np.sin(np.linspace(0.0, 8.0 * np.pi, 500))
    r0 = _orthogonal_residual(stim, 3)
    r1 = _orthogonal_residual(stim, 4)
    traces = np.column_stack([
        5.0 * stim + 0.2 * r0,
        4.0 * stim + 0.2 * r1,
    ])
    raw_corr = float(np.corrcoef(traces.T)[0, 1])

    out = residualize_region_traces_against_stimulus(traces, stim)
    controlled_corr = float(np.corrcoef(out.residual_traces.T)[0, 1])

    assert raw_corr > 0.99
    assert abs(controlled_corr) < 0.1
