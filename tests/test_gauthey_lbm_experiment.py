import numpy as np
import pytest

from dashi.analysis.gauthey_lbm_experiment import (
    LBM_CLUSTERS_PER_PLANE,
    LBM_N_PLANES,
    LBM_TRIALS,
    pooled_lbm_row_to_trial_plane_cluster,
    source_faithful_top_correlation_selection,
)


def test_source_faithful_top_correlation_selects_highest_rows():
    # 400 ROIs at 0.5% => exactly two selected rows.
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    stim = np.sin(t)
    traces = np.zeros((400, 100), dtype=float)
    rng = np.random.default_rng(7)
    traces[:] = rng.normal(scale=0.2, size=traces.shape)
    traces[17] = stim
    traces[233] = 0.8 * stim + 0.02 * np.cos(t)

    result = source_faithful_top_correlation_selection(traces, stim, cutoff_percent=0.5)
    assert result.selected_rows.shape == (2,)
    assert set(result.selected_rows.tolist()) == {17, 233}
    assert np.all(np.diff(result.selected_correlations) >= 0)


def test_source_faithful_top_correlation_drops_constant_nan_rows():
    stim = np.linspace(-1.0, 1.0, 20)
    traces = np.tile(stim, (200, 1))
    traces[3] = 1.0
    result = source_faithful_top_correlation_selection(traces, stim, cutoff_percent=0.5)
    assert 3 not in result.selected_rows


def test_pooled_row_decodes_trial_plane_cluster_boundaries():
    rows_per_trial = LBM_N_PLANES * LBM_CLUSTERS_PER_PLANE
    assert pooled_lbm_row_to_trial_plane_cluster(0) == (LBM_TRIALS[0], 0, 0)
    assert pooled_lbm_row_to_trial_plane_cluster(LBM_CLUSTERS_PER_PLANE) == (LBM_TRIALS[0], 1, 0)
    assert pooled_lbm_row_to_trial_plane_cluster(rows_per_trial) == (LBM_TRIALS[1], 0, 0)
    assert pooled_lbm_row_to_trial_plane_cluster(len(LBM_TRIALS) * rows_per_trial - 1) == (
        LBM_TRIALS[-1],
        LBM_N_PLANES - 1,
        LBM_CLUSTERS_PER_PLANE - 1,
    )


def test_pooled_row_rejects_out_of_range():
    total = len(LBM_TRIALS) * LBM_N_PLANES * LBM_CLUSTERS_PER_PLANE
    with pytest.raises(ValueError):
        pooled_lbm_row_to_trial_plane_cluster(-1)
    with pytest.raises(ValueError):
        pooled_lbm_row_to_trial_plane_cluster(total)
