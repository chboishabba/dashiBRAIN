import pickle

import numpy as np

from dashi.io.gauthey_2p_source_recovery import decode_global_source_row
from dashi.io.gauthey_compact import (
    GAUTHEY_2P_CANDIDATE_ROIS,
    GAUTHEY_2P_EXPECTED_SELECTED_ROIS,
    GAUTHEY_2P_TIME_SAMPLES,
    load_audio_correlated_pickle,
)


def test_source_arithmetic_explains_deposited_940_rows():
    assert GAUTHEY_2P_CANDIDATE_ROIS == 4 * 47 * 1000 == 188_000
    assert GAUTHEY_2P_EXPECTED_SELECTED_ROIS == 940
    assert GAUTHEY_2P_TIME_SAMPLES == 668


def test_audio_correlated_axis_semantics_are_selected_roi_by_time(tmp_path):
    source = np.arange(940 * 668, dtype=np.float32).reshape(940, 668)
    path = tmp_path / "dffs_audio_2p_corr_top05_all.pkl"
    with path.open("wb") as f:
        pickle.dump({"audio_correlated": source}, f)

    matrix = load_audio_correlated_pickle(path)
    assert matrix.source_array_shape == (940, 668)
    assert matrix.source_axis_semantics == "selected_roi_x_time"
    assert matrix.traces.shape == (668, 940)
    assert len(matrix.unit_ids) == 940
    assert matrix.unit_ids[0] == "selected_roi_0000"
    assert matrix.identity_kind == "pooled_selected_roi_row_unmapped_to_source_roi"
    np.testing.assert_array_equal(matrix.traces[:, 17], source[17])


def test_global_source_row_decodes_trial_plane_cluster():
    first = decode_global_source_row(0)
    assert first.trial_ordinal == 0
    assert first.plane_index == 0
    assert first.cluster_index == 0

    end_first_trial = decode_global_source_row(46_999)
    assert end_first_trial.trial_ordinal == 0
    assert end_first_trial.plane_index == 46
    assert end_first_trial.cluster_index == 999

    second_trial = decode_global_source_row(47_000)
    assert second_trial.trial_ordinal == 1
    assert second_trial.plane_index == 0
    assert second_trial.cluster_index == 0

    last = decode_global_source_row(187_999)
    assert last.trial_ordinal == 3
    assert last.plane_index == 46
    assert last.cluster_index == 999
