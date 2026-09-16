from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from dashi.io.gauthey_compact import load_audio_correlated_pickle


def test_gauthey_pickle_loads_known_orientation_and_keeps_identity_unregistered(tmp_path: Path):
    p = tmp_path / "dffs_audio_2p_corr_top05_all.pkl"
    arr = np.arange(940 * 668, dtype=float).reshape(940, 668)
    with open(p, "wb") as f:
        pickle.dump({"audio_correlated": arr}, f)

    loaded = load_audio_correlated_pickle(p)
    assert loaded.traces.shape == (668, 940)
    assert loaded.source_array_shape == (940, 668)
    assert loaded.source_axis_semantics == "selected_roi_x_time"
    assert len(loaded.unit_ids) == 940
    assert loaded.unit_ids[0] == "selected_roi_0000"
    assert loaded.unit_ids[-1] == "selected_roi_0939"
    assert loaded.identity_kind == "pooled_selected_roi_row_unmapped_to_source_roi"


def test_gauthey_pickle_rejects_unreceipted_shape(tmp_path: Path):
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as f:
        pickle.dump({"audio_correlated": np.zeros((10, 10))}, f)
    try:
        load_audio_correlated_pickle(p)
    except ValueError as exc:
        assert "unexpected Gauthey 2p audio_correlated shape" in str(exc)
    else:
        raise AssertionError("unreceipted functional shape must not be silently accepted")
