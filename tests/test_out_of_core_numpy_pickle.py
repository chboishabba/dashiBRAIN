from __future__ import annotations

from io import BytesIO
import pickle

import numpy as np
import pytest

from dashi.io.out_of_core_numpy_pickle import load_numpy_pickle_out_of_core


class ReadSizeGuard(BytesIO):
    """Fail if the unpickler attempts one large read instead of chunked streaming."""

    def __init__(self, payload: bytes, max_read: int):
        super().__init__(payload)
        self.max_read = int(max_read)
        self.max_observed_read = 0

    def read(self, n: int = -1) -> bytes:
        if n >= 0:
            self.max_observed_read = max(self.max_observed_read, n)
            if n > self.max_read:
                raise AssertionError(f"unpickler requested an oversized read: {n}")
        return super().read(n)


@pytest.mark.parametrize("protocol", [4, 5])
def test_large_numpy_payload_is_memmapped_and_exact(tmp_path, protocol):
    array = np.arange(128 * 64, dtype=np.float64).reshape(128, 64)
    encoded = pickle.dumps({"dffs_aligned": array, "metadata": "ok"}, protocol=protocol)

    with load_numpy_pickle_out_of_core(
        BytesIO(encoded),
        spill_dir=tmp_path,
        spill_threshold_bytes=1024,
        copy_chunk_bytes=4096,
    ) as payload:
        observed = payload["dffs_aligned"]
        assert isinstance(observed, np.memmap)
        assert np.array_equal(observed, array)
        assert payload["metadata"] == "ok"

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("protocol", [4, 5])
def test_large_pickle_frame_and_array_payload_are_never_read_as_one_ram_blob(tmp_path, protocol):
    array = np.arange(256 * 64, dtype=np.float64).reshape(256, 64)
    encoded = pickle.dumps({"dffs_aligned": array}, protocol=protocol)
    guarded = ReadSizeGuard(encoded, max_read=8192)

    with load_numpy_pickle_out_of_core(
        guarded,
        spill_dir=tmp_path,
        spill_threshold_bytes=1024,
        copy_chunk_bytes=4096,
    ) as payload:
        assert np.array_equal(payload["dffs_aligned"], array)

    assert guarded.max_observed_read <= 8192


def test_protocol4_aliases_follow_replaced_memmap_object(tmp_path):
    array = np.arange(128 * 64, dtype=np.float64).reshape(128, 64)
    encoded = pickle.dumps({"a": array, "b": array}, protocol=4)

    with load_numpy_pickle_out_of_core(
        BytesIO(encoded),
        spill_dir=tmp_path,
        spill_threshold_bytes=1024,
        copy_chunk_bytes=4096,
    ) as payload:
        assert isinstance(payload["a"], np.memmap)
        assert payload["a"] is payload["b"]


def test_small_pickle_payload_retains_ordinary_numpy_semantics(tmp_path):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    encoded = pickle.dumps({"dffs_aligned": array}, protocol=4)

    with load_numpy_pickle_out_of_core(
        BytesIO(encoded),
        spill_dir=tmp_path,
        spill_threshold_bytes=1 << 20,
    ) as payload:
        assert isinstance(payload["dffs_aligned"], np.ndarray)
        assert not isinstance(payload["dffs_aligned"], np.memmap)
        assert np.array_equal(payload["dffs_aligned"], array)
