"""Out-of-core loading for trusted pickles containing very large NumPy arrays.

The Gauthey aligned-trial source dictionaries contain multi-gigabyte ndarrays.
A normal ``pickle.load`` must materialize the complete ndarray payload in RAM
before downstream code can score it, which can OOM even when the later analysis
is blockwise.

This module keeps ordinary pickle semantics for small objects, but spills large
BINBYTES/BYTEARRAY8 payloads directly to files and reconstructs NumPy arrays as
``np.memmap`` objects.  Pickle FRAME opcodes are treated only as framing hints;
the pure-Python unpickler reads the underlying stream incrementally instead of
preloading an entire frame into ``BytesIO``.

This is intentionally a narrow ingestion adapter for trusted scientific
artifacts.  It is not a general safe-unpickling mechanism and must not be used
for untrusted pickle input.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
import pickle
import struct
import tempfile
from typing import BinaryIO, Iterator

import numpy as np


@dataclass(frozen=True)
class SpilledPickleBuffer:
    path: Path
    length: int
    original_kind: str


class OutOfCoreNumpyUnpickler(pickle._Unpickler):  # type: ignore[attr-defined]
    """Pure-Python unpickler that memmaps large embedded NumPy payloads."""

    dispatch = pickle._Unpickler.dispatch.copy()  # type: ignore[attr-defined]

    def __init__(
        self,
        file: BinaryIO,
        *,
        spill_dir: str | Path,
        spill_threshold_bytes: int = 64 << 20,
        copy_chunk_bytes: int = 8 << 20,
    ) -> None:
        super().__init__(file)
        self.spill_dir = Path(spill_dir)
        self.spill_dir.mkdir(parents=True, exist_ok=True)
        self.spill_threshold_bytes = int(spill_threshold_bytes)
        self.copy_chunk_bytes = int(copy_chunk_bytes)
        if self.spill_threshold_bytes <= 0 or self.copy_chunk_bytes <= 0:
            raise ValueError("spill threshold and copy chunk must be positive")
        self.spilled_paths: list[Path] = []

    def _spill(self, length: int, *, original_kind: str) -> SpilledPickleBuffer:
        fd, raw_path = tempfile.mkstemp(prefix="pickle-array-", suffix=".bin", dir=self.spill_dir)
        path = Path(raw_path)
        self.spilled_paths.append(path)
        remaining = int(length)
        try:
            with os.fdopen(fd, "wb", closefd=True) as out:
                while remaining:
                    want = min(remaining, self.copy_chunk_bytes)
                    chunk = self.read(want)
                    if len(chunk) != want:
                        raise pickle.UnpicklingError(
                            f"pickle payload truncated while spilling: wanted {want}, got {len(chunk)}"
                        )
                    out.write(chunk)
                    remaining -= want
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return SpilledPickleBuffer(path=path, length=int(length), original_kind=original_kind)

    def load_frame(self) -> None:
        # ``pickle._Unframer.load_frame`` normally reads the complete frame into
        # an in-memory BytesIO.  Frames are semantically optional boundaries, so
        # consume only the declared size and continue streaming opcodes directly.
        frame_size, = struct.unpack("<Q", self.read(8))
        if frame_size > (1 << 63) - 1:
            raise pickle.UnpicklingError(f"implausible pickle frame size: {frame_size}")

    def load_binbytes(self) -> None:
        length, = struct.unpack("<I", self.read(4))
        if length >= self.spill_threshold_bytes:
            self.append(self._spill(length, original_kind="bytes"))
        else:
            self.append(self.read(length))

    def load_binbytes8(self) -> None:
        length, = struct.unpack("<Q", self.read(8))
        if length >= self.spill_threshold_bytes:
            self.append(self._spill(length, original_kind="bytes"))
        else:
            self.append(self.read(length))

    def load_bytearray8(self) -> None:
        length, = struct.unpack("<Q", self.read(8))
        if length >= self.spill_threshold_bytes:
            self.append(self._spill(length, original_kind="bytearray"))
        else:
            self.append(bytearray(self.read(length)))

    @staticmethod
    def _memmap_from_spill(
        spill: SpilledPickleBuffer,
        *,
        dtype: np.dtype,
        shape: tuple[int, ...],
        order: str,
    ) -> np.memmap:
        dtype = np.dtype(dtype)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != spill.length:
            raise pickle.UnpicklingError(
                "spilled ndarray payload length disagrees with declared shape/dtype: "
                f"payload={spill.length}, expected={expected}, shape={shape}, dtype={dtype}"
            )
        return np.memmap(spill.path, mode="r", dtype=dtype, shape=shape, order=order)

    def load_build(self) -> None:
        state = self.stack.pop()
        inst = self.stack[-1]
        # Protocol <=4 NumPy ndarray state:
        # (version, shape, dtype, is_fortran, raw-bytes).
        if (
            isinstance(inst, np.ndarray)
            and isinstance(state, tuple)
            and len(state) == 5
            and isinstance(state[4], SpilledPickleBuffer)
        ):
            _version, shape, dtype, is_fortran, spill = state
            mapped = self._memmap_from_spill(
                spill,
                dtype=np.dtype(dtype),
                shape=tuple(int(v) for v in shape),
                order="F" if bool(is_fortran) else "C",
            )
            self.stack[-1] = mapped
            # The placeholder ndarray may already have been memoized before its
            # BUILD opcode.  Preserve subsequent pickle aliasing semantics.
            for key, value in tuple(self.memo.items()):
                if value is inst:
                    self.memo[key] = mapped
            return

        self.stack.append(state)
        super().load_build()

    def load_reduce(self) -> None:
        args = self.stack.pop()
        func = self.stack[-1]
        # Protocol 5 NumPy uses numpy.*._frombuffer(raw, dtype, shape, order).
        if (
            getattr(func, "__name__", None) == "_frombuffer"
            and str(getattr(func, "__module__", "")).startswith("numpy")
            and isinstance(args, tuple)
            and len(args) == 4
            and isinstance(args[0], SpilledPickleBuffer)
        ):
            spill, dtype, shape, order = args
            mapped = self._memmap_from_spill(
                spill,
                dtype=np.dtype(dtype),
                shape=tuple(int(v) for v in shape),
                order=str(order),
            )
            self.stack[-1] = mapped
            return
        self.stack[-1] = func(*args)


OutOfCoreNumpyUnpickler.dispatch[pickle.FRAME[0]] = OutOfCoreNumpyUnpickler.load_frame
OutOfCoreNumpyUnpickler.dispatch[pickle.BINBYTES[0]] = OutOfCoreNumpyUnpickler.load_binbytes
OutOfCoreNumpyUnpickler.dispatch[pickle.BINBYTES8[0]] = OutOfCoreNumpyUnpickler.load_binbytes8
OutOfCoreNumpyUnpickler.dispatch[pickle.BYTEARRAY8[0]] = OutOfCoreNumpyUnpickler.load_bytearray8
OutOfCoreNumpyUnpickler.dispatch[pickle.BUILD[0]] = OutOfCoreNumpyUnpickler.load_build
OutOfCoreNumpyUnpickler.dispatch[pickle.REDUCE[0]] = OutOfCoreNumpyUnpickler.load_reduce


def _close_memmaps(value, seen: set[int]) -> None:
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, np.memmap):
        mmap_obj = getattr(value, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _close_memmaps(key, seen)
            _close_memmaps(item, seen)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            _close_memmaps(item, seen)


@contextmanager
def load_numpy_pickle_out_of_core(
    file: BinaryIO,
    *,
    spill_dir: str | Path,
    spill_threshold_bytes: int = 64 << 20,
    copy_chunk_bytes: int = 8 << 20,
) -> Iterator[object]:
    """Yield a trusted pickle while large NumPy buffers live on disk as memmaps."""
    unpickler = OutOfCoreNumpyUnpickler(
        file,
        spill_dir=spill_dir,
        spill_threshold_bytes=spill_threshold_bytes,
        copy_chunk_bytes=copy_chunk_bytes,
    )
    payload = unpickler.load()
    try:
        yield payload
    finally:
        _close_memmaps(payload, set())
        for path in unpickler.spilled_paths:
            path.unlink(missing_ok=True)
