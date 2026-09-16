from __future__ import annotations

import http.client
import io
from pathlib import Path
import urllib.error
import zlib

import pytest

from dashi.io import remote_zip
from dashi.io.remote_zip import RemoteZipMember


class _Response:
    def __init__(self, data: bytes, headers: dict[str, str] | None = None):
        self._data = data
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._data


class _IncompleteResponse(_Response):
    def read(self) -> bytes:
        raise http.client.IncompleteRead(self._data, 7)


def _raw_deflate(payload: bytes) -> bytes:
    c = zlib.compressobj(level=6, wbits=-15)
    return c.compress(payload) + c.flush()


def test_request_retries_transient_504(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 504, "Gateway Time-out", {}, io.BytesIO())
        return _Response(b"ok")

    monkeypatch.setattr(remote_zip.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_zip.time, "sleep", lambda _: None)

    data, _ = remote_zip._request("https://example.test/archive", max_attempts=2)
    assert data == b"ok"
    assert calls["n"] == 2


def test_request_retries_incomplete_read(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _IncompleteResponse(b"partial")
        return _Response(b"complete")

    monkeypatch.setattr(remote_zip.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(remote_zip.time, "sleep", lambda _: None)

    data, _ = remote_zip._request("https://example.test/archive", max_attempts=2)
    assert data == b"complete"
    assert calls["n"] == 2


def test_stream_resumes_compressed_sidecar_after_interruption(tmp_path, monkeypatch):
    raw = (b"abcdef0123456789" * 1024) + b"tail"
    compressed = _raw_deflate(raw)
    member = RemoteZipMember(
        name="nested.zip",
        compressed_size=len(compressed),
        uncompressed_size=len(raw),
        compression_method=8,
        local_header_offset=0,
        crc32=zlib.crc32(raw) & 0xFFFFFFFF,
    )
    payload_start = 1000
    payload_end = payload_start + len(compressed) - 1
    output = tmp_path / "nested.zip"
    sidecar = output.with_name(output.name + ".compressed.part")

    first = max(1, len(compressed) // 3)
    sidecar.write_bytes(compressed[:first])

    requested: list[tuple[int, int]] = []

    monkeypatch.setattr(remote_zip, "_payload_bounds", lambda url, m: (payload_start, payload_end))

    def fake_request(url, start=None, end=None, **kwargs):
        assert start is not None and end is not None
        requested.append((start, end))
        lo = start - payload_start
        hi = end - payload_start + 1
        return compressed[lo:hi], {}

    monkeypatch.setattr(remote_zip, "_request", fake_request)

    remote_zip.stream_remote_member_to_file(
        "https://example.test/archive",
        member,
        output,
        compressed_chunk_bytes=97,
        resume=True,
    )

    assert output.read_bytes() == raw
    assert not sidecar.exists()
    assert requested
    assert requested[0][0] == payload_start + first


def test_stream_keeps_compressed_sidecar_when_request_fails(tmp_path, monkeypatch):
    raw = b"0123456789" * 1000
    compressed = _raw_deflate(raw)
    member = RemoteZipMember(
        name="nested.zip",
        compressed_size=len(compressed),
        uncompressed_size=len(raw),
        compression_method=8,
        local_header_offset=0,
        crc32=zlib.crc32(raw) & 0xFFFFFFFF,
    )
    payload_start = 500
    payload_end = payload_start + len(compressed) - 1
    output = tmp_path / "nested.zip"
    sidecar = output.with_name(output.name + ".compressed.part")

    monkeypatch.setattr(remote_zip, "_payload_bounds", lambda url, m: (payload_start, payload_end))
    calls = {"n": 0}

    def fake_request(url, start=None, end=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise urllib.error.URLError("temporary failure")
        lo = start - payload_start
        hi = end - payload_start + 1
        return compressed[lo:hi], {}

    monkeypatch.setattr(remote_zip, "_request", fake_request)

    with pytest.raises(urllib.error.URLError):
        remote_zip.stream_remote_member_to_file(
            "https://example.test/archive",
            member,
            output,
            compressed_chunk_bytes=max(1, len(compressed) // 3),
            resume=True,
        )

    assert sidecar.exists()
    assert 0 < sidecar.stat().st_size < len(compressed)
    assert not output.exists()
