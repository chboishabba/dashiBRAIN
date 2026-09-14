from __future__ import annotations

import json
from pathlib import Path
import urllib.error
import zlib

import pytest

from dashi.io import remote_zip
from dashi.io.remote_zip import RemoteZipMember


def _raw_deflate(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, wbits=-15)
    return compressor.compress(payload) + compressor.flush()


def _member(raw: bytes, compressed: bytes) -> RemoteZipMember:
    return RemoteZipMember(
        name="nested.zip",
        compressed_size=len(compressed),
        uncompressed_size=len(raw),
        compression_method=8,
        local_header_offset=0,
        crc32=zlib.crc32(raw) & 0xFFFFFFFF,
    )


def test_stream_halves_request_range_at_same_cursor_after_retry_budget_failure(tmp_path, monkeypatch):
    raw = (b"fruit-fly-transport" * 4096) + b"tail"
    compressed = _raw_deflate(raw)
    member = _member(raw, compressed)
    payload_start = 1000
    payload_end = payload_start + len(compressed) - 1
    output = tmp_path / "nested.zip"

    monkeypatch.setattr(remote_zip, "_payload_bounds", lambda url, m: (payload_start, payload_end))
    requested: list[tuple[int, int]] = []

    def fake_request(url, start=None, end=None, **kwargs):
        assert start is not None and end is not None
        requested.append((start, end))
        requested_size = end - start + 1
        if requested_size > 31:
            raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)
        lo = start - payload_start
        hi = end - payload_start + 1
        return compressed[lo:hi], {"Content-Range": f"bytes {start}-{end}/{payload_end + 1}"}

    monkeypatch.setattr(remote_zip, "_request", fake_request)

    remote_zip.stream_remote_member_to_file(
        "https://example.test/archive",
        member,
        output,
        compressed_chunk_bytes=64,
        minimum_compressed_chunk_bytes=16,
        adaptive_chunking=True,
    )

    assert output.read_bytes() == raw
    assert requested[0] == (payload_start, min(payload_end, payload_start + 63))
    # The failed 64-byte request must not advance the cursor; the smaller retry
    # starts at the exact same compressed byte.
    assert requested[1][0] == requested[0][0]
    assert requested[1][1] < requested[0][1]


def test_terminal_transfer_failure_preserves_partial_and_writes_receipt(tmp_path, monkeypatch):
    raw = b"0123456789" * 4096
    compressed = _raw_deflate(raw)
    member = _member(raw, compressed)
    payload_start = 500
    payload_end = payload_start + len(compressed) - 1
    output = tmp_path / "nested.zip"
    sidecar = output.with_name(output.name + ".compressed.part")
    receipt = tmp_path / "transport_failure.json"

    already_paid = min(23, len(compressed))
    sidecar.write_bytes(compressed[:already_paid])
    monkeypatch.setattr(remote_zip, "_payload_bounds", lambda url, m: (payload_start, payload_end))

    def always_fail(url, start=None, end=None, **kwargs):
        raise urllib.error.HTTPError(url, 504, "Gateway Time-out", {}, None)

    monkeypatch.setattr(remote_zip, "_request", always_fail)

    with pytest.raises(urllib.error.HTTPError):
        remote_zip.stream_remote_member_to_file(
            "https://example.test/archive",
            member,
            output,
            compressed_chunk_bytes=64,
            minimum_compressed_chunk_bytes=16,
            adaptive_chunking=True,
            failure_receipt_path=receipt,
        )

    assert sidecar.read_bytes() == compressed[:already_paid]
    assert not output.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["member"] == "nested.zip"
    assert payload["compressed_bytes_persisted"] == already_paid
    assert payload["compressed_bytes_total"] == len(compressed)
    assert payload["next_remote_payload_byte"] == payload_start + already_paid
    assert payload["http_status"] == 504
    assert payload["adaptive_chunking"] is True
    assert payload["minimum_compressed_chunk_bytes"] == 16
