from __future__ import annotations

from pathlib import Path
import zlib

from dashi.io import remote_zip
from dashi.io.remote_zip import RemoteZipMember


def _raw_deflate(payload: bytes) -> bytes:
    c = zlib.compressobj(level=6, wbits=-15)
    return c.compress(payload) + c.flush()


def test_stream_forwards_long_transfer_retry_policy_to_each_payload_range(tmp_path: Path, monkeypatch):
    raw = (b"retry-policy" * 4096) + b"tail"
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

    monkeypatch.setattr(remote_zip, "_payload_bounds", lambda url, m: (payload_start, payload_end))
    policies: list[dict[str, object]] = []

    def fake_request(url, start=None, end=None, **kwargs):
        assert start is not None and end is not None
        policies.append(dict(kwargs))
        lo = start - payload_start
        hi = end - payload_start + 1
        return compressed[lo:hi], {}

    monkeypatch.setattr(remote_zip, "_request", fake_request)

    remote_zip.stream_remote_member_to_file(
        "https://example.test/archive",
        member,
        output,
        compressed_chunk_bytes=max(1, len(compressed) // 3),
        request_max_attempts=12,
        request_retry_base_seconds=2.0,
        request_retry_cap_seconds=120.0,
        request_timeout_seconds=180.0,
    )

    assert output.read_bytes() == raw
    assert policies
    assert all(policy["max_attempts"] == 12 for policy in policies)
    assert all(policy["retry_base_seconds"] == 2.0 for policy in policies)
    assert all(policy["retry_cap_seconds"] == 120.0 for policy in policies)
    assert all(policy["timeout_seconds"] == 180.0 for policy in policies)
