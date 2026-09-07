"""Selective HTTP-range access to very large ZIP/Zip64 scientific deposits.

The reader fetches the central directory and requested compressed members only;
it never downloads the full archive. It supports stored and deflated members,
checks CRC-32, and can stream very large members to disk without holding their
compressed or uncompressed payloads in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import binascii
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    local_header_offset: int
    crc32: int


class HTTPRangeError(RuntimeError):
    pass


_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def _retry_delay_seconds(headers: object | None, attempt: int, base: float, cap: float) -> float:
    retry_after = headers.get("Retry-After") if headers is not None and hasattr(headers, "get") else None
    if retry_after:
        try:
            return min(cap, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return min(cap, max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds()))
            except Exception:
                pass
    return min(cap, base * (2 ** attempt))


def _request(
    url: str,
    start: int | None = None,
    end: int | None = None,
    *,
    max_attempts: int = 6,
    retry_base_seconds: float = 1.0,
    retry_cap_seconds: float = 30.0,
    timeout_seconds: float = 120.0,
) -> tuple[bytes, object]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    headers = {"User-Agent": "dashiBRAIN-RemoteZip/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    req = urllib.request.Request(url, headers=headers)

    last_error: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRYABLE_HTTP_STATUS or attempt + 1 >= max_attempts:
                raise
            delay = _retry_delay_seconds(exc.headers, attempt, retry_base_seconds, retry_cap_seconds)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                raise
            delay = min(retry_cap_seconds, retry_base_seconds * (2 ** attempt))
        time.sleep(delay)

    raise HTTPRangeError(f"remote request failed after {max_attempts} attempts: {last_error}")


def remote_size(url: str) -> int:
    data, headers = _request(url, 0, 0)
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    length = headers.get("Content-Length")
    if length is not None and len(data) <= 1:
        return int(length)
    raise HTTPRangeError("remote server did not expose archive size through Range semantics")


def _zip64_values(extra: bytes, uncomp32: int, comp32: int, offset32: int) -> tuple[int, int, int]:
    uncomp = uncomp32
    comp = comp32
    offset = offset32
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        body = memoryview(extra)[pos + 4 : pos + 4 + size]
        if tag == 0x0001:
            cursor = 0
            if uncomp32 == 0xFFFFFFFF:
                uncomp = struct.unpack_from("<Q", body, cursor)[0]
                cursor += 8
            if comp32 == 0xFFFFFFFF:
                comp = struct.unpack_from("<Q", body, cursor)[0]
                cursor += 8
            if offset32 == 0xFFFFFFFF:
                offset = struct.unpack_from("<Q", body, cursor)[0]
            break
        pos += 4 + size
    return int(uncomp), int(comp), int(offset)


def list_remote_zip(url: str, *, tail_bytes: int = 1 << 20) -> tuple[RemoteZipMember, ...]:
    total = remote_size(url)
    start = max(0, total - tail_bytes)
    tail, _ = _request(url, start, total - 1)

    locator_pos = tail.rfind(b"PK\x06\x07")
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if locator_pos >= 0:
        _, _, zip64_eocd_offset, _ = struct.unpack_from("<IIQI", tail, locator_pos)
        record, _ = _request(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        if record[:4] != b"PK\x06\x06":
            raise HTTPRangeError("invalid Zip64 EOCD signature")
        (_, _, _, _, _, _, _, total_entries, cd_size, cd_offset) = struct.unpack_from(
            "<IQHHIIQQQQ", record, 0
        )
    elif eocd_pos >= 0:
        fields = struct.unpack_from("<IHHHHIIH", tail, eocd_pos)
        total_entries = fields[4]
        cd_size = fields[5]
        cd_offset = fields[6]
    else:
        raise HTTPRangeError("ZIP end-of-central-directory record not found")

    cd, _ = _request(url, int(cd_offset), int(cd_offset + cd_size - 1))
    members: list[RemoteZipMember] = []
    pos = 0
    while pos + 46 <= len(cd) and len(members) < int(total_entries):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", cd, pos)
        method = fields[4]
        crc = fields[7]
        comp32 = fields[8]
        uncomp32 = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        offset32 = fields[16]
        name_start = pos + 46
        name = cd[name_start : name_start + name_len].decode("utf-8", errors="replace")
        extra_start = name_start + name_len
        extra = cd[extra_start : extra_start + extra_len]
        uncomp, comp, offset = _zip64_values(extra, uncomp32, comp32, offset32)
        members.append(RemoteZipMember(name, comp, uncomp, method, offset, crc))
        pos += 46 + name_len + extra_len + comment_len
    return tuple(members)


def _payload_bounds(url: str, member: RemoteZipMember) -> tuple[int, int]:
    header, _ = _request(url, member.local_header_offset, member.local_header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise HTTPRangeError(f"invalid local header for {member.name}")
    fields = struct.unpack_from("<IHHHHHIIIHH", header, 0)
    name_len, extra_len = fields[9], fields[10]
    payload_start = member.local_header_offset + 30 + name_len + extra_len
    return payload_start, payload_start + member.compressed_size - 1


def fetch_remote_member(url: str, member: RemoteZipMember) -> bytes:
    payload_start, payload_end = _payload_bounds(url, member)
    compressed, _ = _request(url, payload_start, payload_end)
    if len(compressed) != member.compressed_size:
        raise HTTPRangeError(f"short range read for {member.name}")
    if member.compression_method == 0:
        raw = compressed
    elif member.compression_method == 8:
        raw = zlib.decompress(compressed, -15)
    else:
        raise HTTPRangeError(
            f"unsupported compression method {member.compression_method} for {member.name}"
        )
    if len(raw) != member.uncompressed_size:
        raise HTTPRangeError(f"uncompressed-size mismatch for {member.name}")
    if (binascii.crc32(raw) & 0xFFFFFFFF) != member.crc32:
        raise HTTPRangeError(f"CRC-32 mismatch for {member.name}")
    return raw


def _inflate_compressed_file(
    compressed_path: Path,
    output_path: Path,
    member: RemoteZipMember,
    *,
    io_chunk_bytes: int = 8 << 20,
) -> Path:
    crc = 0
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj(-15) if member.compression_method == 8 else None
    if member.compression_method not in (0, 8):
        raise HTTPRangeError(
            f"unsupported compression method {member.compression_method} for {member.name}"
        )

    try:
        with compressed_path.open("rb") as src, output_path.open("wb") as dst:
            while True:
                chunk = src.read(io_chunk_bytes)
                if not chunk:
                    break
                raw = chunk if decompressor is None else decompressor.decompress(chunk)
                if raw:
                    dst.write(raw)
                    written += len(raw)
                    crc = binascii.crc32(raw, crc)
            if decompressor is not None:
                tail = decompressor.flush()
                if tail:
                    dst.write(tail)
                    written += len(tail)
                    crc = binascii.crc32(tail, crc)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    if written != member.uncompressed_size:
        output_path.unlink(missing_ok=True)
        raise HTTPRangeError(f"uncompressed-size mismatch for {member.name}")
    if (crc & 0xFFFFFFFF) != member.crc32:
        output_path.unlink(missing_ok=True)
        raise HTTPRangeError(f"CRC-32 mismatch for {member.name}")
    return output_path


def stream_remote_member_to_file(
    url: str,
    member: RemoteZipMember,
    output_path: str | Path,
    *,
    compressed_chunk_bytes: int = 8 << 20,
    resume: bool = True,
) -> Path:
    """Download one member with exact compressed-byte resume, then inflate/CRC-check.

    The resumable state is the member's compressed payload, not partially inflated
    bytes. This is exact across process crashes because HTTP ranges address the
    compressed stream directly. Once the payload is complete it is inflated to
    ``output_path`` and verified against the central-directory size and CRC-32.
    """
    if compressed_chunk_bytes <= 0:
        raise ValueError("compressed_chunk_bytes must be positive")
    payload_start, payload_end = _payload_bounds(url, member)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    compressed_part = out.with_name(out.name + ".compressed.part")

    if not resume:
        compressed_part.unlink(missing_ok=True)
        out.unlink(missing_ok=True)

    existing = compressed_part.stat().st_size if compressed_part.exists() else 0
    if existing > member.compressed_size:
        compressed_part.unlink(missing_ok=True)
        existing = 0
    cursor = payload_start + existing

    with compressed_part.open("ab" if existing else "wb") as handle:
        while cursor <= payload_end:
            end = min(payload_end, cursor + compressed_chunk_bytes - 1)
            chunk, _ = _request(url, cursor, end)
            expected = end - cursor + 1
            if len(chunk) != expected:
                raise HTTPRangeError(f"short range read for {member.name} at {cursor}")
            handle.write(chunk)
            handle.flush()
            cursor = end + 1

    if compressed_part.stat().st_size != member.compressed_size:
        raise HTTPRangeError(f"compressed-size mismatch for {member.name}")

    result = _inflate_compressed_file(
        compressed_part,
        out,
        member,
        io_chunk_bytes=compressed_chunk_bytes,
    )
    compressed_part.unlink(missing_ok=True)
    return result


def fetch_named_member(url: str, name: str) -> tuple[RemoteZipMember, bytes]:
    by_name = {m.name: m for m in list_remote_zip(url)}
    if name not in by_name:
        raise KeyError(f"archive member not found: {name}")
    member = by_name[name]
    return member, fetch_remote_member(url, member)
