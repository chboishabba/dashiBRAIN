"""Selective HTTP-range access to very large ZIP/Zip64 scientific deposits.

The reader fetches the central directory and requested compressed members only;
it never downloads the full archive.  It supports stored and deflated members,
checks CRC-32, and returns exact member metadata for provenance receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import binascii
import struct
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


def _request(url: str, start: int | None = None, end: int | None = None) -> tuple[bytes, object]:
    headers = {"User-Agent": "dashiBRAIN-RemoteZip/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return resp.read(), resp.headers


def remote_size(url: str) -> int:
    # Range 0-0 is more reliable than HEAD on some repository gateways.
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


def fetch_remote_member(url: str, member: RemoteZipMember) -> bytes:
    header, _ = _request(url, member.local_header_offset, member.local_header_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise HTTPRangeError(f"invalid local header for {member.name}")
    fields = struct.unpack_from("<IHHHHHIIIHH", header, 0)
    name_len, extra_len = fields[9], fields[10]
    payload_start = member.local_header_offset + 30 + name_len + extra_len
    payload_end = payload_start + member.compressed_size - 1
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


def fetch_named_member(url: str, name: str) -> tuple[RemoteZipMember, bytes]:
    by_name = {m.name: m for m in list_remote_zip(url)}
    if name not in by_name:
        raise KeyError(f"archive member not found: {name}")
    member = by_name[name]
    return member, fetch_remote_member(url, member)
