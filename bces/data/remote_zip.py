"""Bounded ZIP central-directory inspection over official HTTP byte ranges."""

from __future__ import annotations

import struct
import binascii
import zlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import requests

EOCD = b"PK\x05\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    compression_method: int
    local_header_offset: int


def _range_get(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = None
    for attempt in range(5):
        response = session.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            allow_redirects=True,
            timeout=60,
        )
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        time.sleep(2**attempt)
    assert response is not None
    response.raise_for_status()
    if response.status_code != 206:
        raise ValueError("official server ignored the bounded byte-range request")
    payload = response.content
    if len(payload) != end - start + 1:
        raise ValueError("byte-range response length mismatch")
    return payload


def inspect_remote_zip(
    url: str, *, maximum_metadata_bytes: int = 250_000_000
) -> dict[str, Any]:
    session = requests.Session()
    head = session.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    archive_bytes = int(head.headers["Content-Length"])
    if head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise ValueError("official server does not advertise byte-range support")
    tail_size = min(262_144, archive_bytes)
    tail = _range_get(session, url, archive_bytes - tail_size, archive_bytes - 1)
    marker = tail.rfind(EOCD)
    if marker < 0 or marker + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory record was not found")
    values = struct.unpack_from("<4s4H2LH", tail, marker)
    _, disk, central_disk, entries_disk, entries_total, central_bytes, central_offset, _ = values
    if disk != 0 or central_disk != 0 or entries_disk != entries_total:
        raise ValueError("multi-disk ZIP archives are unsupported")
    if entries_total == 0xFFFF or central_bytes == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 archive indexing requires explicit review")
    if tail_size + central_bytes > maximum_metadata_bytes:
        raise ValueError("central-directory inspection exceeds the metadata cap")
    central = _range_get(
        session, url, central_offset, central_offset + central_bytes - 1
    )
    members = []
    cursor = 0
    while cursor < len(central):
        if central[cursor : cursor + 4] != CENTRAL:
            raise ValueError("invalid ZIP central-directory entry")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        method = fields[4]
        crc32 = fields[7]
        compressed_bytes = fields[8]
        uncompressed_bytes = fields[9]
        name_bytes, extra_bytes, comment_bytes = fields[10:13]
        local_offset = fields[16]
        start = cursor + 46
        name = central[start : start + name_bytes].decode("utf-8", errors="strict")
        members.append(
            RemoteZipMember(
                name=name,
                compressed_bytes=compressed_bytes,
                uncompressed_bytes=uncompressed_bytes,
                crc32=crc32,
                compression_method=method,
                local_header_offset=local_offset,
            )
        )
        cursor = start + name_bytes + extra_bytes + comment_bytes
    if len(members) != entries_total:
        raise ValueError("central-directory entry count mismatch")
    return {
        "archive_bytes": archive_bytes,
        "accept_ranges": "bytes",
        "central_directory_offset": central_offset,
        "central_directory_bytes": central_bytes,
        "metadata_range_bytes_consumed": tail_size + central_bytes,
        "member_count": len(members),
        "members": [asdict(item) for item in members],
    }


def download_member(
    session: requests.Session, url: str, member: RemoteZipMember
) -> bytes:
    header = _range_get(
        session, url, member.local_header_offset, member.local_header_offset + 29
    )
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != LOCAL:
        raise ValueError(f"invalid local ZIP header for {member.name}")
    flags, method, name_bytes, extra_bytes = fields[2], fields[3], fields[9], fields[10]
    if flags & 0x1:
        raise ValueError(f"encrypted ZIP member is unsupported: {member.name}")
    if method != member.compression_method:
        raise ValueError(f"compression method mismatch for {member.name}")
    data_start = member.local_header_offset + 30 + name_bytes + extra_bytes
    if member.compressed_bytes == 0:
        compressed = b""
    else:
        compressed = _range_get(
            session,
            url,
            data_start,
            data_start + member.compressed_bytes - 1,
        )
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported compression method {method}: {member.name}")
    if len(payload) != member.uncompressed_bytes:
        raise ValueError(f"uncompressed length mismatch for {member.name}")
    if binascii.crc32(payload) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"CRC mismatch for {member.name}")
    return payload


def download_contiguous_members(
    session: requests.Session,
    url: str,
    members: list[RemoteZipMember],
    *,
    archive_bytes: int,
    maximum_range_bytes: int,
) -> tuple[dict[str, bytes], int]:
    if not members:
        return {}, 0
    start = min(item.local_header_offset for item in members)
    end = min(
        archive_bytes - 1,
        max(item.local_header_offset + item.compressed_bytes + 131_100 for item in members),
    )
    if end - start + 1 > maximum_range_bytes:
        raise ValueError("contiguous member range exceeds its byte cap")
    block = _range_get(session, url, start, end)
    payloads = {}
    for member in members:
        cursor = member.local_header_offset - start
        fields = struct.unpack_from("<4s5H3L2H", block, cursor)
        if fields[0] != LOCAL or fields[2] & 0x1:
            raise ValueError(f"invalid or encrypted ZIP member: {member.name}")
        method, name_bytes, extra_bytes = fields[3], fields[9], fields[10]
        data_start = cursor + 30 + name_bytes + extra_bytes
        compressed = block[data_start : data_start + member.compressed_bytes]
        payload = compressed if method == 0 else zlib.decompress(compressed, -zlib.MAX_WBITS)
        if len(payload) != member.uncompressed_bytes or binascii.crc32(payload) & 0xFFFFFFFF != member.crc32:
            raise ValueError(f"length or CRC mismatch for {member.name}")
        payloads[member.name] = payload
    return payloads, len(block)
