#!/usr/bin/env python3
"""Metadata-only index of official CooperScene per-scene ZIP64 archives."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.data.remote_zip import _range_get
from bces.utils.reproducibility import write_json_atomic

BASE = "https://data.ucr.edu/datasets/cooperscene"
SCENES = [("train", i) for i in (1, 3, 4, 5, 6, 7, 9, 10, 11, 12)] + [
    ("validate", i) for i in (13, 14)
] + [("test", i) for i in (2, 8)]
OUT = ROOT / "data/manifests/cooperscene_scene_remote_index_v1.json"


def zip64_extra(extra: bytes, raw_uncompressed: int, raw_compressed: int, raw_offset: int):
    vals = [raw_uncompressed, raw_compressed, raw_offset]
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        blob = extra[cursor + 4:cursor + 4 + size]
        if kind == 1:
            offset = 0
            for j, old in enumerate(vals):
                if old == 0xFFFFFFFF:
                    if offset + 8 > len(blob):
                        raise ValueError("truncated ZIP64 member extra")
                    vals[j] = struct.unpack_from("<Q", blob, offset)[0]
                    offset += 8
            return vals
        cursor += 4 + size
    if 0xFFFFFFFF in vals:
        raise ValueError("missing ZIP64 member extra")
    return vals


def index_one(session: requests.Session, url: str) -> dict:
    head = session.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    length = int(head.headers["Content-Length"])
    if head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise ValueError("server does not advertise ranges")
    tail_size = min(262_144, length)
    tail = _range_get(session, url, length - tail_size, length - 1)
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0:
        raise ValueError("missing EOCD")
    _, disk, c_disk, n_disk, n, size, offset, _ = struct.unpack_from("<4s4H2LH", tail, pos)
    if disk or c_disk or n_disk != n:
        raise ValueError("unsupported multi-disk ZIP")
    locator_absolute = length - tail_size + pos - 20
    if n == 0xFFFF or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        locator = _range_get(session, url, locator_absolute, locator_absolute + 19)
        sig, disk, record_offset, disks = struct.unpack("<4sLQL", locator)
        if sig != b"PK\x06\x07" or disk or disks != 1:
            raise ValueError("invalid ZIP64 locator")
        record = _range_get(session, url, record_offset, record_offset + 55)
        sig, _, _, _, disk, c_disk, n_disk, n, size, offset = struct.unpack("<4sQHHLLQQQQ", record)
        if sig != b"PK\x06\x06" or disk or c_disk or n_disk != n:
            raise ValueError("invalid ZIP64 EOCD")
    if size > 20_000_000:
        raise ValueError("central directory exceeds metadata cap")
    central = _range_get(session, url, offset, offset + size - 1)
    cursor = 0
    yaml_members = []
    suffix_counts = {}
    while cursor < len(central):
        if central[cursor:cursor + 4] != b"PK\x01\x02":
            raise ValueError("invalid central entry")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        method, crc = fields[4], fields[7]
        compressed, uncompressed = fields[8:10]
        name_len, extra_len, comment_len = fields[10:13]
        local_offset = fields[16]
        name_start = cursor + 46
        name = central[name_start:name_start + name_len].decode("utf-8")
        extra = central[name_start + name_len:name_start + name_len + extra_len]
        uncompressed, compressed, local_offset = zip64_extra(extra, uncompressed, compressed, local_offset)
        suffix = Path(name).suffix.lower()
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix == ".yaml":
            yaml_members.append({"name": name, "compressed_bytes": compressed,
                                 "uncompressed_bytes": uncompressed, "crc32": crc,
                                 "compression_method": method,
                                 "local_header_offset": local_offset})
        cursor = name_start + name_len + extra_len + comment_len
    if cursor != size:
        raise ValueError("central-directory length mismatch")
    if sum(suffix_counts.values()) != n:
        raise ValueError("entry count mismatch")
    return {"url": url, "archive_bytes": length, "central_directory_bytes": size,
            "metadata_range_bytes_consumed": tail_size + size + 76,
            "member_count": n, "suffix_counts": suffix_counts,
            "yaml_members": yaml_members}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    session = requests.Session()
    results = []
    for split, scene in SCENES:
        item = index_one(session, f"{BASE}/{split}/{scene}.zip")
        item.update({"split": split, "scene": scene})
        results.append(item)
        print(f"{split}/{scene}: {len(item['yaml_members'])} yaml; "
              f"compressed {sum(m['compressed_bytes'] for m in item['yaml_members'])} bytes", flush=True)
    write_json_atomic(OUT, {"provider_config_url": f"{BASE}/config.json",
                            "license_url": "https://github.com/UCR-CISL/CooperScene/blob/main/DATA_LICENSE",
                            "status": "metadata_only_no_member_download", "scenes": results})


if __name__ == "__main__":
    main()
