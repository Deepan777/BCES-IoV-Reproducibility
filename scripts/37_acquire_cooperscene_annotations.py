#!/usr/bin/env python3
"""Acquire central, paired YAML-only windows; never read sensor ZIP members."""
from __future__ import annotations

import binascii
import hashlib
import json
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.data.remote_zip import _range_get
from bces.utils.reproducibility import write_json_atomic

INDEX = ROOT / "data/manifests/cooperscene_scene_remote_index_v1.json"
DEST = ROOT / "data/raw/cooperscene_annotations_v1"
MANIFEST = ROOT / "data/manifests/cooperscene_annotations_v1.json"
FRAMES_PER_TAKE = 60
AGENTS = (0, 1)


def selection(index):
    chosen = []
    for scene in index["scenes"]:
        by_agent = {agent: {} for agent in AGENTS}
        for member in scene["yaml_members"]:
            parts = member["name"].split("/")
            if len(parts) != 3 or not parts[-1].endswith(".yaml"):
                continue
            agent = int(parts[1])
            if agent in by_agent:
                by_agent[agent][int(Path(parts[-1]).stem)] = member
        common = sorted(set.intersection(*(set(d) for d in by_agent.values())))
        if len(common) < FRAMES_PER_TAKE:
            raise ValueError(f"too few paired frames in scene {scene['scene']}")
        offset = (len(common) - FRAMES_PER_TAKE) // 2
        frames = common[offset:offset + FRAMES_PER_TAKE]
        if any(b - a != 1 for a, b in zip(frames, frames[1:])):
            raise ValueError(f"non-contiguous selected frames in scene {scene['scene']}")
        for frame in frames:
            for agent in AGENTS:
                chosen.append((scene, by_agent[agent][frame], frame, agent))
    return chosen


def fetch_one(item):
    scene, member, frame, agent = item
    destination = DEST / scene["split"] / str(scene["scene"]) / str(agent) / f"{frame}.yaml"
    if destination.exists():
        payload = destination.read_bytes()
        if len(payload) != member["uncompressed_bytes"] or binascii.crc32(payload) & 0xFFFFFFFF != member["crc32"]:
            raise ValueError(f"existing file does not match official ZIP CRC: {destination}")
    else:
        session = requests.Session()
        offset = member["local_header_offset"]
        # One bounded range contains the local header, its short filename/extra,
        # and this YAML's compressed bytes. It may include <512 trailing bytes.
        block = _range_get(session, scene["url"], offset,
                           min(scene["archive_bytes"] - 1,
                               offset + 30 + member["compressed_bytes"] + 511))
        fields = struct.unpack_from("<4s5H3L2H", block)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise ValueError(f"invalid/encrypted member {member['name']}")
        method, name_len, extra_len = fields[3], fields[9], fields[10]
        if method != member["compression_method"] or method not in (0, 8):
            raise ValueError(f"unsupported compression {member['name']}")
        name = block[30:30 + name_len].decode("utf-8")
        if name != member["name"]:
            raise ValueError(f"local/central member mismatch: {name}")
        start = 30 + name_len + extra_len
        compressed = block[start:start + member["compressed_bytes"]]
        if len(compressed) != member["compressed_bytes"]:
            raise ValueError(f"range did not include full member {name}")
        payload = compressed if method == 0 else zlib.decompress(compressed, -zlib.MAX_WBITS)
        if len(payload) != member["uncompressed_bytes"] or binascii.crc32(payload) & 0xFFFFFFFF != member["crc32"]:
            raise ValueError(f"ZIP integrity mismatch: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return {"scene": scene["scene"], "split": scene["split"], "frame": frame,
            "agent": agent, "archive_member": member["name"],
            "relative_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    index_bytes = INDEX.read_bytes()
    index = json.loads(index_bytes)
    if index["status"] != "metadata_only_no_member_download" or len(index["scenes"]) != 14:
        raise ValueError("unexpected official scene index")
    chosen = selection(index)
    if sum(member["compressed_bytes"] + 542 for _, member, _, _ in chosen) > 250_000_000:
        raise ValueError("selected ranges exceed single-tranche cap")
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_one, item): item for item in chosen}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 120 == 0:
                print(f"downloaded and CRC-checked {len(records)}/{len(chosen)} YAML records", flush=True)
    records.sort(key=lambda x: (x["scene"], x["frame"], x["agent"]))
    manifest = {"source_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "selection": "central 60 contiguous common frames per take; agents 0 and 1",
                "status": "annotations_acquired_no_bces_outcome_computed",
                "license_url": index["license_url"],
                "file_count": len(records), "total_annotation_bytes": sum(x["bytes"] for x in records),
                "records": records}
    write_json_atomic(MANIFEST, manifest)
    print(f"complete: {len(records)} YAML records, {manifest['total_annotation_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
