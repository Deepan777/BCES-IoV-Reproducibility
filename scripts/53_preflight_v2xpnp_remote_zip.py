#!/usr/bin/env python3
"""Read only public V2XPnP ZIP directory bytes through bounded HTTP ranges.

No archive payloads, BCES outputs, or labels are read. Publisher SHA-1 is
metadata only here; full archive integrity is NOT verified by this preflight.
"""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import PurePosixPath
import re
import zipfile

import requests


SOURCES = {
    "val": ("0p1asmx3ueh06hxg6ndzubhk695gbge9", "1820532498186", 9_214_934_051,
            "ffe7bc07a4bd799d251fb9ff76e72d36d3f1f267"),
    "test": ("9ef9sl5hfp64hqypkmx0pub9ega5u37n", "1820530882960", 26_133_240_810,
             "b10f54d2437f288b1803190824745a537f95c6c0"),
    "map": ("eapz852kkjzov95gxoxl6p613u63j14s", "1820521730296", 4_386_892,
            "7b1c467178b4663c840f1c82c2b66888c3dc59bb"),
    "train1": ("zfbeizdrt9pfayf3oc3zi8fc9d1osjsd", "1820540096649", 22_310_216_171,
               "c07c51dfe73c52f279a09a80a17d282a8c2b7d58"),
    "train2": ("zxwmm5ohx0xw60wafe9b14kw8uzfyvfl", "1820539381910", 20_762_957_440,
               "035385581fe22b6005a3f5d1c730ec9348081ed5"),
    "train3": ("pcus09ic7xm87smz2ndjnr27sl67hymf", "1820535640282", 19_795_370_955,
               "ad6011596ac021e58d9064536bd18ce9105bf9e1"),
    "train4": ("b4m4mi2ulls2l5dl3wd5f0ikabn010zs", "1820526259082", 15_240_989_930,
               "5d528e51c3bda9f687d702eff96ea8d70da61ff6"),
}
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 48 * 1024 * 1024
INTERESTING = re.compile(r"traj|track|map|observ|visib|label|annot|info|pose|route|sensor", re.I)


class BoundedRemoteZip(io.RawIOBase):
    def __init__(self, share: str, file_id: str, size: int):
        self.url = ("https://ucla.app.box.com/index.php?rm=box_download_shared_file"
                    f"&shared_name={share}&file_id=f_{file_id}")
        self.size = size
        self.position = 0
        self.transferred = 0
        self.requests = 0
        self.session = requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid seek mode")
        if position < 0:
            raise ValueError("negative seek")
        self.position = position
        return position

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self.position
        length = min(length, max(self.size - self.position, 0))
        if not length:
            return b""
        blocks = []
        while length:
            count = min(length, MAX_REQUEST_BYTES)
            if self.transferred + count > MAX_TOTAL_BYTES:
                raise RuntimeError("remote preflight transfer cap reached")
            start, end = self.position, self.position + count - 1
            response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                        timeout=90, stream=True)
            try:
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status_code != 206 or response.headers.get("content-range", "").lower() != expected:
                    raise RuntimeError(f"range rejected: HTTP {response.status_code}, "
                                       f"content-range={response.headers.get('content-range')!r}")
                block = response.content
                if len(block) != count:
                    raise RuntimeError("short remote range")
            finally:
                response.close()
            blocks.append(block)
            self.position += count
            self.transferred += count
            self.requests += 1
            length -= count
        return b"".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=sorted(SOURCES))
    args = parser.parse_args()
    share, file_id, size, publisher_sha1 = SOURCES[args.split]
    source = BoundedRemoteZip(share, file_id, size)
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        unsafe = [name for name in names if PurePosixPath(name).is_absolute()
                  or ".." in PurePosixPath(name).parts or "\\" in name or ":" in name]
        if duplicates or unsafe:
            raise RuntimeError(f"unsafe member names: duplicate={duplicates[:5]}, unsafe={unsafe[:5]}")
        result = {
            "status": "PUBLIC_REMOTE_ZIP_DIRECTORY_ONLY_NOT_FULL_ARCHIVE_VERIFICATION",
            "split": args.split,
            "publisher_share": f"https://ucla.box.com/s/{share}",
            "publisher_file_id": file_id,
            "publisher_size_bytes": size,
            "publisher_sha1_unverified": publisher_sha1,
            "range_requests": source.requests,
            "range_bytes_transferred": source.transferred,
            "member_count": len(members),
            "member_compressed_bytes": sum(item.compress_size for item in members),
            "extensions": dict(Counter(PurePosixPath(name).suffix.lower() for name in names if not name.endswith("/"))),
            "top_directories": dict(Counter(PurePosixPath(name).parts[0] for name in names if name)),
            "scene_directories": dict(Counter(PurePosixPath(name).parts[1] for name in names
                                              if len(PurePosixPath(name).parts) >= 2
                                              and PurePosixPath(name).parts[1] != "")),
            "first_members": names[:30],
            "interesting_members": [
                {"name": item.filename, "bytes": item.file_size,
                 "compressed_bytes": item.compress_size}
                for item in members if INTERESTING.search(item.filename)
            ][:200],
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
