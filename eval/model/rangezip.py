"""Read selected members of a (split) zip archive over HTTP byte ranges.

Owner: Shreeya (Role 1). Part S7d.

FSD50K ships its eval audio as a split zip (FSD50K.eval_audio.z01 + .zip,
6.3 GB). The usual route, downloading both parts and re-joining them with
``zip -s 0``, needs ~12.5 GB. Zenodo serves HTTP Range requests, so this
reads the central directory from the end of the last part and then fetches
only the wanted members' bytes, at their (disk, offset) positions. Nothing
but the kept files is written to disk.

Supports: split/multi-disk archives (parts given in disk order), zip64,
members stored (method 0) or deflated (8), members that straddle a part
boundary. Every member's CRC-32 is checked.

    rz = RangeZip([(url_z01, size_z01), (url_zip, size_zip)])
    entries = rz.entries()                       # {name: Entry}
    for name, data in rz.fetch([entries[n] for n in wanted]):
        ...
"""

from __future__ import annotations

import struct
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

USER_AGENT = "soundsense-eval"
EOCD_SIG, EOCD64_SIG, LOC64_SIG = b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07"
CDIR_SIG, LOCAL_SIG = b"PK\x01\x02", b"PK\x03\x04"


@dataclass(frozen=True)
class Entry:
    name: str
    method: int  # 0 stored, 8 deflated
    crc: int
    comp_size: int
    size: int
    disk: int
    offset: int  # local header offset within its disk


Fetcher = Callable[[int, int, int], bytes]
"""fetch(part_index, start, end_inclusive) -> bytes."""


def _rate_limit_wait(headers, default: float) -> float:
    """Seconds to wait after HTTP 429: Retry-After, else until X-RateLimit-Reset (epoch s)."""
    if headers.get("Retry-After"):
        return float(headers["Retry-After"])
    reset = headers.get("X-RateLimit-Reset")
    if reset:
        return max(1.0, float(reset) - time.time() + 1.0)
    return default


def http_fetcher(urls: list[str], retries: int = 10, stats: dict | None = None) -> Fetcher:
    """Range GETs with retries; on HTTP 429 waits for the rate-limit window to reset."""

    def fetch(part: int, start: int, end: int) -> bytes:
        req = urllib.request.Request(urls[part], headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        delay = 2.0
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"server ignored the Range header (HTTP {resp.status})")
                    data = resp.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short read: {len(data)} of {end - start + 1} bytes")
                if stats is not None:
                    stats["bytes"] = stats.get("bytes", 0) + len(data)
                    stats["requests"] = stats.get("requests", 0) + 1
                return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise
                wait = _rate_limit_wait(exc.headers, delay) if exc.code == 429 else delay
            except (urllib.error.URLError, TimeoutError, RuntimeError, ConnectionError):
                if attempt == retries - 1:
                    raise
                wait = delay
            time.sleep(wait)
            delay = min(delay * 2, 60)
        raise RuntimeError("unreachable")

    return fetch


class RangeZip:
    def __init__(self, parts: list[tuple[str, int]], fetcher: Fetcher | None = None, stats: dict | None = None):
        """``parts``: (url, size) per disk, in disk order (.z01, .z02, ..., .zip last)."""
        self.sizes = [size for _, size in parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.stats = stats if stats is not None else {}
        self._fetch = fetcher or http_fetcher([url for url, _ in parts], stats=self.stats)
        self._entries: dict[str, Entry] | None = None

    # --- reading across parts ------------------------------------------------------

    def read(self, disk: int, offset: int, length: int) -> bytes:
        """``length`` bytes from (disk, offset), continuing into later parts if needed."""
        out = bytearray()
        pos = self.starts[disk] + offset
        while length > 0:
            part = max(i for i, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[part]
            n = min(length, self.sizes[part] - local)
            if n <= 0:
                raise ValueError("read past the end of the archive")
            out += self._fetch(part, local, local + n - 1)
            pos += n
            length -= n
        return bytes(out)

    # --- central directory -------------------------------------------------------

    def entries(self) -> dict[str, Entry]:
        if self._entries is None:
            self._entries = self._read_central_directory()
        return self._entries

    def _read_central_directory(self) -> dict[str, Entry]:
        last = len(self.sizes) - 1
        tail_len = min(self.sizes[last], 1 << 16)
        tail = self.read(last, self.sizes[last] - tail_len, tail_len)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise ValueError("no end-of-central-directory record: not a zip")
        (_, _disk, cd_disk, _n_disk, n_total, cd_size, cd_offset, _clen) = struct.unpack("<4sHHHHIIH", tail[i:i + 22])
        if i >= 20 and tail[i - 20:i - 16] == LOC64_SIG:  # zip64
            _, z64_disk, z64_off, _disks = struct.unpack("<4sIQI", tail[i - 20:i])
            rec = self.read(z64_disk, z64_off, 56)
            if rec[:4] != EOCD64_SIG:
                raise ValueError("bad zip64 end-of-central-directory record")
            (_, _sz, _vm, _vn, _disk, cd_disk, _n_disk, n_total, cd_size, cd_offset) = struct.unpack("<4sQHHIIQQQQ", rec)
        cd = self.read(cd_disk, cd_offset, cd_size)
        entries, p = {}, 0
        for _ in range(n_total):
            if cd[p:p + 4] != CDIR_SIG:
                raise ValueError(f"bad central directory entry at {p}")
            (_, _vm, _vn, _flags, method, _t, _d, crc, csize, usize, nlen, xlen, clen, disk, _ia, _ea, off) = \
                struct.unpack("<4sHHHHHHIIIHHHHHII", cd[p:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            usize, csize, off, disk = _zip64_extra(extra, usize, csize, off, disk)
            entries[name] = Entry(name, method, crc, csize, usize, disk, off)
            p += 46 + nlen + xlen + clen
        return entries

    # --- members ---------------------------------------------------------------------

    def _data_start(self, e: Entry, header: bytes) -> int:
        if header[:4] != LOCAL_SIG:
            raise ValueError(f"{e.name}: bad local header")
        nlen, xlen = struct.unpack("<HH", header[26:30])
        return 30 + nlen + xlen

    def fetch(self, entries: Iterable[Entry], max_gap: int = 1 << 20, max_span: int = 64 << 20,
              on_member: Callable[[Entry], None] | None = None) -> Iterator[tuple[str, bytes]]:
        """Yield (name, uncompressed bytes) for each entry, in archive order.

        Nearby members are fetched in one request (gap <= ``max_gap``, span
        <= ``max_span``) to keep the request count low.
        """
        ordered = sorted(entries, key=lambda e: self.starts[e.disk] + e.offset)
        groups: list[list[Entry]] = []
        for e in ordered:
            start = self.starts[e.disk] + e.offset
            if groups:
                g = groups[-1]
                g_start = self.starts[g[0].disk] + g[0].offset
                g_end = self._approx_end(g[-1])
                if start - g_end <= max_gap and self._approx_end(e) - g_start <= max_span:
                    g.append(e)
                    continue
            groups.append([e])
        for g in groups:
            g_start = self.starts[g[0].disk] + g[0].offset
            blob = self._read_global(g_start, self._approx_end(g[-1]) - g_start)
            for e in g:
                rel = self.starts[e.disk] + e.offset - g_start
                head = blob[rel:rel + 30]
                nlen, xlen = struct.unpack("<HH", head[26:30])
                need = rel + 30 + nlen + xlen + e.comp_size
                if need > len(blob):  # local extra field longer than estimated: top up
                    blob += self._read_global(g_start + len(blob), need - len(blob))
                data_start = rel + self._data_start(e, blob[rel:rel + 30])
                yield e.name, _decompress(e, blob[data_start:data_start + e.comp_size])
                if on_member:
                    on_member(e)

    def _approx_end(self, e: Entry) -> int:
        """Global end of a member: header (30 + name + generous extra) + data."""
        return self.starts[e.disk] + e.offset + 30 + len(e.name.encode()) + 64 + e.comp_size

    def _read_global(self, pos: int, length: int) -> bytes:
        total = sum(self.sizes)
        length = min(length, total - pos)
        part = max(i for i, s in enumerate(self.starts) if s <= pos)
        return self.read(part, pos - self.starts[part], length)


def _zip64_extra(extra: bytes, usize: int, csize: int, off: int, disk: int) -> tuple[int, int, int, int]:
    p = 0
    while p + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[p:p + 4])
        if hid == 0x0001:
            q = p + 4
            vals = []
            for _ in range(hlen // 8):
                vals.append(struct.unpack("<Q", extra[q:q + 8])[0])
                q += 8
            it = iter(vals)
            if usize == 0xFFFFFFFF:
                usize = next(it)
            if csize == 0xFFFFFFFF:
                csize = next(it)
            if off == 0xFFFFFFFF:
                off = next(it)
            if disk == 0xFFFF and hlen % 8 == 4:
                disk = struct.unpack("<I", extra[q:q + 4])[0]
            break
        p += 4 + hlen
    return usize, csize, off, disk


def _decompress(e: Entry, raw: bytes) -> bytes:
    if e.method == 0:
        data = raw
    elif e.method == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise ValueError(f"{e.name}: unsupported compression method {e.method}")
    if len(data) != e.size or zlib.crc32(data) != e.crc:
        raise ValueError(f"{e.name}: CRC/size mismatch (corrupt download?)")
    return data
