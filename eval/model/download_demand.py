"""Download DEMAND noise (kitchen, living room, washing) for the noise-robustness tests.

Owner: Shreeya (Role 1). Part S7d.

DEMAND (Thiemann, Ito & Vincent, 2013; Zenodo record 1227121, CC BY 4.0):
16-microphone, 5-minute recordings of real environments. Only DKITCHEN,
DLIVING and DWASHING at 16 kHz, and only 2 of the 16 channels each (ch01 and
ch09: mics on opposite sides of the array, so two different noise samples)
are fetched, by HTTP range requests from inside the zips (eval/model/rangezip.py).
No zip is ever written to disk: ~40 MB transferred, ~58 MB kept.

    python -m eval.model.download_demand
    python -m eval.model.download_demand --channels 1 5 9 13

Writes data/demand/<ENV>/chNN.wav and data/demand/DEMAND_LICENSE_AND_ATTRIBUTION.txt.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from eval.model.datasets import DEMAND_ENVS, DEMAND_ROOT
from eval.model.rangezip import USER_AGENT, RangeZip

RECORD = 1227121
API = f"https://zenodo.org/api/records/{RECORD}"
CONTENT = API + "/files/{}/content"
DEFAULT_CHANNELS = (1, 9)

ATTRIBUTION = """DEMAND noise used by SoundSense for evaluation only (noise-robustness mixes).

J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database
(DEMAND): A database of multichannel environmental noise recordings", 21st International Congress on
Acoustics, Montreal, 2013. Zenodo record 1227121 (https://zenodo.org/records/1227121).
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

Kept: {kept} at 16 kHz (original files, unchanged).
"""


def zenodo_sizes() -> dict[str, int]:
    req = urllib.request.Request(API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {f["key"]: int(f["size"]) for f in json.load(resp)["files"]}


def download(out: Path, envs=DEMAND_ENVS, channels=DEFAULT_CHANNELS, sizes: dict | None = None,
             rangezip_factory=None) -> list[Path]:
    sizes = sizes or zenodo_sizes()
    written = []
    for env in envs:
        key = f"{env}_16k.zip"
        stats: dict = {}
        rz = (rangezip_factory or (lambda k: RangeZip([(CONTENT.format(k), sizes[k])], stats=stats)))(key)
        entries = rz.entries()
        wanted = [entries[f"{env}/ch{c:02d}.wav"] for c in channels]
        (out / env).mkdir(parents=True, exist_ok=True)
        todo = [e for e in wanted if not (out / e.name).exists()]
        for name, data in rz.fetch(todo):
            tmp = out / (name + ".part")
            tmp.write_bytes(data)
            tmp.replace(out / name)
            written.append(out / name)
        print(f"  {env}: {', '.join(Path(e.name).name for e in wanted)}"
              + (f" ({stats.get('bytes', 0) / 1e6:.1f} MB fetched)" if stats else ""), flush=True)
    kept = ", ".join(f"{env} ch{', ch'.join(f'{c:02d}' for c in channels)}" for env in envs)
    (out / "DEMAND_LICENSE_AND_ATTRIBUTION.txt").write_text(ATTRIBUTION.format(kept=kept), encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DEMAND kitchen/living/washing noise (16 kHz, 2 channels).")
    parser.add_argument("--out", type=Path, default=DEMAND_ROOT)
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS), help="1-16")
    args = parser.parse_args()
    if not all(1 <= c <= 16 for c in args.channels):
        raise SystemExit("channels are 1-16")
    download(args.out, channels=tuple(args.channels))
    kept = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"Done: {kept / 1e6:.0f} MB in {args.out}. DEMAND is CC BY 4.0 (cite Thiemann et al., 2013).")


if __name__ == "__main__":
    main()
