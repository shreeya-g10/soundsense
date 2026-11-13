"""Download a small FSD50K eval subset by HTTP range requests (no full archive on disk).

Owner: Shreeya (Role 1). Part S7d.

FSD50K (Fonseca et al., 2022; Zenodo record 4060432): Freesound clips
labelled with 200 AudioSet-ontology classes. The eval audio is a 6.3 GB split
zip (.z01 + .zip); re-joining it would need ~12.5 GB. Instead this reads the
archive's central directory over HTTP ranges and fetches only the selected
clips (eval/model/rangezip.py), converting each to 16 kHz mono 16-bit WAV in
memory. Peak disk = the kept clips (~380 MB); nothing else is written.

Selected (eval split only):
  targets        clips of the FSD50K classes in FSD50K_TO_CATEGORY, mapped to
                 our category IDs; clips that map to two of our categories
                 are skipped; Crying_and_sobbing counts as baby_cry only if
                 the Freesound title/tags say baby/infant/newborn
  TV backgrounds up to 300 Speech and 300 Music clips (seed 0) whose labels
                 contain no target-like class: the TV-guard test sound (FSD50K
                 has no Television class). A clip labelled both is used once,
                 as speech.
  licences       CC0, CC-BY, CC-BY-NC (Sampling+ is skipped); each clip's
                 licence, uploader, title and Freesound URL go to metadata.csv

    python -m eval.model.download_fsd50k --list     # mapping + counts + sizes, downloads metadata only
    python -m eval.model.download_fsd50k            # ~1 GB of range requests, ~380 MB kept

Writes data/fsd50k/audio/<fsd50k id>.wav, data/fsd50k/metadata.csv and
data/fsd50k/FSD50K_LICENSE_AND_ATTRIBUTION.txt. data/ is git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contracts.types import (
    APPLIANCE_BEEP,
    BABY_CRY,
    DOG_BARK,
    DOOR_KNOCK,
    DOORBELL,
    GLASS_BREAK,
    WATER_RUNNING,
)
from eval.model.datasets import FSD50K_ROOT
from eval.model.rangezip import USER_AGENT, RangeZip

RECORD = 4060432
API = f"https://zenodo.org/api/records/{RECORD}"
CONTENT = API + "/files/{}/content"
AUDIO_PARTS = ("FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip")  # disk order
SAMPLE_RATE = 16000

# FSD50K class -> our category ID (exact class names from FSD50K vocabulary.csv).
FSD50K_TO_CATEGORY: dict[str, str] = {
    "Doorbell": DOORBELL,
    "Knock": DOOR_KNOCK,  # any knock, not only on doors
    "Shatter": GLASS_BREAK,  # "Glass" alone is mostly clinking, not breaking
    "Bark": DOG_BARK,
    "Crying_and_sobbing": BABY_CRY,  # only clips whose title/tags mention a baby (see BABY)
    "Microwave_oven": APPLIANCE_BEEP,  # hum + beeps; FSD50K has no Beep class
    "Water_tap_and_faucet": WATER_RUNNING,
    "Sink_(filling_or_washing)": WATER_RUNNING,
}
NOT_IN_FSD50K = {
    "smoke_alarm": "no smoke detector class; generic 'Alarm' (clock/car/other alarms) is not used",
    "pressure_cooker": "no whistle or steam-whistle class; generic 'Hiss' is not used",
    "television": "no Television class; Speech and Music stand in for TV sound",
}
BABY = re.compile(r"\b(baby|babies|infant|infants|newborn)\b", re.I)
TV_CLASSES = {"Speech": "speech", "Music": "music"}
TV_PER_CLASS = 300
# A TV background must not contain anything that could legitimately alert.
NOT_IN_TV = set(FSD50K_TO_CATEGORY) | {"Glass", "Alarm", "Door", "Dog", "Hiss", "Ringtone", "Chime", "Bell"}
LICENSES = ("CC0", "CC-BY", "CC-BY-NC")
METADATA_FIELDS = ["fname", "file", "role", "our_label", "fsd50k_labels", "license", "license_url",
                   "uploader", "title", "freesound_url"]


@dataclass(frozen=True)
class Pick:
    fname: str
    role: str  # "target" | "tv_background"
    our_label: str  # category ID, or "speech"/"music"
    labels: tuple[str, ...]
    license: str
    license_url: str
    uploader: str
    title: str


def short_license(url: str) -> str:
    u = url.lower()
    if "publicdomain/zero" in u:
        return "CC0"
    if "/by-nc/" in u:
        return "CC-BY-NC"
    if "sampling" in u:
        return "Sampling+"
    if "/by/" in u:
        return "CC-BY"
    return url


def select(eval_rows: list[dict], clip_info: dict, licenses=LICENSES, tv_per_class: int = TV_PER_CLASS,
           seed: int = 0) -> tuple[list[Pick], Counter]:
    """The clips to keep, and why others were skipped (Counter of reasons)."""
    picks, skipped = [], Counter()

    def make(row, role, label):
        meta = clip_info[row["fname"]]
        return Pick(row["fname"], role, label, tuple(row["labels"].split(",")), short_license(meta["license"]),
                    meta["license"], meta.get("uploader", ""), meta.get("title", ""))

    tv_pool: dict[str, list[dict]] = {c: [] for c in TV_CLASSES}
    for row in eval_rows:
        labels = row["labels"].split(",")
        meta = clip_info[row["fname"]]
        lic = short_license(meta["license"])
        cats = {FSD50K_TO_CATEGORY[l] for l in labels if l in FSD50K_TO_CATEGORY}
        if BABY_CRY in cats and not BABY.search(meta.get("title", "") + " " + " ".join(meta.get("tags", []))):
            cats.discard(BABY_CRY)
            skipped["Crying_and_sobbing without baby/infant in title or tags"] += 1
        if len(cats) > 1:
            skipped["maps to 2+ of our categories"] += 1
            continue
        if cats:
            if lic not in licenses:
                skipped[f"licence {lic}"] += 1
                continue
            picks.append(make(row, "target", cats.pop()))
            continue
        for cls in TV_CLASSES:
            if cls in labels and not NOT_IN_TV & set(labels) and lic in licenses:
                tv_pool[cls].append(row)
    rng = random.Random(seed)
    taken = {p.fname for p in picks}
    for cls, rows in tv_pool.items():
        rows = sorted(rows, key=lambda r: r["fname"])
        for row in rng.sample(rows, min(tv_per_class, len(rows))):
            if row["fname"] in taken:  # labelled both Speech and Music: use it once (as the first class)
                skipped["in both Speech and Music samples (kept once)"] += 1
                continue
            taken.add(row["fname"])
            picks.append(make(row, "tv_background", TV_CLASSES[cls]))
    return picks, skipped


def zenodo_sizes() -> dict[str, int]:
    req = urllib.request.Request(API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return {f["key"]: int(f["size"]) for f in json.load(resp)["files"]}


def fetch_metadata(tmp: Path) -> tuple[list[dict], dict, list[str], dict[str, bytes]]:
    """(eval.csv rows, eval clip info, vocabulary class names, doc files) from the small metadata zips."""
    out = {}
    for key in ("FSD50K.ground_truth.zip", "FSD50K.metadata.zip", "FSD50K.doc.zip"):
        path = tmp / key
        req = urllib.request.Request(CONTENT.format(key), headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
            shutil.copyfileobj(resp, f)
        out[key] = zipfile.ZipFile(path)
    gt, md, doc = out["FSD50K.ground_truth.zip"], out["FSD50K.metadata.zip"], out["FSD50K.doc.zip"]
    find = lambda z, suffix: next(n for n in z.namelist() if n.endswith(suffix))
    rows = list(csv.DictReader(io.TextIOWrapper(gt.open(find(gt, "eval.csv")), encoding="utf-8")))
    vocab = [r[1] for r in csv.reader(io.TextIOWrapper(gt.open(find(gt, "vocabulary.csv")), encoding="utf-8"))]
    info = json.load(md.open(find(md, "eval_clips_info_FSD50K.json")))
    docs = {Path(n).name: doc.read(n) for n in doc.namelist() if not n.endswith("/")}
    for z in out.values():
        z.close()
    return rows, info, vocab, docs


def print_mapping(vocab: list[str]) -> None:
    print("FSD50K class -> SoundSense category (exact FSD50K names, all found in vocabulary.csv)")
    for cls, ours in FSD50K_TO_CATEGORY.items():
        print(f"  {cls:28s} -> {ours}" + ("" if cls in vocab else "   (NOT FOUND in vocabulary!)"))
    for cls, ours in TV_CLASSES.items():
        print(f"  {cls:28s} -> TV-guard background '{ours}'" + ("" if cls in vocab else "   (NOT FOUND!)"))
    print("Not available in FSD50K:")
    for ours, why in NOT_IN_FSD50K.items():
        print(f"  {ours:16s} {why}")


def to_16k_wav(data: bytes, sr: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf
    import soxr

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != sr:
        audio = soxr.resample(audio, rate, sr).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def attribution_text(picks: list[Pick], docs: dict[str, bytes]) -> str:
    lic = Counter(p.license for p in picks)
    return (
        "FSD50K subset used by SoundSense for evaluation only.\n\n"
        "Dataset: E. Fonseca, X. Favory, J. Pons, F. Font, X. Serra, \"FSD50K: An Open Dataset of "
        "Human-Labeled Sound Events\", IEEE/ACM TASLP 30, 2022. Zenodo record 4060432 "
        "(https://zenodo.org/records/4060432). The dataset as a whole is CC BY 4.0; every clip keeps its "
        "own Freesound licence, listed per clip in metadata.csv (license, license_url, uploader, title, "
        "freesound_url).\n\n"
        f"Licences in this subset: {', '.join(f'{k} {v}' for k, v in sorted(lic.items()))}. "
        "CC-BY / CC-BY-NC clips require attribution (uploader + Freesound URL); CC-BY-NC clips may not be "
        "used commercially. Sampling+ clips were not downloaded. Audio converted to 16 kHz mono 16-bit WAV.\n"
        + ("\n--- FSD50K documentation files kept: " + ", ".join(sorted(docs)) + "\n" if docs else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the SoundSense FSD50K eval subset (range requests).")
    parser.add_argument("--out", type=Path, default=FSD50K_ROOT)
    parser.add_argument("--licenses", nargs="+", default=list(LICENSES), choices=["CC0", "CC-BY", "CC-BY-NC", "Sampling+"])
    parser.add_argument("--tv-per-class", type=int, default=TV_PER_CLASS)
    parser.add_argument("--max-kept-gb", type=float, default=2.0, help="refuse if the kept subset would exceed this")
    parser.add_argument("--list", action="store_true", help="print the mapping, counts and sizes; download no audio")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:  # ~7 MB of metadata, deleted on exit
        rows, info, vocab, docs = fetch_metadata(Path(tmp))
    print_mapping(vocab)
    picks, skipped = select(rows, info, tuple(args.licenses), args.tv_per_class)

    sizes = zenodo_sizes()
    stats: dict = {}
    rz = RangeZip([(CONTENT.format(k), sizes[k]) for k in AUDIO_PARTS], stats=stats)
    entries = {Path(n).stem: e for n, e in rz.entries().items() if n.endswith(".wav")}
    raw = sum(entries[p.fname].size for p in picks)
    kept_est = raw * SAMPLE_RATE / 44100  # 44.1 kHz 16-bit mono -> 16 kHz 16-bit mono
    transfer = sum(entries[p.fname].comp_size for p in picks)

    counts = Counter((p.role, p.our_label) for p in picks)
    print(f"\nSelected {len(picks)} eval clips (licences {', '.join(args.licenses)}):")
    for (role, label), n in sorted(counts.items()):
        print(f"  {role:14s} {label:16s} {n:4d}")
    print("Skipped: " + ", ".join(f"{k} {v}" for k, v in skipped.most_common()))
    print(f"Transfer ~{transfer / 1e6:.0f} MB (range requests); kept ~{kept_est / 1e6:.0f} MB at 16 kHz "
          f"(raw 44.1 kHz would be {raw / 1e6:.0f} MB)")
    if args.list:
        return
    if kept_est > args.max_kept_gb * 1e9:
        raise SystemExit(f"Refusing: ~{kept_est / 1e9:.2f} GB would exceed --max-kept-gb {args.max_kept_gb}")
    free = shutil.disk_usage(args.out.parent if args.out.parent.exists() else Path.home()).free
    if free < kept_est * 1.5 + 1e9:
        raise SystemExit(f"Refusing: only {free / 1e9:.1f} GB free")

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in picks if not (audio_dir / f"{p.fname}.wav").exists()]
    by_name = {entries[p.fname].name: p for p in todo}
    done, t0, written = 0, time.monotonic(), 0
    for name, data in rz.fetch([entries[p.fname] for p in todo]):
        p = by_name[name]
        wav = to_16k_wav(data)
        tmp_path = audio_dir / f".{p.fname}.wav.part"
        tmp_path.write_bytes(wav)
        tmp_path.replace(audio_dir / f"{p.fname}.wav")
        written += len(wav)
        done += 1
        if done % 50 == 0 or done == len(todo):
            print(f"\r  {done}/{len(todo)} clips, {stats.get('bytes', 0) / 1e6:.0f} MB fetched in "
                  f"{stats.get('requests', 0)} requests, {written / 1e6:.0f} MB written "
                  f"({time.monotonic() - t0:.0f} s)", end="", flush=True)
    print()
    with open(args.out / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        for p in picks:
            w.writerow({"fname": p.fname, "file": f"audio/{p.fname}.wav", "role": p.role, "our_label": p.our_label,
                        "fsd50k_labels": ",".join(p.labels), "license": p.license, "license_url": p.license_url,
                        "uploader": p.uploader, "title": p.title,
                        "freesound_url": f"https://freesound.org/s/{p.fname}/"})
    (args.out / "FSD50K_LICENSE_AND_ATTRIBUTION.txt").write_text(attribution_text(picks, docs), encoding="utf-8")
    for name, content in docs.items():
        (args.out / f"FSD50K_{name}").write_bytes(content)
    kept = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"Done: {len(picks)} clips, {kept / 1e6:.0f} MB in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
