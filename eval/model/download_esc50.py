"""Download ESC-50 and keep only the categories SoundSense evaluates.

Owner: Shreeya (Role 1).

ESC-50 (Piczak, 2015; https://github.com/karolpiczak/ESC-50) is 2000 five-second
clips in 50 categories, licensed CC BY-NC 3.0 (non-commercial, attribution).
The official download is the repository archive (~600 MB). This script:
  1. downloads it to data/esc50/_download/ (or uses --zip if you already have it),
  2. extracts only the clips of the categories in eval/model/datasets.py
     (targets mapped to our category IDs, plus negatives for false-alert tests),
  3. writes data/esc50/audio/*.wav and a filtered data/esc50/meta/esc50.csv
     (same columns as the original, so build_background_bank.py can read it),
  4. deletes the archive (unless --keep-zip).

    python -m eval.model.download_esc50            # download + extract (~600 MB download, 1240 clips ~550 MB kept)
    python -m eval.model.download_esc50 --list     # just print the category mapping
    python -m eval.model.download_esc50 --zip ~/Downloads/ESC-50-master.zip

data/ is git-ignored: never commit the audio.
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from eval.model.datasets import ESC50_KEEP, ESC50_NEGATIVES, ESC50_ROOT, ESC50_TO_CATEGORY, esc50_label

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
META_MEMBER = "meta/esc50.csv"


def print_mapping(out=None) -> None:
    out = out or sys.stdout
    print("ESC-50 category -> SoundSense category ID", file=out)
    for esc, ours in sorted(ESC50_TO_CATEGORY.items()):
        print(f"  {esc:18s} -> {ours}", file=out)
    print("  (no ESC-50 class for: doorbell, pressure_cooker, smoke_alarm, appliance_beep;"
          " use data/indian/)", file=out)
    print("\nESC-50 negative categories (label 'negative', for false-alert testing)", file=out)
    for esc, why in sorted(ESC50_NEGATIVES.items()):
        print(f"  {esc:18s}    {why}", file=out)


def download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (via a .part file), printing progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "soundsense-eval"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(part, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = done * 100 // total if total else -1
            if pct != last and (pct % 5 == 0 or pct < 0):
                size = f"{done / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB ({pct}%)" if total else "")
                print(f"\r  downloading {size}", end="", flush=True)
                last = pct
    print()
    part.replace(dest)
    return dest


def extract(zip_path: Path, out: Path, keep: set[str] | None) -> Counter:
    """Extract the kept categories' clips + a filtered meta CSV. Returns clips per category."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        meta = next((n for n in names if n.endswith(META_MEMBER)), None)
        if meta is None:
            raise SystemExit(f"{zip_path}: no {META_MEMBER} inside; is this the ESC-50 archive?")
        prefix = meta[: -len(META_MEMBER)]  # e.g. "ESC-50-master/"
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(meta), encoding="utf-8")))
        fields = list(rows[0].keys()) if rows else []
        kept = [r for r in rows if keep is None or r["category"] in keep]
        missing = sorted((keep or set()) - {r["category"] for r in rows})
        if missing:
            print(f"  warning: not in ESC-50: {', '.join(missing)}")

        (out / "audio").mkdir(parents=True, exist_ok=True)
        (out / "meta").mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(kept, 1):
            target = out / "audio" / r["filename"]
            if not target.exists():
                with z.open(f"{prefix}audio/{r['filename']}") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if i % 100 == 0 or i == len(kept):
                print(f"\r  extracted {i}/{len(kept)} clips", end="", flush=True)
        print()
        with open(out / "meta" / "esc50.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        for extra in ("LICENSE", "README.md"):  # keep the licence/attribution next to the audio
            if f"{prefix}{extra}" in names:
                (out / f"ESC50_{extra}").write_bytes(z.read(f"{prefix}{extra}"))
    return Counter(r["category"] for r in kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESC-50 and keep the SoundSense categories.")
    parser.add_argument("--out", type=Path, default=ESC50_ROOT, help="output folder (default data/esc50)")
    parser.add_argument("--zip", type=Path, help="use an already-downloaded ESC-50 archive")
    parser.add_argument("--url", default=ESC50_URL)
    parser.add_argument("--all", action="store_true", help="keep all 50 categories")
    parser.add_argument("--keep-zip", action="store_true", help="don't delete the downloaded archive")
    parser.add_argument("--list", action="store_true", help="print the category mapping and exit")
    args = parser.parse_args()

    print_mapping()
    if args.list:
        return
    print()
    zip_path = args.zip
    downloaded = zip_path is None
    if downloaded:
        zip_path = args.out / "_download" / "ESC-50-master.zip"
        if zip_path.exists():
            print(f"Using {zip_path} from an earlier run")
        else:
            print(f"Downloading {args.url} (~600 MB)")
            download(args.url, zip_path)
    counts = extract(zip_path, args.out, None if args.all else ESC50_KEEP)

    print(f"\nKept {sum(counts.values())} clips in {args.out}:")
    for esc, n in sorted(counts.items()):
        print(f"  {esc:18s} {n:3d}  -> {esc50_label(esc)}")
    if downloaded and not args.keep_zip:
        shutil.rmtree(zip_path.parent)
        print("Deleted the downloaded archive (use --keep-zip to keep it).")
    print("ESC-50 is CC BY-NC 3.0: non-commercial use, cite Piczak (2015). See ESC50_LICENSE.")


if __name__ == "__main__":
    main()
