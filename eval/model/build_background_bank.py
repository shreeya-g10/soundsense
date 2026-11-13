"""Build the generic background bank from ESC-50 non-target clips (Part S5f).

Owner: Shreeya (Role 1).

Fills the "generic" source of data/profiles/background.npz with YAMNet
embeddings of ESC-50 clips whose classes are NOT SoundSense targets, so
centered scoring and the margin rule know what "ordinary sounds" look like
even before the user's own room has been heard much.

    python -m eval.model.download_esc50          # ESC-50 subset into data/esc50/ (git-ignored)
    python -m eval.model.build_background_bank   # default --esc50 data/esc50
    python -m eval.model.build_background_bank --esc50 ... --per-class 4 --dry-run
    python -m eval.model.build_background_bank --clear     # empty generic bank (the default state)

Only embeddings are written, never audio.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.custom.profile_store import BACKGROUND_CAPS, DEFAULT_ROOT, YamnetProfileStore
from eval.model.datasets import ESC50_ROOT

# ESC-50 classes too close to a SoundSense category to count as background.
EXCLUDE = {
    "dog",              # dog_bark
    "crying_baby",      # baby_cry
    "door_wood_knock",  # door_knock
    "door_wood_creaks",  # "Door" is mapped to door_knock
    "glass_breaking",   # glass_break
    "clock_alarm",      # smoke_alarm / appliance_beep
    "siren",            # alarm-like
    "church_bells",     # doorbell-like chimes
    "water_drops",      # water_running
    "pouring_water",    # water_running
    "toilet_flush",     # water_running
}


def esc50_clips(root: Path, per_class: int, exclude: set[str]) -> dict[str, list[Path]]:
    """Up to ``per_class`` clips of each non-excluded class, from meta/esc50.csv."""
    meta = root / "meta" / "esc50.csv"
    if not meta.exists():
        raise SystemExit(f"{meta} not found; pass the ESC-50 folder that contains meta/ and audio/")
    clips: dict[str, list[Path]] = defaultdict(list)
    with meta.open(newline="") as f:
        for row in sorted(csv.DictReader(f), key=lambda r: (int(r["fold"]), r["filename"])):
            cat = row["category"]
            if cat not in exclude and len(clips[cat]) < per_class:
                clips[cat].append(root / "audio" / row["filename"])
    return dict(clips)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ESC-50 generic background bank.")
    parser.add_argument("--esc50", type=Path, default=ESC50_ROOT,
                        help="ESC-50 folder with meta/esc50.csv and audio/ (default data/esc50)")
    parser.add_argument("--per-class", type=int, default=8, help="clips per ESC-50 class (default 8)")
    parser.add_argument("--exclude", nargs="*", default=sorted(EXCLUDE), help="ESC-50 classes to leave out")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="profile folder")
    parser.add_argument("--clear", action="store_true", help="empty the generic bank and exit")
    parser.add_argument("--dry-run", action="store_true", help="embed and report, but don't save")
    args = parser.parse_args()

    store = YamnetProfileStore(args.root)
    if args.clear:
        store.set_background("generic", np.zeros((0, 1024), np.float32))
        print(f"Generic background bank cleared. Bank now: {store.background_counts()}")
        return

    from backend.ml.demo_yamnet import load_wav

    clips = esc50_clips(args.esc50, args.per_class, set(args.exclude))
    print(f"{len(clips)} ESC-50 classes x up to {args.per_class} clips "
          f"(excluding {len(set(args.exclude))} target-like classes)")
    embeddings = []
    for cat, paths in sorted(clips.items()):
        n = 0
        for path in paths:
            e = store.embed_audio(load_wav(str(path)))
            embeddings.append(e)
            n += len(e)
        print(f"  {cat:20s} {len(paths)} clips, {n} windows")
    bank = np.vstack(embeddings) if embeddings else np.zeros((0, 1024), np.float32)
    cap = BACKGROUND_CAPS["generic"]
    if len(bank) > cap:
        idx = np.random.default_rng(0).choice(len(bank), cap, replace=False)
        bank = bank[np.sort(idx)]
        print(f"Sampled {cap} of the windows (BACKGROUND_CAPS['generic'])")
    if args.dry_run:
        print(f"Dry run: {len(bank)} embeddings, not saved")
        return
    store.set_background("generic", bank)
    print(f"Saved {len(bank)} generic embeddings. Bank now: {store.background_counts()}")


if __name__ == "__main__":
    main()
