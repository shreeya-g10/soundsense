"""Evaluation datasets: our Indian recordings and the ESC-50 subset (loader + layout rules).

Owner: Shreeya (Role 1).

Layout (same as the team Drive folder; everything under data/ is git-ignored):

    data/indian/profile/<category>/<file>            enrolment clips for custom profiles
    data/indian/profile/background/<file>            optional: normal room sound (no target), for
                                                     the background bank. Name it
                                                     background_<name>_<condition>_<distance>_<n> to
                                                     mark whose home it is; any other name = shared
    data/indian/test/<category>/<file>               test clips of a target sound
    data/indian/test/hard_negatives/<type>/<file>    sounds that must NOT alert (mixer, kettle, tv_ads, ...)
    data/esc50/audio/*.wav + data/esc50/meta/esc50.csv   (eval/model/download_esc50.py)
    data/fsd50k/audio/*.wav + data/fsd50k/metadata.csv   (eval/model/download_fsd50k.py)
    data/freesound/<category>/<file> + data/freesound/metadata.csv   (by hand; CC0/CC-BY only)
    data/demand/<ENV>/chNN.wav                           (eval/model/download_demand.py; noise)

<category> is a SoundSense category ID (contracts.types.CATEGORIES). An extra
<condition>/ sub-folder (test/<category>/<condition>/<file>) is also accepted.

File names (the team Drive naming guide): <category>_<name>_<condition>_<distance>_<n>.<ext>
    e.g. pressure_cooker_shreeya_tv_near_2.m4a, door_knock_vaidehi_quiet_far_1.wav,
         hard_negatives/kettle/kettle_sneha_fan_3m_1.mp3 (<type> instead of <category>)
    name       who recorded it: shreeya | vaidehi | sneha (RECORDERS). Each person's home
               is a different device/room, so the name is also the device.
    condition  quiet | tv | fan | music
    distance   near | far, or metres such as 3m / 2.5m
    n          take number 1, 2, 3, ...

Phone recordings: m4a, mp3, wav, flac, ogg, aac, aiff and caf are accepted
and decoded to 16 kHz mono float32. WAV/FLAC/OGG/MP3 are read by soundfile;
m4a/aac need ffmpeg (``brew install ffmpeg``; on macOS Core Audio also works).

    from eval.model.datasets import iter_test_files
    for audio, label, condition, source in iter_test_files():
        ...   # label: category ID or "negative"; source: "indian" or "esc50"

    python -m eval.model.datasets            # check names and print a summary of what's there
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from contracts.types import (
    APPLIANCE_BEEP,
    BABY_CRY,
    CATEGORIES,
    DOG_BARK,
    DOOR_KNOCK,
    GLASS_BREAK,
    SMOKE_ALARM,
    WATER_RUNNING,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
INDIAN_ROOT = DATA_ROOT / "indian"
PROFILE_ROOT = INDIAN_ROOT / "profile"
TEST_ROOT = INDIAN_ROOT / "test"
ESC50_ROOT = DATA_ROOT / "esc50"
FSD50K_ROOT = DATA_ROOT / "fsd50k"  # eval/model/download_fsd50k.py
FREESOUND_ROOT = DATA_ROOT / "freesound"  # added by hand: <category>/ + metadata.csv
DEMAND_ROOT = DATA_ROOT / "demand"  # eval/model/download_demand.py: <ENV>/chNN.wav

SAMPLE_RATE = 16000
HARD_NEGATIVES = "hard_negatives"
BACKGROUND = "background"  # profile/background/: room sound for the background bank
NEGATIVE = "negative"  # label of every clip that must not alert
CONDITIONS = ("quiet", "tv", "fan", "music")
DISTANCES = ("near", "far")  # or metres: 3m, 2.5m
RECORDERS = ("shreeya", "vaidehi", "sneha")  # <name>: who recorded it = whose home/device
ESC50_CONDITION = "clean"  # ESC-50 clips: isolated sounds, no added room condition
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".aiff", ".aif", ".caf"}

# --- ESC-50 ----------------------------------------------------------------------

# ESC-50 category -> SoundSense category ID. ESC-50 has no doorbell, pressure
# cooker or smoke alarm; those come from our own recordings.
ESC50_TO_CATEGORY: dict[str, str] = {
    "crying_baby": BABY_CRY,
    "door_wood_knock": DOOR_KNOCK,
    "glass_breaking": GLASS_BREAK,
    "dog": DOG_BARK,
    "pouring_water": WATER_RUNNING,
}

# ESC-50 categories kept as negatives (false-alert testing), with why each matters.
ESC50_NEGATIVES: dict[str, str] = {
    # close to a target: the hardest negatives
    "clock_alarm": f"alarm clock: must not raise {SMOKE_ALARM} (generic 'Alarm' is unmapped, S4)",
    "siren": f"street siren, alarm-like: {SMOKE_ALARM} confusion",
    "church_bells": "chimes: doorbell confusion",
    "door_wood_creaks": f"'Door' is mapped to {DOOR_KNOCK} (S5d): opening/closing must not alert",
    "toilet_flush": f"rushing water: {WATER_RUNNING} confusion",
    "water_drops": f"dripping: {WATER_RUNNING} confusion",
    "washing_machine": f"appliance noise: {WATER_RUNNING} / {APPLIANCE_BEEP} confusion",
    "vacuum_cleaner": "loud appliance noise",
    "clock_tick": "ticking: knock/beep confusion",
    "keyboard_typing": f"tapping: {DOOR_KNOCK} confusion",
    "mouse_click": f"clicking: {DOOR_KNOCK} confusion",
    "footsteps": f"thuds: {DOOR_KNOCK} confusion",
    "can_opening": f"metallic hiss/click: {GLASS_BREAK} / pressure_cooker confusion",
    "cat": f"animal: {DOG_BARK} / {BABY_CRY} confusion",
    # speech-like human sounds (ESC-50 has no plain speech)
    "laughing": "speech-like",
    "coughing": "speech-like",
    "sneezing": "speech-like",
    "breathing": "speech-like",
    "snoring": "speech-like",
    # everyday background
    "rain": "background",
    "wind": "background",
    "crackling_fire": "background (frying/tadka-like crackle)",
    "drinking_sipping": "background",
    "brushing_teeth": "background",
    "car_horn": "street noise through a window",
    "fireworks": "Diwali crackers: glass_break / knock confusion",
}

ESC50_KEEP = set(ESC50_TO_CATEGORY) | set(ESC50_NEGATIVES)


def esc50_label(esc_category: str) -> str:
    """Our label for an ESC-50 category: a category ID or NEGATIVE."""
    return ESC50_TO_CATEGORY.get(esc_category, NEGATIVE)


# --- audio -----------------------------------------------------------------------

class AudioDecodeError(RuntimeError):
    pass


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any supported file to mono float32 at ``sr`` Hz, values -1..1.

    soundfile first (wav/flac/ogg/mp3/aiff), then audioread, which uses
    ffmpeg (or Core Audio on macOS) for m4a/aac.

    Raises:
        AudioDecodeError: if no decoder can read the file (with a hint to
            install ffmpeg).
    """
    import librosa

    path = Path(path)
    try:
        with warnings.catch_warnings():  # librosa warns when it falls back to audioread
            warnings.simplefilter("ignore", (UserWarning, FutureWarning))
            audio, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        hint = "" if shutil.which("ffmpeg") else " Install ffmpeg (macOS: brew install ffmpeg) for m4a/aac."
        raise AudioDecodeError(f"cannot decode {path}: {exc}.{hint}") from exc
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


# --- file names --------------------------------------------------------------------

_METRES = re.compile(r"^\d+(\.\d+)?m$")


@dataclass(frozen=True)
class NameParts:
    prefix: str  # category ID or hard-negative type
    name: str  # who recorded it (RECORDERS)
    condition: str
    distance: str  # near | far | <metres>m
    take: int


def parse_name(stem: str, prefix: str) -> NameParts:
    """Parse ``<prefix>_<name>_<condition>_<distance>_<n>`` (the Drive naming guide).

    Case-insensitive (Shreeya_TV_Near -> shreeya, tv, near). A name that is
    not in RECORDERS is accepted here; list clips reports it as a problem.

    Raises:
        ValueError: with the reason, if the name doesn't follow the rule.
    """
    stem = stem.lower()
    if not stem.startswith(prefix + "_"):
        raise ValueError(f"should start with '{prefix}_' (its folder)")
    parts = stem[len(prefix) + 1:].split("_")
    if len(parts) != 4:
        raise ValueError(f"expected <name>_<condition>_<distance>_<n> after '{prefix}_' "
                         f"(name = {' | '.join(RECORDERS)})")
    name, condition, distance, take = parts
    if not name:
        raise ValueError(f"missing <name> (who recorded it: {' | '.join(RECORDERS)})")
    if condition not in CONDITIONS:
        raise ValueError(f"condition {condition!r} is not one of {', '.join(CONDITIONS)}")
    if distance not in DISTANCES and not _METRES.match(distance):
        raise ValueError(f"distance {distance!r} should be near, far or metres like 3m")
    if not take.isdigit() or int(take) < 1:
        raise ValueError(f"take {take!r} should be 1, 2, 3, ...")
    return NameParts(prefix, name, condition, distance, int(take))


# --- clips ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Clip:
    """One evaluation file (metadata only; ``load()`` decodes the audio)."""

    path: Path
    label: str  # category ID, or NEGATIVE
    condition: str  # quiet | tv | fan | music | clean (ESC-50) | unknown
    source: str  # "indian" | "esc50"
    category: str | None  # same as label for targets; None for negatives
    negative_type: str | None  # hard-negative <type> or ESC-50 category; None for targets
    name: str = ""  # Indian: who recorded it; ESC-50: its ESC-50 category
    distance: str | None = None  # near | far | <metres>m
    take: int | None = None
    split: str = "test"  # "test" | "profile"
    recorder: str | None = None  # who recorded it = whose home/device (Indian); None for ESC-50,
                                 # badly named files and shared background clips

    def load(self, sr: int = SAMPLE_RATE) -> np.ndarray:
        return load_audio(self.path, sr)


def _audio_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))


def _indian_clips(root: Path, split: str, problems: list[str]) -> list[Clip]:
    clips: list[Clip] = []
    if not root.is_dir():
        return clips
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name == HARD_NEGATIVES and split == "test":
            groups = [(t.name, t, True) for t in sorted(folder.iterdir()) if t.is_dir()]
        elif folder.name == BACKGROUND and split == "profile":
            for p in _audio_files(folder):  # any name; background_<name>_... marks whose home it is
                try:
                    parts = parse_name(p.stem, BACKGROUND)
                except ValueError:
                    parts = None
                clips.append(Clip(path=p, label=BACKGROUND, condition=parts.condition if parts else "unknown",
                                  source="indian", category=None, negative_type=None,
                                  name=parts.name if parts else "", distance=parts.distance if parts else None,
                                  take=parts.take if parts else None, split=split,
                                  recorder=parts.name if parts else None))
            continue
        elif folder.name in CATEGORIES:
            groups = [(folder.name, folder, False)]
        else:
            problems.append(f"{folder}: not a category ID ({', '.join(CATEGORIES)})"
                            + (f" or {HARD_NEGATIVES}/" if split == "test" else ""))
            continue
        for prefix, group, negative in groups:
            for path in _audio_files(group):
                sub = path.parent.relative_to(group).parts
                try:
                    parts = parse_name(path.stem, prefix)
                except ValueError as exc:
                    problems.append(f"{path.relative_to(root)}: {exc}")
                    parts = None
                if parts and parts.name not in RECORDERS:
                    problems.append(f"{path.relative_to(root)}: name {parts.name!r} is not one of "
                                    f"{', '.join(RECORDERS)} (add new recorders to RECORDERS)")
                condition = parts.condition if parts else "unknown"
                if sub and sub[0] in CONDITIONS and parts and sub[0] != parts.condition:
                    problems.append(f"{path.relative_to(root)}: folder says {sub[0]!r}, name says {parts.condition!r}")
                if sub and sub[0] in CONDITIONS and not parts:
                    condition = sub[0]
                clips.append(Clip(
                    path=path,
                    label=NEGATIVE if negative else prefix,
                    condition=condition,
                    source="indian",
                    category=None if negative else prefix,
                    negative_type=prefix if negative else None,
                    name=parts.name if parts else "",
                    distance=parts.distance if parts else None,
                    take=parts.take if parts else None,
                    split=split,
                    recorder=parts.name if parts else None,
                ))
    return clips


def list_esc50_clips(root: Path = ESC50_ROOT, problems: list[str] | None = None) -> list[Clip]:
    """Clips listed in <root>/meta/esc50.csv whose audio file exists."""
    meta = root / "meta" / "esc50.csv"
    if not meta.exists():
        return []
    clips = []
    with meta.open(newline="") as f:
        for row in csv.DictReader(f):
            path = root / "audio" / row["filename"]
            if not path.exists():
                if problems is not None:
                    problems.append(f"{path}: listed in esc50.csv but missing")
                continue
            esc = row["category"]
            label = esc50_label(esc)
            clips.append(Clip(path=path, label=label, condition=ESC50_CONDITION, source="esc50",
                              category=None if label == NEGATIVE else label,
                              negative_type=esc if label == NEGATIVE else None,
                              name=esc, take=ord(row["take"].upper()) - ord("A") + 1 if row.get("take") else None))
    return clips


# --- FSD50K subset, Freesound, DEMAND (Part S7d) ----------------------------------------

TV_BACKGROUND_TYPES = ("speech", "music")  # FSD50K Speech / Music clips: TV-guard backgrounds
FREESOUND_LICENSES = ("CC0", "CC-BY")
FREESOUND_FIELDS = ["file", "author", "url", "license"]
DEMAND_ENVS = ("DKITCHEN", "DLIVING", "DWASHING")


def list_fsd50k_clips(root: Path = FSD50K_ROOT, problems: list[str] | None = None) -> list[Clip]:
    """The FSD50K eval subset listed in <root>/metadata.csv (download_fsd50k.py).

    Targets have label = our category. Speech/Music clips (TV-guard
    backgrounds) have label "negative" and negative_type "speech"/"music":
    on their own they must not alert. condition "clean", source "fsd50k".
    """
    meta = Path(root) / "metadata.csv"
    if not meta.exists():
        return []
    clips = []
    with meta.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(root) / row["file"]
            if not path.exists():
                if problems is not None:
                    problems.append(f"{path}: listed in fsd50k metadata.csv but missing")
                continue
            target = row["role"] == "target"
            clips.append(Clip(path=path, label=row["our_label"] if target else NEGATIVE, condition=ESC50_CONDITION,
                              source="fsd50k", category=row["our_label"] if target else None,
                              negative_type=None if target else row["our_label"], name=row["fsd50k_labels"]))
    return clips


def list_tv_backgrounds(root: Path = FSD50K_ROOT) -> dict[str, list[Clip]]:
    """FSD50K Speech and Music clips, by type ("speech", "music")."""
    clips = list_fsd50k_clips(root)
    return {t: [c for c in clips if c.negative_type == t] for t in TV_BACKGROUND_TYPES}


def init_freesound(root: Path = FREESOUND_ROOT) -> Path:
    """Create <root>/<category>/ for every category and a metadata.csv template (if missing)."""
    root = Path(root)
    for c in CATEGORIES:
        (root / c).mkdir(parents=True, exist_ok=True)
    meta = root / "metadata.csv"
    if not meta.exists():
        with meta.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(FREESOUND_FIELDS)
    return meta


def _short_license(value: str) -> str:
    v = value.strip().lower().replace(" ", "")
    if v in ("cc0", "cc0-1.0") or "publicdomain/zero" in v:
        return "CC0"
    if v.startswith("cc-by-nc") or "/by-nc" in v:
        return "CC-BY-NC"
    if v in ("cc-by", "cc-by-3.0", "cc-by-4.0", "ccby") or "licenses/by/" in v:
        return "CC-BY"
    return value.strip()


def list_freesound_clips(root: Path = FREESOUND_ROOT, problems: list[str] | None = None) -> list[Clip]:
    """Hand-collected Freesound test clips: <root>/<category>/<file>, test-only, source "freesound".

    Every file needs a row in <root>/metadata.csv (file, author, url,
    license). Only CC0 and CC-BY are used; anything else, or a file without
    a row, is skipped and reported in ``problems``.
    """
    root = Path(root)
    problems = [] if problems is None else problems
    meta = root / "metadata.csv"
    rows = {}
    if meta.exists():
        with meta.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file"):
                    rows[row["file"].strip().replace("\\", "/")] = row
    clips = []
    if not root.is_dir():
        return clips
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if folder.name not in CATEGORIES:
            problems.append(f"{folder}: not a category ID")
            continue
        for path in _audio_files(folder):
            rel = path.relative_to(root).as_posix()
            row = rows.get(rel)
            if row is None:
                problems.append(f"freesound/{rel}: no row in metadata.csv (file, author, url, license); skipped")
                continue
            lic = _short_license(row.get("license", ""))
            if lic not in FREESOUND_LICENSES:
                problems.append(f"freesound/{rel}: licence {row.get('license')!r} is not CC0/CC-BY; skipped")
                continue
            clips.append(Clip(path=path, label=folder.name, condition=ESC50_CONDITION, source="freesound",
                              category=folder.name, negative_type=None, name=row.get("author", "")))
    return clips


def list_demand_noise(root: Path = DEMAND_ROOT) -> dict[str, list[Path]]:
    """DEMAND noise files by environment: {"DKITCHEN": [ch01.wav, ...], ...} (present ones only)."""
    out = {}
    for env in DEMAND_ENVS:
        files = sorted((Path(root) / env).glob("ch*.wav"))
        if files:
            out[env] = files
    return out


def list_test_clips(test_root: Path = TEST_ROOT, esc50_root: Path | None = ESC50_ROOT,
               problems: list[str] | None = None) -> list[Clip]:
    """Every test clip: our recordings (test/ + hard_negatives/) then ESC-50 (if present)."""
    problems = [] if problems is None else problems
    clips = _indian_clips(Path(test_root), "test", problems)
    if esc50_root is not None:
        clips += list_esc50_clips(Path(esc50_root), problems)
    return clips


def list_profile_clips(profile_root: Path = PROFILE_ROOT, category: str | None = None,
                  problems: list[str] | None = None) -> list[Clip]:
    """Enrolment clips from data/indian/profile/<category>/ (optionally one category).

    Room-sound clips from profile/background/ are included with label "background".
    """
    problems = [] if problems is None else problems
    clips = _indian_clips(Path(profile_root), "profile", problems)
    return [c for c in clips if category is None or c.category == category]


def iter_test_files(test_root: Path = TEST_ROOT, esc50_root: Path | None = ESC50_ROOT,
                    sr: int = SAMPLE_RATE) -> Iterator[tuple[np.ndarray, str, str, str]]:
    """Yield ``(audio, label, condition, source)`` for every test file.

    audio: mono float32 at ``sr`` Hz. label: category ID or "negative".
    condition: quiet/tv/fan/music, "clean" for ESC-50, "unknown" if the file
    name breaks the naming rule. source: "indian" or "esc50".
    Files that can't be decoded are logged and skipped.
    """
    for clip in list_test_clips(test_root, esc50_root):
        try:
            audio = clip.load(sr)
        except AudioDecodeError as exc:
            log.error("%s", exc)
            continue
        yield audio, clip.label, clip.condition, clip.source


# --- CLI: check the dataset ------------------------------------------------------------

def summary(clips: list[Clip]) -> str:
    if not clips:
        return "  (none)"
    conds = list(CONDITIONS) + [ESC50_CONDITION, "unknown"]
    counts = Counter((c.source + (f"/{c.recorder}" if c.recorder else ""),
                      c.label if c.label != NEGATIVE else f"negative:{c.negative_type}", c.condition)
                     for c in clips)
    rows = sorted({(s, l) for s, l, _ in counts})
    used = [c for c in conds if any(k[2] == c for k in counts)]
    width = max(len(f"{s}/{l}") for s, l in rows)
    lines = ["  " + " " * width + "".join(f"{c:>8s}" for c in used) + "   total"]
    for s, l in rows:
        n = [counts[(s, l, c)] for c in used]
        lines.append(f"  {f'{s}/{l}':{width}s}" + "".join(f"{x or '-':>8}" for x in n) + f"{sum(n):>8d}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the evaluation datasets and summarise them.")
    parser.add_argument("--indian", type=Path, default=INDIAN_ROOT)
    parser.add_argument("--esc50", type=Path, default=ESC50_ROOT)
    parser.add_argument("--fsd50k", type=Path, default=FSD50K_ROOT)
    parser.add_argument("--freesound", type=Path, default=FREESOUND_ROOT)
    parser.add_argument("--demand", type=Path, default=DEMAND_ROOT)
    parser.add_argument("--decode", action="store_true", help="also decode every file (finds unreadable ones)")
    parser.add_argument("--init-freesound", action="store_true",
                        help="create data/freesound/<category>/ and an empty metadata.csv template, then exit")
    args = parser.parse_args()

    if args.init_freesound:
        meta = init_freesound(args.freesound)
        print(f"Created {args.freesound}/<category>/ for {len(CATEGORIES)} categories and {meta} "
              f"(columns: {', '.join(FREESOUND_FIELDS)}; licence must be CC0 or CC-BY)")
        return
    problems: list[str] = []
    profile = list_profile_clips(args.indian / "profile", problems=problems)
    test = list_test_clips(args.indian / "test", args.esc50, problems=problems)
    extra = list_fsd50k_clips(args.fsd50k, problems) + list_freesound_clips(args.freesound, problems)
    print(f"Profile clips ({args.indian / 'profile'}): {len(profile)}\n{summary(profile)}\n")
    print(f"Test clips ({args.indian / 'test'} + {args.esc50}): {len(test)}\n{summary(test)}\n")
    print(f"Supplementary test clips (FSD50K + Freesound): {len(extra)}\n{summary(extra)}\n")
    demand = list_demand_noise(args.demand)
    print("DEMAND noise: " + (", ".join(f"{e} ({len(f)} channels)" for e, f in demand.items()) or "(none)") + "\n")
    test += extra
    if args.decode:
        for clip in profile + test:
            try:
                audio = clip.load()
                if audio.size == 0:
                    problems.append(f"{clip.path}: empty audio")
            except AudioDecodeError as exc:
                problems.append(str(exc))
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No naming problems.")
    if not shutil.which("ffmpeg"):
        print("\nNote: ffmpeg not found. m4a/aac phone recordings need it on Windows/Linux "
              "(macOS can use Core Audio): brew install ffmpeg")


if __name__ == "__main__":
    main()
