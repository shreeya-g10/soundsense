"""Shared plumbing for the model experiments (exp1/exp4/exp6/exp8, run_all).

Owner: Shreeya (Role 1). Part S7.

- DataPaths / get_*: where the datasets are; raise DataMissing (with a clear
  message) when one isn't there, so scripts can skip it.
- FeatureExtractor: per-window YAMNet category scores + embedding (+ MFCC),
  computed once per file and cached in memory, so run_all doesn't run
  YAMNet twice on the same window.
- write_csv: CSVs go to eval/model/results/ (columns: eval/model/README.md).

Windows are 0.96 s every 0.48 s, as in the app. Silent windows (RMS below
the silence gate's default threshold, no calibration) are dropped, because
the app never classifies them; their count is reported.
"""

from __future__ import annotations

import csv
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES, Windower
from backend.audio.silence import DEFAULT_THRESHOLD, rms
from contracts.types import CATEGORIES
from eval.model import datasets as ds

RESULTS_DIR = Path(__file__).resolve().parent / "results"
NONE = "none"  # "no alert": the right answer for negatives
LABELS = list(CATEGORIES) + [NONE]
DETECTION_THRESHOLD = 0.15  # a category is predicted when its score >= this (as the listener demo's label)

ESC50 = "esc50"
INDIAN = "indian"
ESC50_PROFILE_FOLD = 1  # ESC-50 stand-in for exp4/exp6: profiles from fold 1, test on folds 2-5


class DataMissing(Exception):
    """A dataset an experiment needs isn't there. The message says how to get it."""


@dataclass(frozen=True)
class DataPaths:
    indian: Path = ds.INDIAN_ROOT
    esc50: Path = ds.ESC50_ROOT

    @property
    def profile(self) -> Path:
        return Path(self.indian) / "profile"

    @property
    def test(self) -> Path:
        return Path(self.indian) / "test"


def true_label(clip: ds.Clip) -> str:
    return clip.category or NONE


def esc50_fold(clip: ds.Clip) -> int:
    """ESC-50 file names are <fold>-<source id>-<take>-<target>.wav."""
    return int(clip.path.name.split("-")[0])


def get_esc50(paths: DataPaths) -> list[ds.Clip]:
    clips = ds.list_esc50_clips(Path(paths.esc50))
    if not clips:
        raise DataMissing(f"ESC-50 not found in {paths.esc50}. Run: python -m eval.model.download_esc50")
    return clips


def get_indian_test(paths: DataPaths) -> list[ds.Clip]:
    clips = ds.list_test_clips(paths.test, esc50_root=None)
    if not clips:
        raise DataMissing(f"No Indian test recordings in {paths.test} (see eval/model/README.md, "
                          f"'Our Indian recordings').")
    return clips


def get_indian_profile(paths: DataPaths) -> list[ds.Clip]:
    clips = [c for c in ds.list_profile_clips(paths.profile) if c.category]
    if not clips:
        raise DataMissing(f"No Indian profile recordings in {paths.profile} (see eval/model/README.md).")
    return clips


def get_indian_background(paths: DataPaths) -> list[ds.Clip]:
    return [c for c in ds.list_profile_clips(paths.profile) if c.label == ds.BACKGROUND]


def subset(clips: list, limit: int | None) -> list:
    """Every k-th clip so at most ``limit`` remain, spread over all labels (None = all)."""
    if not limit or len(clips) <= limit:
        return clips
    step = len(clips) / limit
    return [clips[int(i * step)] for i in range(limit)]


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disjoint(profile: list[ds.Clip], test: list[ds.Clip]) -> tuple[list[ds.Clip], list[ds.Clip]]:
    """Test clips with any profile file (same path or same bytes) removed: (kept, removed)."""
    used_paths = {c.path.resolve() for c in profile}
    used_hashes = {file_hash(c.path) for c in profile}
    kept, removed = [], []
    for c in test:
        (removed if c.path.resolve() in used_paths or file_hash(c.path) in used_hashes else kept).append(c)
    return kept, removed


# --- per-window features ------------------------------------------------------------

@dataclass
class WindowFeat:
    start_s: float
    rms: float
    scores: np.ndarray  # (9,) category scores, CATEGORIES order
    embedding: np.ndarray  # (1024,)
    inference_ms: float
    mfcc: np.ndarray | None = None  # (40,) mfcc_vector, if requested

    def category_scores(self) -> dict[str, float]:
        return dict(zip(CATEGORIES, map(float, self.scores)))


@dataclass
class ClipFeat:
    clip: ds.Clip
    windows: list[WindowFeat]
    total_windows: int  # including silent ones
    duration_s: float = 0.0

    @property
    def silent(self) -> int:
        return self.total_windows - len(self.windows)


def windows_of(audio: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(start s, window) for every 0.96 s window (0.48 s hop); short audio is zero-padded to one."""
    audio = np.asarray(audio, np.float32).ravel()
    if audio.size < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
    return [((end - WINDOW_SAMPLES) / SAMPLE_RATE, w) for end, w in Windower().feed(audio)]


@dataclass
class FeatureExtractor:
    """Runs YAMNet + category mapping (+ MFCC) once per window, cached per file."""

    silence_threshold: float = DEFAULT_THRESHOLD
    model: object = None
    mapper: object = None
    verbose: bool = True
    _cache: dict = field(default_factory=dict)

    def load(self) -> None:
        if self.model is None:
            from backend.audio.listener import load_pipeline

            self.model, self.mapper = load_pipeline()

    def features(self, clip: ds.Clip, mfcc: bool = False) -> ClipFeat:
        key = clip.path.resolve()
        cached = self._cache.get(key)
        if cached is not None and (not mfcc or all(w.mfcc is not None for w in cached.windows)):
            return cached
        self.load()
        audio = clip.load()
        if cached is not None:  # add MFCCs to the cached windows
            by_start = {round(s, 3): w for s, w in windows_of(audio)}
            from backend.custom.profile_store import mfcc_vector

            for wf in cached.windows:
                wf.mfcc = mfcc_vector(by_start[round(wf.start_s, 3)])
            return cached
        feats, total = [], 0
        for start, w in windows_of(audio):
            total += 1
            level = rms(w)
            if level < self.silence_threshold:
                continue
            out = self.model.infer(w)
            cats = self.mapper.category_scores(out.scores)
            wf = WindowFeat(start, level, np.array([cats[c] for c in CATEGORIES], np.float32),
                            np.asarray(out.embedding, np.float32), float(out.inference_ms))
            if mfcc:
                from backend.custom.profile_store import mfcc_vector

                wf.mfcc = mfcc_vector(w)
            feats.append(wf)
        result = ClipFeat(clip, feats, total, audio.size / SAMPLE_RATE)
        self._cache[key] = result
        return result

    def many(self, clips: list[ds.Clip], mfcc: bool = False, label: str = "") -> list[ClipFeat]:
        out, t0 = [], time.monotonic()
        for i, clip in enumerate(clips, 1):
            out.append(self.features(clip, mfcc))
            if self.verbose and (i % 200 == 0 or i == len(clips)) and len(clips) > 20:
                print(f"    {label} features: {i}/{len(clips)} files ({time.monotonic() - t0:.0f} s)", flush=True)
        return out


# --- output -------------------------------------------------------------------------

def fmt(value) -> str:
    """CSV cell: None/NaN -> empty, floats to 4 decimals."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_csv(out_dir: Path, name: str, fields: list[str], rows: Iterable[dict]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})
    return path


def skip(experiment: str, what: str, exc: DataMissing) -> None:
    print(f"[{experiment}] SKIP {what}: {exc}", flush=True)


# --- custom profiles (exp4, exp6) -------------------------------------------------

TARGET = "target"
SAME_CATEGORY_OTHER_DEVICE = "same_category_other_device"
UNRELATED = "unrelated"


@dataclass
class Split:
    """Files for a custom-profile experiment. Profile and test files never overlap.

    Indian data gives one Split per recorder: that person's profiles (their
    home = their device) are tested against ALL test clips.
    """

    dataset: str  # "indian" or "esc50" (stand-in: fold 1 profiles, folds 2-5 test)
    profile: list[ds.Clip]  # enrolment clips (with a category)
    test: list[ds.Clip]
    room: list[ds.Clip] = field(default_factory=list)  # room sound -> background bank (room + calibration)
    generic: list[ds.Clip] = field(default_factory=list)  # non-target clips -> background bank (generic)
    removed: int = 0  # test files dropped because they were also profile files
    recorder: str | None = None  # whose profiles these are (Indian); None for ESC-50

    @property
    def categories(self) -> list[str]:
        return [c for c in CATEGORIES if any(p.category == c for p in self.profile)]

    @property
    def who(self) -> str:
        return self.recorder or "all"


def relation(clip: ds.Clip, category: str, recorder: str | None) -> str:
    """How a test clip relates to ``recorder``'s profile for ``category``.

    TARGET: same category, recorded by the same person (their device/home);
    without recorders (ESC-50) every clip of the category is a target.
    SAME_CATEGORY_OTHER_DEVICE: same category, someone else's recording.
    UNRELATED: any other category, or a negative.
    """
    if clip.category != category:
        return UNRELATED
    if recorder is None or clip.recorder == recorder:
        return TARGET
    return SAME_CATEGORY_OTHER_DEVICE


def device(clip: ds.Clip, recorder: str | None) -> str:
    """"own" / "other" device relative to ``recorder``; "all" without recorders."""
    if recorder is None:
        return "all"
    return "own" if clip.recorder == recorder else "other"


def custom_splits(dataset: str, paths: DataPaths, limit: int | None = None,
                  esc50_negatives_in_test: bool = False, echo=print) -> list[Split]:
    """Profile/test splits for ``dataset``; raises DataMissing.

    indian: one Split per recorder with profile clips: profiles from that
            person's profile/<category>/ clips only; room bank from their
            profile/background/ clips (+ shared ones); test = every test/
            clip (+ ESC-50 negatives if ``esc50_negatives_in_test``).
            Profile clips without a valid recorder name are left out.
    esc50:  one Split: fold 1 targets -> profiles, fold 1 negatives ->
            generic bank, folds 2-5 -> test.
    """
    if dataset == INDIAN:
        profile, test = get_indian_profile(paths), get_indian_test(paths)
        room = get_indian_background(paths)
        if esc50_negatives_in_test:
            test += [c for c in ds.list_esc50_clips(Path(paths.esc50)) if c.label == ds.NEGATIVE]
        test, removed = disjoint(profile + room, test)
        test = subset(test, limit)
        unnamed = [c for c in profile if c.recorder is None]
        if unnamed:
            echo(f"    {len(unnamed)} profile clip(s) without a recorder name left out "
                 f"(run python -m eval.model.datasets)")
        recorders = sorted({c.recorder for c in profile if c.recorder})
        if not recorders:
            raise DataMissing("No Indian profile clip has a recorder name (<category>_<name>_...).")
        return [Split(INDIAN, [c for c in profile if c.recorder == r], test,
                      room=[c for c in room if c.recorder in (r, None)], removed=len(removed), recorder=r)
                for r in recorders]
    clips = get_esc50(paths)
    fold1 = [c for c in clips if esc50_fold(c) == ESC50_PROFILE_FOLD]
    test = [c for c in clips if esc50_fold(c) != ESC50_PROFILE_FOLD]
    return [Split(ESC50, [c for c in fold1 if c.category], subset(test, limit),
                  generic=[c for c in fold1 if not c.category])]


def build_store(split: Split, extractor: FeatureExtractor, root: Path, method: str = "embedding",
                **store_kwargs):
    """A YamnetProfileStore in ``root`` with one profile per category in ``split.profile``.

    Returns (store, {category: profile_id}). Background bank: room clips
    (all windows as "calibration", active windows as "room") and generic
    clips (active windows as "generic").
    """
    from backend.custom.profile_store import YamnetProfileStore

    extractor.load()
    store = YamnetProfileStore(root, model=extractor.model, **store_kwargs)
    for clip in split.room:
        audio = clip.load()
        store.add_background(store.embed_audio(audio, skip_silent=False), source="calibration")
        store.add_background_from_audio(audio)
    generic = [w.embedding for f in extractor.many(split.generic, label="generic bank") for w in f.windows]
    if generic:
        store.set_background("generic", np.stack(generic))
    pids = {}
    for category in split.categories:
        pid = store.create(category, category, method=method)
        for clip in split.profile:
            if clip.category == category:
                store.add_example_from_audio(pid, clip.load())
        pids[category] = pid
    return store, pids
