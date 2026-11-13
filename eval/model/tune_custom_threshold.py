"""Choose the custom-sound match threshold for raw vs centered scoring (Parts S5f, Experiment 6).

Owner: Shreeya (Role 1).

Dataset layout (e.g. data/indian/test/, recorded with demo_profiles --record-test):

    <root>/<label>/*.wav       one folder per sound (cooker, doorbell, speech_amma, ...)
    <root>/background/*.wav    normal room sound, no target (optional, recommended)

For each label with >= 2 files: a fresh profile is enrolled on the first
--enrol files (default: half), with the background files as the room sound
(background bank + augmentation noise). Every other file is then a test:
same label = should match, other labels and background = should not.

A file counts as detected at threshold t if any window has similarity >= t
(and, with the margin rule, beats its background score), like one alert.
For each scoring the threshold with the best F1 is reported. Every window
of the background files also goes in as "calibration" background, like the
listener's calibration windows (which are embedded however quiet they are).

    python -m eval.model.tune_custom_threshold data/indian/test
    python -m eval.model.tune_custom_threshold data/indian/test --no-margin --no-augment

Only embeddings are kept (in a temporary profile folder); no audio is written.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.custom.demo_profiles import AUDIO_EXTS
from backend.custom.profile_store import SCORING_CENTERED, SCORING_RAW, YamnetProfileStore

THRESHOLDS = np.round(np.arange(0.20, 0.991, 0.01), 2)
MFCC_THRESHOLDS = np.round(np.arange(0.900, 0.9995, 0.001), 3)  # MFCC cosines are all close to 1


@dataclass
class Trial:
    profile_label: str
    file: Path
    positive: bool
    best: float  # best window similarity that passes the margin rule (-inf if none)


def load_dataset(root: Path, background_label: str) -> tuple[dict[str, list[Path]], list[Path]]:
    labels = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        if files:
            labels[d.name] = files
    background = labels.pop(background_label, [])
    return labels, background


def run(root: Path, scoring: str, margin: bool, augment: bool, enrol: int | None,
        background_label: str = "background", load=None, model=None, method: str = "embedding") -> list[Trial]:
    if load is None:
        from backend.ml.demo_yamnet import load_wav as load
    labels, background = load_dataset(root, background_label)
    audio = {f: load(str(f)) for files in labels.values() for f in files}
    bg_audio = [load(str(f)) for f in background]
    all_files = [f for files in labels.values() for f in files] + background
    trials: list[Trial] = []
    cache: dict[Path, list[tuple[np.ndarray, np.ndarray]]] = {}  # file -> [(window, embedding)]
    for label, files in labels.items():
        if len(files) < 2:
            continue
        k = enrol or max(1, len(files) // 2)
        with tempfile.TemporaryDirectory() as tmp:
            store = YamnetProfileStore(tmp, scoring=scoring, margin=margin, augment=augment,
                                       model=model, threshold=0.0)
            for a in bg_audio:
                # Like the live app: the listener's calibration windows (any loudness)
                # plus the active windows of the enrolment room recording.
                store.add_background(store.embed_audio(a, skip_silent=False), source="calibration")
                store.add_background_from_audio(a)
            pid = store.create(label, "doorbell", method=method)  # category is irrelevant here
            for f in files[:k]:
                store.add_example_from_audio(pid, audio[f])
            enrolled = set(files[:k])
            for f in all_files:
                if f in enrolled:
                    continue
                if f not in cache:
                    windows = [w for _, w in store.active_windows(audio[f] if f in audio else load(str(f)))]
                    cache[f] = list(zip(windows, store.features(windows, "embedding")))
                best = -np.inf
                for w, e in cache[f]:
                    d = store.score(pid, e, samples=w)
                    if not margin or d.background is None or d.similarity > d.background:
                        best = max(best, d.similarity)
                trials.append(Trial(label, f, f in files, best))
    return trials


def sweep(trials: list[Trial], thresholds=None) -> list[tuple[float, float, float, float]]:
    """(threshold, precision, recall, f1) for each threshold."""
    thresholds = THRESHOLDS if thresholds is None else thresholds
    best = np.array([t.best for t in trials])
    pos = np.array([t.positive for t in trials])
    rows = []
    for th in thresholds:
        pred = best >= th
        tp, fp, fn = int((pred & pos).sum()), int((pred & ~pos).sum()), int((~pred & pos).sum())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((float(th), p, r, f1))
    return rows


def best_threshold(rows) -> tuple[float, float, float, float]:
    """Highest F1; ties -> middle of the tied range (most room on both sides)."""
    top = max(r[3] for r in rows)
    tied = [r for r in rows if r[3] >= top - 1e-9]
    return tied[len(tied) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the custom-sound threshold (raw vs centered).")
    parser.add_argument("root", type=Path, help="dataset folder: <label>/*.wav, background/*.wav")
    parser.add_argument("--enrol", type=int, default=None, help="files per label to enrol (default half)")
    parser.add_argument("--background-label", default="background")
    parser.add_argument("--no-margin", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--method", choices=["embedding", "mfcc"], default="embedding",
                        help="mfcc: the S6 baseline (scoring/margin don't apply, so it runs once)")
    args = parser.parse_args()

    settings = [("mfcc", "raw")] if args.method == "mfcc" else [("embedding", SCORING_RAW), ("embedding", SCORING_CENTERED)]
    for method, scoring in settings:
        trials = run(args.root, scoring, not args.no_margin, not args.no_augment, args.enrol,
                     args.background_label, method=method)
        scoring = "mfcc" if method == "mfcc" else scoring
        if not trials:
            raise SystemExit("No label has >= 2 files to enrol and test.")
        rows = sweep(trials, MFCC_THRESHOLDS if method == "mfcc" else None)
        th, p, r, f1 = best_threshold(rows)
        n_pos = sum(t.positive for t in trials)
        pos = [t.best for t in trials if t.positive and np.isfinite(t.best)]
        neg = [t.best for t in trials if not t.positive and np.isfinite(t.best)]
        print(f"\n{scoring}: {n_pos} positive / {len(trials) - n_pos} negative trials "
              f"({n_pos - len(pos)} positives had no scorable window: all silent or below background)")
        if pos:
            print(f"  positives best-window similarity: min {min(pos):.2f}  median {np.median(pos):.2f}")
        if neg:
            print(f"  negatives best-window similarity: median {np.median(neg):.2f}  "
                  f"95th pct {np.percentile(neg, 95):.2f}  max {max(neg):.2f}")
        print(f"  best threshold {th:.3f}: precision {p:.2f}, recall {r:.2f}, F1 {f1:.2f}")
        for t, pp, rr, ff in rows[::5]:
            print(f"    t={t:.3f}  P={pp:.2f} R={rr:.2f} F1={ff:.2f}")


if __name__ == "__main__":
    main()
