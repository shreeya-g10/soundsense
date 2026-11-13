"""Run every model experiment (1, 4, 6, 8) and write eval/model/results/*.csv.

Owner: Shreeya (Role 1). Part S7.

    python -m eval.model.run_all                 # real data; missing datasets are skipped with a message
    python -m eval.model.run_all --smoke         # synthetic mini-dataset in a temp folder: checks every
                                                 # script runs end to end (~1 min, results not kept)
    python -m eval.model.run_all --only exp1 exp8 --exp8-minutes 2

YAMNet runs once per window: exp1/exp4/exp6 share one FeatureExtractor.
exp8 runs in its own process so its RAM figures are clean.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from eval.model import common as cm
from eval.model import exp1_yamnet, exp4_custom, exp6_methods, exp8_resources

EXPERIMENTS = ("exp1", "exp4", "exp6", "exp8")
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FIELDS = {
    "exp1_summary.csv": exp1_yamnet.SUMMARY_FIELDS,
    "exp1_per_class.csv": exp1_yamnet.PER_CLASS_FIELDS,
    "exp1_confusion.csv": exp1_yamnet.CONFUSION_FIELDS,
    "exp1_threshold_sweep.csv": exp1_yamnet.SWEEP_FIELDS,
    "exp4_summary.csv": exp4_custom.SUMMARY_FIELDS,
    "exp4_per_class.csv": exp4_custom.PER_CLASS_FIELDS,
    "exp4_confusion.csv": exp4_custom.CONFUSION_FIELDS,
    "exp6_per_profile.csv": exp6_methods.PER_PROFILE_FIELDS,
    "exp6_summary.csv": exp6_methods.SUMMARY_FIELDS,
    "exp8_resources.csv": exp8_resources.FIELDS,
}


def run_all(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
            only: tuple[str, ...] = EXPERIMENTS, limit: int | None = None, exp8_minutes: float = 10.0,
            extractor: cm.FeatureExtractor | None = None) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    written: list[Path] = []
    for name in only:
        t0 = time.monotonic()
        print(f"\n=== {name} ===", flush=True)
        if name == "exp1":
            written += exp1_yamnet.run(out_dir, paths, extractor, limit=limit)
        elif name == "exp4":
            written += exp4_custom.run(out_dir, paths, extractor, limit=limit)
        elif name == "exp6":
            written += exp6_methods.run(out_dir, paths, extractor, limit=limit)
        elif name == "exp8":
            written += run_exp8_fresh(out_dir, paths, exp8_minutes)
        else:
            raise ValueError(f"unknown experiment {name!r}")
        print(f"=== {name} done in {time.monotonic() - t0:.0f} s ===", flush=True)
    return written


def run_exp8_fresh(out_dir: Path, paths: cm.DataPaths, minutes: float) -> list[Path]:
    """Experiment 8 in a new Python process, so its RAM figures don't include the
    other experiments' feature caches."""
    cmd = [sys.executable, "-m", "eval.model.exp8_resources", "--out", str(out_dir), "--minutes", str(minutes),
           "--indian", str(paths.indian), "--esc50", str(paths.esc50)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("[exp8]"):
            print(line, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exp8 failed:\n{proc.stderr[-2000:]}")
    return [Path(out_dir) / "exp8_resources.csv"]


def check_csvs(paths: list[Path]) -> None:
    """Every written CSV has exactly the documented columns and at least one row."""
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            n = sum(1 for _ in reader)
        expected = EXPECTED_FIELDS[p.name]
        if header != expected:
            raise AssertionError(f"{p.name}: columns {header} != documented {expected}")
        if n == 0:
            raise AssertionError(f"{p.name}: no rows")


# --- smoke test: a tiny synthetic dataset in both layouts --------------------------------

def _sound(kind: str, seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    if kind == "cry":  # wailing harmonic tone
        f = 450 + 120 * np.sin(2 * np.pi * 1.5 * t)
        x = sum(np.sin(2 * np.pi * k * np.cumsum(f) / sr) / k for k in (1, 2, 3))
    elif kind == "bark":  # short noisy bursts
        x = rng.standard_normal(t.size) * (np.sin(2 * np.pi * 2 * t) > 0.7)
    elif kind == "knock":  # decaying clicks
        x = np.zeros(t.size)
        for s in np.arange(0.2, seconds, 0.35):
            i = int(s * sr)
            x[i:i + 800] += np.exp(-np.arange(800) / 120) * rng.standard_normal(800)
    elif kind == "beep":  # alarm-clock beeps
        x = np.sin(2 * np.pi * 2800 * t) * (np.sin(2 * np.pi * 4 * t) > 0)
    else:  # "rain"/"hiss": broadband noise
        x = rng.standard_normal(t.size)
    return (0.25 * x / (np.abs(x).max() or 1)).astype(np.float32)


SMOKE_ESC50 = {"crying_baby": "cry", "dog": "bark", "door_wood_knock": "knock", "clock_alarm": "beep", "rain": "rain"}


def make_smoke_data(root: Path, seed: int = 0) -> cm.DataPaths:
    """Write a mini ESC-50 (2 folds) and a mini Indian set (profile/test/hard_negatives/background)."""
    import soundfile as sf

    rng = np.random.default_rng(seed)
    esc = root / "esc50"
    (esc / "audio").mkdir(parents=True)
    (esc / "meta").mkdir(parents=True)
    rows = []
    for target, (cat, kind) in enumerate(SMOKE_ESC50.items()):
        for fold in (1, 2):
            for take in "AB":
                name = f"{fold}-{100 + target}-{take}-{target}.wav"
                sf.write(esc / "audio" / name, _sound(kind, 2.0, 44100, rng), 44100)
                rows.append({"filename": name, "fold": fold, "target": target, "category": cat,
                             "esc10": "False", "src_file": str(100 + target), "take": take})
    with open(esc / "meta" / "esc50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Indian: two recorders (two homes/devices). Both record baby_cry, so each person's
    # profile also meets "same category, different device" clips.
    ind = root / "indian"
    homes = {"shreeya": {"baby_cry": "cry", "door_knock": "knock"},
             "vaidehi": {"baby_cry": "cry", "dog_bark": "bark"}}
    distances = {"quiet": "near", "tv": "far", "fan": "3m", "music": "near"}
    for who, sounds in homes.items():
        for cat, kind in sounds.items():
            for n in (1, 2):
                p = ind / "profile" / cat / f"{cat}_{who}_quiet_near_{n}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, _sound(kind, 2.0, 16000, rng), 16000)
            for cond, dist in distances.items():
                p = ind / "test" / cat / f"{cat}_{who}_{cond}_{dist}_1.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                x = _sound(kind, 2.0, 48000, rng) * (0.3 if dist == "far" else 1.0)
                if cond in ("tv", "fan", "music"):
                    x = x + 0.02 * rng.standard_normal(x.size).astype(np.float32)
                sf.write(p, x, 48000)  # 48 kHz like a phone
        p = ind / "test" / "hard_negatives" / "kettle" / f"kettle_{who}_fan_near_1.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, _sound("hiss", 2.0, 16000, rng), 16000)
        p = ind / "profile" / "background" / f"background_{who}_quiet_near_1.wav"  # their home's room sound
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.01 * rng.standard_normal(3 * 16000)).astype(np.float32), 16000)
    sf.write(ind / "profile" / "background" / "living room.wav",  # shared (no recorder in the name)
             (0.01 * rng.standard_normal(3 * 16000)).astype(np.float32), 16000)
    return cm.DataPaths(indian=ind, esc50=esc)


def smoke(work_dir: Path, out_dir: Path | None = None) -> list[Path]:
    """Generate the mini-dataset in ``work_dir``, run everything, check the CSVs."""
    paths = make_smoke_data(Path(work_dir) / "data")
    written = run_all(out_dir or Path(work_dir) / "results", paths, exp8_minutes=0.25)
    check_csvs(written)
    missing = set(EXPECTED_FIELDS) - {p.name for p in written}
    if missing:
        raise AssertionError(f"not written: {sorted(missing)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model experiments.")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--out", type=Path, default=None, help="results folder (default eval/model/results)")
    parser.add_argument("--limit", type=int, default=None, help="at most N files per dataset (quick check)")
    parser.add_argument("--exp8-minutes", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true", help="run on a synthetic mini-dataset in a temp folder")
    args = parser.parse_args()

    if args.smoke:
        with tempfile.TemporaryDirectory() as tmp:
            written = smoke(Path(tmp), args.out)
            print(f"\nSmoke test OK: {len(written)} CSVs with the documented columns"
                  + (f" in {args.out}" if args.out else " (temporary, deleted)"))
        return
    written = run_all(args.out or cm.RESULTS_DIR, only=tuple(args.only), limit=args.limit,
                      exp8_minutes=args.exp8_minutes)
    check_csvs(written)
    print(f"\nWrote {len(written)} CSVs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
