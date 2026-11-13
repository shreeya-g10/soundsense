"""Noise robustness: F1 vs SNR with DEMAND household noise, YAMNet alone and with custom profiles.

Owner: Shreeya (Role 1). Part S7d.

Every clean test clip is analysed as is ("clean") and mixed with each DEMAND
noise (DKITCHEN, DLIVING, DWASHING) at 20 / 10 / 5 / 0 dB SNR (eval/model/
mixer.py: active-part SNR, seeded, in memory only). Systems:
    yamnet          top category if >= 0.15, else "none" (Experiment 1)
    yamnet+custom   best matching custom profile's category, else yamnet
                    (Experiment 4; only where profiles exist)

Datasets (each skipped if missing):
    indian     per person: their own profiles (profile/), their own quiet-
               condition test clips (targets + hard negatives) as the clean set
    esc50      stand-in: profiles from fold 1, folds 2-5 as the clean set
    fsd50k     yamnet only (no profile split): targets + Speech/Music negatives
    freesound  yamnet only

    python -m eval.model.exp_noise
    python -m eval.model.exp_noise --limit 200      # quick look

Writes noise_summary.csv and noise_per_class.csv.
"""

from __future__ import annotations

import argparse
import tempfile
import zlib
from pathlib import Path

import numpy as np

from eval.model import common as cm
from eval.model import datasets as ds
from eval.model.exp4_custom import predictions
from eval.model.metrics import classification_report
from eval.model.mixer import SNRS, mix

NAME = "noise"
CLEAN = "clean"
ALL_NOISES = "all"
SUMMARY_FIELDS = ["dataset", "recorder", "noise", "snr_db", "system", "level", "n", "files", "accuracy",
                  "macro_f1", "macro_f1_clean", "delta_vs_clean"]
PER_CLASS_FIELDS = ["dataset", "recorder", "noise", "snr_db", "system", "level", "label", "support", "tp", "fp",
                    "fn", "precision", "recall", "f1"]


class _NoProfiles:
    def match(self, embedding):
        return []


def load_noise(paths: cm.DataPaths) -> dict[str, np.ndarray]:
    """One long noise signal per DEMAND environment (its channels joined)."""
    return {env: np.concatenate([ds.load_audio(f) for f in files]) for env, files in cm.get_demand(paths).items()}


def seed_for(clip: ds.Clip, noise: str, snr) -> int:
    return zlib.crc32(f"{clip.path.name}|{noise}|{snr}".encode())


def conditions(noises: dict[str, np.ndarray]) -> list[tuple[str, int | None]]:
    return [(CLEAN, None)] + [(env, snr) for env in noises for snr in SNRS]


def run_clips(clips: list[ds.Clip], noises: dict[str, np.ndarray], extractor: cm.FeatureExtractor, store,
              threshold: float, recorder: str | None, label: str) -> dict[tuple[str, int | None], list[dict]]:
    """{(noise, snr): exp4-style predictions for every clip}."""
    out: dict[tuple[str, int | None], list[dict]] = {c: [] for c in conditions(noises)}
    for i, clip in enumerate(clips, 1):
        audio = clip.load()
        for noise, snr in out:
            if noise == CLEAN:
                feat = extractor.features(clip)
            else:
                rng = np.random.default_rng(seed_for(clip, noise, snr))
                feat = extractor.clip_features(clip, mix(audio, noises[noise], snr, rng))
            out[(noise, snr)] += predictions([feat], store, threshold, recorder)
        if extractor.verbose and (i % 100 == 0 or i == len(clips)):
            print(f"    {label}: {i}/{len(clips)} clips x {len(out)} conditions", flush=True)
    return out


def evaluate(dataset: str, who: str, preds: dict, systems: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    summary, per_class = [], []
    groups = dict(preds)
    for snr in SNRS:  # every noise together, per SNR
        groups[(ALL_NOISES, snr)] = [p for (noise, s), ps in preds.items() if noise != CLEAN and s == snr for p in ps]
    clean_f1: dict[tuple[str, str], float | None] = {}
    order = lambda kv: (kv[0][0] != CLEAN, str(kv[0][0]), -(99 if kv[0][1] is None else kv[0][1]))  # 0 dB is not None
    for (noise, snr), group in sorted(groups.items(), key=order):
        for system in systems:
            for level in ("window", "clip"):
                if level == "window":
                    y_true = [p["true"] for p in group for _ in p["windows"]]
                    y_pred = [w[system] for p in group for w in p["windows"]]
                else:
                    y_true = [p["true"] for p in group]
                    y_pred = [p["clip"][system] for p in group]
                if not y_true:
                    continue
                rep = classification_report(y_true, y_pred, cm.LABELS)
                if noise == CLEAN:
                    clean_f1[(system, level)] = rep["macro_f1"]
                base_f1 = clean_f1.get((system, level))
                base = {"dataset": dataset, "recorder": who, "noise": noise,
                        "snr_db": "" if snr is None else snr, "system": system, "level": level}
                summary.append({**base, "n": rep["n"], "files": len(group), "accuracy": rep["accuracy"],
                                "macro_f1": rep["macro_f1"], "macro_f1_clean": base_f1,
                                "delta_vs_clean": None if rep["macro_f1"] is None or base_f1 is None
                                else rep["macro_f1"] - base_f1})
                per_class += [{**base, **r} for r in rep["per_class"]]
    return summary, per_class


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, threshold: float = cm.DETECTION_THRESHOLD,
        limit: int | None = None) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    try:
        noises = load_noise(paths)
    except cm.DataMissing as exc:
        cm.skip(NAME, "all datasets", exc)
        return []
    print(f"[{NAME}] noise: {', '.join(f'{k} {v.size / 16000:.0f} s' for k, v in noises.items())}; "
          f"SNRs {', '.join(map(str, SNRS))} dB", flush=True)
    summary, per_class = [], []
    jobs = []  # (dataset, who, clips, store_split or None)
    for dataset in (cm.INDIAN, cm.ESC50):
        try:
            for split in cm.custom_splits(dataset, paths):
                clean = split.test
                if dataset == cm.INDIAN:
                    clean = [c for c in clean if c.recorder == split.recorder and c.condition == "quiet"
                             and c.source == "indian"]
                jobs.append((dataset, split.who, cm.subset(clean, limit), split))
        except cm.DataMissing as exc:
            cm.skip(NAME, dataset, exc)
    for dataset, get in ((cm.FSD50K, cm.get_fsd50k), (cm.FREESOUND, cm.get_freesound)):
        try:
            jobs.append((dataset, "all", cm.subset(get(paths), limit), None))
        except cm.DataMissing as exc:
            cm.skip(NAME, dataset, exc)
    for dataset, who, clips, split in jobs:
        if not clips:
            print(f"[{NAME}] {dataset} {who}: no clean clips; skipped", flush=True)
            continue
        print(f"[{NAME}] {dataset} {who}: {len(clips)} clean clips x {1 + len(noises) * len(SNRS)} conditions"
              + ("" if split else " (yamnet only: no profiles)"), flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            if split is not None:
                store, _ = cm.build_store(split, extractor, Path(tmp))
                systems = ("yamnet", "yamnet+custom")
            else:
                store, systems = _NoProfiles(), ("yamnet",)
            preds = run_clips(clips, noises, extractor, store, threshold, split.recorder if split else None,
                              f"{dataset} {who}")
        s, pc = evaluate(dataset, who, preds, systems)
        summary += s
        per_class += pc
        for r in s:
            if r["level"] == "clip" and r["noise"] in (CLEAN, ALL_NOISES):
                print(f"[{NAME}] {dataset} {who} {r['system']:14s} clip macro F1 "
                      f"{'clean' if r['noise'] == CLEAN else str(r['snr_db']) + ' dB':>6s}: {cm.fmt(r['macro_f1'])}",
                      flush=True)
    if not summary:
        print(f"[{NAME}] nothing to evaluate; no CSVs written.")
        return []
    return [cm.write_csv(out_dir, "noise_summary.csv", SUMMARY_FIELDS, summary),
            cm.write_csv(out_dir, "noise_per_class.csv", PER_CLASS_FIELDS, per_class)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=cm.DETECTION_THRESHOLD)
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="at most N clean clips per dataset/person")
    args = parser.parse_args()
    for p in run(args.out, threshold=args.threshold, limit=args.limit):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
