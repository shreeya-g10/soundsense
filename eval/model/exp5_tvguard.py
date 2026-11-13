"""TV-guard test mixes for Experiment 5 (Vaidehi's TV guard): target sounds under speech/music.

Owner: Shreeya (Role 1). Part S7d. Results feed Vaidehi's Experiment 5 (eval/system/).

Target clips (from every available dataset: FSD50K targets, ESC-50 targets,
Indian quiet-condition targets, Freesound; at most --per-category per source
and category, spread over the files) are analysed:
    clean      the target alone
    speech/music at 20, 10, 5, 0 dB: the target mixed under an FSD50K Speech
               or Music clip (TV stand-in; FSD50K has no Television class),
               SNR = target (active part) vs background, eval/model/mixer.py
and the backgrounds alone ("tv_only": up to --tv-only clips per type; these
must NOT alert). Mixes are rebuilt from the seed, never saved.

For every non-silent window the CSV has YAMNet's 9 category scores and 3
context scores (speech, music, television), so the TV guard can be replayed
offline on exactly these mixes.

    python -m eval.model.exp5_tvguard

Writes exp5_tvguard_windows.csv (one row per window) and exp5_tvguard_summary.csv.
"""

from __future__ import annotations

import argparse
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from contracts.types import CATEGORIES, CONTEXTS
from eval.model import common as cm
from eval.model import datasets as ds
from eval.model.exp1_yamnet import clip_prediction
from eval.model.metrics import predict_top
from eval.model.mixer import SNRS, mix

NAME = "exp5"
PER_CATEGORY = 30
TV_ONLY = 100
WINDOW_FIELDS = (["mix_id", "source", "target_file", "target_label", "background_type", "background_file",
                  "snr_db", "window", "start_s", "rms"] + [f"score_{c}" for c in CATEGORIES]
                 + [f"ctx_{c}" for c in CONTEXTS] + ["yamnet_pred"])
SUMMARY_FIELDS = ["background_type", "snr_db", "target_label", "mixes", "windows", "detection_rate",
                  "any_alert_rate", "mean_best_target_score", "mean_ctx_speech", "mean_ctx_music",
                  "mean_ctx_television"]


def target_clips(paths: cm.DataPaths, per_category: int) -> list[ds.Clip]:
    clips: list[ds.Clip] = []
    sources = [(cm.FSD50K, cm.get_fsd50k), (cm.ESC50, cm.get_esc50), (cm.INDIAN, cm.get_indian_test),
               (cm.FREESOUND, cm.get_freesound)]
    for name, get in sources:
        try:
            found = [c for c in get(paths) if c.category and (name != cm.INDIAN or c.condition == "quiet")]
        except cm.DataMissing as exc:
            cm.skip(NAME, f"{name} targets", exc)
            continue
        by_cat = defaultdict(list)
        for c in found:
            by_cat[c.category].append(c)
        for cat in CATEGORIES:
            clips += cm.subset(sorted(by_cat[cat], key=lambda c: c.path.name), per_category)
    return clips


def snr_order(snr) -> float:
    """Sort key: clean first, then 20 -> 0 dB, then tv_only."""
    s = str(snr)
    return -1e9 if s == "clean" else 1e9 if s == "tv_only" else -float(s)


def _rows(mix_id: str, meta: dict, feat: cm.ClipFeat, threshold: float) -> list[dict]:
    rows = []
    for i, w in enumerate(feat.windows):
        row = {"mix_id": mix_id, **meta, "window": i, "start_s": w.start_s, "rms": w.rms,
               "yamnet_pred": predict_top(w.category_scores(), threshold, cm.NONE)}
        row.update({f"score_{c}": float(v) for c, v in zip(CATEGORIES, w.scores)})
        row.update({f"ctx_{c}": float(v) for c, v in zip(CONTEXTS, w.context)})
        rows.append(row)
    return rows


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, threshold: float = cm.DETECTION_THRESHOLD,
        per_category: int = PER_CATEGORY, tv_only: int = TV_ONLY) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    try:
        backgrounds = cm.get_tv_backgrounds(paths)
    except cm.DataMissing as exc:
        cm.skip(NAME, "TV-guard mixes", exc)
        return []
    targets = target_clips(paths, per_category)
    if not targets:
        print(f"[{NAME}] no target clips; no CSVs written.")
        return []
    print(f"[{NAME}] {len(targets)} target clips x (clean + {len(backgrounds)} backgrounds x {len(SNRS)} SNRs), "
          f"+ {min(tv_only, min(map(len, backgrounds.values())))} TV-only clips per type", flush=True)
    windows: list[dict] = []
    clips_out: list[dict] = []  # per mix: summary inputs

    def record(mix_id, meta, feat, target_label):
        windows.extend(_rows(mix_id, meta, feat, threshold))
        best = max((float(w.scores[CATEGORIES.index(target_label)]) for w in feat.windows), default=0.0) \
            if target_label != cm.NONE else None
        clips_out.append({"background_type": meta["background_type"], "snr_db": meta["snr_db"],
                          "target_label": target_label, "pred": clip_prediction(feat, threshold),
                          "n_windows": len(feat.windows), "best": best,
                          "ctx": [w.context for w in feat.windows]})

    for i, clip in enumerate(targets, 1):
        audio = clip.load()
        base = {"source": clip.source, "target_file": clip.path.name, "target_label": clip.category}
        record(f"{clip.source}:{clip.path.stem}:clean", {**base, "background_type": "none", "background_file": "",
                                                          "snr_db": "clean"}, extractor.features(clip), clip.category)
        for bg_type, pool in backgrounds.items():
            rng = np.random.default_rng(zlib.crc32(f"{clip.path.name}|{bg_type}".encode()))
            bg = pool[int(rng.integers(len(pool)))]
            bg_audio = bg.load()
            for snr in SNRS:
                mixed = mix(audio, bg_audio, snr, rng)
                meta = {**base, "background_type": bg_type, "background_file": bg.path.name, "snr_db": snr}
                record(f"{clip.source}:{clip.path.stem}:{bg_type}:{snr}", meta,
                       extractor.clip_features(clip, mixed), clip.category)
        if extractor.verbose and (i % 50 == 0 or i == len(targets)):
            print(f"    {i}/{len(targets)} targets mixed", flush=True)
    for bg_type, pool in backgrounds.items():
        for bg in cm.subset(sorted(pool, key=lambda c: c.path.name), tv_only):
            meta = {"source": bg.source, "target_file": "", "target_label": cm.NONE, "background_type": bg_type,
                    "background_file": bg.path.name, "snr_db": "tv_only"}
            record(f"{bg.source}:{bg.path.stem}:tv_only", meta, extractor.features(bg), cm.NONE)

    summary = []
    keys = sorted({(c["background_type"], str(c["snr_db"]), c["target_label"]) for c in clips_out},
                  key=lambda k: (k[0], snr_order(k[1]), k[2]))
    for bg_type, snr, label in keys:
        group = [c for c in clips_out if (c["background_type"], str(c["snr_db"]), c["target_label"]) == (bg_type, snr, label)]
        ctx = np.array([v for c in group for v in c["ctx"]]) if any(c["ctx"] for c in group) else np.zeros((0, 3))
        n = len(group)
        summary.append({
            "background_type": bg_type, "snr_db": snr, "target_label": label, "mixes": n,
            "windows": sum(c["n_windows"] for c in group),
            "detection_rate": None if label == cm.NONE else sum(c["pred"] == label for c in group) / n,
            "any_alert_rate": sum(c["pred"] != cm.NONE for c in group) / n,
            "mean_best_target_score": None if label == cm.NONE else float(np.mean([c["best"] for c in group])),
            **{f"mean_ctx_{name}": float(ctx[:, j].mean()) if len(ctx) else None for j, name in enumerate(CONTEXTS)},
        })
    for r in summary:
        if r["target_label"] == cm.NONE:
            print(f"[{NAME}] TV only ({r['background_type']}): false alerts {cm.fmt(r['any_alert_rate'])} "
                  f"of {r['mixes']} clips", flush=True)
    det = defaultdict(list)
    for c in clips_out:
        if c["target_label"] != cm.NONE:
            det[(c["background_type"], str(c["snr_db"]))].append(c["pred"] == c["target_label"])
    for (bg_type, snr), hits in sorted(det.items(), key=lambda kv: (kv[0][0], snr_order(kv[0][1]))):
        print(f"[{NAME}] targets {bg_type:6s} {snr:>5s}: detected {np.mean(hits):.2f} of {len(hits)}", flush=True)
    return [cm.write_csv(out_dir, "exp5_tvguard_windows.csv", WINDOW_FIELDS, windows),
            cm.write_csv(out_dir, "exp5_tvguard_summary.csv", SUMMARY_FIELDS, summary)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=cm.DETECTION_THRESHOLD)
    parser.add_argument("--per-category", type=int, default=PER_CATEGORY)
    parser.add_argument("--tv-only", type=int, default=TV_ONLY)
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    args = parser.parse_args()
    for p in run(args.out, threshold=args.threshold, per_category=args.per_category, tv_only=args.tv_only):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
