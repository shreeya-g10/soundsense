"""Experiment 6: custom sounds, YAMNet-embedding kNN vs MFCC baseline.

Owner: Shreeya (Role 1). Parts S7, S7c.

For each method ("embedding": centered top-3 cosine + margin rule; "mfcc":
40-d MFCC mean/std, top-3 cosine) one profile per category is built from the
same enrolment files, with the store's default thresholds. Then every test
window is scored against every profile:
    target windows     the profile's own sound: same category, recorded by
                       the same person (their device/home)
                       -> tp (matched) / fn (missed); recall = hit rate
    same category,     the same category recorded by someone else (another
    different device   device/home). Reported separately (match rate, mean
                       similarity); NOT counted in tp/fp/F1/AUC
    unrelated windows  windows of every other category and of negatives
                       (hard_negatives/, plus ESC-50 negatives when
                       downloaded) -> fp ("false match") / tn
Clip level: a clip counts as matched if any of its windows matched.
``auc`` (ROC AUC of target vs unrelated similarities; per clip: its best
window) compares the methods without depending on their thresholds.

indian: per person, profiles built only from that person's
data/indian/profile/ clips; tested on all of data/indian/test/ + ESC-50
negatives. Until those exist an ESC-50 stand-in runs (fold 1 profiles,
folds 2-5 test; recorder "all", so every same-category clip is a target),
reported as dataset "esc50".

    python -m eval.model.exp6_methods

Writes exp6_per_profile.csv and exp6_summary.csv.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from eval.model import common as cm
from eval.model.metrics import detection, roc_auc

NAME = "exp6"
METHODS = ("embedding", "mfcc")
PER_PROFILE_FIELDS = ["dataset", "recorder", "method", "category", "level", "threshold", "n_target", "n_unrelated",
                      "tp", "fn", "fp", "tn", "accuracy", "precision", "recall", "f1", "false_match_rate",
                      "fp_other_categories", "n_other_categories", "fp_negatives", "n_negatives",
                      "n_same_category_other_device", "matched_same_category_other_device",
                      "same_category_other_device_match_rate",
                      "mean_sim_target", "mean_sim_same_category_other_device", "mean_sim_unrelated",
                      "auc", "auc_other_device_vs_unrelated"]
SUMMARY_FIELDS = ["dataset", "recorder", "method", "level", "profiles", "macro_f1", "mean_recall", "mean_auc",
                  "false_match_rate", "false_match_rate_negatives", "same_category_other_device_match_rate",
                  "tp", "fn", "fp", "tn"]
SIDES = (cm.TARGET, cm.SAME_CATEGORY_OTHER_DEVICE, cm.UNRELATED)


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def score_split(split: cm.Split, feats: list[cm.ClipFeat], store, pids: dict[str, str], method: str) -> list[dict]:
    """Rows for one method: per (category, level)."""
    threshold = store.threshold_for(next(iter(pids.values()))) if pids else None
    rows = []
    for category, pid in pids.items():
        counts = {lvl: dict(tp=0, fn=0, fp=0, tn=0, fp_other=0, n_other=0, fp_neg=0, n_neg=0, n_dev=0, hit_dev=0)
                  for lvl in ("window", "clip")}
        sims = {lvl: {side: [] for side in SIDES} for lvl in ("window", "clip")}
        for f in feats:
            side = cm.relation(f.clip, category, split.recorder)
            negative = cm.true_label(f.clip) == cm.NONE
            hits, clip_sims = [], []
            for w in f.windows:
                d = store.score(pid, w.embedding, mfcc=w.mfcc)
                hits.append(d.matched)
                clip_sims.append(d.similarity)
            sims["window"][side].extend(clip_sims)
            if clip_sims:
                sims["clip"][side].append(max(clip_sims))
            for lvl, n_items, n_hit in (("window", len(hits), sum(hits)), ("clip", 1 if hits else 0, int(any(hits)))):
                c = counts[lvl]
                if side == cm.TARGET:
                    c["tp"] += n_hit
                    c["fn"] += n_items - n_hit
                elif side == cm.SAME_CATEGORY_OTHER_DEVICE:
                    c["hit_dev"] += n_hit
                    c["n_dev"] += n_items
                else:
                    c["fp"] += n_hit
                    c["tn"] += n_items - n_hit
                    key = "neg" if negative else "other"
                    c[f"fp_{key}"] += n_hit
                    c[f"n_{key}"] += n_items
        for lvl, c in counts.items():
            m = detection(c["tp"], c["fp"], c["fn"], c["tn"])
            s = sims[lvl]
            rows.append({"dataset": split.dataset, "recorder": split.who, "method": method, "category": category,
                         "level": lvl, "threshold": threshold, "n_target": c["tp"] + c["fn"],
                         "n_unrelated": c["fp"] + c["tn"], "tp": c["tp"], "fn": c["fn"], "fp": c["fp"],
                         "tn": c["tn"], **m,
                         "fp_other_categories": c["fp_other"], "n_other_categories": c["n_other"],
                         "fp_negatives": c["fp_neg"], "n_negatives": c["n_neg"],
                         "n_same_category_other_device": c["n_dev"],
                         "matched_same_category_other_device": c["hit_dev"],
                         "same_category_other_device_match_rate": c["hit_dev"] / c["n_dev"] if c["n_dev"] else None,
                         "mean_sim_target": _mean(s[cm.TARGET]),
                         "mean_sim_same_category_other_device": _mean(s[cm.SAME_CATEGORY_OTHER_DEVICE]),
                         "mean_sim_unrelated": _mean(s[cm.UNRELATED]),
                         "auc": roc_auc(s[cm.TARGET], s[cm.UNRELATED]),
                         "auc_other_device_vs_unrelated": roc_auc(s[cm.SAME_CATEGORY_OTHER_DEVICE], s[cm.UNRELATED])})
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Per (dataset, recorder, method, level), plus recorder "all" when a dataset has several recorders."""
    keys = sorted({(r["dataset"], r["recorder"], r["method"], r["level"]) for r in rows})
    for dataset in sorted({r["dataset"] for r in rows}):
        if len({r["recorder"] for r in rows if r["dataset"] == dataset}) > 1:
            keys += sorted({(dataset, "all", r["method"], r["level"]) for r in rows if r["dataset"] == dataset})
    out = []
    for dataset, recorder, method, level in keys:
        group = [r for r in rows if (r["dataset"], r["method"], r["level"]) == (dataset, method, level)
                 and (recorder == "all" or r["recorder"] == recorder)]
        tot = {k: sum(r[k] for r in group) for k in ("tp", "fn", "fp", "tn")}
        fp_neg, n_neg = sum(r["fp_negatives"] for r in group), sum(r["n_negatives"] for r in group)
        n_dev = sum(r["n_same_category_other_device"] for r in group)
        hit_dev = sum(r["matched_same_category_other_device"] for r in group)
        present = [r for r in group if r["n_target"] > 0]
        aucs = [r["auc"] for r in present if r["auc"] is not None]
        out.append({"dataset": dataset, "recorder": recorder, "method": method, "level": level,
                    "profiles": len(group),
                    "macro_f1": _mean([r["f1"] or 0.0 for r in present]),
                    "mean_recall": _mean([r["recall"] for r in present]),
                    "mean_auc": _mean(aucs),
                    "false_match_rate": detection(**tot)["false_match_rate"],
                    "false_match_rate_negatives": fp_neg / n_neg if n_neg else None,
                    "same_category_other_device_match_rate": hit_dev / n_dev if n_dev else None, **tot})
    return out


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, limit: int | None = None,
        datasets: tuple[str, ...] = (cm.INDIAN, cm.ESC50)) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    rows: list[dict] = []
    for dataset in datasets:
        try:
            splits = cm.custom_splits(dataset, paths, limit, esc50_negatives_in_test=True)
        except cm.DataMissing as exc:
            cm.skip(NAME, dataset, exc)
            continue
        note = " (ESC-50 stand-in: fold 1 profiles, folds 2-5 test)" if dataset == cm.ESC50 else ""
        for split in splits:
            print(f"[{NAME}] {dataset}{note} recorder {split.who}: {len(split.categories)} categories, "
                  f"{len(split.test)} test files", flush=True)
            feats = extractor.many(split.test, mfcc=True, label=f"{dataset} test (+MFCC)")
            for method in METHODS:
                with tempfile.TemporaryDirectory() as tmp:
                    store, pids = cm.build_store(split, extractor, Path(tmp), method=method)
                    method_rows = score_split(split, feats, store, pids, method)
                rows += method_rows
                for s in summarise(method_rows):
                    if s["level"] == "window":
                        dev = s["same_category_other_device_match_rate"]
                        print(f"[{NAME}] {dataset} {split.who} {method:9s} window: macro F1 {cm.fmt(s['macro_f1'])}, "
                              f"recall {cm.fmt(s['mean_recall'])}, false matches {cm.fmt(s['false_match_rate'])}, "
                              f"AUC {cm.fmt(s['mean_auc'])}"
                              + (f", other-device matches {cm.fmt(dev)}" if dev is not None else ""), flush=True)
    if not rows:
        print(f"[{NAME}] nothing to evaluate; no CSVs written.")
        return []
    return [cm.write_csv(out_dir, "exp6_per_profile.csv", PER_PROFILE_FIELDS, rows),
            cm.write_csv(out_dir, "exp6_summary.csv", SUMMARY_FIELDS, summarise(rows))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[cm.INDIAN, cm.ESC50], action="append",
                        help="default: indian (skipped if missing) and the esc50 stand-in")
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="at most N test files per dataset")
    args = parser.parse_args()
    for p in run(args.out, limit=args.limit, datasets=tuple(args.dataset or (cm.INDIAN, cm.ESC50))):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
