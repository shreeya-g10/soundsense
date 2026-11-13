"""Experiment 1: YAMNet alone. Top category per window vs the file's label.

Owner: Shreeya (Role 1). Part S7.

For every non-silent window: prediction = the category with the highest
score if it is >= --threshold (default 0.15), else "none". The truth is the
file's category, or "none" for negatives (ESC-50 negatives, hard_negatives/).
Also at clip level (as one alert): the category of the clip's single highest
window score, if >= threshold, else "none".

Reported for ESC-50 and the Indian test set separately, and for the Indian
set per condition (quiet / tv / fan / far). Missing datasets are skipped.

    python -m eval.model.exp1_yamnet
    python -m eval.model.exp1_yamnet --threshold 0.3

Writes (eval/model/results/): exp1_summary.csv, exp1_per_class.csv,
exp1_confusion.csv, exp1_threshold_sweep.csv (columns: eval/model/README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.model import common as cm
from eval.model.metrics import classification_report, confusion_rows, predict_top

NAME = "exp1"
SUMMARY_FIELDS = ["dataset", "condition", "level", "threshold", "n", "files", "silent_windows", "accuracy", "macro_f1"]
PER_CLASS_FIELDS = ["dataset", "condition", "level", "threshold", "label", "support", "tp", "fp", "fn",
                    "precision", "recall", "f1"]
CONFUSION_FIELDS = ["dataset", "condition", "level", "threshold", "true", "pred", "count"]
SWEEP_FIELDS = ["dataset", "level", "threshold", "accuracy", "macro_f1"]
SWEEP = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def window_predictions(feats: list[cm.ClipFeat], threshold: float) -> tuple[list[str], list[str]]:
    y_true, y_pred = [], []
    for f in feats:
        t = cm.true_label(f.clip)
        for w in f.windows:
            y_true.append(t)
            y_pred.append(predict_top(w.category_scores(), threshold, cm.NONE))
    return y_true, y_pred


def clip_prediction(f: cm.ClipFeat, threshold: float) -> str:
    if not f.windows:
        return cm.NONE
    best = max(f.windows, key=lambda w: float(w.scores.max()))
    return predict_top(best.category_scores(), threshold, cm.NONE)


def clip_predictions(feats: list[cm.ClipFeat], threshold: float) -> tuple[list[str], list[str]]:
    return [cm.true_label(f.clip) for f in feats], [clip_prediction(f, threshold) for f in feats]


def groups(dataset: str, feats: list[cm.ClipFeat]) -> list[tuple[str, list[cm.ClipFeat]]]:
    """("all", every file) plus one group per condition present (if more than one kind)."""
    out = [("all", feats)]
    conditions = sorted({f.clip.condition for f in feats})
    if dataset == cm.INDIAN or len(conditions) > 1:
        out += [(c, [f for f in feats if f.clip.condition == c]) for c in conditions]
    return out


def evaluate(dataset: str, feats: list[cm.ClipFeat], threshold: float) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"summary": [], "per_class": [], "confusion": [], "sweep": []}
    for condition, group in groups(dataset, feats):
        for level, predict in (("window", window_predictions), ("clip", clip_predictions)):
            y_true, y_pred = predict(group, threshold)
            if not y_true:
                continue
            rep = classification_report(y_true, y_pred, cm.LABELS)
            base = {"dataset": dataset, "condition": condition, "level": level, "threshold": threshold}
            rows["summary"].append({**base, "n": rep["n"], "files": len(group),
                                    "silent_windows": sum(f.silent for f in group),
                                    "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]})
            rows["per_class"] += [{**base, **r} for r in rep["per_class"]]
            rows["confusion"] += [{**base, **r} for r in confusion_rows(rep["confusion"], cm.LABELS)]
    for level, predict in (("window", window_predictions), ("clip", clip_predictions)):
        for t in SWEEP:
            y_true, y_pred = predict(feats, t)
            if y_true:
                rep = classification_report(y_true, y_pred, cm.LABELS)
                rows["sweep"].append({"dataset": dataset, "level": level, "threshold": t,
                                      "accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]})
    return rows


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, threshold: float = cm.DETECTION_THRESHOLD,
        limit: int | None = None) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    all_rows: dict[str, list[dict]] = {"summary": [], "per_class": [], "confusion": [], "sweep": []}
    for dataset, get in ((cm.ESC50, cm.get_esc50), (cm.INDIAN, cm.get_indian_test)):
        try:
            clips = get(paths)
        except cm.DataMissing as exc:
            cm.skip(NAME, dataset, exc)
            continue
        clips = cm.subset(clips, limit)
        print(f"[{NAME}] {dataset}: {len(clips)} files", flush=True)
        feats = extractor.many(clips, label=dataset)
        for k, v in evaluate(dataset, feats, threshold).items():
            all_rows[k] += v
        for r in all_rows["summary"]:
            if r["dataset"] == dataset and r["condition"] == "all":
                print(f"[{NAME}] {dataset} {r['level']:6s}: n={r['n']}, accuracy {cm.fmt(r['accuracy'])}, "
                      f"macro F1 {cm.fmt(r['macro_f1'])}", flush=True)
    if not all_rows["summary"]:
        print(f"[{NAME}] nothing to evaluate; no CSVs written.")
        return []
    return [
        cm.write_csv(out_dir, "exp1_summary.csv", SUMMARY_FIELDS, all_rows["summary"]),
        cm.write_csv(out_dir, "exp1_per_class.csv", PER_CLASS_FIELDS, all_rows["per_class"]),
        cm.write_csv(out_dir, "exp1_confusion.csv", CONFUSION_FIELDS, all_rows["confusion"]),
        cm.write_csv(out_dir, "exp1_threshold_sweep.csv", SWEEP_FIELDS, all_rows["sweep"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=cm.DETECTION_THRESHOLD)
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="first N files per dataset (quick check)")
    args = parser.parse_args()
    for p in run(args.out, threshold=args.threshold, limit=args.limit):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

