"""Experiment 4: YAMNet + custom profiles vs YAMNet alone.

Owner: Shreeya (Role 1). Part S7.

One custom profile per category is built from data/indian/profile/<category>/
(embedding method, centered scoring, default thresholds; profile/background/
room sound, if present, fills the background bank). It is tested on
data/indian/test/ only. Profile files are never test files: any test file
whose path or bytes equal a profile file is dropped (and counted).

Per window, three systems:
    yamnet          top category if >= threshold, else "none" (= Experiment 1)
    custom_only     category of the best matching profile, else "none"
    yamnet+custom   the best matching profile's category if any profile
                    matches, else the YAMNet prediction
and at clip level (one alert per file: the best-similarity match over the
clip's windows, else the YAMNet clip prediction).

Until the Indian recordings exist this runs an ESC-50 stand-in (profiles
from fold 1, test on folds 2-5), reported as dataset "esc50". Results for the
two datasets are never mixed.

    python -m eval.model.exp4_custom
    python -m eval.model.exp4_custom --dataset indian

Writes exp4_summary.csv, exp4_per_class.csv, exp4_confusion.csv.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from eval.model import common as cm
from eval.model.exp1_yamnet import clip_prediction
from eval.model.metrics import classification_report, confusion_rows, predict_top

NAME = "exp4"
SYSTEMS = ("yamnet", "custom_only", "yamnet+custom")
SUMMARY_FIELDS = ["dataset", "system", "condition", "level", "threshold", "n", "files", "profiles",
                  "profile_files", "removed_overlap", "accuracy", "macro_f1", "macro_f1_yamnet", "delta_macro_f1"]
PER_CLASS_FIELDS = ["dataset", "system", "condition", "level", "label", "support", "tp", "fp", "fn",
                    "precision", "recall", "f1"]
CONFUSION_FIELDS = ["dataset", "system", "condition", "level", "true", "pred", "count"]


def predictions(feats: list[cm.ClipFeat], store, threshold: float) -> list[dict]:
    """Per clip: truth, condition, and per-window / clip predictions of every system."""
    out = []
    for f in feats:
        windows = []
        best_match = None  # (similarity, category) over the clip
        for w in f.windows:
            matches = store.match(w.embedding)
            yam = predict_top(w.category_scores(), threshold, cm.NONE)
            custom = matches[0].category if matches else cm.NONE
            windows.append({"yamnet": yam, "custom_only": custom,
                            "yamnet+custom": custom if matches else yam})
            if matches and (best_match is None or matches[0].similarity > best_match[0]):
                best_match = (matches[0].similarity, matches[0].category)
        yam_clip = clip_prediction(f, threshold)
        clip = {"yamnet": yam_clip,
                "custom_only": best_match[1] if best_match else cm.NONE,
                "yamnet+custom": best_match[1] if best_match else yam_clip}
        out.append({"true": cm.true_label(f.clip), "condition": f.clip.condition, "windows": windows, "clip": clip})
    return out


def evaluate(split: cm.Split, preds: list[dict], threshold: float) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"summary": [], "per_class": [], "confusion": []}
    conditions = ["all"] + (sorted({p["condition"] for p in preds}) if split.dataset == cm.INDIAN else [])
    for condition in conditions:
        group = [p for p in preds if condition == "all" or p["condition"] == condition]
        for level in ("window", "clip"):
            reports = {}
            for system in SYSTEMS:
                if level == "window":
                    y_true = [p["true"] for p in group for _ in p["windows"]]
                    y_pred = [w[system] for p in group for w in p["windows"]]
                else:
                    y_true = [p["true"] for p in group]
                    y_pred = [p["clip"][system] for p in group]
                if y_true:
                    reports[system] = classification_report(y_true, y_pred, cm.LABELS)
            base_f1 = reports["yamnet"]["macro_f1"] if "yamnet" in reports else None
            for system, rep in reports.items():
                base = {"dataset": split.dataset, "system": system, "condition": condition, "level": level}
                delta = None if rep["macro_f1"] is None or base_f1 is None else rep["macro_f1"] - base_f1
                rows["summary"].append({**base, "threshold": threshold, "n": rep["n"], "files": len(group),
                                        "profiles": len(split.categories), "profile_files": len(split.profile),
                                        "removed_overlap": split.removed, "accuracy": rep["accuracy"],
                                        "macro_f1": rep["macro_f1"], "macro_f1_yamnet": base_f1,
                                        "delta_macro_f1": delta})
                rows["per_class"] += [{**base, **r} for r in rep["per_class"]]
                rows["confusion"] += [{**base, **r} for r in confusion_rows(rep["confusion"], cm.LABELS)]
    return rows


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, threshold: float = cm.DETECTION_THRESHOLD,
        limit: int | None = None, datasets: tuple[str, ...] = (cm.INDIAN, cm.ESC50)) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    all_rows: dict[str, list[dict]] = {"summary": [], "per_class": [], "confusion": []}
    for dataset in datasets:
        try:
            split = cm.custom_split(dataset, paths, limit)
        except cm.DataMissing as exc:
            cm.skip(NAME, dataset, exc)
            continue
        note = " (ESC-50 stand-in: fold 1 profiles, folds 2-5 test)" if dataset == cm.ESC50 else ""
        print(f"[{NAME}] {dataset}{note}: {len(split.categories)} profiles from {len(split.profile)} files, "
              f"{len(split.test)} test files"
              + (f", {split.removed} overlapping test files dropped" if split.removed else ""), flush=True)
        with tempfile.TemporaryDirectory() as tmp:  # embeddings only, deleted afterwards
            store, _ = cm.build_store(split, extractor, Path(tmp))
            feats = extractor.many(split.test, label=f"{dataset} test")
            preds = predictions(feats, store, threshold)
        for k, v in evaluate(split, preds, threshold).items():
            all_rows[k] += v
        for r in all_rows["summary"]:
            if r["dataset"] == dataset and r["condition"] == "all" and r["level"] == "window":
                delta = "" if r["system"] == "yamnet" or r["delta_macro_f1"] is None else \
                    f" (delta vs YAMNet {r['delta_macro_f1']:+.3f})"
                print(f"[{NAME}] {dataset} {r['system']:14s} window macro F1 {cm.fmt(r['macro_f1'])}{delta}",
                      flush=True)
    if not all_rows["summary"]:
        print(f"[{NAME}] nothing to evaluate; no CSVs written.")
        return []
    return [
        cm.write_csv(out_dir, "exp4_summary.csv", SUMMARY_FIELDS, all_rows["summary"]),
        cm.write_csv(out_dir, "exp4_per_class.csv", PER_CLASS_FIELDS, all_rows["per_class"]),
        cm.write_csv(out_dir, "exp4_confusion.csv", CONFUSION_FIELDS, all_rows["confusion"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=[cm.INDIAN, cm.ESC50], action="append",
                        help="default: indian (skipped if missing) and the esc50 stand-in")
    parser.add_argument("--threshold", type=float, default=cm.DETECTION_THRESHOLD)
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="at most N test files per dataset")
    args = parser.parse_args()
    for p in run(args.out, threshold=args.threshold, limit=args.limit,
                 datasets=tuple(args.dataset or (cm.INDIAN, cm.ESC50))):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
