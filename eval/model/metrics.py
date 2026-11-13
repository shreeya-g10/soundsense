"""Classification and detection metrics for the model experiments (pure numpy).

Owner: Shreeya (Role 1). Part S7.

Undefined values are None (written as empty CSV cells), never a silent 0:
    precision  None when nothing was predicted as the class (tp + fp == 0)
    recall     None when the class never occurs (tp + fn == 0)
    f1         None if precision or recall is None; 0.0 if both are 0
macro F1 averages F1 over the classes that occur in y_true (support > 0),
counting an undefined F1 there (never predicted) as 0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """``cm[i, j]`` = number of items with true label ``labels[i]`` predicted as ``labels[j]``.

    Raises:
        ValueError: if the lengths differ or a label is not in ``labels``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} items, y_pred {len(y_pred)}")
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"label {t if t not in index else p!r} not in labels")
        cm[index[t], index[p]] += 1
    return cm


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float | None:
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} items, y_pred {len(y_pred)}")
    return _div(sum(t == p for t, p in zip(y_true, y_pred)), len(y_true))


def per_class(cm: np.ndarray, labels: Sequence[str]) -> list[dict]:
    """One dict per label: label, support, tp, fp, fn, precision, recall, f1."""
    rows = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        rows.append({"label": label, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f1_score(p, r)})
    return rows


def macro_f1(rows: list[dict]) -> float | None:
    """Mean F1 over classes with support > 0 (undefined F1 there counts as 0)."""
    present = [r for r in rows if r["support"] > 0]
    if not present:
        return None
    return float(np.mean([r["f1"] or 0.0 for r in present]))


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    """{"n", "accuracy", "macro_f1", "per_class": [...], "confusion": cm}."""
    cm = confusion_matrix(y_true, y_pred, labels)
    rows = per_class(cm, labels)
    return {"n": len(y_true), "accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(rows),
            "per_class": rows, "confusion": cm}


def confusion_rows(cm: np.ndarray, labels: Sequence[str]) -> list[dict]:
    """Long format for charts: one dict per (true, pred) cell, including zeros."""
    return [{"true": t, "pred": p, "count": int(cm[i, j])}
            for i, t in enumerate(labels) for j, p in enumerate(labels)]


def detection(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Binary detection metrics: accuracy, precision, recall, f1, false_match_rate (fp / negatives)."""
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"accuracy": _div(tp + tn, tp + fp + fn + tn), "precision": p, "recall": r,
            "f1": f1_score(p, r), "false_match_rate": _div(fp, fp + tn)}


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Area under the ROC curve: P(random positive score > random negative score), ties count 1/2.

    Threshold-free separation of two score sets; 0.5 = no better than chance.
    None if either set is empty.
    """
    pos, neg = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size, np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    for value in np.unique(scores):  # average ranks of ties
        tied = scores == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def predict_top(scores: dict[str, float], threshold: float, none: str = "none") -> str:
    """The highest-scoring label if its score >= threshold, else ``none``.

    Ties go to the first label in ``scores`` order.
    """
    if not scores:
        return none
    best = max(scores, key=scores.get)
    return best if scores[best] >= threshold else none
