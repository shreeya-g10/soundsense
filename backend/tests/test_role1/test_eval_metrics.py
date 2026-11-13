"""Tests for eval/model/metrics.py (Part S7). Hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest

from eval.model.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    confusion_rows,
    detection,
    f1_score,
    macro_f1,
    per_class,
    predict_top,
    roc_auc,
)

LABELS = ["doorbell", "dog_bark", "none"]
Y_TRUE = ["doorbell", "doorbell", "doorbell", "dog_bark", "dog_bark", "none", "none", "none"]
Y_PRED = ["doorbell", "doorbell", "none", "dog_bark", "doorbell", "none", "none", "dog_bark"]


def test_confusion_matrix_rows_are_truth_columns_are_predictions():
    cm = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    np.testing.assert_array_equal(cm, [[2, 0, 1],
                                       [1, 1, 0],
                                       [0, 1, 2]])
    assert cm.sum() == len(Y_TRUE)


def test_confusion_matrix_validates_input():
    with pytest.raises(ValueError, match="items"):
        confusion_matrix(["a"], [], ["a"])
    with pytest.raises(ValueError, match="'b'"):
        confusion_matrix(["a"], ["b"], ["a"])


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(5 / 8)
    assert accuracy([], []) is None
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_per_class_precision_recall_f1():
    rows = {r["label"]: r for r in per_class(confusion_matrix(Y_TRUE, Y_PRED, LABELS), LABELS)}
    d = rows["doorbell"]  # tp 2, fp 1 (a dog bark), fn 1
    assert (d["tp"], d["fp"], d["fn"], d["support"]) == (2, 1, 1, 3)
    assert d["precision"] == pytest.approx(2 / 3) and d["recall"] == pytest.approx(2 / 3)
    assert d["f1"] == pytest.approx(2 / 3)
    g = rows["dog_bark"]  # tp 1, fp 1, fn 1
    assert g["precision"] == pytest.approx(0.5) and g["recall"] == pytest.approx(0.5) and g["f1"] == pytest.approx(0.5)
    n = rows["none"]  # tp 2, fp 1, fn 1
    assert n["f1"] == pytest.approx(2 / 3)


def test_undefined_values_are_none_not_zero():
    labels = ["doorbell", "smoke_alarm", "none"]
    rows = {r["label"]: r for r in per_class(confusion_matrix(["doorbell", "none"], ["doorbell", "doorbell"], labels), labels)}
    s = rows["smoke_alarm"]  # never occurs, never predicted
    assert s["support"] == 0 and s["precision"] is None and s["recall"] is None and s["f1"] is None
    n = rows["none"]  # occurs once, never predicted: recall 0, precision undefined
    assert n["recall"] == 0.0 and n["precision"] is None and n["f1"] is None
    assert f1_score(0.0, 0.0) == 0.0 and f1_score(None, 1.0) is None


def test_macro_f1_averages_present_classes_only():
    rows = per_class(confusion_matrix(Y_TRUE, Y_PRED, LABELS + ["smoke_alarm"]), LABELS + ["smoke_alarm"])
    assert macro_f1(rows) == pytest.approx((2 / 3 + 0.5 + 2 / 3) / 3)  # smoke_alarm (support 0) ignored
    labels = ["a", "b"]  # "b" present but never predicted -> counts as 0
    assert macro_f1(per_class(confusion_matrix(["a", "b"], ["a", "a"], labels), labels)) == pytest.approx((2 / 3 + 0) / 2)
    assert macro_f1([]) is None


def test_classification_report_and_long_confusion():
    rep = classification_report(Y_TRUE, Y_PRED, LABELS)
    assert rep["n"] == 8 and rep["accuracy"] == pytest.approx(5 / 8)
    assert rep["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 2 / 3) / 3)
    rows = confusion_rows(rep["confusion"], LABELS)
    assert len(rows) == 9 and sum(r["count"] for r in rows) == 8
    assert {"true": "dog_bark", "pred": "doorbell", "count": 1} in rows
    assert {"true": "dog_bark", "pred": "none", "count": 0} in rows  # zeros kept for heatmaps


def test_detection_metrics():
    m = detection(tp=8, fp=2, fn=2, tn=88)
    assert m["precision"] == pytest.approx(0.8) and m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(0.8) and m["accuracy"] == pytest.approx(0.96)
    assert m["false_match_rate"] == pytest.approx(2 / 90)
    empty = detection(0, 0, 0, 0)
    assert all(v is None for v in empty.values())
    assert detection(0, 0, 5, 10)["precision"] is None and detection(0, 0, 5, 10)["recall"] == 0.0


def test_predict_top():
    scores = {"doorbell": 0.4, "dog_bark": 0.7, "baby_cry": 0.1}
    assert predict_top(scores, 0.5) == "dog_bark"
    assert predict_top(scores, 0.7) == "dog_bark"  # >= threshold
    assert predict_top(scores, 0.71) == "none"
    assert predict_top({}, 0.1) == "none"
    assert predict_top({"a": 0.5, "b": 0.5}, 0.1) == "a"  # tie -> first


def test_roc_auc():
    assert roc_auc([0.9, 0.8], [0.1, 0.2, 0.3]) == 1.0  # perfectly separated
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5  # ties count half
    assert roc_auc([0.9, 0.4], [0.5, 0.1]) == pytest.approx(3 / 4)  # 3 of 4 pairs ordered right
    assert roc_auc([], [0.1]) is None and roc_auc([0.1], []) is None
