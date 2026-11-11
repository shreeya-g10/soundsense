"""Tests for backend/ml/yamnet.py (Part S3).

Needs the YAMNet model: downloaded into data/models/ on first run (internet
required once), offline afterwards. Skipped if it can't be loaded.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES
from backend.ml import yamnet
from backend.ml.yamnet import EMBEDDING_SIZE, NUM_CLASSES, YamnetModel, top_k


@pytest.fixture(scope="module")
def model():
    try:
        return YamnetModel.instance()
    except Exception as exc:  # no cache and no internet
        pytest.skip(f"YAMNet unavailable: {exc}")


def test_output_shapes(model):
    out = model.infer(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    assert out.scores.shape == (NUM_CLASSES,) == (521,)
    assert out.embedding.shape == (EMBEDDING_SIZE,) == (1024,)
    assert out.scores.dtype == np.float32 and out.embedding.dtype == np.float32
    assert out.inference_ms > 0


@pytest.mark.parametrize("kind", ["silence", "noise"])
def test_inference_on_096s_runs(model, kind):
    rng = np.random.default_rng(0)
    window = (
        np.zeros(WINDOW_SAMPLES, np.float32)
        if kind == "silence"
        else (rng.standard_normal(WINDOW_SAMPLES) * 0.3).astype(np.float32)
    )
    out = model.infer(window)
    assert np.all(np.isfinite(out.scores)) and np.all(np.isfinite(out.embedding))
    assert np.all((out.scores >= 0) & (out.scores <= 1))


def test_silence_is_classified_as_silence(model):
    (name, _), *_ = top_k(model.infer(np.zeros(WINDOW_SAMPLES, np.float32)).scores, 1)
    assert name == "Silence"


def test_longer_input_is_averaged_over_frames(model):
    out = model.infer(np.zeros(3 * SAMPLE_RATE, np.float32))
    assert out.scores.shape == (521,) and out.embedding.shape == (1024,)


def test_class_names(model):
    assert len(model.class_names) == 521
    assert model.class_names[0] == "Speech"
    assert "Smoke detector, smoke alarm" in model.class_names


def test_top_k(model):
    scores = np.zeros(521, np.float32)
    scores[[0, 5, 9]] = [0.2, 0.9, 0.5]
    assert top_k(scores, 3) == [
        (model.class_names[5], pytest.approx(0.9)),
        (model.class_names[9], pytest.approx(0.5)),
        (model.class_names[0], pytest.approx(0.2)),
    ]
    with pytest.raises(ValueError):
        top_k(np.zeros(10), 3)


def test_bad_input_rejected(model):
    with pytest.raises(ValueError):
        model.infer(np.zeros((2, 100), np.float32))
    with pytest.raises(ValueError):
        model.infer(np.zeros(0, np.float32))


def test_model_loads_only_once(model):
    loads = YamnetModel.load_count
    got = []
    threads = [threading.Thread(target=lambda: got.append(YamnetModel.instance())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(m is model for m in got)
    assert YamnetModel.instance() is model
    assert YamnetModel.load_count == loads == 1


def test_model_cached_under_data_models(model):
    assert (yamnet.MODEL_DIR / "saved_model.pb").exists()
    assert yamnet.MODEL_DIR.parent.name == "models" and yamnet.MODEL_DIR.parents[1].name == "data"
