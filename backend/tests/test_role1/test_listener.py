"""Tests for backend/ml/category_map.py and backend/audio/listener.py (Part S4)."""

from __future__ import annotations

import dataclasses
import inspect
import threading
import time
import typing

import numpy as np
import pytest

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES
from backend.audio.listener import YamnetListener
from backend.ml import category_map, yamnet
from backend.ml.category_map import CATEGORY_CLASSES, CONTEXT_CLASSES, CategoryMapper
from contracts import types as contract
from contracts.types import CATEGORIES, CONTEXTS, EMBEDDING_SIZE, WindowResult

CSV = yamnet.MODEL_DIR / "assets" / "yamnet_class_map.csv"


@pytest.fixture(scope="module")
def class_names():
    if not CSV.exists():
        try:
            yamnet.YamnetModel.instance()
        except Exception as exc:
            pytest.skip(f"YAMNet unavailable: {exc}")
    return yamnet._read_class_names(CSV)


@pytest.fixture(scope="module")
def model():
    try:
        return yamnet.YamnetModel.instance()
    except Exception as exc:
        pytest.skip(f"YAMNet unavailable: {exc}")


# --- category mapping ------------------------------------------------------

def test_every_mapped_name_exists_in_real_class_map(class_names):
    assert len(class_names) == 521
    for target, names in {**CATEGORY_CLASSES, **CONTEXT_CLASSES}.items():
        for name in names:
            assert name in class_names, f"{target}: {name!r} not in yamnet_class_map.csv"
    CategoryMapper(class_names)  # must not raise


def test_mapping_covers_exactly_the_contract():
    assert set(CATEGORY_CLASSES) == set(CATEGORIES)
    assert set(CONTEXT_CLASSES) == set(CONTEXTS)


def test_missing_name_fails_loudly(class_names, monkeypatch):
    monkeypatch.setitem(category_map.CATEGORY_CLASSES, "door_knock", ["Knock", "Knok (typo)"])
    with pytest.raises(ValueError, match="Knok"):
        CategoryMapper(class_names)


def test_category_score_is_max_of_mapped_classes(class_names):
    mapper = CategoryMapper(class_names)
    scores = np.zeros(521, np.float32)
    scores[class_names.index("Whistle")] = 0.3
    scores[class_names.index("Hiss")] = 0.7
    scores[class_names.index("Speech")] = 0.9
    scores[class_names.index("Television")] = 0.4
    cats = mapper.category_scores(scores)
    assert list(cats) == CATEGORIES
    assert cats["pressure_cooker"] == pytest.approx(0.7)
    assert all(v == 0 for k, v in cats.items() if k != "pressure_cooker")
    assert mapper.context_scores(scores) == pytest.approx({"speech": 0.9, "music": 0.0, "television": 0.4})
    with pytest.raises(ValueError):
        mapper.category_scores(np.zeros(10))


# --- listener: driven by a fake mic -------------------------------------

class FakeMic:
    """Replaces MicCapture: plays given windows through the callback from a thread."""

    def __init__(self, callback, windows):
        self.callback, self.windows = callback, windows
        self.overflows = 0
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for i, w in enumerate(self.windows):
            if self._stop.is_set():
                return
            self.callback(w, 1000.0 + 0.48 * i)
        self._stop.wait()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()


def run_listener(windows, **kwargs) -> list[WindowResult]:
    got: list[WindowResult] = []
    listener = YamnetListener(got.append, **kwargs)
    listener._mic = FakeMic(listener._on_capture, windows)
    listener.start()
    deadline = time.monotonic() + 30
    while len(got) < len(windows) and time.monotonic() < deadline:
        time.sleep(0.01)
    listener.stop()
    n = len(got)
    time.sleep(0.05)
    assert len(got) == n, "on_window called after stop()"
    assert not listener.is_running()
    return got


def silent():
    return np.zeros(WINDOW_SAMPLES, np.float32)


def loud():
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def test_listener_matches_audio_listener_protocol():
    for name in ("start", "stop", "is_running"):
        assert callable(getattr(YamnetListener, name))
    params = inspect.signature(YamnetListener.__init__).parameters
    assert list(params)[:3] == ["self", "on_window", "device"]
    assert params["device"].default is None


def test_window_result_fields_match_contract(model):
    expected = {
        "timestamp": float, "rms": float, "skipped": bool,
        "category_scores": dict[str, float], "context_scores": dict[str, float],
        "embedding": list[float] | None, "custom_matches": list[contract.CustomMatch],
        "inference_ms": float, "room": str,
    }
    assert typing.get_type_hints(WindowResult) == expected

    (w,) = run_listener([loud()], room="Kitchen")
    assert [f.name for f in dataclasses.fields(w)] == list(expected)
    assert isinstance(w.timestamp, float) and w.timestamp == 1000.0
    assert isinstance(w.rms, float) and w.rms == pytest.approx(0.3 / np.sqrt(2), rel=1e-3)
    assert w.skipped is False
    assert list(w.category_scores) == CATEGORIES
    assert list(w.context_scores) == CONTEXTS
    for scores in (w.category_scores, w.context_scores):
        assert all(type(v) is float and 0.0 <= v <= 1.0 for v in scores.values())
    assert isinstance(w.embedding, list) and len(w.embedding) == EMBEDDING_SIZE
    assert all(type(v) is float for v in w.embedding)
    assert w.custom_matches == []
    assert isinstance(w.inference_ms, float) and w.inference_ms > 0
    assert w.room == "Kitchen"


def test_skipped_windows_follow_contract(model):
    results = run_listener([silent(), loud(), silent()])
    assert [w.skipped for w in results] == [True, False, True]
    for w in (results[0], results[2]):
        assert w.category_scores == {} and w.context_scores == {}
        assert w.embedding is None
        assert w.custom_matches == []
        assert w.inference_ms == 0.0
        assert w.rms == 0.0 and w.room == "Laptop"


def test_windows_delivered_in_capture_order(model):
    windows = [silent() if i % 2 else loud() for i in range(8)]
    results = run_listener(windows)
    assert [w.timestamp for w in results] == [1000.0 + 0.48 * i for i in range(8)]


def test_cpu_saver_off_never_skips(model):
    results = run_listener([silent(), silent()], cpu_saver=False)
    assert [w.skipped for w in results] == [False, False]
    assert all(len(w.category_scores) == 9 for w in results)


def test_inference_runs_on_worker_thread(model):
    threads = set()
    listener = YamnetListener(lambda w: threads.add(threading.current_thread().name))
    listener._mic = FakeMic(listener._on_capture, [loud()])
    listener.start()
    deadline = time.monotonic() + 30
    while not threads and time.monotonic() < deadline:
        time.sleep(0.01)
    listener.stop()
    assert threads == {"YamnetListener"}


def test_callback_errors_do_not_stop_listener(model):
    calls = []

    def bad(w):
        calls.append(w)
        raise ValueError("boom")

    listener = YamnetListener(bad)
    listener._mic = FakeMic(listener._on_capture, [loud(), silent(), loud()])
    listener.start()
    deadline = time.monotonic() + 30
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert listener.is_running()
    listener.stop()
    assert len(calls) == 3


def test_start_raises_runtime_error_when_mic_fails():
    class BrokenMic(FakeMic):
        def start(self):
            raise RuntimeError("no mic")

    listener = YamnetListener(lambda w: None)
    listener._mic = BrokenMic(listener._on_capture, [])
    with pytest.raises(RuntimeError):
        listener.start()
    assert not listener.is_running()
    listener.stop()  # safe after failed start
