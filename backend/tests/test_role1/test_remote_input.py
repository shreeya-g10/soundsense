"""Tests for backend/audio/remote_input.py (Part S6, Phase 5 RemoteAudioInput).

A fake YAMNet records every window it is given, so these tests check the
windowing exactly and need no model download.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np
import pytest

from backend.audio import remote_input
from backend.audio.capture import HOP_SAMPLES, WINDOW_SAMPLES
from backend.audio.remote_input import YamnetRemoteInput
from contracts import types as contract
from contracts.types import CATEGORIES, CONTEXTS, EMBEDDING_SIZE, LISTENER_SAMPLE_RATE, WindowResult

SR = LISTENER_SAMPLE_RATE
CHUNK = SR // 2  # 0.5 s, like the phone page


class FakeModel:
    def __init__(self, delay: float = 0.0):
        self.windows: list[np.ndarray] = []
        self.delay = delay

    def infer(self, samples):
        if self.delay:
            time.sleep(self.delay)
        self.windows.append(samples.copy())
        emb = np.full(EMBEDDING_SIZE, float(np.abs(samples).mean()), np.float32)
        return type("Out", (), {"scores": np.zeros(521, np.float32), "embedding": emb, "inference_ms": 1.5})()


class FakeMapper:
    def category_scores(self, scores):
        return {c: 0.0 for c in CATEGORIES}

    def context_scores(self, scores):
        return {c: 0.0 for c in CONTEXTS}


@pytest.fixture
def model(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(remote_input, "load_pipeline", lambda: (m, FakeMapper()))
    return m


def pcm(audio: np.ndarray) -> bytes:
    return np.asarray(audio, dtype="<f4").tobytes()


def loud(n: int, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def collect(**kwargs):
    got: list[WindowResult] = []
    lock = threading.Lock()

    def on_window(w):
        with lock:
            got.append(w)

    return YamnetRemoteInput(on_window, **kwargs), got


def push_chunks(remote, audio: np.ndarray, room: str, chunk: int = CHUNK) -> None:
    for i in range(0, audio.size, chunk):
        remote.push(pcm(audio[i:i + chunk]), room)


def n_windows(n_samples: int) -> int:
    return 0 if n_samples < WINDOW_SAMPLES else 1 + (n_samples - WINDOW_SAMPLES) // HOP_SAMPLES


def test_matches_remote_audio_input_protocol():
    params = list(inspect.signature(YamnetRemoteInput.__init__).parameters)
    assert params[:2] == ["self", "on_window"]
    push = inspect.signature(YamnetRemoteInput.push).parameters
    assert list(push) == ["self", "samples", "room"]
    assert set(n for n, _ in inspect.getmembers(contract.RemoteAudioInput, inspect.isfunction)) <= set(dir(YamnetRemoteInput))


def test_push_validates_bytes():
    remote, _ = collect()
    with pytest.raises(ValueError, match="multiple of 4"):
        remote.push(b"\x00" * 7, "Kitchen")
    with pytest.raises(TypeError):
        remote.push(np.zeros(4, np.float32), "Kitchen")
    with pytest.raises(TypeError):
        remote.push(b"\x00" * 8, None)
    remote.push(b"", "Kitchen")  # empty chunk: nothing to do, no worker started
    assert not remote.is_running()


def test_windows_are_096s_with_50_percent_overlap(model):
    remote, got = collect(calibrate=False)
    stream = (np.arange(3 * SR) / (3 * SR) * 0.5 + 0.1).astype(np.float32)  # ramp: each sample unique
    push_chunks(remote, stream, "Kitchen")
    remote.close()
    assert len(got) == n_windows(stream.size) == 5
    assert HOP_SAMPLES * 2 == WINDOW_SAMPLES  # 50% overlap
    for k, window in enumerate(model.windows):
        np.testing.assert_array_equal(window, stream[k * HOP_SAMPLES: k * HOP_SAMPLES + WINDOW_SAMPLES])
    assert all(w.room == "Kitchen" and not w.skipped for w in got)


def test_window_result_fields_follow_contract(model):
    remote, got = collect(calibrate=False)
    push_chunks(remote, loud(SR), "Kitchen")  # 1 s -> 1 window
    remote.close()
    (w,) = got
    assert isinstance(w.timestamp, float) and w.rms == pytest.approx(0.3 / np.sqrt(2), rel=1e-2)
    assert list(w.category_scores) == CATEGORIES and list(w.context_scores) == CONTEXTS
    assert isinstance(w.embedding, list) and len(w.embedding) == EMBEDDING_SIZE
    assert w.custom_matches == [] and w.inference_ms == 1.5 and w.room == "Kitchen"


def test_timestamps_follow_the_audio(model):
    remote, got = collect(calibrate=False)
    t0 = time.time()
    push_chunks(remote, loud(4 * SR), "Hall")
    remote.close()
    stamps = [w.timestamp for w in got]
    assert len(stamps) == n_windows(4 * SR)
    gaps = np.diff(stamps)
    assert np.all(gaps >= 0) and np.all(gaps < 0.48 + 0.25)  # monotonic, even pushed faster than real time
    assert max(stamps) <= time.time()  # never in the future
    assert t0 - 0.1 <= stamps[0] <= time.time()


def test_each_room_is_buffered_separately(model):
    remote, got = collect(calibrate=False)
    kitchen, hall = loud(3 * SR, 1000), loud(2 * SR, 500)
    for i in range(0, 3 * SR, CHUNK):  # interleaved frames from two phones
        remote.push(pcm(kitchen[i:i + CHUNK]), "Kitchen")
        if i < hall.size:
            remote.push(pcm(hall[i:i + CHUNK]), "Hall")
    remote.push(pcm(loud(CHUNK)), "Garden")  # 0.5 s: not a full window yet
    remote.close()
    by_room = {r: [w for w in got if w.room == r] for r in ("Kitchen", "Hall", "Garden")}
    assert len(by_room["Kitchen"]) == n_windows(kitchen.size)
    assert len(by_room["Hall"]) == n_windows(hall.size)
    assert by_room["Garden"] == []
    assert sorted(remote.rooms()) == ["Garden", "Hall", "Kitchen"]
    for room in ("Kitchen", "Hall"):  # in order within a room
        ts = [w.timestamp for w in by_room[room]]
        assert ts == sorted(ts)


def test_each_room_calibrates_its_own_silence_gate(model):
    remote, got = collect()  # calibrate=True by default
    rng = np.random.default_rng(0)
    quiet = (rng.standard_normal(4 * SR) * 0.005).astype(np.float32)
    noisy = (rng.standard_normal(4 * SR) * 0.02).astype(np.float32)  # x2 margin stays under MAX_THRESHOLD
    push_chunks(remote, quiet, "Bedroom")
    push_chunks(remote, noisy, "Kitchen")
    remote.close()
    for room, level in (("Bedroom", 0.005), ("Kitchen", 0.02)):
        ws = [w for w in got if w.room == room]
        assert [w.skipped for w in ws[:3]] == [False] * 3  # calibration windows run the model
        assert all(w.skipped for w in ws[3:])  # then that room's own noise is skipped
        assert remote.gate(room).ambient_level == pytest.approx(level, rel=0.1)
    stats = remote.stats()
    assert set(stats["rooms"]) == {"Bedroom", "Kitchen"} and stats["dropped_chunks"] == 0


def test_skipped_windows_follow_contract(model):
    remote, got = collect(calibrate=False)
    push_chunks(remote, np.zeros(SR, np.float32), "Kitchen")
    remote.close()
    (w,) = got
    assert w.skipped and w.category_scores == {} and w.context_scores == {}
    assert w.embedding is None and w.custom_matches == [] and w.inference_ms == 0.0 and w.room == "Kitchen"
    assert len(model.windows) == 0


def test_runs_custom_matching_and_feeds_calibration_background(model):
    calls = {"match": 0, "background": []}

    class Store:
        def match(self, embedding, samples=None):
            calls["match"] += 1
            assert samples is not None and samples.size == WINDOW_SAMPLES
            return [contract.CustomMatch("p1", "My Cooker", "pressure_cooker", 0.93)]

        def add_background(self, embedding, source):
            calls["background"].append(source)

    remote, got = collect(profile_store=Store())
    push_chunks(remote, loud(4 * SR), "Kitchen")
    remote.close()
    assert calls["match"] == len(got) == n_windows(4 * SR)
    assert calls["background"] == ["calibration"] * 3
    assert all(w.custom_matches[0].name == "My Cooker" for w in got)


def test_push_returns_quickly_and_work_runs_on_worker_thread(monkeypatch):
    slow = FakeModel(delay=0.05)
    monkeypatch.setattr(remote_input, "load_pipeline", lambda: (slow, FakeMapper()))
    threads = set()
    remote = YamnetRemoteInput(lambda w: threads.add(threading.current_thread().name), calibrate=False)
    t0 = time.perf_counter()
    push_chunks(remote, loud(4 * SR), "Kitchen")  # 7 windows x 50 ms of "inference"
    assert time.perf_counter() - t0 < 0.1
    remote.close()
    assert threads == {"RemoteAudioInput"} and len(slow.windows) == n_windows(4 * SR)


def test_close_waits_and_push_after_close_restarts(model):
    remote, got = collect(calibrate=False)
    push_chunks(remote, loud(SR), "Kitchen")
    remote.close()
    n = len(got)
    time.sleep(0.05)
    assert len(got) == n == 1 and not remote.is_running()
    remote.close()  # safe twice
    push_chunks(remote, loud(SR), "Kitchen")  # buffer continues: 0.04 s left over + 1 s
    remote.close()
    assert len(got) == 1 + n_windows(2 * SR) - 1


def test_callback_errors_do_not_stop_the_worker(model):
    calls = []

    def bad(w):
        calls.append(w)
        raise RuntimeError("boom")

    remote = YamnetRemoteInput(bad, calibrate=False)
    push_chunks(remote, loud(2 * SR), "Kitchen")
    remote.close()
    assert len(calls) == n_windows(2 * SR)


def test_full_queue_drops_oldest_chunks_and_restarts_room_buffer(monkeypatch):
    slow = FakeModel(delay=0.02)
    monkeypatch.setattr(remote_input, "load_pipeline", lambda: (slow, FakeMapper()))
    monkeypatch.setattr(remote_input, "_QUEUE_MAX_CHUNKS", 4)
    remote = YamnetRemoteInput(lambda w: None, calibrate=False)
    remote._queue = remote_input.queue.Queue(maxsize=4)
    push_chunks(remote, loud(20 * SR), "Kitchen")
    remote.close()
    assert remote.dropped_chunks > 0
    # Every analysed window is still made of contiguous audio (no stitching across a gap).
    for w in slow.windows:
        assert w.size == WINDOW_SAMPLES


def test_model_load_failure_sets_error(monkeypatch):
    def boom():
        raise OSError("no model")

    monkeypatch.setattr(remote_input, "load_pipeline", boom)
    remote, got = collect()
    remote.push(pcm(loud(CHUNK)), "Kitchen")
    deadline = time.monotonic() + 5
    while remote.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert isinstance(remote.error, OSError) and got == []
