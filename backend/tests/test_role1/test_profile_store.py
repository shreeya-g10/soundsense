"""Tests for backend/custom/profile_store.py (Part S5).

Most tests use synthetic, YAMNet-like (non-negative) embeddings and a fake
model/recorder, so they need no mic. One integration test uses real YAMNet.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES
from backend.audio.listener import YamnetListener
from backend.audio.silence import GateDecision
from backend.custom.profile_store import YamnetProfileStore
from contracts.types import EMBEDDING_SIZE, CustomMatch

rng = np.random.default_rng(42)


def base_vector(half: int = 0) -> np.ndarray:
    """Non-negative 1024-d vector supported on the first (0) or second (1) half."""
    v = np.zeros(EMBEDDING_SIZE, np.float32)
    sl = slice(0, 512) if half == 0 else slice(512, 1024)
    v[sl] = np.abs(rng.standard_normal(512))
    return v


def jitter(v: np.ndarray, amount: float = 0.1) -> np.ndarray:
    return np.clip(v + rng.normal(0, amount, v.shape), 0, None).astype(np.float32)


class FakeModel:
    """Embedding = spectrum magnitude, so different tones give different embeddings."""

    def infer(self, samples):
        spec = np.abs(np.fft.rfft(samples))[:EMBEDDING_SIZE].astype(np.float32)
        return type("Out", (), {"embedding": spec})()


def tone(freq: float, seconds: float = 3.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeRecorder:
    def __init__(self, audio):
        self.audio, self.calls = audio, 0

    def __call__(self, seconds):
        self.calls += 1
        return self.audio


@pytest.fixture
def store(tmp_path):
    return YamnetProfileStore(tmp_path, model=FakeModel(), recorder=FakeRecorder(tone(440)))


def enrol(store, pid, base, n=5):
    for _ in range(n):
        store.add_feedback(jitter(base).tolist(), "doorbell", pid, positive=True)


# --- contract basics ---------------------------------------------------

def test_create_list_and_errors(store):
    pid = store.create("My Bell", "doorbell")
    assert store.list() == [{"id": pid, "name": "My Bell", "category": "doorbell", "examples": 0}]
    with pytest.raises(ValueError):
        store.create("x", "not_a_category")
    with pytest.raises(ValueError):
        store.create("  ", "doorbell")
    for call in (
        lambda: store.delete("nope"),
        lambda: store.add_example_from_mic("nope"),
        lambda: store.test_from_mic("nope"),
        lambda: store.add_feedback([0.0] * 1024, "doorbell", "nope", True),
    ):
        with pytest.raises(KeyError):
            call()
    with pytest.raises(ValueError):
        store.add_feedback([0.0] * 1024, "bad", None, True)
    with pytest.raises(ValueError):
        store.set_method("svm")


def test_unknown_profile_fails_before_recording(store):
    with pytest.raises(KeyError):
        store.add_example_from_mic("nope")
    assert store._recorder.calls == 0


# --- matching ------------------------------------------------------------

def test_self_match_is_high(store):
    pid = store.create("My Bell", "doorbell")
    base = base_vector()
    enrol(store, pid, base)
    assert store.similarity(pid, base) > 0.95
    (m,) = store.match(base)
    assert m == CustomMatch(profile_id=pid, name="My Bell", category="doorbell", similarity=m.similarity)
    assert m.similarity > 0.95


def test_unrelated_vector_does_not_match(store):
    pid = store.create("My Bell", "doorbell")
    enrol(store, pid, base_vector(half=0))
    assert store.similarity(pid, base_vector(half=1)) < 0.1  # only jitter noise overlaps
    assert store.match(base_vector(half=1)) == []
    # A random non-negative vector (like an unrelated YAMNet embedding) is also below 0.80.
    other = np.abs(rng.standard_normal(EMBEDDING_SIZE)).astype(np.float32)
    assert store.similarity(pid, other) < store.threshold
    assert store.match(other) == []


def test_similarity_is_mean_of_top_k(store):
    pid = store.create("p", "doorbell")
    e = np.eye(EMBEDDING_SIZE, dtype=np.float32)
    q = (e[0] + e[1]) / np.sqrt(2)
    # cos(q, e0) = cos(q, e1) = 0.707; cos(q, q) = 1; cos(q, e2) = 0
    for v in (e[0], e[1], q, e[2], e[2]):
        store.add_feedback(v.tolist(), "doorbell", pid, True)
    assert store.similarity(pid, q) == pytest.approx((1 + 0.7071 + 0.7071) / 3, abs=1e-3)


def test_threshold_configurable_and_matches_sorted(tmp_path):
    store = YamnetProfileStore(tmp_path, model=FakeModel(), threshold=0.5)
    a, b = store.create("A", "doorbell"), store.create("B", "door_knock")
    base = base_vector()
    enrol(store, a, base)
    mix = jitter(base, 0.6)
    store.add_feedback(mix.tolist(), "door_knock", b, True)
    matches = store.match(base)
    assert [m.profile_id for m in matches][0] == a
    assert all(m.similarity >= 0.5 for m in matches)
    assert matches == sorted(matches, key=lambda m: m.similarity, reverse=True)
    store.threshold = 0.999
    assert store.match(base) == []


def test_profile_without_examples_never_matches(store):
    pid = store.create("empty", "doorbell")
    assert store.similarity(pid, base_vector()) == 0.0
    assert store.match(base_vector()) == []


# --- recording (fake mic + fake model) --------------------------------

def test_add_example_from_mic_drops_silent_windows(tmp_path):
    audio = np.concatenate([np.zeros(SAMPLE_RATE * 2, np.float32), tone(440, 3.0)])
    store = YamnetProfileStore(tmp_path, model=FakeModel(), recorder=FakeRecorder(audio))
    pid = store.create("Beep", "appliance_beep")
    n1 = store.add_example_from_mic(pid, seconds=5)
    # 5 s -> 9 windows; the first ~2 s are silent and dropped
    assert 0 < n1 < 9
    assert store.add_example_from_mic(pid, seconds=5) == 2 * n1


def test_all_silent_recording_adds_nothing(tmp_path):
    store = YamnetProfileStore(tmp_path, model=FakeModel(), recorder=FakeRecorder(np.zeros(3 * SAMPLE_RATE, np.float32)))
    pid = store.create("x", "doorbell")
    assert store.add_example_from_mic(pid) == 0
    assert store.test_from_mic(pid) == 0.0


def test_test_from_mic_returns_best_similarity(tmp_path):
    rec = FakeRecorder(tone(440))
    store = YamnetProfileStore(tmp_path, model=FakeModel(), recorder=rec)
    pid = store.create("A440", "appliance_beep")
    store.add_example_from_mic(pid)
    assert store.test_from_mic(pid) > 0.95
    rec.audio = tone(3000)
    assert store.test_from_mic(pid) < 0.5


# --- persistence -------------------------------------------------------------

def test_persistence_survives_restart(tmp_path):
    s1 = YamnetProfileStore(tmp_path, model=FakeModel())
    pid = s1.create("My Bell", "doorbell")
    base = base_vector()
    enrol(s1, pid, base)
    s1.add_feedback(base_vector(half=1).tolist(), "doorbell", pid, positive=False)
    s1.set_method("mfcc")
    sim = s1.similarity(pid, base)

    s2 = YamnetProfileStore(tmp_path, model=FakeModel())
    assert s2.list() == s1.list()
    assert s2.method == "mfcc"
    assert s2.similarity(pid, base) == pytest.approx(sim)
    assert len(s2._profiles[pid].negative) == 1


def test_files_hold_only_embeddings(tmp_path, store):
    pid = store.create("My Bell", "doorbell")
    store.add_example_from_mic(pid)
    index = json.loads((tmp_path / "profiles.json").read_text())
    assert index["profiles"] == [{"id": pid, "name": "My Bell", "category": "doorbell", "method": "embedding"}]
    with np.load(tmp_path / f"{pid}.npz") as npz:
        assert sorted(npz.files) == ["negative", "positive"]
        assert npz["positive"].shape[1] == EMBEDDING_SIZE
    assert not list(tmp_path.glob("*.wav")) and not list(tmp_path.glob("*.tmp"))


def test_delete_removes_files(tmp_path, store):
    keep, gone = store.create("Keep", "doorbell"), store.create("Gone", "doorbell")
    store.add_example_from_mic(gone)
    assert (tmp_path / f"{gone}.npz").exists()
    store.delete(gone)
    assert not (tmp_path / f"{gone}.npz").exists()
    assert [p["id"] for p in store.list()] == [keep]
    ids = [p["id"] for p in json.loads((tmp_path / "profiles.json").read_text())["profiles"]]
    assert ids == [keep]
    assert [p["id"] for p in YamnetProfileStore(tmp_path, model=FakeModel()).list()] == [keep]
    with pytest.raises(KeyError):
        store.delete(gone)


def test_base_model_feedback_saved(tmp_path, store):
    store.add_feedback(base_vector().tolist(), "smoke_alarm", None, positive=False)
    with np.load(tmp_path / "feedback.npz") as npz:
        assert npz["smoke_alarm_negative"].shape == (1, EMBEDDING_SIZE)


# --- listener integration (real YAMNet) -------------------------------

def test_listener_fills_custom_matches(tmp_path):
    try:
        from backend.ml.yamnet import YamnetModel

        YamnetModel.instance()
    except Exception as exc:
        pytest.skip(f"YAMNet unavailable: {exc}")

    store = YamnetProfileStore(tmp_path)
    pid = store.create("Whistle", "pressure_cooker")
    assert store.add_example_from_audio(pid, tone(1500, 3.0)) > 0

    listener = YamnetListener(lambda w: None, profile_store=store)
    listener._load_model()
    window = tone(1500, WINDOW_SAMPLES / SAMPLE_RATE)
    w = listener._analyse(window, 0.0, GateDecision(skipped=False, rms=0.2))
    assert [m.profile_id for m in w.custom_matches] == [pid]
    assert w.custom_matches[0].category == "pressure_cooker" and w.custom_matches[0].similarity > 0.9

    silent = listener._analyse(np.zeros(WINDOW_SAMPLES, np.float32), 0.0, GateDecision(skipped=True, rms=0.0))
    assert silent.custom_matches == []

    no_store = YamnetListener(lambda w: None)
    no_store._load_model()
    assert no_store._analyse(window, 0.0, GateDecision(skipped=False, rms=0.2)).custom_matches == []
