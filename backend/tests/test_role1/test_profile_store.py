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
        assert sorted(npz.files) == ["augmented", "negative", "positive"]
        assert all(npz[k].shape[1] == EMBEDDING_SIZE for k in npz.files)
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


# --- demo_profiles helpers (S5e) -------------------------------------------

def _window(skipped: bool = False):
    from contracts.types import WindowResult

    return WindowResult(timestamp=1000.0, rms=0.0 if skipped else 0.05, skipped=skipped,
                        category_scores={}, context_scores={}, embedding=None,
                        custom_matches=[], inference_ms=0.0, room="Laptop")


def test_demo_resolve_profile_by_id_or_name(store):
    from backend.custom.demo_profiles import resolve_profile

    pid = store.create("My Kitchen Cooker", "pressure_cooker")
    store.create("Doorbell", "doorbell")
    assert resolve_profile(store, pid)["id"] == pid
    assert resolve_profile(store, "my kitchen cooker")["id"] == pid
    with pytest.raises(SystemExit, match="--list"):
        resolve_profile(store, "nope")
    store.create("Doorbell", "doorbell")
    with pytest.raises(SystemExit, match="pass the id"):
        resolve_profile(store, "Doorbell")


def test_demo_format_profiles(store):
    from backend.custom.demo_profiles import format_profiles

    assert format_profiles([]) == "No saved profiles."
    pid = store.create("Cooker", "pressure_cooker")
    store.add_example_from_audio(pid, tone(440))
    lines = format_profiles(store.list()).splitlines()
    assert lines[0].split() == ["ID", "NAME", "CATEGORY", "EXAMPLES"]
    assert lines[1].split() == [pid, "Cooker", "pressure_cooker", str(store.list()[0]["examples"])]


def test_demo_line_shows_top_yamnet_class():
    from backend.custom.demo_profiles import format_line

    line = format_line(_window(), 0.84, True, [("Steam", 0.41), ("Hiss", 0.2)])
    assert "similarity 0.84 MATCH  [Steam 0.41]" in line
    assert "MATCH" not in format_line(_window(), 0.3, False, [("Speech", 0.9)])
    assert format_line(_window(skipped=True), None, False, []).endswith("skipped")


def test_demo_live_summary():
    from backend.custom.demo_profiles import LiveSummary

    assert "no windows tested" in str(LiveSummary())
    s = LiveSummary()
    for sim, hit in [(0.9, True), (0.5, False), (0.7, False)]:
        s.add(sim, hit)
    assert str(s) == "Summary: 3 windows tested, 1 matches (33%), similarity max 0.90, mean 0.70"


# --- S5f: background-contrast (centered) scoring ---------------------------

COMMON = np.abs(rng.standard_normal(EMBEDDING_SIZE)).astype(np.float32) * 3  # shared by every sound


def sound(seed: int, amount: float = 1.0) -> np.ndarray:
    """COMMON + a sound-specific non-negative part (YAMNet-like)."""
    r = np.random.default_rng(seed)
    return COMMON + amount * np.abs(r.standard_normal(EMBEDDING_SIZE)).astype(np.float32)


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_store(tmp_path, **kw):
    return YamnetProfileStore(tmp_path, model=FakeModel(), recorder=FakeRecorder(tone(440)), **kw)


def test_scoring_options_and_default_thresholds(tmp_path):
    from backend.custom.profile_store import CENTERED_MATCH_THRESHOLD, RAW_MATCH_THRESHOLD

    assert make_store(tmp_path).scoring == "centered"
    assert make_store(tmp_path).threshold == CENTERED_MATCH_THRESHOLD
    assert make_store(tmp_path, scoring="raw").threshold == RAW_MATCH_THRESHOLD == 0.80
    assert make_store(tmp_path, scoring="raw", threshold=0.5).threshold == 0.5
    with pytest.raises(ValueError):
        make_store(tmp_path, scoring="mfcc")
    s = make_store(tmp_path)
    with pytest.raises(ValueError):
        s.scoring = "bogus"


def test_centered_equals_raw_when_background_empty(tmp_path):
    raw, cen = make_store(tmp_path / "r", scoring="raw"), make_store(tmp_path / "c")
    for s in (raw, cen):
        pid = s.create("A", "doorbell")
        for i in range(4):
            s.add_feedback(sound(i).tolist(), "doorbell", pid, positive=True)
    q = sound(99)
    assert cen.similarity(cen.list()[0]["id"], q) == pytest.approx(raw.similarity(raw.list()[0]["id"], q))
    assert cen.background_similarity(q) is None


def test_centered_score_is_cosine_after_subtracting_background_mean(tmp_path):
    store = make_store(tmp_path, top_k=3)
    pid = store.create("A", "doorbell")
    positives = np.stack([sound(i, 0.5) + sound(100 + i, 0.2) - COMMON for i in range(5)])
    for p in positives:
        store.add_feedback(p.tolist(), "doorbell", pid, positive=True)
    background = np.stack([sound(200 + i, 0.3) for i in range(6)])
    store.add_background(background, source="room")
    q = sound(1, 0.5)

    mean = background.mean(axis=0)
    sims = unit(positives - mean) @ unit(q - mean)
    expected = np.sort(sims)[-3:].mean()
    assert store.similarity(pid, q) == pytest.approx(np.clip(expected, 0, 1), abs=1e-5)
    bg_expected = np.sort(unit(background - mean) @ unit(q - mean))[-3:].mean()
    assert store.background_similarity(q) == pytest.approx(bg_expected, abs=1e-5)


def test_centering_separates_sounds_that_share_a_common_component(tmp_path):
    raw, cen = make_store(tmp_path / "r", scoring="raw"), make_store(tmp_path / "c")
    bank = np.stack([sound(300 + i, 0.3) for i in range(20)])
    for s in (raw, cen):
        s.add_background(bank, source="room")
        pid = s.create("A", "doorbell")
        s.add_feedback(sound(1).tolist(), "doorbell", pid, positive=True)
    other = sound(2)  # a different sound with the same room/common component
    r = raw.similarity(raw.list()[0]["id"], other)
    c = cen.similarity(cen.list()[0]["id"], other)
    assert r > 0.8  # raw cosine is fooled by the shared component
    assert c < r - 0.3
    assert cen.similarity(cen.list()[0]["id"], sound(1)) == pytest.approx(1.0, abs=1e-5)


def test_background_bank_persists_reloads_and_is_capped(tmp_path, monkeypatch):
    from backend.custom import profile_store as ps

    monkeypatch.setitem(ps.BACKGROUND_CAPS, "calibration", 4)
    s1 = make_store(tmp_path)
    for i in range(6):
        s1.add_background(sound(i), source="calibration")
    s1.add_background(np.stack([sound(10), sound(11)]), source="room")
    s1.set_background("generic", np.stack([sound(20)]))
    assert s1.background_counts() == {"calibration": 4, "room": 2, "generic": 1}
    with np.load(tmp_path / "background.npz") as npz:
        assert sorted(npz.files) == ["calibration", "generic", "room"]
        np.testing.assert_allclose(npz["calibration"][-1], sound(5), rtol=1e-6)  # newest kept
    s2 = make_store(tmp_path)
    assert s2.background_counts() == s1.background_counts()
    with pytest.raises(ValueError):
        s2.add_background(sound(1), source="kitchen")


def test_bank_change_invalidates_cached_scores(tmp_path):
    store = make_store(tmp_path)
    pid = store.create("A", "doorbell")
    store.add_feedback(sound(1).tolist(), "doorbell", pid, positive=True)
    before = store.similarity(pid, sound(2))
    store.add_background(np.stack([sound(400 + i, 0.3) for i in range(10)]), source="room")
    assert store.similarity(pid, sound(2)) < before


# --- S5f: margin rule --------------------------------------------------------

def test_margin_rule_rejects_window_closer_to_background(tmp_path):
    for scoring in ("raw", "centered"):
        kw = dict(scoring=scoring, threshold=0.5, top_k=1)
        strict = make_store(tmp_path / scoring / "m", **kw)
        loose = make_store(tmp_path / scoring / "n", margin=False, **kw)
        tv = sound(7)  # e.g. the TV, which is always on
        for s in (strict, loose):
            s.add_background(np.stack([tv, sound(8, 0.3), sound(9, 0.3)]), source="room")
            pid = s.create("Cooker", "pressure_cooker")
            s.add_feedback((0.6 * tv + 0.4 * sound(1)).tolist(), "pressure_cooker", pid, positive=True)
        d = strict.score(strict.list()[0]["id"], tv)
        assert d.similarity >= 0.5 and d.background > d.similarity and not d.matched, scoring
        assert strict.match(tv) == []
        assert [m.name for m in loose.match(tv)] == ["Cooker"], scoring


def test_margin_rule_accepts_window_that_beats_background(tmp_path):
    store = make_store(tmp_path, threshold=0.5)
    store.add_background(np.stack([sound(500 + i, 0.3) for i in range(5)]), source="room")
    pid = store.create("Bell", "doorbell")
    for i in range(3):
        store.add_feedback(sound(1, 1.0 + 0.01 * i).tolist(), "doorbell", pid, positive=True)
    d = store.score(pid, sound(1))
    assert d.matched and d.similarity > d.background
    (m,) = store.match(sound(1))
    assert m.profile_id == pid and m.similarity == pytest.approx(d.similarity)


def test_threshold_still_required_with_empty_background(tmp_path):
    store = make_store(tmp_path, threshold=0.99)
    pid = store.create("Bell", "doorbell")
    store.add_feedback(sound(1).tolist(), "doorbell", pid, positive=True)
    d = store.score(pid, sound(2))
    assert d.background is None and not d.matched


# --- S5f: augmentation -------------------------------------------------------

def test_augmentation_adds_gain_and_noise_copies(tmp_path):
    from backend.custom.profile_store import AUG_GAINS_DB

    store = make_store(tmp_path)
    pid = store.create("Bell", "doorbell")
    n = store.add_example_from_audio(pid, tone(440, seconds=2.0, amp=0.2))
    p = store._profiles[pid]
    assert n == len(p.positive) == store.list()[0]["examples"]  # contract count: recorded windows only
    assert len(p.augmented) == 3 * n
    orig, quiet, loud, noisy = p.positive[0], *p.augmented[:3]
    # FakeModel embedding = spectrum magnitude, so it scales with the gain.
    np.testing.assert_allclose(quiet, orig * 10 ** (AUG_GAINS_DB[0] / 20), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(loud, orig * 10 ** (AUG_GAINS_DB[1] / 20), rtol=1e-3, atol=1e-3)
    assert not np.allclose(noisy, orig) and np.corrcoef(noisy, orig)[0, 1] > 0.9
    with np.load(tmp_path / f"{pid}.npz") as npz:
        assert npz["augmented"].shape == (3 * n, EMBEDDING_SIZE)


def test_augment_off_stores_recorded_windows_only(tmp_path):
    store = make_store(tmp_path, augment=False)
    pid = store.create("Bell", "doorbell")
    n = store.add_example_from_audio(pid, tone(440))
    assert n > 0 and len(store._profiles[pid].augmented) == 0


def test_mix_at_snr_hits_target_snr():
    from backend.custom.profile_store import mix_at_snr, pink_noise

    sig = tone(440, seconds=1.0, amp=0.1)
    noise = pink_noise(sig.size, np.random.default_rng(0)) * 0.5
    mixed = mix_at_snr(sig, noise, 10.0)
    added = mixed - sig
    snr = 20 * np.log10(np.sqrt(np.mean(sig ** 2)) / np.sqrt(np.mean(added ** 2)))
    assert snr == pytest.approx(10.0, abs=0.05)
    np.testing.assert_array_equal(mix_at_snr(sig, np.zeros_like(sig), 10.0), sig)


def test_noise_augmentation_uses_room_audio_in_memory(tmp_path):
    store = make_store(tmp_path)
    room = tone(3000, seconds=5.0, amp=0.05)
    added = store.add_background_from_audio(room)
    assert added > 0 and store.background_counts()["room"] == added
    window = tone(440, seconds=0.96, amp=0.2)[:WINDOW_SAMPLES]
    noisy = store.augmentations(window)[-1]
    spectrum = np.abs(np.fft.rfft(noisy - window))
    peak_hz = np.argmax(spectrum) * SAMPLE_RATE / window.size
    assert peak_hz == pytest.approx(3000, abs=20)
    assert not list(tmp_path.glob("*.wav"))  # room audio is never written


def test_pre_s5f_profiles_still_load(tmp_path):
    pid = "abcd1234"
    (tmp_path / "profiles.json").write_text(json.dumps(
        {"method": "embedding", "profiles": [{"id": pid, "name": "Old", "category": "doorbell", "method": "embedding"}]}))
    np.savez(tmp_path / f"{pid}.npz", positive=np.stack([sound(1)]), negative=np.zeros((0, EMBEDDING_SIZE), np.float32))
    store = make_store(tmp_path)
    assert store.list() == [{"id": pid, "name": "Old", "category": "doorbell", "examples": 1}]
    assert store.similarity(pid, sound(1)) == pytest.approx(1.0, abs=1e-5)


# --- S5f: demo --test-wav and --record-test ---------------------------------

def _write_wav(path, audio):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="FLOAT")
    return path


def test_demo_audio_paths(tmp_path):
    from backend.custom.demo_profiles import audio_paths

    a = _write_wav(tmp_path / "set" / "b" / "2.wav", tone(440, 1))
    b = _write_wav(tmp_path / "set" / "a" / "1.wav", tone(440, 1))
    (tmp_path / "set" / "notes.txt").write_text("x")
    assert audio_paths(tmp_path / "set") == [b, a]
    assert audio_paths(a) == [a]
    with pytest.raises(SystemExit):
        audio_paths(tmp_path / "missing")


def test_demo_test_wav_scores_files_per_window(tmp_path):
    from backend.custom.demo_profiles import evaluate_paths, overall_summary
    from backend.ml.demo_yamnet import load_wav

    store = make_store(tmp_path / "profiles", scoring="raw", threshold=0.9)
    pid = store.create("Bell", "doorbell")
    store.add_example_from_audio(pid, tone(440, 3.0))
    same = _write_wav(tmp_path / "wav" / "same.wav", tone(440, 2.0, amp=0.1))
    other = _write_wav(tmp_path / "wav" / "other.wav", tone(2500, 2.0))
    silent = _write_wav(tmp_path / "wav" / "silent.wav", np.zeros(SAMPLE_RATE * 2, np.float32))
    lines: list[str] = []
    results = evaluate_paths(store, pid, [same, other, silent], FakeModel(), load=load_wav, echo=lines.append)

    r_same, r_other, r_silent = results
    assert len(r_same.windows) == r_same.total_windows == 3  # 2 s -> 3 windows
    assert r_same.match_rate == 1.0 and r_same.max > 0.99
    assert r_other.match_rate == 0.0 and r_other.max < 0.5
    assert r_silent.windows == [] and "all 3 windows silent" in r_silent.summary()
    assert "same.wav: 3 windows (0 silent), max 1.00" in r_same.summary()
    assert "matched 100%" in r_same.summary()
    window_lines = [l for l in lines if l.lstrip().startswith(("0.00s", "0.48s", "0.96s"))]
    assert len(window_lines) == 6 and "MATCH" in window_lines[0]
    assert overall_summary(results).startswith("Overall: 3 files (1 with a match), 6 windows")


def test_demo_record_test_paths_and_consent(tmp_path):
    from backend.custom.demo_profiles import TEST_ROOT, clip_path, confirm_consent, record_test, safe_label, save_test_clip

    assert TEST_ROOT.parts[-3:] == ("data", "indian", "test")
    assert safe_label(" pressure cooker #2 ") == "pressure_cooker_2"
    with pytest.raises(SystemExit):
        safe_label("///")
    path = clip_path("door knock", root=tmp_path, now=0)
    assert path.parent == tmp_path / "door_knock" and path.suffix == ".wav"

    assert confirm_consent(lambda _: " YES ") is True
    assert confirm_consent(lambda _: "y") is False

    def eof(_):
        raise EOFError

    assert confirm_consent(eof) is False
    assert record_test("x", 1, ask=lambda _: "no") is None  # refuses before touching the mic

    audio = tone(440, 1.0, amp=0.2)
    import soundfile as sf

    saved = save_test_clip(audio, path)
    back, rate = sf.read(str(saved), dtype="float32")
    assert rate == SAMPLE_RATE and back.shape == audio.shape
    np.testing.assert_allclose(back, audio, atol=1e-4)


def test_tune_sweep_picks_best_f1():
    from pathlib import Path

    from eval.model.tune_custom_threshold import Trial, best_threshold, sweep

    trials = [Trial("a", Path("p1"), True, 0.9), Trial("a", Path("p2"), True, 0.8),
              Trial("a", Path("n1"), False, 0.7), Trial("a", Path("n2"), False, -np.inf)]
    th, p, r, f1 = best_threshold(sweep(trials))
    assert 0.7 < th <= 0.8 and p == r == f1 == 1.0


def test_default_threshold_falls_back_to_raw_while_bank_is_empty(tmp_path):
    from backend.custom.profile_store import CENTERED_MATCH_THRESHOLD, RAW_MATCH_THRESHOLD

    store = make_store(tmp_path)
    assert store.effective_threshold == RAW_MATCH_THRESHOLD  # empty bank: centered == raw
    store.add_background(sound(1), source="calibration")
    assert store.effective_threshold == CENTERED_MATCH_THRESHOLD
    store.scoring = "raw"
    assert store.threshold == store.effective_threshold == RAW_MATCH_THRESHOLD
    fixed = make_store(tmp_path / "f", threshold=0.95)
    assert fixed.effective_threshold == 0.95  # explicit threshold is never changed
    fixed.scoring = "raw"
    assert fixed.threshold == 0.95
