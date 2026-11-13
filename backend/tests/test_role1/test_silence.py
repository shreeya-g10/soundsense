"""Tests for backend/audio/silence.py (Part S2)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES
from backend.audio.silence import (
    DEFAULT_THRESHOLD,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    SilenceGate,
    ambient_rms,
)

rng = np.random.default_rng(0)


def noise(level: float, n: int = WINDOW_SAMPLES) -> np.ndarray:
    """Gaussian noise with RMS ~= level."""
    return (rng.standard_normal(n) * level).astype(np.float32)


def tone(amplitude: float = 0.3, n: int = WINDOW_SAMPLES) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def test_default_threshold_is_sensible():
    assert MIN_THRESHOLD < DEFAULT_THRESHOLD < MAX_THRESHOLD


def test_default_threshold_skips_typical_room_noise():
    # Laptop mic at ~65% input volume: room noise around 0.005 RMS.
    assert DEFAULT_THRESHOLD == pytest.approx(0.008)
    assert SilenceGate().check(noise(0.005)).skipped


def test_silent_array_is_skipped():
    gate = SilenceGate()
    d = gate.check(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    assert d.skipped and d.rms == 0.0
    assert gate.check(noise(DEFAULT_THRESHOLD / 4)).skipped


def test_loud_array_is_active():
    gate = SilenceGate()
    d = gate.check(tone(0.3))
    assert not d.skipped
    assert d.rms == pytest.approx(0.3 / np.sqrt(2), rel=1e-3)


def test_custom_threshold():
    gate = SilenceGate(threshold=0.1)
    assert gate.check(noise(0.05)).skipped
    assert not gate.check(noise(0.2)).skipped
    gate.threshold = 0.5
    assert gate.check(noise(0.2)).skipped
    with pytest.raises(ValueError):
        gate.threshold = -1


def test_disabled_flag_never_skips():
    gate = SilenceGate(enabled=False)
    for window in (np.zeros(WINDOW_SAMPLES, np.float32), noise(1e-5), tone(0.3)):
        assert not gate.check(window).skipped
    s = gate.stats()
    assert s["skipped"] == 0 and s["active"] == 3
    assert s["below_threshold"] == 2  # still counted for Experiment 8


def test_enable_toggle_at_runtime():
    gate = SilenceGate(enabled=False)
    silent = np.zeros(WINDOW_SAMPLES, np.float32)
    assert not gate.check(silent).skipped
    gate.enabled = True
    assert gate.check(silent).skipped


def test_stats_count_skipped_and_active():
    gate = SilenceGate()
    for _ in range(3):
        gate.check(np.zeros(WINDOW_SAMPLES, np.float32))
    gate.check(tone())
    assert gate.stats() == {
        "total": 4,
        "skipped": 3,
        "active": 1,
        "below_threshold": 3,
        "skip_rate": 0.75,
        "threshold": DEFAULT_THRESHOLD,
        "enabled": True,
    }
    gate.reset_stats()
    assert gate.stats()["total"] == 0 and gate.stats()["skip_rate"] == 0.0


def test_calibrate_sets_threshold_above_ambient():
    ambient = noise(0.01, n=2 * SAMPLE_RATE)
    gate = SilenceGate(margin=2.0)
    thr = gate.calibrate(ambient)
    assert thr == gate.threshold
    assert thr > ambient_rms(ambient)
    assert thr == pytest.approx(0.02, rel=0.05)
    # Room noise itself is now skipped; a real sound is not.
    assert gate.check(noise(0.01)).skipped
    assert not gate.check(tone(0.3)).skipped


def test_calibration_ignores_short_transient():
    ambient = noise(0.005, n=2 * SAMPLE_RATE)
    ambient[8000:9600] = tone(0.8, n=1600)  # 0.1 s loud click
    thr = SilenceGate(margin=2.0).calibrate(ambient)
    assert thr == pytest.approx(0.01, rel=0.1)


def test_calibration_is_clamped():
    gate = SilenceGate()
    assert gate.calibrate(np.zeros(SAMPLE_RATE, np.float32)) == MIN_THRESHOLD
    assert gate.calibrate(noise(0.5, n=SAMPLE_RATE)) == MAX_THRESHOLD


def test_streaming_calibration_from_windows():
    gate = SilenceGate(margin=2.0)
    gate.start_calibration(seconds=2.0)
    assert gate.calibrating
    decisions = [gate.check(noise(0.01)) for _ in range(3)]  # 2 s = 3 windows
    assert all(d.calibrating and not d.skipped for d in decisions)
    assert not gate.calibrating
    assert gate.threshold > 0.01 and gate.ambient_level == pytest.approx(0.01, rel=0.05)
    after = gate.check(noise(0.01))
    assert after.skipped and not after.calibrating


def test_thread_safe_counting():
    gate = SilenceGate()
    silent = np.zeros(1600, np.float32)

    def work():
        for _ in range(500):
            gate.check(silent)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert gate.stats()["total"] == gate.stats()["skipped"] == 2000
