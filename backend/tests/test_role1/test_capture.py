"""Tests for backend/audio/capture.py (Part S1). No real microphone needed."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from backend.audio import capture
from backend.audio.capture import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, MicCapture, Windower, rms


def test_constants():
    assert SAMPLE_RATE == 16000
    assert WINDOW_SAMPLES == 15360  # 0.96 s
    assert HOP_SAMPLES == 7680  # 0.48 s, 50% overlap


def test_rms():
    assert rms(np.zeros(100, dtype=np.float32)) == 0.0
    assert rms(np.ones(100, dtype=np.float32)) == pytest.approx(1.0)
    assert rms(np.array([], dtype=np.float32)) == 0.0


@pytest.mark.parametrize("block_size", [1000, 7680, 15360, 20000])
def test_windower_overlap_any_block_size(block_size):
    stream = np.arange(5 * SAMPLE_RATE, dtype=np.float32)  # 5 s, values = sample index
    w = Windower()
    out = []
    for i in range(0, stream.size, block_size):
        out += w.feed(stream[i : i + block_size])

    expected = (stream.size - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    assert len(out) == expected
    for k, (end, win) in enumerate(out):
        assert win.dtype == np.float32 and win.size == WINDOW_SAMPLES
        assert end == k * HOP_SAMPLES + WINDOW_SAMPLES
        np.testing.assert_array_equal(win, stream[end - WINDOW_SAMPLES : end])
    # 50% overlap: second half of window k == first half of window k+1
    for (_, a), (_, b) in zip(out, out[1:]):
        np.testing.assert_array_equal(a[HOP_SAMPLES:], b[:HOP_SAMPLES])


def test_windower_returns_copies():
    w = Windower(window=4, hop=2)
    (_, first), = w.feed(np.arange(4, dtype=np.float32))
    first[:] = -1
    (_, second), = w.feed(np.arange(4, 6, dtype=np.float32))
    np.testing.assert_array_equal(second, [2, 3, 4, 5])


class FakeStream:
    """Stands in for sd.InputStream: feeds blocks of a sine wave from a thread."""

    def __init__(self, samplerate, blocksize, callback, **_):
        self.rate, self.blocksize, self.callback = samplerate, blocksize, callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        n = 0
        status = SimpleNamespace(input_overflow=False)
        while not self._stop.is_set():
            t = (np.arange(self.blocksize) + n) / self.rate
            block = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)[:, None]
            self.callback(block, self.blocksize, None, status)
            n += self.blocksize
            time.sleep(0.001)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def close(self):
        pass


@pytest.fixture
def fake_mic(monkeypatch):
    def use(rate):
        monkeypatch.setattr(capture.sd, "InputStream", FakeStream)
        monkeypatch.setattr(capture.MicCapture, "_pick_rate", lambda self: rate)

    return use


@pytest.mark.parametrize("device_rate", [16000, 48000, 44100])
def test_mic_capture_delivers_windows_on_background_thread(fake_mic, device_rate):
    fake_mic(device_rate)
    got = []
    threads = set()

    def on_window(samples, ts):
        threads.add(threading.current_thread().name)
        got.append((samples, ts))

    mic = MicCapture(on_window)
    mic.start()
    assert mic.is_running()
    deadline = time.monotonic() + 5
    while len(got) < 6 and time.monotonic() < deadline:
        time.sleep(0.01)
    mic.stop()
    n = len(got)
    time.sleep(0.05)

    assert not mic.is_running()
    assert len(got) == n >= 6, "no windows after stop()"
    assert threads == {"MicCapture"}
    for samples, _ in got:
        assert samples.dtype == np.float32 and samples.size == WINDOW_SAMPLES
    # Resampled or not, a 0.5-amplitude sine has rms ~0.354
    assert rms(got[-1][0]) == pytest.approx(0.5 / np.sqrt(2), abs=0.01)
    gaps = np.diff([ts for _, ts in got])
    np.testing.assert_allclose(gaps, 0.48, atol=1e-6)


def test_callback_errors_do_not_stop_capture(fake_mic):
    fake_mic(16000)
    calls = []

    def bad(samples, ts):
        calls.append(ts)
        raise ValueError("boom")

    mic = MicCapture(bad)
    mic.start()
    deadline = time.monotonic() + 5
    while len(calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert mic.is_running()
    mic.stop()
    assert len(calls) >= 3


def test_start_raises_runtime_error_when_mic_unavailable(monkeypatch):
    def broken(**_):
        raise OSError("no device")

    monkeypatch.setattr(capture.sd, "InputStream", broken)
    monkeypatch.setattr(capture.MicCapture, "_pick_rate", lambda self: 16000)
    with pytest.raises(RuntimeError):
        MicCapture(lambda s, t: None).start()


def test_stop_without_start_is_noop():
    MicCapture(lambda s, t: None).stop()
