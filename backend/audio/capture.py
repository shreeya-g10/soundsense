"""Microphone capture: 16 kHz mono audio, cut into overlapping windows.

Owner: Shreeya (Role 1). Part S1.

``MicCapture`` opens the microphone with sounddevice and delivers
0.96 s windows (15 360 samples) every 0.48 s (50% overlap) to a callback, on
its own background thread. These are YAMNet-sized windows that later parts
turn into ``contracts.types.WindowResult`` objects.

Privacy: audio only ever lives in memory. Nothing here writes audio to disk.

Threads:
  - PortAudio's audio thread runs ``_on_audio``. It only copies the block
    into a queue, because it must never block.
  - Our worker thread (``MicCapture``) takes blocks off the queue, resamples
    if needed, cuts windows and calls ``on_window``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
WINDOW_S = 0.96
HOP_S = 0.48
WINDOW_SAMPLES = round(WINDOW_S * SAMPLE_RATE)  # 15360
HOP_SAMPLES = round(HOP_S * SAMPLE_RATE)  # 7680

_QUEUE_MAX_BLOCKS = 64  # ~30 s of audio at one hop per block

WindowCallback = Callable[[np.ndarray, float], None]
"""on_window(samples, timestamp): float32 array of WINDOW_SAMPLES values in
-1..1, and the Unix time at which the window's last sample was captured."""


def rms(samples: np.ndarray) -> float:
    """Root-mean-square loudness of float audio in -1..1 (result 0..1)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class Windower:
    """Cuts a stream of audio blocks into fixed-size overlapping windows.

    Pure numpy and thread-free, so it can be tested without a microphone.
    Blocks may be any length; a window is emitted as soon as enough samples
    have arrived.
    """

    def __init__(self, window: int = WINDOW_SAMPLES, hop: int = HOP_SAMPLES) -> None:
        if not 0 < hop <= window:
            raise ValueError("need 0 < hop <= window")
        self.window = window
        self.hop = hop
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # stream sample index of self._buf[0]

    def feed(self, block: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Add a block of samples; return the windows it completed.

        Returns:
            A list of ``(end_sample, window)`` pairs. ``end_sample`` is the
            stream index just past the window's last sample, so window k ends
            at ``k * hop + window``. Each window is a fresh float32 copy.
        """
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float32).ravel()])
        out = []
        while self._buf.size >= self.window:
            out.append((self._buf_start + self.window, self._buf[: self.window].copy()))
            self._buf = self._buf[self.hop :]
            self._buf_start += self.hop
        return out

    def reset(self) -> None:
        """Forget any buffered samples and restart sample counting at 0."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0


class _Resampler:
    """Streaming resampler from the device's rate to 16 kHz (no clicks at block edges)."""

    def __init__(self, in_rate: float) -> None:
        import soxr  # installed with librosa

        self._stream = soxr.ResampleStream(in_rate, SAMPLE_RATE, 1, dtype="float32")

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self._stream.resample_chunk(block)


class MicCapture:
    """Captures the microphone and calls ``on_window`` for each 0.96 s window.

    Example::

        def handle(samples, timestamp):
            print(timestamp, rms(samples))

        mic = MicCapture(handle)
        mic.start()
        ...
        mic.stop()

    ``on_window`` runs on the capture worker thread. It should return within
    0.48 s on average or audio backs up (blocks are dropped once ~30 s are
    queued; see ``dropped_blocks``). Exceptions it raises are logged and
    ignored, so one bad window never stops capture.
    """

    def __init__(
        self,
        on_window: WindowCallback,
        device: int | str | None = None,
        window_s: float = WINDOW_S,
        hop_s: float = HOP_S,
    ) -> None:
        """Create the capture. Does not open the microphone yet.

        Args:
            on_window: Called with ``(samples, timestamp)`` for each window.
            device: sounddevice input device index or name; None = default.
            window_s: Window length in seconds (0.96 for YAMNet).
            hop_s: Seconds between window starts (0.48 = 50% overlap).
        """
        self._on_window = on_window
        self.device = device
        self.window_samples = round(window_s * SAMPLE_RATE)
        self.hop_samples = round(hop_s * SAMPLE_RATE)

        self.device_rate: float | None = None  # rate the mic actually runs at
        self.overflows = 0  # PortAudio reported lost input
        self.dropped_blocks = 0  # we dropped blocks because on_window was too slow
        self.last_audio_time: float | None = None  # time.time() of the last block

        self._stream: sd.InputStream | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._t0: float | None = None  # wall time of stream sample 0

    # --- public API -------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and start delivering windows in the background.

        Returns immediately. Does nothing if already running. Raises
        RuntimeError if the microphone cannot be opened.
        """
        with self._lock:
            if self.is_running():
                return
            rate = self._pick_rate()
            resample = None if rate == SAMPLE_RATE else _Resampler(rate)
            self.device_rate = rate
            self._queue = queue.Queue(maxsize=_QUEUE_MAX_BLOCKS)
            self._t0 = None
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    blocksize=round(self.hop_samples * rate / SAMPLE_RATE),
                    callback=self._on_audio,
                )
                stream.start()
            except Exception as exc:
                raise RuntimeError(f"Cannot open microphone {self.device!r}: {exc}") from exc
            self._stream = stream
            self._thread = threading.Thread(
                target=self._run, args=(resample,), name="MicCapture", daemon=True
            )
            self._thread.start()
            log.info("Mic capture started (device=%r, %s Hz)", self.device, rate)

    def stop(self) -> None:
        """Stop capture and release the microphone.

        Blocks until ``on_window`` will not be called again (unless called
        from inside ``on_window`` itself). Does nothing if not running.
        """
        with self._lock:
            stream, self._stream = self._stream, None
            thread, self._thread = self._thread, None
        if stream is not None:
            stream.stop()
            stream.close()
        if thread is not None:
            self._put(None)  # wake the worker so it exits
            if thread is not threading.current_thread():
                thread.join()
        log.info("Mic capture stopped")

    def is_running(self) -> bool:
        """True while the microphone is open and windows are being delivered."""
        return self._thread is not None and self._thread.is_alive()

    # --- internals --------------------------------------------------------

    def _pick_rate(self) -> float:
        """16 kHz if the device supports it, else its default rate (resampled)."""
        try:
            sd.check_input_settings(device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            return SAMPLE_RATE
        except Exception:
            try:
                info = sd.query_devices(self.device, kind="input")
            except Exception as exc:
                raise RuntimeError(f"No usable input device {self.device!r}: {exc}") from exc
            log.info("Device does not support 16 kHz; capturing at %s Hz and resampling", info["default_samplerate"])
            return float(info["default_samplerate"])

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback (audio thread). Must not block."""
        if status.input_overflow:
            self.overflows += 1
        now = time.time()
        self.last_audio_time = now
        if self._t0 is None:
            self._t0 = now - frames / (self.device_rate or SAMPLE_RATE)
        self._put(indata[:, 0].copy())

    def _put(self, item: np.ndarray | None) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            if item is None:  # make room for the stop signal
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(None)
            else:
                self.dropped_blocks += 1

    def _run(self, resample: _Resampler | None) -> None:
        """Worker thread: resample, cut windows, call on_window."""
        windower = Windower(self.window_samples, self.hop_samples)
        while True:
            block = self._queue.get()
            if block is None:
                return
            if resample is not None:
                block = resample(block)
            for end_sample, window in windower.feed(block):
                timestamp = (self._t0 or time.time()) + end_sample / SAMPLE_RATE
                try:
                    self._on_window(window, timestamp)
                except Exception:
                    log.exception("on_window callback raised")
