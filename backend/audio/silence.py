"""Silence gate (CPU saver): skip quiet windows so the model doesn't run on them.

Owner: Shreeya (Role 1). Part S2.

Each 0.96 s window's RMS loudness is compared with a threshold. Below it, the
window is "skipped" (``WindowResult.skipped = True``: YAMNet is not run).
Settings' ``cpu_saver`` maps to ``SilenceGate.enabled``: when disabled no
window is ever skipped, but windows that *would* have been skipped are still
counted, so Experiment 8 can compare both modes.

Threshold:
  - Fixed: ``SilenceGate(threshold=0.003)`` (about -50 dBFS, the default).
  - Auto-calibrated: ``gate.start_calibration()`` measures ~2 s of ambient
    noise from the next windows, then sets
    ``threshold = ambient RMS x margin``, clamped to a safe range.
    Windows seen during calibration are never skipped, so nothing important
    is missed at start-up.

All methods are thread-safe: ``check()`` runs on the capture thread while
``stats()`` may be read from the API thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from backend.audio.capture import HOP_S, SAMPLE_RATE, WINDOW_S, rms

DEFAULT_THRESHOLD = 0.003  # ~ -50 dBFS: quieter than a quiet room with a fan
DEFAULT_MARGIN = 2.0  # calibrated threshold = ambient x 2 (~ +6 dB)
CALIBRATION_S = 2.0
MIN_THRESHOLD = 0.0005  # ~ -66 dBFS: never gate below this (dead-silent rooms)
MAX_THRESHOLD = 0.05  # ~ -26 dBFS: never gate above this (noisy calibration)
_FRAME_SAMPLES = SAMPLE_RATE // 10  # 0.1 s frames for robust ambient estimate


@dataclass(frozen=True)
class GateDecision:
    """Result of SilenceGate.check() for one window."""

    skipped: bool  # True = silent, do not run the model
    rms: float  # window loudness, 0..1
    calibrating: bool = False  # window was used to measure ambient noise


def ambient_rms(samples: np.ndarray) -> float:
    """Robust ambient loudness: median RMS over 0.1 s frames.

    The median ignores a short noise (a cough, a door) during calibration,
    which a plain RMS over the whole clip would not.
    """
    samples = np.asarray(samples, dtype=np.float32).ravel()
    n = samples.size // _FRAME_SAMPLES
    if n == 0:
        return rms(samples)
    frames = samples[: n * _FRAME_SAMPLES].reshape(n, _FRAME_SAMPLES).astype(np.float64)
    return float(np.median(np.sqrt(np.mean(frames**2, axis=1))))


class SilenceGate:
    """Decides per window whether to skip the model (see module docstring)."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        enabled: bool = True,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        """Args:
            threshold: RMS below which a window is skipped.
            enabled: The ``cpu_saver`` setting. False = never skip.
            margin: Multiplier applied to ambient RMS when calibrating.
        """
        if threshold < 0 or margin <= 0:
            raise ValueError("threshold must be >= 0 and margin > 0")
        self._lock = threading.Lock()
        self._threshold = float(threshold)
        self._enabled = bool(enabled)
        self.margin = float(margin)
        self._calib_windows_left = 0
        self._calib_chunks: list[np.ndarray] = []
        self.ambient_level: float | None = None  # set by calibrate()
        self._reset_counts()

    # --- settings ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if value < 0:
            raise ValueError("threshold must be >= 0")
        with self._lock:
            self._threshold = float(value)

    @property
    def calibrating(self) -> bool:
        return self._calib_windows_left > 0

    # --- calibration ------------------------------------------------------

    def calibrate(self, ambient: np.ndarray) -> float:
        """Set the threshold from a clip of ambient (background) audio.

        Returns:
            The new threshold: ``ambient_rms(ambient) x margin``, clamped to
            [MIN_THRESHOLD, MAX_THRESHOLD].
        """
        level = ambient_rms(ambient)
        new = min(MAX_THRESHOLD, max(MIN_THRESHOLD, level * self.margin))
        with self._lock:
            self._threshold = new
            self.ambient_level = level
        return new

    def start_calibration(self, seconds: float = CALIBRATION_S) -> None:
        """Calibrate from the next ~``seconds`` of windows passed to check().

        With 0.96 s windows every 0.48 s, 2 s is 3 windows (1.92 s of audio).
        Those windows are reported as not skipped.
        """
        windows = max(1, round((seconds - WINDOW_S) / HOP_S) + 1)
        with self._lock:
            self._calib_windows_left = windows
            self._calib_chunks = []

    # --- per-window decision ---------------------------------------------

    def check(self, samples: np.ndarray) -> GateDecision:
        """Decide whether to skip this window, and count it."""
        level = rms(samples)

        if self._calib_windows_left > 0:
            with self._lock:
                # Overlapping windows: keep only the new (hop) part after the first.
                chunk = samples if not self._calib_chunks else samples[-round(HOP_S * SAMPLE_RATE):]
                self._calib_chunks.append(np.asarray(chunk, dtype=np.float32))
                self._calib_windows_left -= 1
                done = self._calib_windows_left == 0
                chunks = self._calib_chunks if done else None
                if done:
                    self._calib_chunks = []
                self._count(skipped=False, below=False)
            if chunks is not None:
                self.calibrate(np.concatenate(chunks))
            return GateDecision(skipped=False, rms=level, calibrating=True)

        with self._lock:
            below = level < self._threshold
            skipped = below and self._enabled
            self._count(skipped=skipped, below=below)
        return GateDecision(skipped=skipped, rms=level)

    # --- statistics (Experiment 8) ---------------------------------------

    def stats(self) -> dict:
        """Counts since creation or the last reset_stats().

        Keys:
            total: windows checked
            skipped: windows the model did not run on
            active: windows the model ran on (total - skipped)
            below_threshold: windows quieter than the threshold, counted even
                when the gate is disabled (what *would* have been skipped)
            skip_rate: skipped / total (0.0 if no windows)
            threshold, enabled: current settings
        """
        with self._lock:
            total = self._total
            return {
                "total": total,
                "skipped": self._skipped,
                "active": total - self._skipped,
                "below_threshold": self._below,
                "skip_rate": self._skipped / total if total else 0.0,
                "threshold": self._threshold,
                "enabled": self._enabled,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._reset_counts()

    def _reset_counts(self) -> None:
        self._total = 0
        self._skipped = 0
        self._below = 0

    def _count(self, skipped: bool, below: bool) -> None:
        self._total += 1
        self._skipped += skipped
        self._below += below


def rms_to_db(level: float) -> float:
    """RMS (0..1) to dBFS, floored at -120 dB."""
    return 20 * math.log10(max(level, 1e-6))
