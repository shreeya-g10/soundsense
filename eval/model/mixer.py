"""Mix a clean clip with noise (or TV sound) at a given SNR, in memory.

Owner: Shreeya (Role 1). Part S7d.

    from eval.model.mixer import mix, SNRS
    noisy = mix(clean, noise, snr_db=5, rng=np.random.default_rng(0))

SNR is measured against the clean clip's ACTIVE part: the RMS of its
non-silent 0.96 s windows (RMS >= the silence gate's 0.008), not of the
whole clip. Many test clips are short events padded with silence (a 5 s
ESC-50 clip with one bark); a whole-clip RMS would make the event much
louder than the nominal SNR. Falls back to the whole-clip RMS if no window
is active.

The noise segment starts at a random offset (``rng``), is looped if shorter
than the clip, and is scaled so 20*log10(active_rms(clean) / rms(noise)) =
snr_db. The sum is clipped to [-1, 1]. Mixes are never written to disk by
the experiments; they are rebuilt from the seed.
"""

from __future__ import annotations

import numpy as np

from backend.audio.silence import DEFAULT_THRESHOLD

SNRS = (20, 10, 5, 0)


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def active_rms(clean: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> float:
    """RMS over the clip's non-silent 0.96 s windows (whole-clip RMS if none is active)."""
    from eval.model.common import windows_of

    active = [w for _, w in windows_of(clean) if rms(w) >= threshold]
    return rms(np.concatenate(active)) if active else rms(clean)


def noise_segment(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` samples of ``noise`` from a random offset, looping if needed."""
    noise = np.asarray(noise, np.float32).ravel()
    if noise.size == 0:
        raise ValueError("empty noise")
    start = int(rng.integers(0, noise.size))
    reps = int(np.ceil((start + n) / noise.size))
    return np.tile(noise, reps)[start:start + n]


def mix(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator,
        threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """``clean`` + noise scaled to ``snr_db`` (active-part SNR), clipped to [-1, 1].

    Raises:
        ValueError: if the noise segment is silent (no level to scale).
    """
    clean = np.asarray(clean, np.float32).ravel()
    seg = noise_segment(noise, clean.size, rng)
    n = rms(seg)
    if n == 0:
        raise ValueError("noise segment is silent")
    s = active_rms(clean, threshold)
    if s == 0:  # silent clip: the mix is just noise at the clip's (zero) level
        return np.zeros_like(clean)
    scaled = seg * (s / n) / 10 ** (snr_db / 20)
    return np.clip(clean + scaled, -1.0, 1.0).astype(np.float32)


def measured_snr(clean: np.ndarray, mixed: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> float:
    """Active-part SNR of a mix (for checking): 20*log10(active_rms(clean) / rms(mixed - clean))."""
    return 20 * np.log10(active_rms(clean, threshold) / rms(np.asarray(mixed) - np.asarray(clean)))
