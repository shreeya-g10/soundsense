"""Real ProfileStore: user-taught custom sounds matched by YAMNet embeddings.

Owner: Shreeya (Role 1). Parts S5 and S5f.

Implements ``contracts.types.ProfileStore`` plus ``match(embedding)``, which
``YamnetListener`` calls on every active window to fill
``WindowResult.custom_matches``.

Storage (under data/profiles/, git-ignored). Only embeddings are stored,
never audio:
    profiles.json   {"method": ..., "profiles": [{id, name, category, method}]}
    <id>.npz        positive: (N, 1024) recorded examples
                    augmented: (3N, 1024) the same windows at -6/+6 dB and
                        mixed with room noise at 10 dB SNR (S5f)
                    negative: (M, 1024) counter-examples from feedback
    background.npz  background bank (S5f), one array per source:
                    calibration: silence-gate calibration windows (listener)
                    room: active windows of the "normal room sound" recording
                    generic: ESC-50 non-target clips (eval/model/build_background_bank.py)
    feedback.npz    feedback on base-model alerts (profile_id=None), per category

Scoring (S5f), set with ``scoring=``:
    "raw":      cosine similarity of the raw embeddings.
    "centered": (default) subtract the background mean from every embedding,
                then L2-normalise, then cosine. YAMNet embeddings are all
                non-negative and share a large common component, so raw
                cosines are high for almost any pair (noise vs a voice ~0.3,
                two different voices ~0.87). Centering removes what "this
                room always sounds like" and leaves what is distinctive.
                With an empty background bank it is the same as raw.
Either way a profile's similarity is the mean of the top-3 positive
(recorded + augmented) examples.

Matching: a window matches a profile when its similarity >= threshold AND
(margin rule, ``margin=True``) it is higher than the window's similarity to
the background bank (mean of its top-3 background embeddings, same scoring).
So a window that looks more like the room than like the sound never matches.

Thresholds (see eval/model/README.md, "Custom-sound threshold"):
    raw 0.80 (S5);  centered CENTERED_MATCH_THRESHOLD, chosen with
    eval/model/tune_custom_threshold.py. Re-tune on data/indian/test in
    Experiment 6.

Methods: only "embedding" matching exists so far. set_method("mfcc") is
accepted and saved (it's a valid contract value) but matching still uses
embeddings until the MFCC method is implemented.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES, Windower
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from contracts.types import CATEGORIES, EMBEDDING_SIZE, METHOD_EMBEDDING, METHOD_MFCC, CustomMatch, CustomMethod

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "profiles"
DEFAULT_TOP_K = 3

Scoring = Literal["raw", "centered"]
SCORING_RAW: Scoring = "raw"
SCORING_CENTERED: Scoring = "centered"
DEFAULT_SCORING: Scoring = SCORING_CENTERED

RAW_MATCH_THRESHOLD = 0.80  # S5, raw cosine
CENTERED_MATCH_THRESHOLD = 0.89  # S5f, provisional (synthetic set): see eval/model/README.md
DEFAULT_MATCH_THRESHOLD = CENTERED_MATCH_THRESHOLD
THRESHOLDS: dict[str, float] = {SCORING_RAW: RAW_MATCH_THRESHOLD, SCORING_CENTERED: CENTERED_MATCH_THRESHOLD}

BACKGROUND_SOURCES = ("calibration", "room", "generic")
# Newest embeddings kept per source, so the bank (and its mean) can't grow forever.
BACKGROUND_CAPS = {"calibration": 300, "room": 300, "generic": 3000}
NOISE_AUDIO_S = 30.0  # seconds of room audio kept IN MEMORY for noise augmentation

AUG_GAINS_DB = (-6.0, 6.0)
AUG_SNR_DB = 10.0

Recorder = Callable[[float], np.ndarray]
"""record(seconds) -> 16 kHz mono float32 audio."""


def record_from_mic(seconds: float, device: int | str | None = None) -> np.ndarray:
    """Record ``seconds`` of 16 kHz mono audio into memory (never to disk)."""
    from backend.audio.capture import MicCapture

    chunks: list[np.ndarray] = []
    needed = round(seconds * SAMPLE_RATE)
    done = threading.Event()

    def collect(samples: np.ndarray, _ts: float) -> None:
        if not done.is_set():
            chunks.append(samples)
            if sum(c.size for c in chunks) >= needed:
                done.set()

    # Non-overlapping 0.1 s blocks: just collect contiguous audio.
    mic = MicCapture(collect, device=device, window_s=0.1, hop_s=0.1)
    mic.start()
    try:
        if not done.wait(timeout=seconds + 5):
            raise RuntimeError("microphone delivered no audio")
    finally:
        mic.stop()
    return np.concatenate(chunks)[:needed]


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def _empty() -> np.ndarray:
    return np.zeros((0, EMBEDDING_SIZE), np.float32)


def _rows(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)


def _top_k_mean(matrix: np.ndarray, query: np.ndarray, k: int) -> float | None:
    """Mean of the k highest cosine similarities (rows are unit vectors); None if empty."""
    if len(matrix) == 0:
        return None
    sims = matrix @ query
    k = min(k, sims.size)
    return float(np.partition(sims, sims.size - k)[-k:].mean())


# --- augmentation helpers ----------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    """``samples`` scaled by ``gain_db``, clipped to [-1, 1] like a real mic."""
    return np.clip(samples * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise with unit RMS (stand-in when no room audio is in memory)."""
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(spectrum.size)
    spectrum[1:] /= np.sqrt(f[1:])
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, n)
    return (noise / (_rms(noise) or 1.0)).astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """``signal`` + ``noise`` scaled so that RMS(signal)/RMS(noise) = ``snr_db``.

    ``noise`` is tiled or cut to the signal's length. Silent noise -> signal unchanged.
    """
    signal = np.asarray(signal, np.float32)
    noise = np.resize(np.asarray(noise, np.float32), signal.shape)
    s, n = _rms(signal), _rms(noise)
    if s == 0 or n == 0:
        return signal.copy()
    scaled = noise * (s / n) / 10 ** (snr_db / 20)
    return np.clip(signal + scaled, -1.0, 1.0).astype(np.float32)


# --- data ------------------------------------------------------------------

@dataclass(frozen=True)
class ScoreDetail:
    """How one window scored against one profile."""

    similarity: float  # profile score, clipped to 0..1 (the contract's similarity)
    background: float | None  # score to the background bank; None if the bank is empty
    matched: bool  # similarity >= threshold and (margin rule) > background


@dataclass
class _Profile:
    id: str
    name: str
    category: str
    method: str
    positive: np.ndarray = field(default_factory=_empty)
    augmented: np.ndarray = field(default_factory=_empty)
    negative: np.ndarray = field(default_factory=_empty)
    _cache: tuple | None = None  # (key, matrix) for matching

    def add(self, embeddings: np.ndarray, positive: bool, augmented: np.ndarray | None = None) -> None:
        if positive:
            self.positive = np.vstack([self.positive, _rows(embeddings)])
            if augmented is not None:
                self.augmented = np.vstack([self.augmented, _rows(augmented)])
            self._cache = None
        else:
            self.negative = np.vstack([self.negative, _rows(embeddings)])


class YamnetProfileStore:
    """Implements ``contracts.types.ProfileStore`` (see module docstring)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        threshold: float | None = None,
        scoring: Scoring = DEFAULT_SCORING,
        margin: bool = True,
        augment: bool = True,
        top_k: int = DEFAULT_TOP_K,
        silence_threshold: float = DEFAULT_THRESHOLD,
        recorder: Recorder | None = None,
        device: int | str | None = None,
        model=None,
        seed: int | None = 0,
    ) -> None:
        """Open (or create) the store in ``root`` and load saved profiles.

        Args:
            root: Folder for profiles.json and the .npz files.
            threshold: Minimum similarity for match(). None = the default
                for ``scoring`` (THRESHOLDS); centered scoring with an empty
                background bank is identical to raw, so it then uses the raw
                default (see effective_threshold).
            scoring: "centered" (default) or "raw"; see module docstring.
            margin: Also require similarity > background similarity.
            augment: Store -6/+6 dB and 10 dB-SNR noisy copies of each
                recorded window (as extra matching examples).
            top_k: Number of nearest examples averaged.
            silence_threshold: RMS below which a recorded window is dropped.
            recorder: record(seconds) -> audio; defaults to the microphone.
                Tests pass a fake.
            device: Mic device for the default recorder.
            model: Object with infer(samples).embedding; defaults to YAMNet.
            seed: Seed for augmentation noise (None = random).
        """
        if scoring not in THRESHOLDS:
            raise ValueError(f"scoring must be 'raw' or 'centered', got {scoring!r}")
        self._auto_threshold = threshold is None  # follow the scoring's default
        threshold = THRESHOLDS[scoring] if threshold is None else threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.root = Path(root)
        self.threshold = threshold
        self._scoring: Scoring = scoring
        self.margin = margin
        self.augment = augment
        self.top_k = top_k
        self.silence_threshold = silence_threshold
        self._recorder = recorder or (lambda s: record_from_mic(s, device=device))
        self._model = model
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._profiles: dict[str, _Profile] = {}
        self._background: dict[str, np.ndarray] = {s: _empty() for s in BACKGROUND_SOURCES}
        self._bg_version = 0
        self._bg_cache: tuple | None = None  # (version, scoring, mean, unit matrix)
        self._noise_audio: np.ndarray | None = None  # room audio, memory only
        self.method: CustomMethod = METHOD_EMBEDDING
        self._load()

    @property
    def scoring(self) -> Scoring:
        return self._scoring

    @property
    def effective_threshold(self) -> float:
        """The threshold match() uses right now.

        With a default threshold, centered scoring and an empty background
        bank (nothing to center by, so scores are raw), the raw default.
        """
        with self._lock:
            if self._auto_threshold and self._scoring == SCORING_CENTERED and self.background_size() == 0:
                return RAW_MATCH_THRESHOLD
            return self.threshold

    @scoring.setter
    def scoring(self, value: Scoring) -> None:
        """Switch scoring. A default threshold follows it; an explicit one is kept."""
        if value not in THRESHOLDS:
            raise ValueError(f"scoring must be 'raw' or 'centered', got {value!r}")
        with self._lock:
            self._scoring = value
            if self._auto_threshold:
                self.threshold = THRESHOLDS[value]

    # --- ProfileStore protocol -------------------------------------------

    def create(self, name: str, category: str) -> str:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        with self._lock:
            profile_id = uuid.uuid4().hex[:8]
            self._profiles[profile_id] = _Profile(profile_id, name.strip(), category, self.method)
            self._save_profile(profile_id)
            self._save_index()
            return profile_id

    def add_example_from_mic(self, profile_id: str, seconds: float = 3) -> int:
        """Record, drop silent windows, store each window's embedding (+ augmentations).

        Returns the profile's total example count (recorded windows only,
        not augmented copies; unchanged if every window was silent).
        """
        self._get(profile_id)  # KeyError before recording
        return self.add_example_from_audio(profile_id, self._recorder(seconds))

    def test_from_mic(self, profile_id: str, seconds: float = 3) -> float:
        """Record and return the best window similarity to the profile (0 if all silent)."""
        self._get(profile_id)
        return self.test_from_audio(profile_id, self._recorder(seconds))

    def add_feedback(
        self,
        embedding: builtins.list[float],
        category: str,
        profile_id: str | None,
        positive: bool,
    ) -> None:
        """Positive feedback adds an example; negative is stored as a counter-example.

        Feedback with profile_id=None (base-model alerts) goes to
        feedback.npz for later model adaptation.
        """
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.size != EMBEDDING_SIZE:
            raise ValueError(f"embedding must have {EMBEDDING_SIZE} values, got {vec.size}")
        with self._lock:
            if profile_id is None:
                self._save_base_feedback(vec, category, positive)
                return
            self._get(profile_id).add(vec, positive)
            self._save_profile(profile_id)

    def list(self) -> builtins.list[dict[str, Any]]:
        with self._lock:
            return [
                {"id": p.id, "name": p.name, "category": p.category, "examples": int(len(p.positive))}
                for p in self._profiles.values()
            ]

    def delete(self, profile_id: str) -> None:
        with self._lock:
            self._get(profile_id)
            del self._profiles[profile_id]
            self._npz_path(profile_id).unlink(missing_ok=True)
            self._save_index()

    def set_method(self, method: CustomMethod) -> None:
        if method not in (METHOD_EMBEDDING, METHOD_MFCC):
            raise ValueError(f"unknown method {method!r}; use 'embedding' or 'mfcc'")
        with self._lock:
            if method == METHOD_MFCC:
                log.warning("MFCC matching is not implemented yet; embeddings are still used")
            self.method = method
            self._save_index()

    # --- matching (called by YamnetListener) ------------------------------

    def similarity(self, profile_id: str, embedding: np.ndarray | builtins.list[float]) -> float:
        """Profile similarity, 0..1: mean of the top-k positive examples (0 if none)."""
        return self.score(profile_id, embedding).similarity

    def background_similarity(self, embedding: np.ndarray | builtins.list[float]) -> float | None:
        """Similarity to the background bank (top-k mean, same scoring); None if empty."""
        with self._lock:
            mean, bg = self._background_state()
            return _top_k_mean(bg, self._transform(embedding, mean), self.top_k)

    def score(self, profile_id: str, embedding: np.ndarray | builtins.list[float]) -> ScoreDetail:
        """Similarity, background similarity and match decision for one window."""
        with self._lock:
            profile = self._get(profile_id)
            mean, bg = self._background_state()
            query = self._transform(embedding, mean)
            return self._detail(profile, query, mean, _top_k_mean(bg, query, self.top_k))

    def match(self, embedding: np.ndarray | builtins.list[float]) -> builtins.list[CustomMatch]:
        """Profiles the window matches (threshold + margin rule), best first."""
        with self._lock:
            mean, bg = self._background_state()
            query = self._transform(embedding, mean)
            bg_score = _top_k_mean(bg, query, self.top_k)
            matches = []
            for p in self._profiles.values():
                d = self._detail(p, query, mean, bg_score)
                if d.matched:
                    matches.append(CustomMatch(profile_id=p.id, name=p.name, category=p.category,
                                               similarity=d.similarity))
        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def _detail(self, profile: _Profile, query: np.ndarray, mean: np.ndarray, bg_score: float | None) -> ScoreDetail:
        raw = _top_k_mean(self._profile_matrix(profile, mean), query, self.top_k)
        if raw is None:
            return ScoreDetail(0.0, bg_score, False)
        matched = raw >= self.effective_threshold and (not self.margin or bg_score is None or raw > bg_score)
        return ScoreDetail(float(np.clip(raw, 0.0, 1.0)), bg_score, matched)

    def _transform(self, embedding, mean: np.ndarray) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float32).reshape(-1)
        return _unit(x - mean if self._scoring == SCORING_CENTERED else x)[0]

    def _background_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean used for centering, unit background matrix) for the current scoring."""
        key = (self._bg_version, self._scoring)
        if self._bg_cache is None or self._bg_cache[0] != key:
            bank = np.vstack(list(self._background.values()))
            if self._scoring == SCORING_CENTERED and len(bank):
                mean = bank.mean(axis=0).astype(np.float32)
            else:
                mean = np.zeros(EMBEDDING_SIZE, np.float32)
            matrix = _unit(bank - mean) if len(bank) else _empty()
            self._bg_cache = (key, mean, matrix)
        return self._bg_cache[1], self._bg_cache[2]

    def _profile_matrix(self, profile: _Profile, mean: np.ndarray) -> np.ndarray:
        key = (self._bg_version, self._scoring)
        if profile._cache is None or profile._cache[0] != key:
            examples = np.vstack([profile.positive, profile.augmented])
            matrix = _unit(examples - mean) if len(examples) else _empty()
            profile._cache = (key, matrix)
        return profile._cache[1]

    # --- background bank (S5f) ------------------------------------------

    def add_background(self, embeddings: np.ndarray | builtins.list[float], source: str = "calibration") -> int:
        """Add background embeddings (e.g. calibration windows). Returns the bank's total size."""
        self._check_source(source)
        rows = _rows(embeddings)
        with self._lock:
            if len(rows):
                merged = np.vstack([self._background[source], rows])
                self._background[source] = merged[-BACKGROUND_CAPS[source]:]
                self._bg_version += 1
                self._save_background()
            return self.background_size()

    def set_background(self, source: str, embeddings: np.ndarray) -> None:
        """Replace one source's embeddings (e.g. a rebuilt ESC-50 "generic" bank)."""
        self._check_source(source)
        with self._lock:
            self._background[source] = _rows(embeddings)[-BACKGROUND_CAPS[source]:]
            self._bg_version += 1
            self._save_background()

    def add_background_from_audio(self, samples: np.ndarray, source: str = "room") -> int:
        """Add the active windows of a "normal room sound" recording to the bank.

        The audio is also kept IN MEMORY (last NOISE_AUDIO_S seconds, never
        saved) as the noise for later augmentation. Returns the number of
        embeddings added (0 if the room was below the silence threshold;
        the listener's calibration windows still cover a quiet room).
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        with self._lock:
            self._noise_audio = samples[-round(NOISE_AUDIO_S * SAMPLE_RATE):].copy()
        embeddings = self.embed_audio(samples)
        self.add_background(embeddings, source)
        return int(len(embeddings))

    def background_counts(self) -> dict[str, int]:
        with self._lock:
            return {s: int(len(a)) for s, a in self._background.items()}

    def background_size(self) -> int:
        return sum(self.background_counts().values())

    @staticmethod
    def _check_source(source: str) -> None:
        if source not in BACKGROUND_SOURCES:
            raise ValueError(f"source must be one of {BACKGROUND_SOURCES}, got {source!r}")

    # --- audio -> embeddings ---------------------------------------------

    def active_windows(self, samples: np.ndarray, skip_silent: bool = True) -> builtins.list[tuple[float, np.ndarray]]:
        """(start seconds, window) for every non-silent 0.96 s window (0.48 s hop).

        ``skip_silent=False`` returns every window.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if samples.size < WINDOW_SAMPLES:
            samples = np.pad(samples, (0, WINDOW_SAMPLES - samples.size))
        gate = SilenceGate(threshold=self.silence_threshold, enabled=skip_silent)
        return [((end - WINDOW_SAMPLES) / SAMPLE_RATE, w)
                for end, w in Windower().feed(samples) if not gate.check(w).skipped]

    def embed_audio(self, samples: np.ndarray, skip_silent: bool = True) -> np.ndarray:
        """(N, 1024) embeddings of the non-silent (or, skip_silent=False, all) windows."""
        model = self._get_model()
        out = [model.infer(w).embedding for _, w in self.active_windows(samples, skip_silent)]
        return np.array(out, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)

    def augmentations(self, window: np.ndarray) -> builtins.list[np.ndarray]:
        """The window at each AUG_GAINS_DB gain, then mixed with noise at AUG_SNR_DB.

        The noise is a random slice of the room audio in memory
        (add_background_from_audio), or pink noise if there is none.
        """
        out = [apply_gain(window, g) for g in AUG_GAINS_DB]
        noise_audio = self._noise_audio
        if noise_audio is not None and noise_audio.size >= window.size and _rms(noise_audio) > 0:
            start = int(self._rng.integers(0, noise_audio.size - window.size + 1))
            noise = noise_audio[start:start + window.size]
        else:
            noise = pink_noise(window.size, self._rng)
        out.append(mix_at_snr(window, noise, AUG_SNR_DB))
        return out

    def add_example_from_audio(self, profile_id: str, samples: np.ndarray) -> int:
        """Like add_example_from_mic, from audio already in memory."""
        self._get(profile_id)
        model = self._get_model()
        windows = [w for _, w in self.active_windows(samples)]
        embeddings = _rows([model.infer(w).embedding for w in windows])
        augmented = None
        if self.augment and windows:
            augmented = _rows([model.infer(a).embedding for w in windows for a in self.augmentations(w)])
        with self._lock:
            profile = self._get(profile_id)
            if len(embeddings) == 0:
                log.warning("No non-silent audio recorded for %s; nothing added", profile_id)
            else:
                profile.add(embeddings, positive=True, augmented=augmented)
                self._save_profile(profile_id)
            return int(len(profile.positive))

    def test_from_audio(self, profile_id: str, samples: np.ndarray) -> float:
        """Like test_from_mic, from audio already in memory."""
        self._get(profile_id)
        embeddings = self.embed_audio(samples)
        if len(embeddings) == 0:
            return 0.0
        return max(self.similarity(profile_id, e) for e in embeddings)

    def _get_model(self):
        if self._model is None:
            from backend.ml.yamnet import YamnetModel

            self._model = YamnetModel.instance()
        return self._model

    # --- persistence -----------------------------------------------------

    def _get(self, profile_id: str) -> _Profile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise KeyError(f"unknown profile_id {profile_id!r}") from None

    @property
    def _index_path(self) -> Path:
        return self.root / "profiles.json"

    @property
    def _background_path(self) -> Path:
        return self.root / "background.npz"

    def _npz_path(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.npz"

    def _load(self) -> None:
        if self._background_path.exists():
            with np.load(self._background_path) as npz:
                for s in BACKGROUND_SOURCES:
                    if s in npz.files:
                        self._background[s] = _rows(npz[s])
        if not self._index_path.exists():
            return
        data = json.loads(self._index_path.read_text())
        self.method = data.get("method", METHOD_EMBEDDING)
        for rec in data.get("profiles", []):
            profile = _Profile(rec["id"], rec["name"], rec["category"], rec.get("method", METHOD_EMBEDDING))
            path = self._npz_path(profile.id)
            if path.exists():
                with np.load(path) as npz:
                    profile.positive = _rows(npz["positive"])
                    profile.negative = _rows(npz["negative"])
                    if "augmented" in npz.files:  # absent in pre-S5f profiles
                        profile.augmented = _rows(npz["augmented"])
            self._profiles[profile.id] = profile

    def _save_index(self) -> None:
        data = {
            "method": self.method,
            "profiles": [
                {"id": p.id, "name": p.name, "category": p.category, "method": p.method}
                for p in self._profiles.values()
            ],
        }
        self._atomic_write(self._index_path, lambda f: f.write(json.dumps(data, indent=2).encode()))

    def _save_profile(self, profile_id: str) -> None:
        p = self._profiles[profile_id]
        self._atomic_write(self._npz_path(profile_id),
                           lambda f: np.savez(f, positive=p.positive, augmented=p.augmented, negative=p.negative))

    def _save_background(self) -> None:
        arrays = dict(self._background)
        self._atomic_write(self._background_path, lambda f: np.savez(f, **arrays))

    def _save_base_feedback(self, vec: np.ndarray, category: str, positive: bool) -> None:
        path = self.root / "feedback.npz"
        arrays: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as npz:
                arrays = {k: npz[k] for k in npz.files}
        key = f"{category}_{'positive' if positive else 'negative'}"
        old = arrays.get(key, _empty())
        arrays[key] = np.vstack([old, vec[None, :]])
        self._atomic_write(path, lambda f: np.savez(f, **arrays))

    def _atomic_write(self, path: Path, write: Callable) -> None:
        """Write via a temp file + rename, so a crash never leaves a half-written file."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)

