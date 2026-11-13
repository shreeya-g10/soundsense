"""Real ProfileStore: user-taught custom sounds matched by YAMNet embeddings or MFCCs.

Owner: Shreeya (Role 1). Parts S5, S5f and S6.

Implements ``contracts.types.ProfileStore`` plus:
    match(embedding, samples=None)   YamnetListener calls it on every active
                                     window to fill WindowResult.custom_matches
    add_background(embedding, source) the listener feeds calibration windows
    is_negative(embedding, category)  for the engine: has the user marked
                                     sounds like this as false alerts?

Storage (under data/profiles/, git-ignored). Only features (embeddings and
MFCC statistics) are stored, never audio:
    profiles.json   {"method": ..., "profiles": [{id, name, category, method}]}
    <id>.npz        per method, one array per kind:
                      embedding: positive / augmented / negative     (N, 1024)
                      mfcc:      mfcc_positive / mfcc_augmented / mfcc_negative (N, 40)
                    positive = recorded windows (and positive feedback),
                    augmented = the same windows at -6/+6 dB and mixed with
                    room noise at 10 dB SNR (S5f), negative = false-alert
                    feedback (S6).
    background.npz  background bank (S5f), one array per source:
                      calibration: silence-gate calibration windows (listener)
                      room: active windows of the "normal room sound" recording
                      generic: ESC-50 non-target clips (eval/model/build_background_bank.py)
    category_negatives.npz  false-alert feedback on base-model alerts
                    (profile_id=None), one (N, 1024) array per category (S6)
    feedback.npz    correct-alert feedback on base-model alerts, "<category>_positive"

Methods (S6). Each profile is built with, and always matched by, the method
that was selected when it was created (``set_method`` affects new profiles):
    "embedding": the 1024-d YAMNet embedding (default).
    "mfcc":      baseline: librosa MFCC (n_mfcc=20) of the window, mean and
                 std over time = a 40-d vector (stored as is). Needs no YAMNet.
                 Matched with the same top-k cosine, without centering or the
                 margin rule (the background bank holds embeddings), at
                 MFCC_MATCH_THRESHOLD. The cosine ignores the 2 c0 values
                 (mean/std of coefficient 0, i.e. loudness): c0 is ~10x larger
                 than the rest, so with it every pair of sounds scores ~1.00
                 (F1 0.09 vs 0.76 on the tuning set). ``mfcc_drop_c0=False``
                 keeps them, for comparison.

Scoring of embedding profiles (S5f), ``scoring=``:
    "raw":      cosine similarity of the raw embeddings.
    "centered": (default) subtract the background mean from every embedding,
                then L2-normalise, then cosine. YAMNet embeddings are all
                non-negative and share a large common component, so raw
                cosines are high for almost any pair. Centering removes what
                "this room always sounds like". With an empty background bank
                it is the same as raw.
A profile's similarity is the mean of its top-3 positive (recorded +
augmented) examples.

Matching: a window matches a profile when
    1. similarity >= threshold, and
    2. (margin rule, embedding profiles, ``margin=True``) similarity > the
       window's similarity to the background bank (top-3 mean), and
    3. (negatives rule, S6) similarity > the window's similarity to the
       profile's negatives (top-3 mean), i.e. it is not closer to sounds the
       user said were false alerts.

Thresholds (see eval/model/README.md, "Custom-sound threshold"):
    raw 0.80 (S5); centered CENTERED_MATCH_THRESHOLD; mfcc MFCC_MATCH_THRESHOLD.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
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

N_MFCC = 20
MFCC_SIZE = 2 * N_MFCC  # mean + std over time
MFCC_MATCH_THRESHOLD = 0.991  # S6, provisional (synthetic set, c0 ignored): see eval/model/README.md
MFCC_C0_DIMS = [0, N_MFCC]  # mean and std of coefficient 0 in the 40-d vector
FEATURE_SIZES = {METHOD_EMBEDDING: EMBEDDING_SIZE, METHOD_MFCC: MFCC_SIZE}
KINDS = ("positive", "augmented", "negative")

BACKGROUND_SOURCES = ("calibration", "room", "generic")
# Newest embeddings kept per source, so the bank (and its mean) can't grow forever.
BACKGROUND_CAPS = {"calibration": 300, "room": 300, "generic": 3000}
CATEGORY_NEGATIVES_CAP = 500  # newest false-alert embeddings kept per category
NOISE_AUDIO_S = 30.0  # seconds of room audio kept IN MEMORY for noise augmentation
RECENT_FEATURES = 4096  # windows (~33 min) whose MFCC is remembered, for feedback

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


def mfcc_vector(samples: np.ndarray) -> np.ndarray:
    """40-d MFCC feature of one window: mean and std over time of 20 MFCCs."""
    import librosa

    y = np.asarray(samples, dtype=np.float32).ravel()
    m = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    return np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


def _empty(size: int = EMBEDDING_SIZE) -> np.ndarray:
    return np.zeros((0, size), np.float32)


def _rows(x, size: int = EMBEDDING_SIZE) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1, size)


def _top_k_mean(matrix: np.ndarray, query: np.ndarray, k: int) -> float | None:
    """Mean of the k highest cosine similarities (rows are unit vectors); None if empty."""
    if len(matrix) == 0:
        return None
    sims = matrix @ query
    k = min(k, sims.size)
    return float(np.partition(sims, sims.size - k)[-k:].mean())


def _key(embedding) -> bytes:
    """Identity of a window's embedding (WindowResult.embedding round-trips float32 exactly)."""
    return hashlib.blake2b(np.asarray(embedding, np.float32).tobytes(), digest_size=16).digest()


def _npz_key(method: str, kind: str) -> str:
    return kind if method == METHOD_EMBEDDING else f"{method}_{kind}"


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
    background: float | None  # score to the background bank; None if empty or not used
    matched: bool  # threshold + margin rule + negatives rule
    negative: float | None = None  # score to the profile's negatives; None if it has none


@dataclass
class _Profile:
    id: str
    name: str
    category: str
    method: str
    arrays: dict = field(default_factory=dict)  # (method, kind) -> (N, size) array
    _cache: dict = field(default_factory=dict)  # (kind group, bank version, scoring) -> unit matrix

    def get(self, kind: str, method: str | None = None) -> np.ndarray:
        method = method or self.method
        return self.arrays.get((method, kind), _empty(FEATURE_SIZES[method]))

    def add(self, kind: str, rows: np.ndarray, method: str | None = None) -> None:
        method = method or self.method
        rows = _rows(rows, FEATURE_SIZES[method])
        if len(rows):
            self.arrays[(method, kind)] = np.vstack([self.get(kind, method), rows])
            self._cache.clear()

    # The profile's own-method arrays.
    positive = property(lambda self: self.get("positive"))
    augmented = property(lambda self: self.get("augmented"))
    negative = property(lambda self: self.get("negative"))


class YamnetProfileStore:
    """Implements ``contracts.types.ProfileStore`` (see module docstring)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        threshold: float | None = None,
        mfcc_threshold: float = MFCC_MATCH_THRESHOLD,
        mfcc_drop_c0: bool = True,
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
            threshold: Minimum similarity for match() (embedding profiles).
                None = the default for ``scoring`` (THRESHOLDS); centered
                scoring with an empty background bank is identical to raw,
                so it then uses the raw default (see effective_threshold).
            mfcc_threshold: Minimum similarity for MFCC profiles.
            mfcc_drop_c0: Ignore coefficient 0 (loudness) in MFCC cosines.
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
        for t in (threshold, mfcc_threshold):
            if not 0.0 <= t <= 1.0:
                raise ValueError("threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.root = Path(root)
        self.threshold = threshold
        self.mfcc_threshold = mfcc_threshold
        self.mfcc_drop_c0 = mfcc_drop_c0
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
        self._bg_cache: tuple | None = None  # (key, mean, unit matrix)
        self._category_negatives: dict[str, np.ndarray] = {}
        self._category_positives: dict[str, np.ndarray] = {}
        self._noise_audio: np.ndarray | None = None  # room audio, memory only
        self._recent_mfcc: OrderedDict[bytes, np.ndarray] = OrderedDict()  # features, memory only
        self.method: CustomMethod = METHOD_EMBEDDING
        self._load()

    @property
    def scoring(self) -> Scoring:
        return self._scoring

    @scoring.setter
    def scoring(self, value: Scoring) -> None:
        """Switch scoring. A default threshold follows it; an explicit one is kept."""
        if value not in THRESHOLDS:
            raise ValueError(f"scoring must be 'raw' or 'centered', got {value!r}")
        with self._lock:
            self._scoring = value
            if self._auto_threshold:
                self.threshold = THRESHOLDS[value]

    @property
    def effective_threshold(self) -> float:
        """The threshold match() uses for embedding profiles right now.

        With a default threshold, centered scoring and an empty background
        bank (nothing to center by, so scores are raw), the raw default.
        """
        with self._lock:
            if self._auto_threshold and self._scoring == SCORING_CENTERED and self.background_size() == 0:
                return RAW_MATCH_THRESHOLD
            return self.threshold

    def threshold_for(self, profile_id: str) -> float:
        """The match threshold of one profile (depends on its method)."""
        with self._lock:
            method = self._get(profile_id).method
        return self.mfcc_threshold if method == METHOD_MFCC else self.effective_threshold

    # --- ProfileStore protocol -------------------------------------------

    def create(self, name: str, category: str, method: CustomMethod | None = None) -> str:
        """Create a profile; it is built and matched with ``method`` (default: the current one)."""
        method = method or self.method
        if method not in FEATURE_SIZES:
            raise ValueError(f"unknown method {method!r}; use 'embedding' or 'mfcc'")
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        with self._lock:
            profile_id = uuid.uuid4().hex[:8]
            self._profiles[profile_id] = _Profile(profile_id, name.strip(), category, method)
            self._save_profile(profile_id)
            self._save_index()
            return profile_id

    def add_example_from_mic(self, profile_id: str, seconds: float = 3) -> int:
        """Record, drop silent windows, store each window's features (+ augmentations).

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
        """Store feedback on an alert.

        With a profile: positive -> appended to its positives, negative ->
        to its negatives (used by the negatives rule in match()). MFCC
        profiles need the window's MFCC, which is looked up from recently
        matched windows (memory only); if it is no longer known, the
        embedding is kept under the profile's embedding arrays instead and a
        warning is logged.

        profile_id=None (base-model alert): negative -> category_negatives.npz
        (see is_negative()); positive -> feedback.npz.
        """
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        vec = np.asarray(embedding, dtype=np.float32).ravel()
        if vec.size != EMBEDDING_SIZE:
            raise ValueError(f"embedding must have {EMBEDDING_SIZE} values, got {vec.size}")
        kind = "positive" if positive else "negative"
        with self._lock:
            if profile_id is None:
                store = self._category_positives if positive else self._category_negatives
                old = store.get(category, _empty())
                store[category] = np.vstack([old, vec[None, :]])[-CATEGORY_NEGATIVES_CAP:]
                if positive:
                    self._save_category("feedback.npz", self._category_positives, suffix="_positive")
                else:
                    self._save_category("category_negatives.npz", self._category_negatives)
                return
            profile = self._get(profile_id)
            if profile.method == METHOD_MFCC:
                feature = self._recent_mfcc.get(_key(vec))
                if feature is None:
                    log.warning("Feedback for MFCC profile %s: window's MFCC no longer known; "
                                "kept as an embedding (not used by MFCC matching)", profile_id)
                    profile.add(kind, vec, method=METHOD_EMBEDDING)
                else:
                    profile.add(kind, feature)
            else:
                profile.add(kind, vec)
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
        """Feature extraction for profiles created from now on: "embedding" or "mfcc".

        Existing profiles keep the method they were built with.
        """
        if method not in (METHOD_EMBEDDING, METHOD_MFCC):
            raise ValueError(f"unknown method {method!r}; use 'embedding' or 'mfcc'")
        with self._lock:
            self.method = method
            self._save_index()

    # --- matching (called by YamnetListener) ------------------------------

    def similarity(self, profile_id: str, embedding: np.ndarray | builtins.list[float],
                   samples: np.ndarray | None = None) -> float:
        """Profile similarity, 0..1: mean of the top-k positive examples (0 if none)."""
        return self.score(profile_id, embedding, samples).similarity

    def background_similarity(self, embedding: np.ndarray | builtins.list[float]) -> float | None:
        """Similarity to the background bank (top-k mean, same scoring); None if empty."""
        with self._lock:
            mean, bg = self._background_state()
            return _top_k_mean(bg, self._transform(embedding, mean), self.top_k)

    def score(self, profile_id: str, embedding: np.ndarray | builtins.list[float] | None,
              samples: np.ndarray | None = None) -> ScoreDetail:
        """Similarity, background/negative similarity and match decision for one window.

        MFCC profiles use ``samples`` (the window's audio) if given, else the
        MFCC remembered for ``embedding`` from a recent match(); if neither
        is available the score is 0.
        """
        with self._lock:
            profile = self._get(profile_id)
            ctx = self._query_context(embedding, samples, need_mfcc=profile.method == METHOD_MFCC)
            return self._detail(profile, ctx)

    def match(self, embedding: np.ndarray | builtins.list[float] | None,
              samples: np.ndarray | None = None) -> builtins.list[CustomMatch]:
        """Profiles the window matches (threshold + margin + negatives rules), best first.

        ``samples`` (the window's audio, kept in memory only) is needed for
        MFCC profiles; without it they are skipped unless the window's MFCC
        is already known.
        """
        with self._lock:
            need_mfcc = any(p.method == METHOD_MFCC for p in self._profiles.values())
            ctx = self._query_context(embedding, samples, need_mfcc)
            matches = []
            for p in self._profiles.values():
                d = self._detail(p, ctx)
                if d.matched:
                    matches.append(CustomMatch(profile_id=p.id, name=p.name, category=p.category,
                                               similarity=d.similarity))
        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def is_negative(self, embedding: np.ndarray | builtins.list[float], category: str) -> bool:
        """True if the user has marked base-model alerts like this one as false.

        For the engine: before raising a base-model alert for ``category``,
        skip it if this returns True. The window counts as negative when its
        similarity to that category's false-alert embeddings (top-k mean,
        same scoring as embedding profiles) reaches effective_threshold and
        is higher than its similarity to the category's correct-alert
        feedback (if any). False when the category has no negatives.
        """
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")
        with self._lock:
            negs = self._category_negatives.get(category)
            if negs is None or len(negs) == 0:
                return False
            mean, _ = self._background_state()
            query = self._transform(embedding, mean)
            neg = _top_k_mean(_unit(negs - mean), query, self.top_k)
            pos_rows = self._category_positives.get(category)
            pos = _top_k_mean(_unit(pos_rows - mean), query, self.top_k) if pos_rows is not None and len(pos_rows) else None
            return neg >= self.effective_threshold and (pos is None or neg > pos)

    def category_negative_counts(self) -> dict[str, int]:
        with self._lock:
            return {c: int(len(a)) for c, a in self._category_negatives.items()}

    def _query_context(self, embedding, samples, need_mfcc: bool) -> dict:
        """Transformed query vectors shared by all profiles for one window."""
        ctx: dict[str, Any] = {"emb": None, "bg": None, "mean": None, "mfcc": None}
        if embedding is not None:
            mean, bg = self._background_state()
            q = self._transform(embedding, mean)
            ctx.update(emb=q, mean=mean, bg=_top_k_mean(bg, q, self.top_k))
        if need_mfcc:
            feature = None
            if samples is not None:
                feature = mfcc_vector(samples)
                if embedding is not None:
                    self._remember_mfcc(embedding, feature)
            elif embedding is not None:
                feature = self._recent_mfcc.get(_key(embedding))
            if feature is not None:
                ctx["mfcc"] = _unit(self._mfcc_view(feature))[0]
        return ctx

    def _mfcc_view(self, x: np.ndarray) -> np.ndarray:
        """The MFCC values compared by the cosine (c0 dropped unless mfcc_drop_c0=False)."""
        return np.delete(x, MFCC_C0_DIMS, axis=-1) if self.mfcc_drop_c0 else x

    def _remember_mfcc(self, embedding, feature: np.ndarray) -> None:
        self._recent_mfcc[_key(embedding)] = feature
        while len(self._recent_mfcc) > RECENT_FEATURES:
            self._recent_mfcc.popitem(last=False)

    def _detail(self, profile: _Profile, ctx: dict) -> ScoreDetail:
        if profile.method == METHOD_MFCC:
            query, bg_score, threshold, use_margin = ctx["mfcc"], None, self.mfcc_threshold, False
            mean = None
        else:
            query, bg_score, threshold, use_margin = ctx["emb"], ctx["bg"], self.effective_threshold, self.margin
            mean = ctx["mean"]
        if query is None:
            return ScoreDetail(0.0, bg_score, False)
        pos = _top_k_mean(self._profile_matrix(profile, "pos", mean), query, self.top_k)
        neg = _top_k_mean(self._profile_matrix(profile, "neg", mean), query, self.top_k)
        if pos is None:
            return ScoreDetail(0.0, bg_score, False, neg)
        matched = (pos >= threshold
                   and (not use_margin or bg_score is None or pos > bg_score)
                   and (neg is None or pos > neg))
        return ScoreDetail(float(np.clip(pos, 0.0, 1.0)), bg_score, matched, neg)

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

    def _profile_matrix(self, profile: _Profile, group: str, mean: np.ndarray | None) -> np.ndarray:
        """Unit rows of the profile's positives ("pos": recorded + augmented) or negatives ("neg")."""
        key = (group, self._bg_version, self._scoring)
        if key not in profile._cache:
            kinds = ("positive", "augmented") if group == "pos" else ("negative",)
            rows = np.vstack([profile.get(k) for k in kinds])
            if len(rows) and mean is not None:  # embedding profile: same centering as the query
                rows = rows - mean
            if profile.method == METHOD_MFCC:
                rows = self._mfcc_view(rows)
            profile._cache[key] = _unit(rows) if len(rows) else rows
        return profile._cache[key]

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

    # --- audio -> features -----------------------------------------------

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

    def features(self, windows: builtins.list[np.ndarray], method: str) -> np.ndarray:
        """(N, size) features of audio windows for ``method``."""
        if method == METHOD_MFCC:
            return _rows([mfcc_vector(w) for w in windows], MFCC_SIZE)
        model = self._get_model()
        return _rows([model.infer(w).embedding for w in windows])

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
        method = self._get(profile_id).method
        windows = [w for _, w in self.active_windows(samples)]
        examples = self.features(windows, method)
        augmented = None
        if self.augment and windows:
            augmented = self.features([a for w in windows for a in self.augmentations(w)], method)
        with self._lock:
            profile = self._get(profile_id)
            if len(examples) == 0:
                log.warning("No non-silent audio recorded for %s; nothing added", profile_id)
            else:
                profile.add("positive", examples)
                if augmented is not None:
                    profile.add("augmented", augmented)
                self._save_profile(profile_id)
            return int(len(profile.positive))

    def test_from_audio(self, profile_id: str, samples: np.ndarray) -> float:
        """Like test_from_mic, from audio already in memory."""
        profile = self._get(profile_id)
        windows = [w for _, w in self.active_windows(samples)]
        if not windows:
            return 0.0
        if profile.method == METHOD_MFCC:
            return max(self.similarity(profile_id, None, samples=w) for w in windows)
        return max(self.similarity(profile_id, e) for e in self.features(windows, METHOD_EMBEDDING))

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
        negatives = self.root / "category_negatives.npz"
        if negatives.exists():
            with np.load(negatives) as npz:
                self._category_negatives = {c: _rows(npz[c]) for c in npz.files if c in CATEGORIES}
        feedback = self.root / "feedback.npz"
        if feedback.exists():
            with np.load(feedback) as npz:
                self._category_positives = {
                    k[: -len("_positive")]: _rows(npz[k]) for k in npz.files
                    if k.endswith("_positive") and k[: -len("_positive")] in CATEGORIES
                }
        if not self._index_path.exists():
            return
        data = json.loads(self._index_path.read_text())
        self.method = data.get("method", METHOD_EMBEDDING)
        for rec in data.get("profiles", []):
            profile = _Profile(rec["id"], rec["name"], rec["category"], rec.get("method", METHOD_EMBEDDING))
            path = self._npz_path(profile.id)
            if path.exists():
                with np.load(path) as npz:
                    for method, size in FEATURE_SIZES.items():
                        for kind in KINDS:
                            key = _npz_key(method, kind)
                            if key in npz.files and npz[key].size:  # "augmented" absent before S5f
                                profile.arrays[(method, kind)] = _rows(npz[key], size)
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
        # Always the own method's three arrays, plus any other non-empty ones.
        arrays = {_npz_key(p.method, k): p.get(k) for k in KINDS}
        for (method, kind), a in p.arrays.items():
            if len(a):
                arrays[_npz_key(method, kind)] = a
        self._atomic_write(self._npz_path(profile_id), lambda f: np.savez(f, **arrays))

    def _save_background(self) -> None:
        arrays = dict(self._background)
        self._atomic_write(self._background_path, lambda f: np.savez(f, **arrays))

    def _save_category(self, filename: str, arrays: dict[str, np.ndarray], suffix: str = "") -> None:
        path = self.root / filename
        out: dict[str, np.ndarray] = {}
        if path.exists():  # keep keys we don't manage (e.g. old feedback.npz negatives)
            with np.load(path) as npz:
                out = {k: npz[k] for k in npz.files}
        out.update({f"{c}{suffix}": a for c, a in arrays.items()})
        self._atomic_write(path, lambda f: np.savez(f, **out))

    def _atomic_write(self, path: Path, write: Callable) -> None:
        """Write via a temp file + rename, so a crash never leaves a half-written file."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
