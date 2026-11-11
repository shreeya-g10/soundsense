"""Real ProfileStore: user-taught custom sounds matched by YAMNet embeddings.

Owner: Shreeya (Role 1). Part S5.

Implements ``contracts.types.ProfileStore`` plus ``match(embedding)``, which
``YamnetListener`` calls on every active window to fill
``WindowResult.custom_matches``.

Storage (under data/profiles/, git-ignored). Only embeddings are stored,
never audio:
    profiles.json   {"method": ..., "profiles": [{id, name, category, method}]}
    <id>.npz        positive: (N, 1024) float32, negative: (M, 1024) float32
    feedback.npz    feedback on base-model alerts (profile_id=None), per category

Matching: for each profile, similarity = mean cosine similarity of the k=3
nearest positive examples (fewer if the profile has fewer). Profiles at or
above the threshold (default 0.80) are returned, best first.

Measured on real YAMNet embeddings: speech vs chimes ~0.3, noise vs anything
~0.1-0.3, but different chimes 0.76-0.99 and two voices ~0.87. So 0.80
separates *kinds* of sound well; telling two similar sounds apart (my cooker
vs a kettle) relies on good examples plus negative feedback.

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
from typing import Any, Callable

import numpy as np

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES, Windower
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from contracts.types import CATEGORIES, EMBEDDING_SIZE, METHOD_EMBEDDING, METHOD_MFCC, CustomMatch, CustomMethod

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "profiles"
DEFAULT_MATCH_THRESHOLD = 0.80
DEFAULT_TOP_K = 3

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


@dataclass
class _Profile:
    id: str
    name: str
    category: str
    method: str
    positive: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBEDDING_SIZE), np.float32))
    negative: np.ndarray = field(default_factory=lambda: np.zeros((0, EMBEDDING_SIZE), np.float32))
    _pos_unit: np.ndarray | None = None  # cache for matching

    def pos_unit(self) -> np.ndarray:
        if self._pos_unit is None:
            self._pos_unit = _unit(self.positive) if len(self.positive) else self.positive
        return self._pos_unit

    def add(self, embeddings: np.ndarray, positive: bool) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)
        if positive:
            self.positive = np.vstack([self.positive, embeddings])
            self._pos_unit = None
        else:
            self.negative = np.vstack([self.negative, embeddings])


class YamnetProfileStore:
    """Implements ``contracts.types.ProfileStore`` (see module docstring)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        silence_threshold: float = DEFAULT_THRESHOLD,
        recorder: Recorder | None = None,
        device: int | str | None = None,
        model=None,
    ) -> None:
        """Open (or create) the store in ``root`` and load saved profiles.

        Args:
            root: Folder for profiles.json and the .npz files.
            threshold: Minimum similarity for match() to report a profile.
            top_k: Number of nearest positive examples averaged.
            silence_threshold: RMS below which a recorded window is dropped.
            recorder: record(seconds) -> audio; defaults to the microphone.
                Tests pass a fake.
            device: Mic device for the default recorder.
            model: Object with infer(samples).embedding; defaults to YAMNet.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.root = Path(root)
        self.threshold = threshold
        self.top_k = top_k
        self.silence_threshold = silence_threshold
        self._recorder = recorder or (lambda s: record_from_mic(s, device=device))
        self._model = model
        self._lock = threading.RLock()
        self._profiles: dict[str, _Profile] = {}
        self.method: CustomMethod = METHOD_EMBEDDING
        self._load()

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
        """Record, drop silent windows, store each window's embedding.

        Returns the profile's total example count (unchanged if every
        window was silent).
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
        """Mean cosine similarity of the top-k nearest positive examples (0 if none)."""
        with self._lock:
            return self._similarity(self._get(profile_id), _unit(embedding)[0])

    def match(self, embedding: np.ndarray | builtins.list[float]) -> builtins.list[CustomMatch]:
        """Profiles whose similarity >= threshold, best first."""
        query = _unit(embedding)[0]
        with self._lock:
            matches = [
                CustomMatch(profile_id=p.id, name=p.name, category=p.category, similarity=s)
                for p in self._profiles.values()
                if (s := self._similarity(p, query)) >= self.threshold
            ]
        return sorted(matches, key=lambda m: m.similarity, reverse=True)

    def _similarity(self, profile: _Profile, query: np.ndarray) -> float:
        pos = profile.pos_unit()
        if len(pos) == 0:
            return 0.0
        sims = pos @ query
        k = min(self.top_k, sims.size)
        top = np.partition(sims, sims.size - k)[-k:]
        return float(np.clip(top.mean(), 0.0, 1.0))

    # --- audio -> embeddings ---------------------------------------------

    def embed_audio(self, samples: np.ndarray) -> np.ndarray:
        """(N, 1024) embeddings of the non-silent 0.96 s windows in ``samples``."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if samples.size < WINDOW_SAMPLES:
            samples = np.pad(samples, (0, WINDOW_SAMPLES - samples.size))
        gate = SilenceGate(threshold=self.silence_threshold)
        model = self._get_model()
        out = [model.infer(w).embedding for _, w in Windower().feed(samples) if not gate.check(w).skipped]
        return np.array(out, dtype=np.float32).reshape(-1, EMBEDDING_SIZE)

    def add_example_from_audio(self, profile_id: str, samples: np.ndarray) -> int:
        """Like add_example_from_mic, from audio already in memory."""
        embeddings = self.embed_audio(samples)
        with self._lock:
            profile = self._get(profile_id)
            if len(embeddings) == 0:
                log.warning("No non-silent audio recorded for %s; nothing added", profile_id)
            else:
                profile.add(embeddings, positive=True)
                self._save_profile(profile_id)
            return int(len(profile.positive))

    def test_from_audio(self, profile_id: str, samples: np.ndarray) -> float:
        """Like test_from_mic, from audio already in memory."""
        embeddings = self.embed_audio(samples)
        with self._lock:
            profile = self._get(profile_id)
            if len(embeddings) == 0:
                return 0.0
            return max(self._similarity(profile, q) for q in _unit(embeddings))

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

    def _npz_path(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.npz"

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        data = json.loads(self._index_path.read_text())
        self.method = data.get("method", METHOD_EMBEDDING)
        for rec in data.get("profiles", []):
            profile = _Profile(rec["id"], rec["name"], rec["category"], rec.get("method", METHOD_EMBEDDING))
            path = self._npz_path(profile.id)
            if path.exists():
                with np.load(path) as npz:
                    profile.positive = npz["positive"].astype(np.float32)
                    profile.negative = npz["negative"].astype(np.float32)
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
        self._atomic_write(self._npz_path(profile_id), lambda f: np.savez(f, positive=p.positive, negative=p.negative))

    def _save_base_feedback(self, vec: np.ndarray, category: str, positive: bool) -> None:
        path = self.root / "feedback.npz"
        arrays: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as npz:
                arrays = {k: npz[k] for k in npz.files}
        key = f"{category}_{'positive' if positive else 'negative'}"
        old = arrays.get(key, np.zeros((0, EMBEDDING_SIZE), np.float32))
        arrays[key] = np.vstack([old, vec[None, :]])
        self._atomic_write(path, lambda f: np.savez(f, **arrays))

    def _atomic_write(self, path: Path, write: Callable) -> None:
        """Write via a temp file + rename, so a crash never leaves a half-written file."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            write(f)
        os.replace(tmp, path)
