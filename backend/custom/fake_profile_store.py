"""In-memory fake ProfileStore, so the backend can run before the real one exists.

Owner: Shreeya (Role 1).

Implements the ``ProfileStore`` protocol from ``contracts/types.py``. Nothing
is recorded or saved: "recording" adds a random embedding, and "testing"
returns a plausible random similarity. Everything is lost when the process
exits.

Usage::

    from backend.custom.fake_profile_store import FakeProfileStore

    store = FakeProfileStore()
    pid = store.create("My Kitchen Cooker", "pressure_cooker")
    store.add_example_from_mic(pid)   # -> 1
    store.test_from_mic(pid)          # -> e.g. 0.87
"""

from __future__ import annotations

import builtins
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from contracts.types import CATEGORIES, EMBEDDING_SIZE, METHOD_EMBEDDING, METHOD_MFCC, CustomMethod


@dataclass
class _Profile:
    id: str
    name: str
    category: str
    examples: list[list[float]] = field(default_factory=list)


@dataclass
class FeedbackRecord:
    """One add_feedback() call, kept so tests can inspect what was sent."""

    embedding: list[float]
    category: str
    profile_id: str | None
    positive: bool


class FakeProfileStore:
    """In-memory stand-in for the real ProfileStore (same interface)."""

    def __init__(self, simulate_recording: bool = False, seed: int | None = None) -> None:
        """Create an empty store.

        Args:
            simulate_recording: If True, the *_from_mic methods sleep for
                ``seconds`` like the real store does. Default False, so tests
                run fast.
            seed: Random seed for repeatable embeddings and similarities.
        """
        self.simulate_recording = simulate_recording
        self.method: CustomMethod = METHOD_EMBEDDING
        self.feedback: builtins.list[FeedbackRecord] = []
        self._profiles: dict[str, _Profile] = {}
        self._rng = random.Random(seed)

    # --- ProfileStore protocol --------------------------------------------

    def create(self, name: str, category: str) -> str:
        self._check_category(category)
        profile_id = uuid.uuid4().hex[:8]
        self._profiles[profile_id] = _Profile(id=profile_id, name=name, category=category)
        return profile_id

    def add_example_from_mic(self, profile_id: str, seconds: float = 3) -> int:
        profile = self._get(profile_id)
        self._record(seconds)
        profile.examples.append(self._random_embedding())
        return len(profile.examples)

    def test_from_mic(self, profile_id: str, seconds: float = 3) -> float:
        profile = self._get(profile_id)
        self._record(seconds)
        if not profile.examples:
            return 0.0
        # More examples -> slightly better matches, like the real thing.
        base = min(0.95, 0.7 + 0.05 * len(profile.examples))
        return round(min(1.0, max(0.0, base + self._rng.uniform(-0.1, 0.05))), 3)

    def add_feedback(
        self,
        embedding: builtins.list[float],
        category: str,
        profile_id: str | None,
        positive: bool,
    ) -> None:
        self._check_category(category)
        if profile_id is not None:
            self._get(profile_id)
        self.feedback.append(FeedbackRecord(builtins.list(embedding), category, profile_id, positive))

    def list(self) -> builtins.list[dict[str, Any]]:
        return [
            {"id": p.id, "name": p.name, "category": p.category, "examples": len(p.examples)}
            for p in self._profiles.values()
        ]

    def delete(self, profile_id: str) -> None:
        self._get(profile_id)
        del self._profiles[profile_id]

    def set_method(self, method: CustomMethod) -> None:
        if method not in (METHOD_EMBEDDING, METHOD_MFCC):
            raise ValueError(f"unknown method {method!r}; use 'embedding' or 'mfcc'")
        self.method = method

    # --- internals --------------------------------------------------------

    def _get(self, profile_id: str) -> _Profile:
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise KeyError(f"unknown profile_id {profile_id!r}") from None

    @staticmethod
    def _check_category(category: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}")

    def _record(self, seconds: float) -> None:
        if self.simulate_recording:
            time.sleep(seconds)

    def _random_embedding(self) -> builtins.list[float]:
        return [self._rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_SIZE)]
