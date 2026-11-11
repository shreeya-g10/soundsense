"""Fake AudioListener for developing the engine without Shreeya's real audio code.

Owner: Vaidehi (Role 2).

Implements the ``AudioListener`` protocol from ``contracts/types.py``. Instead
of reading the microphone, it plays a named, scripted scenario: one
``WindowResult`` every 0.48 s on a background thread.

Usage::

    from backend.engine.fake_listener import FakeListener

    listener = FakeListener(on_window=print, scenario="whistles")
    listener.start()
    ...
    listener.stop()

Try it from the repo root::

    python -m backend.engine.fake_listener whistles

Scenarios (see SCENARIOS):
    whistles          3 pressure-cooker whistles with silent gaps; whistle
                      windows carry a custom match ("My Kitchen Cooker")
    smoke_alarm       about 10 s of a smoke alarm
    doorbell_with_tv  TV playing throughout (high television score), with a
                      short doorbell in the middle
    baby_long         a baby crying for about 2.5 minutes
    silence           silent windows only (all skipped)

When a scenario finishes, the listener keeps sending silent windows until
stop() is called, like a real mic. Pass ``loop=True`` to repeat the scenario
instead.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from contracts.types import (
    BABY_CRY,
    CATEGORIES,
    CONTEXT_MUSIC,
    CONTEXT_SPEECH,
    CONTEXT_TELEVISION,
    CONTEXTS,
    DOORBELL,
    EMBEDDING_SIZE,
    PRESSURE_COOKER,
    SMOKE_ALARM,
    CustomMatch,
    WindowResult,
)

log = logging.getLogger(__name__)

WINDOW_INTERVAL_S = 0.48

FAKE_CUSTOM_PROFILE = CustomMatch(
    profile_id="fake-cooker-1",
    name="My Kitchen Cooker",
    category=PRESSURE_COOKER,
    similarity=0.0,  # set per window
)


# ---------------------------------------------------------------------------
# Scenario scripts
# ---------------------------------------------------------------------------

@dataclass
class _Step:
    """Script for one window. Turned into a WindowResult when it is sent."""

    skipped: bool = False
    rms: float = 0.005
    category: str | None = None  # dominant category, if any
    score: float = 0.0  # score for the dominant category
    context: dict[str, float] = field(default_factory=dict)
    custom_similarity: float | None = None  # add FAKE_CUSTOM_PROFILE match if set


def _silence(n: int) -> list[_Step]:
    return [_Step(skipped=True, rms=0.004) for _ in range(n)]


def _sound(n: int, category: str, score: float, rms: float = 0.3, **kw) -> list[_Step]:
    return [_Step(rms=rms, category=category, score=score, **kw) for _ in range(n)]


def _whistles() -> list[_Step]:
    steps = _silence(6)
    for _ in range(3):
        steps += _sound(5, PRESSURE_COOKER, 0.9, rms=0.45, custom_similarity=0.9)
        steps += _silence(12)  # ~6 s gap between whistles
    return steps


def _smoke_alarm() -> list[_Step]:
    return _silence(4) + _sound(21, SMOKE_ALARM, 0.95, rms=0.6)  # ~10 s


def _doorbell_with_tv() -> list[_Step]:
    tv = {CONTEXT_TELEVISION: 0.85, CONTEXT_SPEECH: 0.6, CONTEXT_MUSIC: 0.3}
    background = [_Step(rms=0.15, context=tv) for _ in range(10)]
    bell = _sound(3, DOORBELL, 0.7, rms=0.3, context=tv)
    return background + bell + background


def _baby_long() -> list[_Step]:
    return _silence(4) + _sound(313, BABY_CRY, 0.85, rms=0.35)  # ~2.5 min


def _silence_only() -> list[_Step]:
    return _silence(60)


SCENARIOS: dict[str, Callable[[], list[_Step]]] = {
    "whistles": _whistles,
    "smoke_alarm": _smoke_alarm,
    "doorbell_with_tv": _doorbell_with_tv,
    "baby_long": _baby_long,
    "silence": _silence_only,
}
"""Scenario name -> function returning that scenario's window scripts."""


# ---------------------------------------------------------------------------
# Listener
# ---------------------------------------------------------------------------

class FakeListener:
    """Scripted stand-in for the real AudioListener (same interface)."""

    def __init__(
        self,
        on_window: Callable[[WindowResult], None],
        device: int | str | None = None,
        scenario: str = "whistles",
        loop: bool = False,
        interval: float = WINDOW_INTERVAL_S,
        seed: int | None = None,
    ) -> None:
        """Create the fake listener. Does not start it.

        Args:
            on_window: Called once per window, from the background thread.
            device: Ignored; accepted to match the AudioListener protocol.
            scenario: A key of SCENARIOS.
            loop: Repeat the scenario forever instead of falling back to silence.
            interval: Seconds between windows (0.48 by default; lower it in tests).
            seed: Random seed for repeatable scores and embeddings.
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
        self._on_window = on_window
        self.device = device
        self.scenario = scenario
        self.loop = loop
        self.interval = interval
        self._rng = random.Random(seed)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # --- AudioListener protocol -------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="FakeListener", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        # Joining from inside on_window would deadlock; the loop exits on its own.
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --- internals --------------------------------------------------------

    def _run(self) -> None:
        steps = SCENARIOS[self.scenario]()
        i = 0
        next_at = time.monotonic()
        while not self._stop.is_set():
            if i < len(steps):
                step = steps[i]
            elif self.loop:
                steps, i = SCENARIOS[self.scenario](), 0
                step = steps[0]
            else:
                step = _Step(skipped=True, rms=0.004)
            i += 1

            try:
                self._on_window(self._make_window(step))
            except Exception:  # never let a callback bug kill the thread
                log.exception("on_window callback raised")

            next_at += self.interval
            self._stop.wait(max(0.0, next_at - time.monotonic()))

    def _jitter(self, value: float, spread: float = 0.05) -> float:
        return min(1.0, max(0.0, value + self._rng.uniform(-spread, spread)))

    def _make_window(self, step: _Step) -> WindowResult:
        rms = self._jitter(step.rms, spread=step.rms * 0.2)
        if step.skipped:
            return WindowResult(
                timestamp=time.time(),
                rms=rms,
                skipped=True,
                category_scores={},
                context_scores={},
                embedding=None,
                custom_matches=[],
                inference_ms=0.0,
            )

        category_scores = {c: self._rng.uniform(0.0, 0.05) for c in CATEGORIES}
        if step.category is not None:
            category_scores[step.category] = self._jitter(step.score)

        context_scores = {c: self._rng.uniform(0.0, 0.1) for c in CONTEXTS}
        for key, value in step.context.items():
            context_scores[key] = self._jitter(value)

        matches: list[CustomMatch] = []
        if step.custom_similarity is not None:
            matches.append(
                CustomMatch(
                    profile_id=FAKE_CUSTOM_PROFILE.profile_id,
                    name=FAKE_CUSTOM_PROFILE.name,
                    category=FAKE_CUSTOM_PROFILE.category,
                    similarity=self._jitter(step.custom_similarity, 0.03),
                )
            )

        return WindowResult(
            timestamp=time.time(),
            rms=rms,
            skipped=False,
            category_scores=category_scores,
            context_scores=context_scores,
            embedding=[self._rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_SIZE)],
            custom_matches=matches,
            inference_ms=self._rng.uniform(30.0, 60.0),
        )


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "whistles"

    def show(w: WindowResult) -> None:
        if w.skipped:
            print(f"{w.timestamp:.2f}  rms={w.rms:.3f}  (silent, skipped)")
            return
        top = max(w.category_scores, key=w.category_scores.get)
        ctx = max(w.context_scores, key=w.context_scores.get)
        custom = f"  custom={w.custom_matches[0].name}" if w.custom_matches else ""
        print(
            f"{w.timestamp:.2f}  rms={w.rms:.3f}  top={top}:{w.category_scores[top]:.2f}"
            f"  ctx={ctx}:{w.context_scores[ctx]:.2f}{custom}"
        )

    fake = FakeListener(show, scenario=name)
    fake.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fake.stop()
