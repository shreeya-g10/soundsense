"""Unit tests for backend/engine/detector.py."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from contracts.types import (
    BABY_CRY,
    CATEGORIES,
    CONTEXT_MUSIC,
    CONTEXT_SPEECH,
    CONTEXT_TELEVISION,
    DOORBELL,
    EMBEDDING_SIZE,
    PRESSURE_COOKER,
    SMOKE_ALARM,
    CustomMatch,
    WindowResult,
)

from backend.engine.detector import DetectionEngine, EngineConfig
from backend.engine.fake_listener import FakeListener
from backend.storage.store import SQLiteStore


def _window(
    *,
    ts: float | None = None,
    rms: float = 0.3,
    skipped: bool = False,
    category: str | None = None,
    score: float = 0.0,
    tv: float = 0.0,
    custom: list[CustomMatch] | None = None,
    room: str = "Laptop",
) -> WindowResult:
    if skipped:
        return WindowResult(
            timestamp=ts or time.time(),
            rms=0.004,
            skipped=True,
            category_scores={},
            context_scores={},
            embedding=None,
            custom_matches=[],
            inference_ms=0.0,
            room=room,
        )
    cat_scores = {c: 0.02 for c in CATEGORIES}
    if category is not None:
        cat_scores[category] = score
    return WindowResult(
        timestamp=ts or time.time(),
        rms=rms,
        skipped=False,
        category_scores=cat_scores,
        context_scores={
            CONTEXT_SPEECH: 0.1,
            CONTEXT_MUSIC: 0.05,
            CONTEXT_TELEVISION: tv,
        },
        embedding=[0.1] * EMBEDDING_SIZE,
        custom_matches=custom or [],
        inference_ms=25.0,
        room=room,
    )


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStore(tmp_path / "engine.db")
    yield s
    s.close()


def test_silent_window_emits_level_only(store: SQLiteStore) -> None:
    levels = []
    alerts = []
    engine = DetectionEngine(
        store,
        on_level=levels.append,
        on_alert=lambda a, _e: alerts.append(a),
    )
    res = engine.process_window(_window(skipped=True))
    assert res is None
    assert alerts == []
    assert levels == [{"type": "level", "rms": 0.004, "top": None, "skipped": True}]


def test_whistle_counter_and_custom_match(store: SQLiteStore) -> None:
    alerts = []
    engine = DetectionEngine(
        store,
        on_alert=lambda a, _e: alerts.append(a),
        config=EngineConfig(cooldowns_s={PRESSURE_COOKER: 3.0}),
    )
    match = CustomMatch(
        profile_id="p1",
        name="My Kitchen Cooker",
        category=PRESSURE_COOKER,
        similarity=0.91,
    )

    t0 = 1_700_000_000.0
    # 2 windows in whistle 1 -> only 1 alert due to cooldown
    engine.process_window(_window(ts=t0, category=PRESSURE_COOKER, score=0.9, custom=[match]))
    engine.process_window(_window(ts=t0 + 0.5, category=PRESSURE_COOKER, score=0.9, custom=[match]))
    # Whistle 2 after cooldown
    engine.process_window(_window(ts=t0 + 5.0, category=PRESSURE_COOKER, score=0.88, custom=[match]))

    assert len(alerts) == 2
    assert alerts[0]["label"] == "Whistle 1 of 3"
    assert alerts[0]["custom_name"] == "My Kitchen Cooker"
    assert alerts[1]["label"] == "Whistle 2 of 3"
    assert engine.get_cached_embedding(alerts[0]["id"]) is not None


def test_tv_guard_suppresses_weak_doorbell_but_passes_strong_or_critical(store: SQLiteStore) -> None:
    alerts = []
    engine = DetectionEngine(
        store,
        on_alert=lambda a, _e: alerts.append(a),
        config=EngineConfig(cooldowns_s={DOORBELL: 2.0, SMOKE_ALARM: 2.0}),
    )
    t0 = 1_700_000_000.0

    # At sensitivity=0.5: base threshold=0.55, TV guard raises it to 0.67
    weak_bell = engine.process_window(_window(ts=t0, category=DOORBELL, score=0.60, tv=0.85))
    assert weak_bell is None
    history = store.get_history()
    assert len(history) == 1
    assert history[0]["alerted"] is False

    # Strong doorbell (0.72) passes TV guard
    strong_bell = engine.process_window(_window(ts=t0 + 3.0, category=DOORBELL, score=0.72, tv=0.85))
    assert strong_bell is not None
    assert strong_bell["category"] == DOORBELL

    # Critical smoke alarm always ignores TV guard
    smoke = engine.process_window(_window(ts=t0 + 4.0, category=SMOKE_ALARM, score=0.60, tv=0.90))
    assert smoke is not None
    assert smoke["priority"] == "critical"


def test_quiet_hours_suppresses_non_critical_and_allows_critical(store: SQLiteStore) -> None:
    now_dt = datetime.now().astimezone()
    start_h = (now_dt.hour - 1) % 24
    end_h = (now_dt.hour + 1) % 24
    settings = store.get_settings()
    settings["quiet_hours"] = {
        "enabled": True,
        "start": f"{start_h:02d}:00",
        "end": f"{end_h:02d}:00",
    }
    store.save_settings(settings)

    alerts = []
    engine = DetectionEngine(store, on_alert=lambda a, _e: alerts.append(a))

    now = now_dt.timestamp()
    assert engine.process_window(_window(ts=now, category=DOORBELL, score=0.9)) is None
    assert engine.process_window(_window(ts=now + 1, category=SMOKE_ALARM, score=0.9)) is not None

    hist = store.get_history()
    assert len(hist) == 2
    assert {h["category"]: h["alerted"] for h in hist} == {
        DOORBELL: False,
        SMOKE_ALARM: True,
    }


def test_escalation_when_unacknowledged(store: SQLiteStore) -> None:
    alerts = []
    engine = DetectionEngine(
        store,
        on_alert=lambda a, _e: alerts.append(a),
        config=EngineConfig(
            cooldowns_s={BABY_CRY: 2.0},
            escalation_delays_s={"high": 3.0},
        ),
    )
    t0 = 1_700_000_000.0
    a1 = engine.process_window(_window(ts=t0, category=BABY_CRY, score=0.85))
    assert a1 is not None and a1["escalated"] is False

    # Still unacknowledged after 3.5s -> escalated repeat
    a2 = engine.process_window(_window(ts=t0 + 3.5, category=BABY_CRY, score=0.85))
    assert a2 is not None and a2["escalated"] is True

    # Acknowledge a2 -> next alert after delay is not escalated
    store.acknowledge_alert(a2["id"])
    a3 = engine.process_window(_window(ts=t0 + 7.5, category=BABY_CRY, score=0.85))
    assert a3 is not None and a3["escalated"] is False


def test_fake_listener_whistles_end_to_end(store: SQLiteStore) -> None:
    alerts = []
    engine = DetectionEngine(
        store,
        on_alert=lambda a, _e: alerts.append(a),
        config=EngineConfig(cooldowns_s={PRESSURE_COOKER: 0.05}),
    )
    engine.start()
    listener = FakeListener(on_window=engine.on_window, scenario="whistles", interval=0.005, seed=7)
    listener.start()
    # Wait up to 2s for 3 whistles (51 scripted steps * 5ms = ~0.26s)
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline and len(alerts) < 3:
        time.sleep(0.02)
    listener.stop()
    engine.stop()

    assert len(alerts) == 3
    assert [a["label"] for a in alerts] == ["Whistle 1 of 3", "Whistle 2 of 3", "Whistle 3 of 3"]
    assert [a["priority"] for a in alerts] == ["high", "high", "critical"]


def test_disabled_category_and_sensitivity_extremes(store: SQLiteStore) -> None:
    settings = store.get_settings()
    settings["categories"][DOORBELL]["enabled"] = False
    settings["categories"][BABY_CRY]["sensitivity"] = 1.0  # threshold 0.30
    settings["categories"][SMOKE_ALARM]["sensitivity"] = 0.0  # threshold 0.80
    store.save_settings(settings)

    engine = DetectionEngine(store)
    t0 = 1_700_000_000.0
    assert engine.process_window(_window(ts=t0, category=DOORBELL, score=0.95)) is None
    assert engine.process_window(_window(ts=t0 + 1, category=SMOKE_ALARM, score=0.75)) is None
    baby = engine.process_window(_window(ts=t0 + 2, category=BABY_CRY, score=0.35))
    assert baby is not None and baby["category"] == BABY_CRY


def test_overnight_quiet_hours_and_queue_overflow(store: SQLiteStore) -> None:
    assert DetectionEngine._is_quiet_hours(
        {"enabled": True, "start": "22:00", "end": "07:00"},
        datetime.now().astimezone().replace(hour=23, minute=30).timestamp(),
    )
    assert not DetectionEngine._is_quiet_hours(
        {"enabled": True, "start": "22:00", "end": "07:00"},
        datetime.now().astimezone().replace(hour=14, minute=0).timestamp(),
    )

    engine = DetectionEngine(store, config=EngineConfig(queue_maxsize=2))
    for i in range(5):
        engine.on_window(_window(ts=1_700_000_000.0 + i, skipped=True))
    assert engine._queue.qsize() == 2
