"""Unit tests for backend/storage/store.py."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from contracts.types import (
    BABY_CRY,
    CATEGORIES,
    DOORBELL,
    PRESSURE_COOKER,
    SMOKE_ALARM,
)

from backend.storage.store import (
    EventNotFoundError,
    SQLiteStore,
    default_settings,
    format_iso_local,
)


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test_soundsense.db")
    yield s
    s.close()


def test_default_settings_and_roundtrip(store: SQLiteStore) -> None:
    settings = store.get_settings()
    assert settings == default_settings()
    assert set(settings["categories"].keys()) == set(CATEGORIES)

    settings["tv_guard"] = False
    settings["alert_style"] = "compact"
    settings["categories"][DOORBELL]["sensitivity"] = 0.8
    saved = store.save_settings(settings)
    assert saved["tv_guard"] is False
    assert store.get_settings()["categories"][DOORBELL]["sensitivity"] == 0.8


def test_events_history_filters_and_clear(store: SQLiteStore) -> None:
    now = time.time()
    store.add_event(
        event_id="e1",
        timestamp_epoch=now - 120,
        category=PRESSURE_COOKER,
        label="Whistle 1 of 3",
        confidence=0.91,
        priority="high",
        alerted=True,
        custom_name="My Kitchen Cooker",
        room="Kitchen",
        latency_ms=42.0,
    )
    store.add_event(
        event_id="e2",
        timestamp_epoch=now - 60,
        category=DOORBELL,
        label="Doorbell",
        confidence=0.77,
        priority="medium",
        alerted=False,
        custom_name=None,
        room="Laptop",
        latency_ms=30.0,
    )
    store.add_event(
        event_id="e3",
        timestamp_epoch=now,
        category=SMOKE_ALARM,
        label="Smoke alarm",
        confidence=0.96,
        priority="critical",
        alerted=True,
        custom_name=None,
        room="Laptop",
        latency_ms=24.0,
    )

    all_events = store.get_history()
    assert [e["id"] for e in all_events] == ["e3", "e2", "e1"]
    assert all_events[0]["alerted"] is True
    assert all_events[1]["alerted"] is False

    assert [e["id"] for e in store.get_history(category=DOORBELL)] == ["e2"]
    assert [e["id"] for e in store.get_history(q="kitchen")] == ["e1"]
    assert [e["id"] for e in store.get_history(q="whistle")] == ["e1"]
    assert len(store.get_history(limit=2)) == 2

    from_iso = format_iso_local(now - 90)
    to_iso = format_iso_local(now - 30)
    assert [e["id"] for e in store.get_history(from_ts=from_iso, to_ts=to_iso)] == ["e2"]

    store.clear_history()
    assert store.get_history() == []


def test_ack_feedback_and_not_found(store: SQLiteStore) -> None:
    now = time.time()
    store.add_event(
        event_id="ev1",
        timestamp_epoch=now,
        category=BABY_CRY,
        label="Baby crying",
        confidence=0.85,
        priority="high",
        alerted=True,
    )
    store.acknowledge_alert("ev1")
    updated = store.set_feedback("ev1", False)
    assert updated["acknowledged"] is True
    assert updated["feedback"] is False

    with pytest.raises(EventNotFoundError):
        store.acknowledge_alert("missing")
    with pytest.raises(EventNotFoundError):
        store.set_feedback("missing", True)


def test_stats_and_events_today(store: SQLiteStore) -> None:
    now_dt = datetime.now().astimezone()
    today_epoch = now_dt.timestamp()
    yesterday_epoch = (now_dt - timedelta(days=1)).timestamp()

    store.add_event(
        event_id="t1",
        timestamp_epoch=today_epoch,
        category=DOORBELL,
        label="Doorbell",
        confidence=0.8,
        priority="medium",
        alerted=True,
        latency_ms=100.0,
    )
    store.add_event(
        event_id="t2",
        timestamp_epoch=today_epoch,
        category=DOORBELL,
        label="Doorbell",
        confidence=0.82,
        priority="medium",
        alerted=True,
        latency_ms=200.0,
    )
    store.add_event(
        event_id="y1",
        timestamp_epoch=yesterday_epoch,
        category=SMOKE_ALARM,
        label="Smoke alarm",
        confidence=0.95,
        priority="critical",
        alerted=True,
        latency_ms=150.0,
    )

    store.set_feedback("t1", True)
    store.set_feedback("t2", False)

    assert store.events_today() == 2

    stats = store.get_stats()
    assert set(stats["per_category"].keys()) == set(CATEGORIES)
    assert stats["per_category"][DOORBELL] == 2
    assert stats["per_category"][SMOKE_ALARM] == 1
    assert len(stats["per_day"]) == 30
    assert stats["per_day"][-1]["count"] == 2
    assert stats["per_day"][-2]["count"] == 1
    assert stats["false_alert_rate"] == 0.5
    assert stats["avg_latency_ms"] == 150.0


def test_whistle_counter_persistence(store: SQLiteStore) -> None:
    wc = store.get_whistle_counter()
    assert wc == {"enabled": True, "target": 3, "count": 0}

    assert store.increment_whistle_counter()["count"] == 1
    assert store.update_whistle_counter(True, 5) == {"enabled": True, "target": 5, "count": 1}
    assert store.reset_whistle_counter() == {"enabled": True, "target": 5, "count": 0}

    with pytest.raises(ValueError):
        store.update_whistle_counter(True, 0)
    with pytest.raises(ValueError):
        store.update_whistle_counter(True, True)  # type: ignore[arg-type]


def test_date_only_to_filter_includes_full_day(store: SQLiteStore) -> None:
    now_dt = datetime.now().astimezone().replace(hour=15, minute=30, second=0, microsecond=0)
    date_str = now_dt.strftime("%Y-%m-%d")
    store.add_event(
        event_id="day1",
        timestamp_epoch=now_dt.timestamp(),
        category=DOORBELL,
        label="Doorbell",
        confidence=0.85,
        priority="medium",
        alerted=True,
    )
    res = store.get_history(from_ts=date_str, to_ts=date_str)
    assert [e["id"] for e in res] == ["day1"]
