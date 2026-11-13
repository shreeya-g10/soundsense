"""Integration tests for FastAPI routes and WebSockets (/ws/live, /ws/listener)."""

from __future__ import annotations

import struct
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from contracts.types import CATEGORIES, DOORBELL, PRESSURE_COOKER

from backend.api.runtime import BackendRuntime
from backend.custom.fake_profile_store import FakeProfileStore
from backend.engine.fake_listener import FakeListener
from backend.main import create_app
from backend.storage.store import SQLiteStore


@pytest.fixture()
def client(tmp_path):
    store = SQLiteStore(tmp_path / "api_test.db")
    profile_store = FakeProfileStore(simulate_recording=False, seed=42)
    runtime_box: dict[str, BackendRuntime] = {}

    def make_listener(on_window):
        return FakeListener(on_window=on_window, scenario="whistles", interval=0.01, seed=42)

    rt = BackendRuntime(
        store=store,
        profile_store=profile_store,
        enable_notifications=False,
    )
    # Replace listener with fast FakeListener wired to rt.engine.on_window
    rt.listener = make_listener(rt.engine.on_window)
    runtime_box["rt"] = rt

    app = create_app(runtime=rt)
    with TestClient(app) as tc:
        yield tc, rt


def test_status_and_monitor_start_stop(client) -> None:
    tc, _rt = client
    r = tc.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "monitoring",
        "mic_ok",
        "uptime_s",
        "events_today",
        "cpu_percent",
        "ram_mb",
    }
    assert body["monitoring"] is False
    assert body["uptime_s"] == 0

    assert tc.post("/api/monitor/start").json() == {"monitoring": True}
    time.sleep(0.05)
    assert tc.get("/api/status").json()["monitoring"] is True
    assert tc.post("/api/monitor/stop").json() == {"monitoring": False}


def test_settings_get_put_and_validation(client) -> None:
    tc, rt = client
    settings = tc.get("/api/settings").json()
    assert set(settings["categories"].keys()) == set(CATEGORIES)

    settings["custom_method"] = "mfcc"
    settings["tv_guard"] = False
    res = tc.put("/api/settings", json=settings)
    assert res.status_code == 200
    assert res.json()["custom_method"] == "mfcc"
    assert rt.profile_store.method == "mfcc"  # type: ignore[attr-defined]

    # Missing fields -> 422
    bad = dict(settings)
    bad.pop("tv_guard")
    assert tc.put("/api/settings", json=bad).status_code == 422

    # Missing category -> 422
    bad_cats = tc.get("/api/settings").json()
    bad_cats["categories"].pop(DOORBELL)
    assert tc.put("/api/settings", json=bad_cats).status_code == 422


def test_history_ack_feedback_and_stats(client) -> None:
    tc, rt = client
    rt.store.add_event(
        event_id="a1",
        timestamp_epoch=time.time(),
        category=PRESSURE_COOKER,
        label="Whistle 1 of 3",
        confidence=0.91,
        priority="high",
        alerted=True,
        custom_name="My Kitchen Cooker",
        room="Laptop",
        latency_ms=55.0,
    )
    # Seed embedding in engine cache to verify feedback forwarding
    rt.engine._remember_embedding("a1", [0.25] * 1024, PRESSURE_COOKER, None)

    hist = tc.get("/api/history", params={"category": PRESSURE_COOKER, "q": "cooker"}).json()
    assert len(hist) == 1
    assert hist[0]["id"] == "a1"

    assert tc.post("/api/alerts/a1/ack").json() == {"ok": True}
    assert tc.post("/api/alerts/a1/feedback", json={"correct": False}).json() == {"ok": True}
    assert len(rt.profile_store.feedback) == 1  # type: ignore[attr-defined]
    assert rt.profile_store.feedback[0].positive is False  # type: ignore[attr-defined]

    assert tc.post("/api/alerts/unknown/ack").status_code == 404
    assert tc.post("/api/alerts/unknown/feedback", json={"correct": True}).status_code == 404

    stats = tc.get("/api/stats").json()
    assert stats["per_category"][PRESSURE_COOKER] == 1
    assert len(stats["per_day"]) == 30
    assert stats["false_alert_rate"] == 1.0
    assert stats["avg_latency_ms"] == 55.0

    assert tc.delete("/api/history").json() == {"ok": True}
    assert tc.get("/api/history").json() == []


def test_custom_sounds_crud_and_errors(client) -> None:
    tc, _rt = client
    assert tc.get("/api/custom-sounds").json() == []

    bad = tc.post("/api/custom-sounds", json={"name": "Bad", "category": "not_a_category"})
    assert bad.status_code == 422

    created = tc.post(
        "/api/custom-sounds",
        json={"name": "My Kitchen Cooker", "category": PRESSURE_COOKER},
    )
    assert created.status_code == 200
    pid = created.json()["id"]

    rec = tc.post(f"/api/custom-sounds/{pid}/record")
    assert rec.status_code == 200
    assert rec.json() == {"examples": 1}

    tested = tc.post(f"/api/custom-sounds/{pid}/test")
    assert tested.status_code == 200
    assert "similarity" in tested.json() and "matched" in tested.json()

    items = tc.get("/api/custom-sounds").json()
    assert items == [
        {"id": pid, "name": "My Kitchen Cooker", "category": PRESSURE_COOKER, "examples": 1}
    ]

    assert tc.delete(f"/api/custom-sounds/{pid}").json() == {"ok": True}
    assert tc.post(f"/api/custom-sounds/{pid}/record").status_code == 404
    assert tc.post(f"/api/custom-sounds/{pid}/test").status_code == 404
    assert tc.delete(f"/api/custom-sounds/{pid}").status_code == 404


def test_whistle_counter_endpoints(client) -> None:
    tc, _rt = client
    assert tc.get("/api/whistle-counter").json() == {"enabled": True, "target": 3, "count": 0}
    assert tc.put("/api/whistle-counter", json={"enabled": True, "target": 4}).json() == {
        "enabled": True,
        "target": 4,
        "count": 0,
    }
    assert tc.put("/api/whistle-counter", json={"enabled": True, "target": 0}).status_code == 422
    assert tc.post("/api/whistle-counter/reset").json()["count"] == 0


def test_ws_live_receives_status_level_and_alert(client) -> None:
    tc, _rt = client
    with tc.websocket_connect("/ws/live") as ws:
        initial = ws.receive_json()
        assert initial == {"type": "status", "monitoring": False}

        tc.post("/api/monitor/start")
        seen_types: set[str] = set()
        for _ in range(40):
            msg = ws.receive_json()
            seen_types.add(msg["type"])
            if {"status", "level", "alert"}.issubset(seen_types):
                break
        assert {"status", "level", "alert"}.issubset(seen_types)
        tc.post("/api/monitor/stop")


def test_ws_listener_phase5(client) -> None:
    tc, rt = client
    # Missing room -> closes with 1008
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with tc.websocket_connect("/ws/listener") as ws:
            ws.receive_text()
    assert exc_info.value.code == 1008

    with tc.websocket_connect("/ws/listener?room=Kitchen") as ws:
        ack = ws.receive_json()
        assert ack == {"type": "ack", "room": "Kitchen"}
        pcm = struct.pack("<4f", 0.1, -0.1, 0.2, -0.2)
        ws.send_bytes(pcm)
        time.sleep(0.05)

    # Verify level event reached history/engine without errors; now test room-filtered alert routing
    with tc.websocket_connect("/ws/listener?room=Kitchen") as ws:
        _ = ws.receive_json()  # ack
        # Trigger a room alert directly through runtime helper
        import asyncio

        rt._on_alert_from_thread(
            {
                "type": "alert",
                "id": "k1",
                "category": DOORBELL,
                "label": "Doorbell",
                "confidence": 0.85,
                "priority": "medium",
                "timestamp": "2026-11-12T19:04:05+05:30",
                "room": "Kitchen",
                "custom_name": None,
                "escalated": False,
            },
            {},
        )
        msg = ws.receive_json()
        assert msg["type"] == "alert"
        assert msg["room"] == "Kitchen"


def test_validation_errors_and_mic_failure(client) -> None:
    tc, rt = client

    # Invalid history filters -> 422
    assert tc.get("/api/history", params={"category": "bad_cat"}).status_code == 422
    assert tc.get("/api/history", params={"from": "not-a-timestamp"}).status_code == 422

    # Invalid settings payloads -> 422
    s = tc.get("/api/settings").json()
    bad_time = dict(s)
    bad_time["quiet_hours"] = {"enabled": True, "start": "25:99", "end": "07:00"}
    assert tc.put("/api/settings", json=bad_time).status_code == 422

    bad_style = dict(s)
    bad_style["alert_style"] = "popup"
    assert tc.put("/api/settings", json=bad_style).status_code == 422

    # Listener start failure -> 500 and mic_ok becomes False
    class BrokenListener:
        def start(self) -> None:
            raise RuntimeError("Microphone not available")

        def stop(self) -> None:
            pass

        def is_running(self) -> bool:
            return False

    rt.listener = BrokenListener()
    res = tc.post("/api/monitor/start")
    assert res.status_code == 500
    status = tc.get("/api/status").json()
    assert status["mic_ok"] is False
    assert status["monitoring"] is False
