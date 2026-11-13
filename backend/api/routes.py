"""FastAPI REST and WebSocket routes matching contracts/API.md.

Owner: Vaidehi (Role 2).
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from contracts.types import CATEGORIES

from backend.api.runtime import BackendRuntime
from backend.api.schemas import (
    CreateCustomSoundRequest,
    FeedbackRequest,
    SettingsModel,
    WhistleCounterUpdateRequest,
)
from backend.storage.store import EventNotFoundError, parse_iso_to_epoch

router = APIRouter()


def _get_runtime(request: Request) -> BackendRuntime:
    return request.app.state.runtime


# ---------------------------------------------------------------------------
# Monitoring & Status
# ---------------------------------------------------------------------------


@router.post("/api/monitor/start")
async def start_monitor(request: Request) -> dict[str, bool]:
    runtime = _get_runtime(request)
    runtime.attach_loop(asyncio.get_running_loop())
    try:
        return await runtime.start_monitoring()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/monitor/stop")
async def stop_monitor(request: Request) -> dict[str, bool]:
    runtime = _get_runtime(request)
    return await runtime.stop_monitoring()


@router.get("/api/status")
async def get_status(request: Request) -> dict[str, Any]:
    return _get_runtime(request).get_status()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/api/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return _get_runtime(request).store.get_settings()


@router.put("/api/settings")
async def put_settings(body: SettingsModel, request: Request) -> dict[str, Any]:
    runtime = _get_runtime(request)
    saved = runtime.store.save_settings(body.model_dump())
    runtime.profile_store.set_method(saved["custom_method"])
    return saved


# ---------------------------------------------------------------------------
# History & Alerts
# ---------------------------------------------------------------------------


@router.get("/api/history")
async def get_history(
    request: Request,
    category: str | None = Query(None),
    from_ts: str | None = Query(None, alias="from"),
    to_ts: str | None = Query(None, alias="to"),
    q: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    if category is not None and category != "" and category not in CATEGORIES:
        raise HTTPException(status_code=422, detail=f"Unknown category {category!r}")
    for label, val in (("from", from_ts), ("to", to_ts)):
        if val:
            try:
                parse_iso_to_epoch(val)
            except ValueError as exc:
                raise HTTPException(
                    status_code=422, detail=f"Invalid ISO 8601 timestamp for {label}"
                ) from exc

    return _get_runtime(request).store.get_history(
        category=category or None,
        from_ts=from_ts or None,
        to_ts=to_ts or None,
        q=q or None,
        limit=limit,
    )


@router.delete("/api/history")
async def delete_history(request: Request) -> dict[str, bool]:
    _get_runtime(request).store.clear_history()
    return {"ok": True}


@router.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str, request: Request) -> dict[str, bool]:
    try:
        _get_runtime(request).store.acknowledge_alert(alert_id)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found") from exc
    return {"ok": True}


@router.post("/api/alerts/{alert_id}/feedback")
async def alert_feedback(
    alert_id: str, body: FeedbackRequest, request: Request
) -> dict[str, bool]:
    runtime = _get_runtime(request)
    try:
        event = runtime.store.set_feedback(alert_id, body.correct)
    except EventNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found") from exc

    cached = runtime.engine.get_cached_embedding(alert_id)
    if cached is not None:
        await asyncio.to_thread(
            runtime.profile_store.add_feedback,
            cached.embedding,
            cached.category,
            cached.profile_id,
            body.correct,
        )
    else:
        _ = event  # event recorded in SQLite; embedding already expired from RAM cache
    return {"ok": True}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/api/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    return _get_runtime(request).get_stats()


# ---------------------------------------------------------------------------
# Custom sounds
# ---------------------------------------------------------------------------


@router.get("/api/custom-sounds")
async def list_custom_sounds(request: Request) -> list[dict[str, Any]]:
    return _get_runtime(request).profile_store.list()


@router.post("/api/custom-sounds")
async def create_custom_sound(
    body: CreateCustomSoundRequest, request: Request
) -> dict[str, str]:
    runtime = _get_runtime(request)
    try:
        profile_id = runtime.profile_store.create(body.name, body.category)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": profile_id}


@router.post("/api/custom-sounds/{profile_id}/record")
async def record_custom_sound(profile_id: str, request: Request) -> dict[str, int]:
    runtime = _get_runtime(request)
    try:
        examples = await asyncio.to_thread(
            runtime.profile_store.add_example_from_mic, profile_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Custom sound {profile_id!r} not found"
        ) from exc
    return {"examples": examples}


@router.post("/api/custom-sounds/{profile_id}/test")
async def test_custom_sound(profile_id: str, request: Request) -> dict[str, Any]:
    runtime = _get_runtime(request)
    try:
        similarity = await asyncio.to_thread(runtime.profile_store.test_from_mic, profile_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Custom sound {profile_id!r} not found"
        ) from exc
    threshold = runtime.engine.config.custom_match_threshold
    return {"similarity": round(float(similarity), 3), "matched": similarity >= threshold}


@router.delete("/api/custom-sounds/{profile_id}")
async def delete_custom_sound(profile_id: str, request: Request) -> dict[str, bool]:
    runtime = _get_runtime(request)
    try:
        runtime.profile_store.delete(profile_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Custom sound {profile_id!r} not found"
        ) from exc
    return {"ok": True}


# ---------------------------------------------------------------------------
# Whistle counter
# ---------------------------------------------------------------------------


@router.get("/api/whistle-counter")
async def get_whistle_counter(request: Request) -> dict[str, Any]:
    return _get_runtime(request).store.get_whistle_counter()


@router.put("/api/whistle-counter")
async def update_whistle_counter(
    body: WhistleCounterUpdateRequest, request: Request
) -> dict[str, Any]:
    try:
        return _get_runtime(request).store.update_whistle_counter(body.enabled, body.target)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/whistle-counter/reset")
async def reset_whistle_counter(request: Request) -> dict[str, Any]:
    return _get_runtime(request).store.reset_whistle_counter()


# ---------------------------------------------------------------------------
# WebSockets
# ---------------------------------------------------------------------------


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    runtime: BackendRuntime = websocket.app.state.runtime
    runtime.attach_loop(asyncio.get_running_loop())
    await websocket.accept()
    runtime.register_live(websocket)
    try:
        await websocket.send_json({"type": "status", "monitoring": runtime.is_monitoring()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unregister_live(websocket)


@router.websocket("/ws/listener")
async def ws_listener(websocket: WebSocket, room: str | None = Query(None)) -> None:
    clean_room = (room or "").strip()
    if not clean_room:
        await websocket.accept()
        await websocket.close(code=1008, reason="room query parameter is required")
        return

    runtime: BackendRuntime = websocket.app.state.runtime
    runtime.attach_loop(asyncio.get_running_loop())
    await websocket.accept()
    runtime.register_listener(websocket, clean_room)
    try:
        await websocket.send_json({"type": "ack", "room": clean_room})
        while True:
            data = await websocket.receive_bytes()
            try:
                runtime.remote_audio.push(data, clean_room)
            except ValueError:
                # Ignore malformed chunk sizes rather than crashing the connection
                pass
    except WebSocketDisconnect:
        pass
    finally:
        runtime.unregister_listener(websocket)
