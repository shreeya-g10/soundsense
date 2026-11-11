"""Backend runtime coordinator tying together listener, engine, storage, alerts, and WebSockets.

Owner: Vaidehi (Role 2).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from array import array
from typing import Any, Callable

from fastapi import WebSocket

from contracts.types import AudioListener, ProfileStore, RemoteAudioInput, WindowResult

from backend.api.metrics import ProcessMetrics
from backend.custom.fake_profile_store import FakeProfileStore
from backend.engine.detector import DetectionEngine, EngineConfig
from backend.engine.fake_listener import FakeListener
from backend.notify.notifier import AlertDispatcher, DesktopNotifierWrapper, ScreenFlasher
from backend.storage.store import SQLiteStore

log = logging.getLogger(__name__)


class FakeRemoteAudioInput:
    """Lightweight stand-in for Phase 5 RemoteAudioInput until Role 1's is wired."""

    def __init__(self, on_window: Callable[[WindowResult], None]) -> None:
        self._on_window = on_window

    def push(self, samples: bytes, room: str) -> None:
        if len(samples) % 4 != 0:
            raise ValueError("samples byte length must be a multiple of 4 (Float32 PCM)")
        if not samples:
            rms = 0.0
        else:
            floats = array("f")
            floats.frombytes(samples)
            sq_sum = sum(x * x for x in floats)
            rms = min(1.0, math.sqrt(sq_sum / len(floats)))

        self._on_window(
            WindowResult(
                timestamp=time.time(),
                rms=rms,
                skipped=True,
                category_scores={},
                context_scores={},
                embedding=None,
                custom_matches=[],
                inference_ms=0.0,
                room=room,
            )
        )


class BackendRuntime:
    """Holds shared backend state and lifecycle for FastAPI routes and WebSockets."""

    def __init__(
        self,
        *,
        store: SQLiteStore | None = None,
        profile_store: ProfileStore | None = None,
        listener: AudioListener | None = None,
        remote_audio: RemoteAudioInput | None = None,
        dispatcher: AlertDispatcher | None = None,
        engine_config: EngineConfig | None = None,
        enable_notifications: bool = True,
    ) -> None:
        self.store = store or SQLiteStore(os.getenv("SOUNDSENSE_DB_PATH", "soundsense.db"))
        self.profile_store: ProfileStore = profile_store or FakeProfileStore()
        self.metrics = ProcessMetrics()

        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            self.dispatcher = AlertDispatcher(
                flasher=ScreenFlasher(enabled=enable_notifications),
                desktop=DesktopNotifierWrapper(enabled=enable_notifications),
            )

        self.engine = DetectionEngine(
            store=self.store,
            on_level=self._on_level_from_thread,
            on_alert=self._on_alert_from_thread,
            config=engine_config,
        )

        if listener is not None:
            self.listener: AudioListener = listener
        else:
            scenario = os.getenv("SOUNDSENSE_FAKE_SCENARIO", "whistles")
            self.listener = FakeListener(on_window=self.engine.on_window, scenario=scenario)

        self.remote_audio: RemoteAudioInput = remote_audio or FakeRemoteAudioInput(
            on_window=self.engine.on_window
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started_at: float | None = None
        self._mic_ok: bool = True

        self._live_clients: set[WebSocket] = set()
        self._listener_clients: dict[WebSocket, str] = {}

        # Sync initial custom_method from stored settings to profile_store
        try:
            method = self.store.get_settings().get("custom_method", "embedding")
            self.profile_store.set_method(method)
        except Exception:
            log.exception("Failed syncing initial custom_method")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop
        self.engine.start()
        self.dispatcher.start()

    def shutdown(self) -> None:
        try:
            self.listener.stop()
        except Exception:
            log.exception("Error stopping listener")
        self._started_at = None
        self.engine.stop()
        self.dispatcher.stop()
        self.store.close()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ------------------------------------------------------------------
    # Monitoring state
    # ------------------------------------------------------------------

    def is_monitoring(self) -> bool:
        return self.listener.is_running()

    def uptime_s(self) -> float:
        if not self.is_monitoring() or self._started_at is None:
            return 0.0
        return round(max(0.0, time.monotonic() - self._started_at), 1)

    async def start_monitoring(self) -> dict[str, bool]:
        if not self.listener.is_running():
            try:
                self.listener.start()
            except RuntimeError as exc:
                self._mic_ok = False
                raise exc
            self._mic_ok = True
            self._started_at = time.monotonic()
            await self.broadcast_live({"type": "status", "monitoring": True})
        return {"monitoring": True}

    async def stop_monitoring(self) -> dict[str, bool]:
        if self.listener.is_running():
            self.listener.stop()
            self._started_at = None
            await self.broadcast_live({"type": "status", "monitoring": False})
        else:
            self._started_at = None
        return {"monitoring": False}

    def get_status(self) -> dict[str, Any]:
        return {
            "monitoring": self.is_monitoring(),
            "mic_ok": self._mic_ok,
            "uptime_s": self.uptime_s(),
            "events_today": self.store.events_today(),
            "cpu_percent": self.metrics.cpu_percent(),
            "ram_mb": self.metrics.ram_mb(),
        }

    def get_stats(self) -> dict[str, Any]:
        stats = self.store.get_stats()
        stats["cpu_percent"] = self.metrics.cpu_percent()
        stats["ram_mb"] = self.metrics.ram_mb()
        return stats

    # ------------------------------------------------------------------
    # WebSocket management & thread-safe callbacks
    # ------------------------------------------------------------------

    def register_live(self, ws: WebSocket) -> None:
        self._live_clients.add(ws)

    def unregister_live(self, ws: WebSocket) -> None:
        self._live_clients.discard(ws)

    def register_listener(self, ws: WebSocket, room: str) -> None:
        self._listener_clients[ws] = room

    def unregister_listener(self, ws: WebSocket) -> None:
        self._listener_clients.pop(ws, None)

    async def broadcast_live(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._live_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._live_clients.discard(ws)

    async def broadcast_room_alert(self, alert_msg: dict[str, Any]) -> None:
        room = alert_msg.get("room")
        dead: list[WebSocket] = []
        for ws, client_room in list(self._listener_clients.items()):
            if client_room == room:
                try:
                    await ws.send_json(alert_msg)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self._listener_clients.pop(ws, None)

    async def handle_alert_async(self, alert_msg: dict[str, Any]) -> None:
        await self.broadcast_live(alert_msg)
        await self.broadcast_room_alert(alert_msg)
        settings = self.store.get_settings()
        await self.dispatcher.dispatch(alert_msg, settings)

    def _schedule(self, coro: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            coro.close()
            return
        loop.call_soon_threadsafe(asyncio.create_task, coro)

    def _on_level_from_thread(self, msg: dict[str, Any]) -> None:
        self._schedule(self.broadcast_live(msg))

    def _on_alert_from_thread(self, alert_msg: dict[str, Any], _event: dict[str, Any]) -> None:
        self._schedule(self.handle_alert_async(alert_msg))
