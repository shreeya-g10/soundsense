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

    def close(self) -> None:
        """No-op close to match RemoteAudioInput lifecycle."""
        pass


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
        use_fakes: bool | None = None,
    ) -> None:
        self.store = store or SQLiteStore(os.getenv("SOUNDSENSE_DB_PATH", "soundsense.db"))
        saved_settings = self.store.get_settings()
        cpu_saver = bool(saved_settings.get("cpu_saver", True))

        if use_fakes is None:
            use_fakes = os.getenv("SOUNDSENSE_USE_FAKES", "false").lower() in ("1", "true", "yes")

        if profile_store is not None:
            self.profile_store: ProfileStore = profile_store
        elif use_fakes:
            self.profile_store = FakeProfileStore()
        else:
            try:
                from backend.custom.profile_store import YamnetProfileStore
                self.profile_store = YamnetProfileStore()
            except Exception as exc:
                log.warning("Could not initialize YamnetProfileStore (%s); falling back to FakeProfileStore", exc)
                self.profile_store = FakeProfileStore()

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
            profile_store=self.profile_store,
            on_level=self._on_level_from_thread,
            on_alert=self._on_alert_from_thread,
            config=engine_config,
        )

        if listener is not None:
            self.listener: AudioListener = listener
        elif use_fakes:
            scenario = os.getenv("SOUNDSENSE_FAKE_SCENARIO", "whistles")
            loop = os.getenv("SOUNDSENSE_FAKE_LOOP", "false").lower() in ("1", "true", "yes")
            self.listener = FakeListener(
                on_window=self.engine.on_window,
                scenario=scenario,
                loop=loop,
            )
        else:
            try:
                from backend.audio.listener import YamnetListener
                self.listener = YamnetListener(
                    self.engine.on_window,
                    profile_store=self.profile_store,
                    cpu_saver=cpu_saver,
                )
            except Exception as exc:
                log.warning("Could not initialize YamnetListener (%s); falling back to FakeListener", exc)
                self.listener = FakeListener(on_window=self.engine.on_window, scenario="whistles")

        if remote_audio is not None:
            self.remote_audio: RemoteAudioInput = remote_audio
        elif use_fakes:
            self.remote_audio = FakeRemoteAudioInput(on_window=self.engine.on_window)
        else:
            try:
                from backend.audio.remote_input import YamnetRemoteInput
                self.remote_audio = YamnetRemoteInput(
                    self.engine.on_window,
                    profile_store=self.profile_store,
                    cpu_saver=cpu_saver,
                )
            except Exception as exc:
                log.warning("Could not initialize YamnetRemoteInput (%s); falling back to FakeRemoteAudioInput", exc)
                self.remote_audio = FakeRemoteAudioInput(on_window=self.engine.on_window)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._started_at: float | None = None
        self._mic_ok: bool = True

        self._live_clients: set[WebSocket] = set()
        self._listener_clients: dict[WebSocket, str] = {}

        self.sync_settings(saved_settings)

    # ------------------------------------------------------------------
    # Settings synchronization
    # ------------------------------------------------------------------

    def sync_settings(self, settings: dict[str, Any]) -> None:
        """Sync dynamic settings to listener, profile store, and remote audio."""
        method = settings.get("custom_method", "embedding")
        try:
            self.profile_store.set_method(method)
        except Exception:
            log.exception("Failed syncing custom_method to profile store")

        cpu_saver = bool(settings.get("cpu_saver", True))
        if hasattr(self.listener, "gate") and hasattr(self.listener.gate, "enabled"):
            self.listener.gate.enabled = cpu_saver
        if hasattr(self.remote_audio, "cpu_saver"):
            self.remote_audio.cpu_saver = cpu_saver
            for r in getattr(self.remote_audio, "_rooms", {}).values():
                if hasattr(r, "gate"):
                    r.gate.enabled = cpu_saver

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
        if hasattr(self.remote_audio, "close"):
            try:
                self.remote_audio.close()
            except Exception:
                log.exception("Error closing remote audio input")
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
