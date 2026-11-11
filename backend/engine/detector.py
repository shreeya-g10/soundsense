"""Detection engine: turns WindowResult outputs into alert and level decisions.

Owner: Vaidehi (Role 2).
Consumes contracts.types.WindowResult without ever accessing raw audio.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from contracts.types import (
    APPLIANCE_BEEP,
    BABY_CRY,
    CATEGORIES,
    CONTEXT_MUSIC,
    CONTEXT_SPEECH,
    CONTEXT_TELEVISION,
    DOG_BARK,
    DOOR_KNOCK,
    DOORBELL,
    GLASS_BREAK,
    PRESSURE_COOKER,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    SMOKE_ALARM,
    WATER_RUNNING,
    CustomMatch,
    WindowResult,
)

from backend.storage.store import EventNotFoundError, SQLiteStore

log = logging.getLogger(__name__)

CATEGORY_LABELS: dict[str, str] = {
    SMOKE_ALARM: "Smoke alarm",
    GLASS_BREAK: "Glass breaking",
    PRESSURE_COOKER: "Pressure cooker",
    BABY_CRY: "Baby crying",
    DOOR_KNOCK: "Door knock",
    DOORBELL: "Doorbell",
    DOG_BARK: "Dog barking",
    APPLIANCE_BEEP: "Appliance beep",
    WATER_RUNNING: "Water running",
}

CONTEXT_LABELS: dict[str, str] = {
    CONTEXT_SPEECH: "Speech",
    CONTEXT_MUSIC: "Music",
    CONTEXT_TELEVISION: "Television",
}

DEFAULT_COOLDOWNS_S: dict[str, float] = {
    SMOKE_ALARM: 10.0,
    GLASS_BREAK: 10.0,
    PRESSURE_COOKER: 4.0,
    BABY_CRY: 8.0,
    DOOR_KNOCK: 6.0,
    DOORBELL: 6.0,
    DOG_BARK: 8.0,
    APPLIANCE_BEEP: 8.0,
    WATER_RUNNING: 12.0,
}


def format_category_label(category: str) -> str:
    """Return human-readable label for a category ID."""
    return CATEGORY_LABELS.get(category, category.replace("_", " ").title())


@dataclass
class EngineConfig:
    """Tunable parameters for DetectionEngine."""

    base_threshold_at_zero_sens: float = 0.80
    threshold_span: float = 0.50  # at sens=0.5 -> 0.55; at sens=1.0 -> 0.30
    custom_match_threshold: float = 0.75
    tv_guard_context_min: float = 0.50
    tv_guard_penalty: float = 0.12
    critical_bypass_quiet_hours: bool = True
    cooldowns_s: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COOLDOWNS_S))
    escalation_delays_s: dict[str, float] = field(
        default_factory=lambda: {PRIORITY_CRITICAL: 6.0, PRIORITY_HIGH: 20.0}
    )
    embedding_cache_size: int = 256
    queue_maxsize: int = 128


@dataclass
class _CategoryState:
    last_alert_ts: float = 0.0
    last_seen_ts: float = 0.0
    last_suppressed_ts: float = 0.0
    last_event_id: str | None = None
    last_escalated_ts: float = 0.0


@dataclass
class CachedEmbedding:
    """In-memory embedding reference kept briefly for /api/alerts/{id}/feedback."""

    embedding: list[float]
    category: str
    profile_id: str | None


class DetectionEngine:
    """Processes WindowResult objects into SQLite events, live levels, and alerts."""

    def __init__(
        self,
        store: SQLiteStore,
        on_level: Callable[[dict[str, Any]], None] | None = None,
        on_alert: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.store = store
        self._on_level = on_level
        self._on_alert = on_alert
        self.config = config or EngineConfig()

        self._state: dict[tuple[str, str], _CategoryState] = {}
        self._state_lock = threading.Lock()
        self._embedding_cache: OrderedDict[str, CachedEmbedding] = OrderedDict()

        self._queue: queue.Queue[WindowResult | None] = queue.Queue(
            maxsize=self.config.queue_maxsize
        )
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Worker lifecycle (optional; tests may call process_window directly)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background window-processing worker thread."""
        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._run_worker, name="DetectionEngine", daemon=True
            )
            self._worker.start()

    def stop(self) -> None:
        """Stop the background worker thread cleanly."""
        with self._state_lock:
            self._stop_event.set()
            worker = self._worker
            self._worker = None
        if worker is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            if worker is not threading.current_thread():
                worker.join(timeout=3.0)

    def on_window(self, result: WindowResult) -> None:
        """AudioListener callback: enqueues a WindowResult quickly and never raises."""
        try:
            self._queue.put_nowait(result)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(result)
            except queue.Full:
                log.warning("DetectionEngine queue full; dropping audio window")

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self.process_window(item)
            except Exception:
                log.exception("Failed processing WindowResult")

    # ------------------------------------------------------------------
    # Embedding cache for feedback
    # ------------------------------------------------------------------

    def get_cached_embedding(self, event_id: str) -> CachedEmbedding | None:
        """Return cached embedding metadata for an event if still in memory."""
        with self._state_lock:
            return self._embedding_cache.get(event_id)

    def _remember_embedding(
        self,
        event_id: str,
        embedding: list[float] | None,
        category: str,
        profile_id: str | None,
    ) -> None:
        if embedding is None:
            return
        with self._state_lock:
            self._embedding_cache[event_id] = CachedEmbedding(
                embedding=list(embedding),
                category=category,
                profile_id=profile_id,
            )
            while len(self._embedding_cache) > self.config.embedding_cache_size:
                self._embedding_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Core window processing
    # ------------------------------------------------------------------

    def process_window(self, result: WindowResult) -> dict[str, Any] | None:
        """Evaluate one WindowResult synchronously. Returns alert dict if alerted."""
        level_msg = self._build_level_message(result)
        if self._on_level is not None:
            try:
                self._on_level(level_msg)
            except Exception:
                log.exception("on_level callback raised")

        if result.skipped:
            return None

        settings = self.store.get_settings()
        candidate = self._pick_best_candidate(result, settings)
        if candidate is None:
            return None

        category, confidence, priority, custom_match = candidate
        base_threshold = self._threshold_for_sensitivity(
            settings["categories"][category]["sensitivity"]
        )
        tv_penalty = self._tv_guard_penalty(result, settings, priority)
        effective_threshold = min(0.95, base_threshold + tv_penalty)

        if confidence < base_threshold:
            return None

        now_epoch = result.timestamp
        if confidence < effective_threshold:
            # Suppressed by TV guard -> log once per cooldown with alerted=False
            return self._record_suppressed(
                result=result,
                category=category,
                confidence=confidence,
                priority=priority,
                custom_match=custom_match,
                now_epoch=now_epoch,
            )

        if self._is_quiet_hours(settings.get("quiet_hours", {}), now_epoch):
            if not (self.config.critical_bypass_quiet_hours and priority == PRIORITY_CRITICAL):
                return self._record_suppressed(
                    result=result,
                    category=category,
                    confidence=confidence,
                    priority=priority,
                    custom_match=custom_match,
                    now_epoch=now_epoch,
                )

        return self._handle_detection(
            result=result,
            category=category,
            confidence=confidence,
            priority=priority,
            custom_match=custom_match,
            now_epoch=now_epoch,
        )

    def _build_level_message(self, result: WindowResult) -> dict[str, Any]:
        if result.skipped:
            return {
                "type": "level",
                "rms": round(float(result.rms), 4),
                "top": None,
                "skipped": True,
            }

        best_name: str | None = None
        best_score = -1.0

        for cat, score in result.category_scores.items():
            if score > best_score:
                best_score = score
                best_name = format_category_label(cat)

        for ctx, score in result.context_scores.items():
            if score > best_score:
                best_score = score
                best_name = CONTEXT_LABELS.get(ctx, ctx.title())

        return {
            "type": "level",
            "rms": round(float(result.rms), 4),
            "top": best_name,
            "skipped": False,
        }

    def _threshold_for_sensitivity(self, sensitivity: float) -> float:
        clamped = min(1.0, max(0.0, float(sensitivity)))
        return self.config.base_threshold_at_zero_sens - self.config.threshold_span * clamped

    def _tv_guard_penalty(
        self,
        result: WindowResult,
        settings: dict[str, Any],
        priority: str,
    ) -> float:
        if not settings.get("tv_guard", True) or priority == PRIORITY_CRITICAL:
            return 0.0
        ctx = result.context_scores or {}
        bg_score = max(
            ctx.get(CONTEXT_TELEVISION, 0.0),
            ctx.get(CONTEXT_SPEECH, 0.0),
            ctx.get(CONTEXT_MUSIC, 0.0),
        )
        if bg_score >= self.config.tv_guard_context_min:
            return self.config.tv_guard_penalty
        return 0.0

    def _pick_best_candidate(
        self,
        result: WindowResult,
        settings: dict[str, Any],
    ) -> tuple[str, float, str, CustomMatch | None] | None:
        cat_settings = settings.get("categories", {})
        best_by_cat_match: dict[str, CustomMatch] = {}
        for match in result.custom_matches:
            if match.similarity >= self.config.custom_match_threshold:
                prev = best_by_cat_match.get(match.category)
                if prev is None or match.similarity > prev.similarity:
                    best_by_cat_match[match.category] = match

        best_choice: tuple[str, float, str, CustomMatch | None] | None = None
        for cat in CATEGORIES:
            cfg = cat_settings.get(cat)
            if not cfg or not cfg.get("enabled", True):
                continue
            base_score = float(result.category_scores.get(cat, 0.0))
            custom_match = best_by_cat_match.get(cat)
            score = max(base_score, custom_match.similarity) if custom_match else base_score
            if best_choice is None or score > best_choice[1]:
                best_choice = (cat, score, str(cfg["priority"]), custom_match)

        return best_choice

    @staticmethod
    def _is_quiet_hours(quiet_hours: dict[str, Any], epoch: float) -> bool:
        if not quiet_hours.get("enabled", False):
            return False
        start_str = str(quiet_hours.get("start", "22:00"))
        end_str = str(quiet_hours.get("end", "07:00"))
        try:
            sh, sm = (int(x) for x in start_str.split(":", 1))
            eh, em = (int(x) for x in end_str.split(":", 1))
        except ValueError:
            return False

        now_dt = datetime.fromtimestamp(epoch).astimezone()
        now_min = now_dt.hour * 60 + now_dt.minute
        start_min = sh * 60 + sm
        end_min = eh * 60 + em

        if start_min == end_min:
            return True
        if start_min < end_min:
            return start_min <= now_min < end_min
        return now_min >= start_min or now_min < end_min

    def _record_suppressed(
        self,
        *,
        result: WindowResult,
        category: str,
        confidence: float,
        priority: str,
        custom_match: CustomMatch | None,
        now_epoch: float,
    ) -> None:
        key = (category, result.room)
        cooldown = self.config.cooldowns_s.get(category, 8.0)
        with self._state_lock:
            st = self._state.setdefault(key, _CategoryState())
            if now_epoch - st.last_suppressed_ts < cooldown:
                return
            st.last_suppressed_ts = now_epoch

        event_id = uuid.uuid4().hex[:8]
        label = format_category_label(category)
        custom_name = custom_match.name if custom_match else None
        latency_ms = max(0.0, (time.time() - result.timestamp) * 1000.0)
        self.store.add_event(
            event_id=event_id,
            timestamp_epoch=now_epoch,
            category=category,
            label=label,
            confidence=confidence,
            priority=priority,
            alerted=False,
            custom_name=custom_name,
            room=result.room,
            latency_ms=latency_ms,
        )
        self._remember_embedding(
            event_id,
            result.embedding,
            category,
            custom_match.profile_id if custom_match else None,
        )
        return None

    def _handle_detection(
        self,
        *,
        result: WindowResult,
        category: str,
        confidence: float,
        priority: str,
        custom_match: CustomMatch | None,
        now_epoch: float,
    ) -> dict[str, Any] | None:
        key = (category, result.room)
        cooldown = self.config.cooldowns_s.get(category, 8.0)
        escalated = False

        with self._state_lock:
            st = self._state.setdefault(key, _CategoryState())
            since_alert = now_epoch - st.last_alert_ts
            since_seen = now_epoch - st.last_seen_ts
            st.last_seen_ts = now_epoch

            if st.last_alert_ts > 0 and since_alert < cooldown:
                return None

            # Check if an earlier unacknowledged high/critical alert should escalate
            esc_delay = self.config.escalation_delays_s.get(priority)
            new_episode = st.last_event_id is None or since_seen >= cooldown * 2.5
            if new_episode:
                st.last_escalated_ts = now_epoch
            elif esc_delay is not None and (now_epoch - st.last_escalated_ts) >= esc_delay:
                try:
                    prev = self.store.get_event(st.last_event_id)
                    if not prev["acknowledged"]:
                        escalated = True
                        st.last_escalated_ts = now_epoch
                    else:
                        st.last_escalated_ts = now_epoch
                except EventNotFoundError:
                    st.last_escalated_ts = now_epoch

            st.last_alert_ts = now_epoch

        label = format_category_label(category)
        if category == PRESSURE_COOKER:
            whistle = self.store.get_whistle_counter()
            if whistle["enabled"]:
                whistle = self.store.increment_whistle_counter()
                label = f"Whistle {whistle['count']} of {whistle['target']}"
                if whistle["count"] >= whistle["target"]:
                    priority = PRIORITY_CRITICAL

        event_id = uuid.uuid4().hex[:8]
        custom_name = custom_match.name if custom_match else None
        latency_ms = max(0.0, (time.time() - result.timestamp) * 1000.0)

        event = self.store.add_event(
            event_id=event_id,
            timestamp_epoch=now_epoch,
            category=category,
            label=label,
            confidence=confidence,
            priority=priority,
            alerted=True,
            custom_name=custom_name,
            room=result.room,
            latency_ms=latency_ms,
        )
        with self._state_lock:
            self._state[key].last_event_id = event_id

        self._remember_embedding(
            event_id,
            result.embedding,
            category,
            custom_match.profile_id if custom_match else None,
        )

        alert_msg = {
            "type": "alert",
            "id": event["id"],
            "category": event["category"],
            "label": event["label"],
            "confidence": event["confidence"],
            "priority": event["priority"],
            "timestamp": event["timestamp"],
            "room": event["room"],
            "custom_name": event["custom_name"],
            "escalated": escalated,
        }

        if self._on_alert is not None:
            try:
                self._on_alert(alert_msg, event)
            except Exception:
                log.exception("on_alert callback raised")

        return alert_msg
