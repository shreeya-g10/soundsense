"""SQLite persistence for settings, event history, and whistle counter.

Owner: Vaidehi (Role 2).
No raw audio or embeddings are ever stored here.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from contracts.types import (
    BABY_CRY,
    CATEGORIES,
    DOOR_KNOCK,
    DOORBELL,
    DOG_BARK,
    GLASS_BREAK,
    METHOD_EMBEDDING,
    PRESSURE_COOKER,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    SMOKE_ALARM,
    WATER_RUNNING,
    APPLIANCE_BEEP,
)

_DEFAULT_PRIORITIES: dict[str, str] = {
    SMOKE_ALARM: PRIORITY_CRITICAL,
    GLASS_BREAK: PRIORITY_CRITICAL,
    PRESSURE_COOKER: PRIORITY_HIGH,
    BABY_CRY: PRIORITY_HIGH,
    DOOR_KNOCK: PRIORITY_MEDIUM,
    DOORBELL: PRIORITY_MEDIUM,
    DOG_BARK: PRIORITY_MEDIUM,
    APPLIANCE_BEEP: PRIORITY_LOW,
    WATER_RUNNING: PRIORITY_LOW,
}


class EventNotFoundError(KeyError):
    """Raised when an alert/event ID does not exist in storage."""


def default_settings() -> dict[str, Any]:
    """Return a fresh copy of the default settings defined in contracts/API.md."""
    return {
        "categories": {
            cat: {
                "enabled": True,
                "priority": _DEFAULT_PRIORITIES[cat],
                "sensitivity": 0.5,
            }
            for cat in CATEGORIES
        },
        "quiet_hours": {"enabled": False, "start": "22:00", "end": "07:00"},
        "alert_style": "full",
        "tv_guard": True,
        "cpu_saver": True,
        "custom_method": METHOD_EMBEDDING,
        "telegram": {"enabled": False, "bot_token": "", "chat_id": ""},
    }


def _local_tz():
    return datetime.now().astimezone().tzinfo


def format_iso_local(epoch: float | None = None) -> str:
    """Format Unix epoch seconds as an ISO 8601 string with local tz offset."""
    ts = time.time() if epoch is None else epoch
    dt = datetime.fromtimestamp(ts, tz=_local_tz())
    return dt.isoformat(timespec="seconds")


def parse_iso_to_epoch(value: str, *, end_of_day: bool = False) -> float:
    """Parse an ISO 8601 string into Unix epoch seconds."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if end_of_day and len(normalized) == 10 and "T" not in normalized and " " not in normalized:
        dt = dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_local_tz())
    return dt.timestamp()


def _local_date_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=_local_tz()).strftime("%Y-%m-%d")


class SQLiteStore:
    """Thread-safe SQLite store for SoundSense configuration and history."""

    def __init__(self, db_path: str | Path = "soundsense.db") -> None:
        """Open (or create) the SQLite database and initialize schema."""
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    timestamp_epoch REAL NOT NULL,
                    local_date TEXT NOT NULL,
                    category TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    priority TEXT NOT NULL,
                    alerted INTEGER NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    feedback INTEGER,
                    custom_name TEXT,
                    room TEXT NOT NULL,
                    latency_ms REAL
                );

                CREATE INDEX IF NOT EXISTS idx_events_epoch
                    ON events (timestamp_epoch DESC);
                CREATE INDEX IF NOT EXISTS idx_events_category
                    ON events (category);
                CREATE INDEX IF NOT EXISTS idx_events_date
                    ON events (local_date);

                CREATE TABLE IF NOT EXISTS whistle_counter (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL,
                    target INTEGER NOT NULL,
                    count INTEGER NOT NULL
                );
                """
            )
            cur = self._conn.execute("SELECT 1 FROM settings WHERE id = 1")
            if cur.fetchone() is None:
                self._conn.execute(
                    "INSERT INTO settings (id, data) VALUES (1, ?)",
                    (json.dumps(default_settings()),),
                )

            cur = self._conn.execute("SELECT 1 FROM whistle_counter WHERE id = 1")
            if cur.fetchone() is None:
                self._conn.execute(
                    "INSERT INTO whistle_counter (id, enabled, target, count) VALUES (1, 1, 3, 0)"
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_settings(self) -> dict[str, Any]:
        """Return the full settings object."""
        with self._lock:
            row = self._conn.execute("SELECT data FROM settings WHERE id = 1").fetchone()
            assert row is not None
            return json.loads(row["data"])

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Replace and persist the full settings object."""
        payload = copy.deepcopy(settings)
        with self._lock:
            self._conn.execute(
                "UPDATE settings SET data = ? WHERE id = 1",
                (json.dumps(payload),),
            )
            self._conn.commit()
        return payload

    # ------------------------------------------------------------------
    # Events & History
    # ------------------------------------------------------------------

    def add_event(
        self,
        *,
        event_id: str,
        timestamp_epoch: float,
        category: str,
        label: str,
        confidence: float,
        priority: str,
        alerted: bool,
        custom_name: str | None = None,
        room: str = "Laptop",
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        """Persist a new detection event and return its API-shaped dict."""
        iso_ts = format_iso_local(timestamp_epoch)
        local_date = _local_date_from_epoch(timestamp_epoch)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events (
                    id, timestamp, timestamp_epoch, local_date, category, label,
                    confidence, priority, alerted, acknowledged, feedback,
                    custom_name, room, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?)
                """,
                (
                    event_id,
                    iso_ts,
                    float(timestamp_epoch),
                    local_date,
                    category,
                    label,
                    float(confidence),
                    priority,
                    1 if alerted else 0,
                    custom_name,
                    room,
                    None if latency_ms is None else float(latency_ms),
                ),
            )
            self._conn.commit()
        return {
            "id": event_id,
            "timestamp": iso_ts,
            "category": category,
            "label": label,
            "confidence": round(float(confidence), 4),
            "priority": priority,
            "alerted": bool(alerted),
            "acknowledged": False,
            "feedback": None,
            "custom_name": custom_name,
            "room": room,
        }

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        fb = row["feedback"]
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "category": row["category"],
            "label": row["label"],
            "confidence": row["confidence"],
            "priority": row["priority"],
            "alerted": bool(row["alerted"]),
            "acknowledged": bool(row["acknowledged"]),
            "feedback": None if fb is None else bool(fb),
            "custom_name": row["custom_name"],
            "room": row["room"],
        }

    def get_event(self, event_id: str) -> dict[str, Any]:
        """Fetch a single event by ID or raise EventNotFoundError."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise EventNotFoundError(event_id)
        return self._row_to_event(row)

    def get_history(
        self,
        *,
        category: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        q: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events newest-first with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if from_ts:
            clauses.append("timestamp_epoch >= ?")
            params.append(parse_iso_to_epoch(from_ts))
        if to_ts:
            clauses.append("timestamp_epoch <= ?")
            params.append(parse_iso_to_epoch(to_ts, end_of_day=True))
        if q:
            clauses.append(
                "(INSTR(LOWER(label), LOWER(?)) > 0 "
                "OR INSTR(LOWER(COALESCE(custom_name, '')), LOWER(?)) > 0)"
            )
            params.extend([q, q])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT * FROM events
            {where}
            ORDER BY timestamp_epoch DESC, rowid DESC
            LIMIT ?
        """
        params.append(max(0, int(limit)))

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def clear_history(self) -> None:
        """Delete all recorded events."""
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()

    def acknowledge_alert(self, event_id: str) -> None:
        """Mark an alert event as acknowledged."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET acknowledged = 1 WHERE id = ?", (event_id,)
            )
            self._conn.commit()
            if cur.rowcount == 0:
                raise EventNotFoundError(event_id)

    def set_feedback(self, event_id: str, correct: bool) -> dict[str, Any]:
        """Record user feedback (True/False) on an event and return the event."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE events SET feedback = ? WHERE id = ?",
                (1 if correct else 0, event_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                raise EventNotFoundError(event_id)
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            assert row is not None
            return self._row_to_event(row)

    def events_today(self) -> int:
        """Count events recorded since local midnight."""
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM events WHERE local_date = ?", (today,)
            ).fetchone()
            assert row is not None
            return int(row["cnt"])

    def get_stats(self) -> dict[str, Any]:
        """Compute aggregate stats required by GET /api/stats (excluding cpu/ram)."""
        with self._lock:
            cat_rows = self._conn.execute(
                "SELECT category, COUNT(*) AS cnt FROM events GROUP BY category"
            ).fetchall()
            per_category = {cat: 0 for cat in CATEGORIES}
            for r in cat_rows:
                if r["category"] in per_category:
                    per_category[r["category"]] = int(r["cnt"])

            day_rows = self._conn.execute(
                "SELECT local_date, COUNT(*) AS cnt FROM events GROUP BY local_date"
            ).fetchall()
            day_map = {r["local_date"]: int(r["cnt"]) for r in day_rows}

            today = datetime.now().astimezone().date()
            per_day: list[dict[str, Any]] = []
            for offset in range(29, -1, -1):
                day_str = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
                per_day.append({"date": day_str, "count": day_map.get(day_str, 0)})

            fb_row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS rated,
                    SUM(CASE WHEN feedback = 0 THEN 1 ELSE 0 END) AS wrong
                FROM events
                WHERE feedback IS NOT NULL
                """
            ).fetchone()
            assert fb_row is not None
            rated = int(fb_row["rated"] or 0)
            wrong = int(fb_row["wrong"] or 0)
            false_alert_rate = round(wrong / rated, 3) if rated > 0 else 0.0

            lat_row = self._conn.execute(
                "SELECT AVG(latency_ms) AS avg_lat FROM events WHERE latency_ms IS NOT NULL"
            ).fetchone()
            assert lat_row is not None
            avg_latency_ms = round(float(lat_row["avg_lat"]), 1) if lat_row["avg_lat"] is not None else 0.0

        return {
            "per_category": per_category,
            "per_day": per_day,
            "false_alert_rate": false_alert_rate,
            "avg_latency_ms": avg_latency_ms,
        }

    # ------------------------------------------------------------------
    # Whistle counter
    # ------------------------------------------------------------------

    def get_whistle_counter(self) -> dict[str, Any]:
        """Return current whistle counter configuration and count."""
        with self._lock:
            row = self._conn.execute(
                "SELECT enabled, target, count FROM whistle_counter WHERE id = 1"
            ).fetchone()
            assert row is not None
            return {
                "enabled": bool(row["enabled"]),
                "target": int(row["target"]),
                "count": int(row["count"]),
            }

    def update_whistle_counter(self, enabled: bool, target: int) -> dict[str, Any]:
        """Update whistle counter enabled flag and target (>= 1)."""
        if not isinstance(target, int) or isinstance(target, bool) or target < 1:
            raise ValueError("target must be an integer >= 1")
        with self._lock:
            self._conn.execute(
                "UPDATE whistle_counter SET enabled = ?, target = ? WHERE id = 1",
                (1 if enabled else 0, target),
            )
            self._conn.commit()
            return self.get_whistle_counter()

    def reset_whistle_counter(self) -> dict[str, Any]:
        """Reset whistle count to 0."""
        with self._lock:
            self._conn.execute("UPDATE whistle_counter SET count = 0 WHERE id = 1")
            self._conn.commit()
            return self.get_whistle_counter()

    def increment_whistle_counter(self) -> dict[str, Any]:
        """Increment whistle count by 1 and return updated state."""
        with self._lock:
            self._conn.execute("UPDATE whistle_counter SET count = count + 1 WHERE id = 1")
            self._conn.commit()
            return self.get_whistle_counter()
