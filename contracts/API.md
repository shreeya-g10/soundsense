# SoundSense API contract

**FROZEN.** See `AGENTS.md` before changing anything here.

- **Implemented by:** Vaidehi (Role 2), in `backend/api/`
- **Used by:** Sneha (Role 3), in `frontend/` and `listener/`
- **Phase 5 audio input:** `/ws/listener` frames are passed to `RemoteAudioInput.push()`, implemented by Shreeya (Role 1)

## Conventions

- **Base URL (dev):** `http://localhost:8000`. All REST routes are under the `/api` prefix.
- **Format:** all request and response bodies are JSON (`Content-Type: application/json`).
- **Timestamps:** ISO 8601 strings with timezone offset, e.g. `"2026-11-12T19:04:05+05:30"`.
- **IDs:** strings.
- **Category IDs and priorities:** exactly the strings in `contracts/types.py` (`smoke_alarm`, …, `critical`, …).
- **Simple actions:** endpoints whose response isn't listed below return `{"ok": true}`.
- **Errors:** non-2xx status with FastAPI's default body `{"detail": "<message>"}`.
  - `404` for an unknown alert or custom-sound `id`
  - `422` for an invalid request body or query parameter

---

## REST endpoints

### Monitoring

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/monitor/start` | none | `{"monitoring": true}` |
| POST | `/api/monitor/stop` | none | `{"monitoring": false}` |

### Status

`GET /api/status`

```json
{
  "monitoring": true,
  "mic_ok": true,
  "uptime_s": 3725.4,
  "events_today": 12,
  "cpu_percent": 8.5,
  "ram_mb": 412.0
}
```

| Field | Type | Meaning |
|---|---|---|
| `monitoring` | bool | Listener is running |
| `mic_ok` | bool | Microphone opened and is delivering audio |
| `uptime_s` | number | Seconds since monitoring started; `0` when not monitoring |
| `events_today` | int | Events recorded since local midnight |
| `cpu_percent` | number | Backend process CPU usage |
| `ram_mb` | number | Backend process memory, MB |

### Settings

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/settings` | none | Full settings object |
| PUT | `/api/settings` | Full settings object (replaces it) | The saved settings object |

PUT always sends the **whole** object (no partial updates). See [Settings schema](#settings-schema).

### History

`GET /api/history?category=&from=&to=&q=&limit=`

All query parameters are optional:

| Param | Type | Meaning |
|---|---|---|
| `category` | category ID | Only events of this category |
| `from` | ISO 8601 | Events at or after this time |
| `to` | ISO 8601 | Events at or before this time |
| `q` | string | Case-insensitive text search over `label` and `custom_name` |
| `limit` | int | Max events returned, default `100` |

Response: a JSON array of events, newest first.

```json
[
  {
    "id": "a1b2c3",
    "timestamp": "2026-11-12T19:04:05+05:30",
    "category": "pressure_cooker",
    "label": "Whistle 2 of 3",
    "confidence": 0.87,
    "priority": "high",
    "alerted": true,
    "acknowledged": false,
    "feedback": null,
    "custom_name": "My Kitchen Cooker",
    "room": "Laptop"
  }
]
```

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Event ID (same as the alert `id` on the WebSocket) |
| `timestamp` | ISO 8601 | When the sound was detected |
| `category` | category ID | |
| `label` | string | Human-readable text shown to the user |
| `confidence` | number | 0.0 to 1.0 |
| `priority` | priority | Priority at the time of the event |
| `alerted` | bool | `true` if the user was alerted; `false` if suppressed (e.g. quiet hours, TV guard) |
| `acknowledged` | bool | User acknowledged the alert |
| `feedback` | bool \| null | `true` = correct, `false` = false alert, `null` = no feedback |
| `custom_name` | string \| null | Name of the matched custom sound, if any |
| `room` | string | Audio source, e.g. `"Laptop"` |

`DELETE /api/history` deletes all events. Response: `{"ok": true}`.

### Alerts

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/alerts/{id}/ack` | none | `{"ok": true}` |
| POST | `/api/alerts/{id}/feedback` | `{"correct": true}` | `{"ok": true}` |

### Stats

`GET /api/stats`

```json
{
  "per_category": {"smoke_alarm": 1, "doorbell": 14, "pressure_cooker": 9},
  "per_day": [
    {"date": "2026-11-11", "count": 18},
    {"date": "2026-11-12", "count": 12}
  ],
  "false_alert_rate": 0.08,
  "avg_latency_ms": 142.5,
  "cpu_percent": 8.5,
  "ram_mb": 412.0
}
```

| Field | Type | Meaning |
|---|---|---|
| `per_category` | object | Event count per category ID (all categories present, `0` if none) |
| `per_day` | array | `{date: "YYYY-MM-DD", count}` for the last 30 days, oldest first, including zero days |
| `false_alert_rate` | number | 0.0 to 1.0: events with `feedback: false` ÷ events with any feedback; `0` if none |
| `avg_latency_ms` | number | Average time from end of audio window to alert |
| `cpu_percent`, `ram_mb` | number | As in `/api/status` |

### Custom sounds

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/custom-sounds` | none | `[{"id", "name", "category", "examples"}]` |
| POST | `/api/custom-sounds` | `{"name": "My Kitchen Cooker", "category": "pressure_cooker"}` | `{"id": "p1"}` |
| POST | `/api/custom-sounds/{id}/record` | none | `{"examples": 3}` |
| POST | `/api/custom-sounds/{id}/test` | none | `{"similarity": 0.91, "matched": true}` |
| DELETE | `/api/custom-sounds/{id}` | none | `{"ok": true}` |

- `record` and `test` capture about 3 seconds from the laptop mic, so the request takes about 3 s to return. Show a recording indicator while waiting.
- `examples`: the total number of examples the profile now has.
- `similarity`: 0.0 to 1.0. `matched` is `true` when the similarity reaches the backend's match threshold.

### Whistle counter

Counts pressure-cooker whistles and alerts when the target is reached.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/whistle-counter` | none | `{"enabled", "target", "count"}` |
| PUT | `/api/whistle-counter` | `{"enabled": true, "target": 3}` | `{"enabled", "target", "count"}` |
| POST | `/api/whistle-counter/reset` | none | `{"enabled", "target", "count"}` (count = 0) |

`target` is an integer ≥ 1. `count` is the number of whistles counted since the last reset.

---

## WebSocket `/ws/live`

URL (dev): `ws://localhost:8000/ws/live`. Messages flow **server → client** only. Each message is a JSON object with a `type` field. Clients should ignore unknown types.

### `level`: sent once per audio window

```json
{"type": "level", "rms": 0.12, "top": "Speech", "skipped": false}
```

- `rms`: loudness, 0.0 to 1.0.
- `top`: human-readable name of the top-scoring sound, or `null` when `skipped` is true.
- `skipped`: `true` if the window was silent and the model was not run.

### `alert`: sent when an alert fires

```json
{
  "type": "alert",
  "id": "a1b2c3",
  "category": "pressure_cooker",
  "label": "Whistle 2 of 3",
  "confidence": 0.87,
  "priority": "high",
  "timestamp": "2026-11-12T19:04:05+05:30",
  "room": "Laptop",
  "custom_name": "My Kitchen Cooker",
  "escalated": false
}
```

- `id`: same ID as in `/api/history`, used for `/api/alerts/{id}/ack` and `/feedback`.
- `custom_name`: `null` unless a custom sound matched.
- `escalated`: `true` when this is a repeat of an alert that wasn't acknowledged.

### `status`: sent on connect and whenever monitoring starts or stops

```json
{"type": "status", "monitoring": true}
```

---

## WebSocket `/ws/listener` (Phase 5, optional)

Lets a phone act as an extra microphone in another room. Sneha's `listener/` page streams the phone's mic to the laptop; Vaidehi's endpoint passes each frame to `RemoteAudioInput.push()` (Shreeya, `contracts/types.py`).

**URL:** `wss://<laptop-ip>:8000/ws/listener?room=<RoomName>`, e.g. `?room=Kitchen`.

- `room` is required. If it's missing or empty, the server closes the connection with code `1008`.
- **HTTPS required.** Phone browsers only allow mic access on a secure origin, so for Phase 5 the backend runs with a self-signed certificate (`uvicorn ... --ssl-keyfile key.pem --ssl-certfile cert.pem`). The user accepts the certificate warning once on the phone. Certificate and key files are never committed.
- The backend serves the `listener/` page from the same HTTPS origin, so the page and the WebSocket share one certificate.

### Client → server: binary frames

Each WebSocket **binary** message is one chunk of raw audio:

| Property | Value |
|---|---|
| Encoding | Float32 PCM, little-endian, values -1.0 to 1.0 |
| Sample rate | 16 000 Hz (resample in the browser if the mic runs at 44.1/48 kHz) |
| Channels | 1 (mono) |
| Chunk size | ~0.5 s = 8000 samples = 32 000 bytes |

The client sends no JSON messages.

### Server → client: JSON text frames

**`ack`:** sent once on connect.

```json
{"type": "ack", "room": "Kitchen"}
```

**`alert`:** sent for every alert whose `room` matches this connection's room. Same format as the `alert` message on `/ws/live`.

```json
{
  "type": "alert",
  "id": "k9x2m1",
  "category": "doorbell",
  "label": "Doorbell",
  "confidence": 0.81,
  "priority": "medium",
  "timestamp": "2026-11-12T19:04:05+05:30",
  "room": "Kitchen",
  "custom_name": null,
  "escalated": false
}
```

Alerts from phone rooms also appear on `/ws/live` and in `/api/history`, with `room` set to the phone's room name.

---

## Settings schema

```json
{
  "categories": {
    "<category>": {"enabled": true, "priority": "critical", "sensitivity": 0.5}
  },
  "quiet_hours": {"enabled": false, "start": "22:00", "end": "07:00"},
  "alert_style": "full",
  "tv_guard": true,
  "cpu_saver": true,
  "custom_method": "embedding",
  "telegram": {"enabled": false, "bot_token": "", "chat_id": ""}
}
```

| Field | Type | Meaning |
|---|---|---|
| `categories` | object | One entry for **every** category ID |
| `categories.*.enabled` | bool | Detect and alert for this category |
| `categories.*.priority` | priority | `critical` \| `high` \| `medium` \| `low` |
| `categories.*.sensitivity` | number | 0.0 to 1.0; higher = alerts on weaker matches |
| `quiet_hours.start`, `.end` | `"HH:MM"` | 24-hour local time; may wrap past midnight |
| `alert_style` | string | `"full"` \| `"compact"` |
| `tv_guard` | bool | Reduce false alerts while TV, music or speech is playing |
| `cpu_saver` | bool | Skip model inference on silent windows |
| `custom_method` | string | `"embedding"` \| `"mfcc"` |
| `telegram` | object | Optional Telegram alert forwarding |

### Default settings

```json
{
  "categories": {
    "smoke_alarm":     {"enabled": true, "priority": "critical", "sensitivity": 0.5},
    "glass_break":     {"enabled": true, "priority": "critical", "sensitivity": 0.5},
    "pressure_cooker": {"enabled": true, "priority": "high",     "sensitivity": 0.5},
    "baby_cry":        {"enabled": true, "priority": "high",     "sensitivity": 0.5},
    "door_knock":      {"enabled": true, "priority": "medium",   "sensitivity": 0.5},
    "doorbell":        {"enabled": true, "priority": "medium",   "sensitivity": 0.5},
    "dog_bark":        {"enabled": true, "priority": "medium",   "sensitivity": 0.5},
    "appliance_beep":  {"enabled": true, "priority": "low",      "sensitivity": 0.5},
    "water_running":   {"enabled": true, "priority": "low",      "sensitivity": 0.5}
  },
  "quiet_hours": {"enabled": false, "start": "22:00", "end": "07:00"},
  "alert_style": "full",
  "tv_guard": true,
  "cpu_saver": true,
  "custom_method": "embedding",
  "telegram": {"enabled": false, "bot_token": "", "chat_id": ""}
}
```
