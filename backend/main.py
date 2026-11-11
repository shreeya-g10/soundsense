"""SoundSense backend entry point (stub).

Owner: Vaidehi (Role 2).

Run from the repo root:

    uvicorn backend.main:app --reload

Only GET /api/status exists so far, returning dummy data in the
contracts/API.md format.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SoundSense")

# Allow the Vite dev server (frontend/) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status() -> dict:
    """Dummy status. Replace with real values once the engine is wired in."""
    return {
        "monitoring": False,
        "mic_ok": True,
        "uptime_s": 0,
        "events_today": 0,
        "cpu_percent": 0.0,
        "ram_mb": 0.0,
    }
