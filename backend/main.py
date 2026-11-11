"""SoundSense backend entry point.

Owner: Vaidehi (Role 2).

Run from the repo root:

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.api.runtime import BackendRuntime


def create_app(runtime: BackendRuntime | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        rt = runtime or BackendRuntime()
        fastapi_app.state.runtime = rt
        rt.startup(asyncio.get_running_loop())
        try:
            yield
        finally:
            rt.shutdown()

    fastapi_app = FastAPI(title="SoundSense", lifespan=lifespan)

    # Allow the Vite dev server (frontend/) to call the API from the browser.
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.include_router(router)

    # Serve Phase 5 phone listener page from the same origin when present.
    listener_dir = Path(__file__).resolve().parent.parent / "listener"
    if listener_dir.is_dir() and (listener_dir / "index.html").is_file():
        fastapi_app.mount(
            "/listener",
            StaticFiles(directory=str(listener_dir), html=True),
            name="listener",
        )

    return fastapi_app


app = create_app()
