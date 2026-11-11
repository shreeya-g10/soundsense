"""Shared types for SoundSense. FROZEN: see AGENTS.md before changing anything here.

This module is the boundary between:
  - Shreeya (Role 1, AI & Audio), who IMPLEMENTS AudioListener and ProfileStore
    and PRODUCES WindowResult / CustomMatch objects, and
  - Vaidehi (Role 2, Backend & Alerts), who CALLS AudioListener and ProfileStore
    and CONSUMES WindowResult / CustomMatch objects in the detection engine.

Import it as ``from contracts.types import ...`` from the repo root.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

# ---------------------------------------------------------------------------
# Category IDs
# ---------------------------------------------------------------------------
# Stable string IDs for the household sounds SoundSense detects. These exact
# strings are used as dict keys in WindowResult.category_scores, in settings,
# in the database and in the HTTP/WebSocket API (see contracts/API.md).

SMOKE_ALARM = "smoke_alarm"
GLASS_BREAK = "glass_break"
PRESSURE_COOKER = "pressure_cooker"
BABY_CRY = "baby_cry"
DOOR_KNOCK = "door_knock"
DOORBELL = "doorbell"
DOG_BARK = "dog_bark"
APPLIANCE_BEEP = "appliance_beep"
WATER_RUNNING = "water_running"

CATEGORIES: list[str] = [
    SMOKE_ALARM,
    GLASS_BREAK,
    PRESSURE_COOKER,
    BABY_CRY,
    DOOR_KNOCK,
    DOORBELL,
    DOG_BARK,
    APPLIANCE_BEEP,
    WATER_RUNNING,
]
"""Every category ID, in display order."""

# ---------------------------------------------------------------------------
# Priorities
# ---------------------------------------------------------------------------
# Alert priority levels, most to least urgent. Used in settings and alerts.

PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"

PRIORITIES: list[str] = [PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]
"""Every priority, most urgent first."""

# ---------------------------------------------------------------------------
# Context classes
# ---------------------------------------------------------------------------
# Keys of WindowResult.context_scores. Not alert categories: they let the engine
# recognise background media/speech (e.g. for the TV guard).

CONTEXT_SPEECH = "speech"
CONTEXT_MUSIC = "music"
CONTEXT_TELEVISION = "television"

CONTEXTS: list[str] = [CONTEXT_SPEECH, CONTEXT_MUSIC, CONTEXT_TELEVISION]
"""Every context key."""

# ---------------------------------------------------------------------------
# Custom-sound matching methods
# ---------------------------------------------------------------------------

METHOD_EMBEDDING = "embedding"
METHOD_MFCC = "mfcc"

CustomMethod = Literal["embedding", "mfcc"]
"""Matching method accepted by ProfileStore.set_method()."""

EMBEDDING_SIZE = 1024
"""Length of a YAMNet embedding (WindowResult.embedding)."""


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class CustomMatch:
    """A user-taught custom sound that matched an audio window.

    Produced by: Shreeya (Role 1), inside the AudioListener pipeline.
    Consumed by: Vaidehi (Role 2), the detection engine, which may raise an
    alert labelled with ``name``.

    Attributes:
        profile_id: ID of the custom-sound profile (as returned by
            ProfileStore.create()).
        name: User-given name, e.g. "My Kitchen Cooker".
        category: The category ID the profile belongs to (one of CATEGORIES).
        similarity: Similarity between the window and the profile, 0.0 to 1.0
            (higher = closer).
    """

    profile_id: str
    name: str
    category: str
    similarity: float


@dataclass
class WindowResult:
    """Analysis result for one window of microphone audio.

    Produced by: Shreeya (Role 1). The AudioListener builds one of these per
    audio window and passes it to its ``on_window`` callback.
    Consumed by: Vaidehi (Role 2). The detection engine turns these into
    alert decisions and live-level updates.

    No raw audio is included. Audio never leaves Role 1's code and is never
    saved to disk.

    Attributes:
        timestamp: Unix time in seconds (``time.time()``) when the window
            finished being captured.
        rms: Root-mean-square loudness of the window (float audio in -1..1, so
            rms is 0.0 to 1.0).
        skipped: True if the window was silent and the model was NOT run.
            When True: category_scores and context_scores are empty dicts,
            embedding is None, custom_matches is empty and inference_ms is 0.0.
        category_scores: Score 0.0 to 1.0 for each category ID in CATEGORIES
            (every category present when skipped is False).
        context_scores: Score 0.0 to 1.0 for each key in CONTEXTS
            ("speech", "music", "television").
        embedding: The 1024-value YAMNet embedding for the window
            (mean-pooled over YAMNet frames), or None when skipped.
        custom_matches: Custom-sound profiles that matched this window, sorted
            by similarity, highest first. Empty if none matched.
        inference_ms: Time spent on model inference for this window, in
            milliseconds.
        room: Where the audio came from. "Laptop" for the built-in mic.
    """

    timestamp: float
    rms: float
    skipped: bool
    category_scores: dict[str, float]
    context_scores: dict[str, float]
    embedding: list[float] | None
    custom_matches: list[CustomMatch]
    inference_ms: float
    room: str = "Laptop"


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class AudioListener(Protocol):
    """Captures microphone audio and emits one WindowResult per window.

    Implemented by: Shreeya (Role 1), using sounddevice + YAMNet.
    Called by: Vaidehi (Role 2). The backend creates one listener, passes an
    ``on_window`` callback, and starts/stops it from /api/monitor/start and
    /api/monitor/stop.

    Threading: ``on_window`` is called from the listener's audio/worker
    thread, not the caller's thread. The callback must return quickly and
    must not raise. Hand heavy work off to another thread or queue.
    """

    def __init__(
        self,
        on_window: Callable[[WindowResult], None],
        device: int | str | None = None,
    ) -> None:
        """Create a listener. Does not start capturing.

        Args:
            on_window: Called once per analysed audio window.
            device: sounddevice input device index or name. None = system
                default input.
        """
        ...

    def start(self) -> None:
        """Start capturing and analysing audio in the background.

        Returns immediately. Calling start() while already running does
        nothing. Raises RuntimeError if the microphone cannot be opened.
        """
        ...

    def stop(self) -> None:
        """Stop capturing and release the microphone.

        Blocks until no further ``on_window`` calls will happen. Calling
        stop() while not running does nothing.
        """
        ...

    def is_running(self) -> bool:
        """Return True while the listener is capturing audio."""
        ...


LISTENER_SAMPLE_RATE = 16000
"""Sample rate (Hz) of audio pushed to RemoteAudioInput: mono Float32 PCM."""


class RemoteAudioInput(Protocol):
    """Audio input from a phone listener page (Phase 5, optional).

    Implemented by: Shreeya (Role 1). Runs the same pipeline as AudioListener
    (silence skip, YAMNet, custom matching) and produces WindowResult objects
    whose ``room`` is the room passed to push().
    Called by: Vaidehi (Role 2), from the /ws/listener WebSocket endpoint,
    once per binary frame received (see contracts/API.md).
    Audio source: Sneha (Role 3), the listener/ page, which sends the frames.

    Results go to the ``on_window`` callback given to the constructor, with
    the same threading rules as AudioListener. Audio is processed in memory
    only and never saved.
    """

    def __init__(self, on_window: Callable[[WindowResult], None]) -> None:
        """Create the input. ``on_window`` receives every WindowResult."""
        ...

    def push(self, samples: bytes, room: str) -> None:
        """Feed one chunk of audio from a phone.

        Args:
            samples: Raw little-endian Float32 PCM, mono, LISTENER_SAMPLE_RATE
                (16 kHz), values -1..1. Usually ~0.5 s (8000 samples = 32000
                bytes), but any length that is a multiple of 4 bytes is
                accepted. Raises ValueError otherwise.
            room: Room name from the phone, e.g. "Kitchen". Each room is
                buffered and windowed separately.

        Returns quickly: chunks are buffered and analysed on a worker thread,
        so this is safe to call from an async WebSocket handler.
        """
        ...


class ProfileStore(Protocol):
    """Stores user-taught custom sounds and the feedback that tunes detection.

    Implemented by: Shreeya (Role 1). Only embeddings / features are saved to
    disk, never audio.
    Called by: Vaidehi (Role 2), from the /api/custom-sounds endpoints and
    /api/alerts/{id}/feedback.

    Methods that record from the mic block for about ``seconds`` while
    recording. All methods raise KeyError for an unknown ``profile_id``.
    """

    def create(self, name: str, category: str) -> str:
        """Create an empty custom-sound profile.

        Args:
            name: User-given name, e.g. "My Kitchen Cooker".
            category: Category ID the sound belongs to (one of CATEGORIES).
                Raises ValueError otherwise.

        Returns:
            The new profile's ID.
        """
        ...

    def add_example_from_mic(self, profile_id: str, seconds: float = 3) -> int:
        """Record ``seconds`` of audio from the mic and add it as an example.

        Returns:
            The number of examples the profile now has.
        """
        ...

    def test_from_mic(self, profile_id: str, seconds: float = 3) -> float:
        """Record ``seconds`` of audio and compare it with the profile.

        Does not change the profile.

        Returns:
            Similarity, 0.0 to 1.0 (higher = closer).
        """
        ...

    def add_feedback(
        self,
        embedding: builtins.list[float],
        category: str,
        profile_id: str | None,
        positive: bool,
    ) -> None:
        """Record user feedback on an alert so future detection improves.

        Args:
            embedding: The WindowResult.embedding that triggered the alert.
            category: Category ID the alert was raised for.
            profile_id: The custom-sound profile that matched, or None if the
                alert came from the base model.
            positive: True if the user said the alert was correct, False if
                it was a false alert.
        """
        ...

    def list(self) -> builtins.list[dict[str, Any]]:
        """Return every profile.

        Each dict has exactly these keys:
            {"id": str, "name": str, "category": str, "examples": int}
        """
        ...

    def delete(self, profile_id: str) -> None:
        """Delete a profile and all of its stored examples."""
        ...

    def set_method(self, method: CustomMethod) -> None:
        """Choose how custom sounds are matched: "embedding" or "mfcc".

        Raises ValueError for any other value.
        """
        ...
