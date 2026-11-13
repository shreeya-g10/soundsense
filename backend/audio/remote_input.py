"""RemoteAudioInput (contracts/types.py, Phase 5): audio from phone listener pages.

Owner: Shreeya (Role 1). Part S6.

    from backend.audio.remote_input import YamnetRemoteInput

    remote = YamnetRemoteInput(on_window=engine.handle, profile_store=store)
    # in Vaidehi's /ws/listener handler, once per binary frame:
    remote.push(frame_bytes, room="Kitchen")
    ...
    remote.close()   # at shutdown

push() only decodes the bytes and queues them, so it is safe to call from an
async WebSocket handler. One "RemoteAudioInput" worker thread (started on
the first push) does the rest, per room:
    buffer -> 0.96 s windows every 0.48 s (50% overlap) -> silence gate
    (calibrated from the room's first ~2 s) -> YAMNet -> custom matching
    -> WindowResult(room=room) -> on_window
It is the same pipeline as YamnetListener (backend.audio.listener.analyse_window).

Timestamps: a window's timestamp is the arrival time of the chunk that
completed it, minus the audio that came after the window's end in that chunk,
and never earlier than the room's previous window (frames that arrive in a
burst, faster than real time, would otherwise step backwards by up to 20 ms).

If analysis falls ~30 s behind, the oldest queued chunk is dropped (counted
in ``dropped_chunks``) and that room's buffer restarts, so alerts never lag
far behind and no window is stitched across a gap.

Audio is processed in memory only and never saved.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from backend.audio.capture import HOP_SAMPLES, WINDOW_SAMPLES, Windower
from backend.audio.listener import analyse_window, load_pipeline
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from contracts.types import LISTENER_SAMPLE_RATE, ProfileStore, WindowResult

log = logging.getLogger(__name__)

_QUEUE_MAX_CHUNKS = 64  # ~32 s of 0.5 s chunks
_STOP = object()


@dataclass
class _Room:
    windower: Windower
    gate: SilenceGate
    samples_seen: int = 0
    pending_gap: bool = False  # a chunk was dropped: restart the buffer
    last_timestamp: float = 0.0  # keeps timestamps monotonic per room
    windows: int = 0  # analysed so far


class YamnetRemoteInput:
    """Implements ``contracts.types.RemoteAudioInput`` with YAMNet."""

    def __init__(
        self,
        on_window: Callable[[WindowResult], None],
        *,
        profile_store: ProfileStore | None = None,
        cpu_saver: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        calibrate: bool = True,
    ) -> None:
        """Create the input. Nothing runs until the first push().

        Args:
            on_window: Called once per window, from the "RemoteAudioInput"
                thread. Must return quickly and not raise.
            profile_store: Custom-sound store (see YamnetListener).
            cpu_saver: Skip silent windows (each room has its own gate;
                change later with ``gate(room).enabled``).
            threshold: Silence-gate threshold until a room is calibrated.
            calibrate: Calibrate each room's gate from its first ~2 s.
        """
        self._on_window = on_window
        self.profile_store = profile_store
        self.cpu_saver = cpu_saver
        self.threshold = threshold
        self.calibrate = calibrate
        self.dropped_chunks = 0
        self.error: Exception | None = None  # set if the model failed to load
        self._rooms: dict[str, _Room] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_CHUNKS)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._model = None
        self._mapper = None

    # --- RemoteAudioInput protocol ----------------------------------------

    def push(self, samples: bytes, room: str) -> None:
        """Queue one chunk of little-endian Float32 mono 16 kHz audio from ``room``.

        Raises:
            ValueError: if the length is not a multiple of 4 bytes.
            TypeError: if ``samples`` is not bytes-like or ``room`` not a str.
        """
        if not isinstance(samples, (bytes, bytearray, memoryview)):
            raise TypeError(f"samples must be bytes, got {type(samples).__name__}")
        if not isinstance(room, str):
            raise TypeError(f"room must be a str, got {type(room).__name__}")
        data = memoryview(samples).cast("B")
        if len(data) % 4:
            raise ValueError(f"audio chunk must be a multiple of 4 bytes (Float32), got {len(data)}")
        if not len(data):
            return
        audio = np.frombuffer(data, dtype="<f4").astype(np.float32)  # copy: caller may reuse the buffer
        audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        self._ensure_worker()
        self._put((room, audio, time.time()))

    # --- extras -----------------------------------------------------------

    def close(self) -> None:
        """Stop the worker after the queued chunks are analysed.

        Blocks until ``on_window`` will not be called again. A later push()
        starts a new worker. Room buffers and calibration are kept.
        """
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None:
            self._put(_STOP)
            if thread is not threading.current_thread():
                thread.join()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def rooms(self) -> list[str]:
        with self._lock:
            return list(self._rooms)

    def gate(self, room: str) -> SilenceGate:
        """The silence gate of ``room`` (KeyError if it never pushed audio)."""
        with self._lock:
            return self._rooms[room].gate

    def stats(self) -> dict:
        """Per-room silence-gate counts (see SilenceGate.stats) plus dropped chunks."""
        with self._lock:
            rooms = {name: r.gate.stats() for name, r in self._rooms.items()}
        return {"rooms": rooms, "dropped_chunks": self.dropped_chunks}

    # --- internals --------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self.error = None
            self._thread = threading.Thread(target=self._run, name="RemoteAudioInput", daemon=True)
            self._thread.start()

    def _put(self, item) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:  # drop the oldest chunk so we stay near real time
                    old = self._queue.get_nowait()
                    if old is not _STOP:
                        self.dropped_chunks += 1
                        self._mark_gap(old[0])
                except queue.Empty:
                    pass

    def _mark_gap(self, room: str) -> None:
        with self._lock:
            if room in self._rooms:
                self._rooms[room].pending_gap = True

    def _room(self, name: str) -> _Room:
        with self._lock:
            r = self._rooms.get(name)
            if r is None:
                gate = SilenceGate(threshold=self.threshold, enabled=self.cpu_saver)
                if self.calibrate:
                    gate.start_calibration()
                r = self._rooms[name] = _Room(Windower(WINDOW_SAMPLES, HOP_SAMPLES), gate)
            return r

    def _load_model(self) -> None:
        if self._model is None:
            self._model, self._mapper = load_pipeline()

    def _run(self) -> None:
        try:
            self._load_model()
        except Exception as exc:
            log.exception("Could not load YAMNet; remote input stopping")
            self.error = exc
            with self._lock:
                self._thread = None
            return
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            name, audio, arrived = item
            try:
                self._process(name, audio, arrived)
            except Exception:
                log.exception("Remote audio processing failed for one chunk from %s", name)

    def _process(self, name: str, audio: np.ndarray, arrived: float) -> None:
        r = self._room(name)
        if r.pending_gap:
            r.windower.reset()
            r.samples_seen = 0
            r.pending_gap = False
        windows = r.windower.feed(audio)
        r.samples_seen += audio.size
        for end, window in windows:
            timestamp = max(r.last_timestamp, arrived - (r.samples_seen - end) / LISTENER_SAMPLE_RATE)
            r.last_timestamp = timestamp
            decision = r.gate.check(window)
            try:
                result = analyse_window(self._model, self._mapper, self.profile_store,
                                        window, timestamp, decision, name)
            except Exception:
                log.exception("Inference failed for one window from %s", name)
                continue
            r.windows += 1
            try:
                self._on_window(result)
            except Exception:
                log.exception("on_window callback raised")


RemoteInput = YamnetRemoteInput  # short alias
