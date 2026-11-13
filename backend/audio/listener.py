"""The real AudioListener (contracts/types.py): mic -> silence gate -> YAMNet -> WindowResult.

Owner: Shreeya (Role 1). Part S4.

    from backend.audio.listener import YamnetListener

    listener = YamnetListener(on_window=engine.handle)   # engine = Vaidehi's code
    listener.start()
    listener.gate.enabled = settings["cpu_saver"]
    ...
    listener.stop()

Threads (none of them blocks PortAudio's audio thread):
  1. PortAudio audio thread -> MicCapture's queue (copy only).
  2. "MicCapture" thread: cuts 0.96 s windows, runs the silence gate (cheap),
     and queues each window for inference.
  3. "YamnetListener" thread: loads YAMNet on first start, runs inference,
     builds the WindowResult and calls ``on_window``.
Windows are delivered in capture order. If inference ever falls ~15 s
behind, the oldest queued window is dropped (counted in ``dropped_windows``)
so alerts never lag far behind real time.

Audio never leaves memory and is never written to disk.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

from backend.audio.capture import MicCapture
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from contracts.types import ProfileStore, WindowResult

log = logging.getLogger(__name__)

_QUEUE_MAX_WINDOWS = 32  # ~15 s at one window per 0.48 s; also covers first model load (~4 s)
_STOP = object()


class YamnetListener:
    """Implements ``contracts.types.AudioListener`` with the real mic and YAMNet."""

    def __init__(
        self,
        on_window: Callable[[WindowResult], None],
        device: int | str | None = None,
        *,
        room: str = "Laptop",
        cpu_saver: bool = True,
        threshold: float = DEFAULT_THRESHOLD,
        calibrate: bool = True,
        profile_store: ProfileStore | None = None,
    ) -> None:
        """Create the listener. Does not open the mic or load the model yet.

        Args:
            on_window: Called once per window from the "YamnetListener"
                thread. Must return quickly and not raise.
            device: sounddevice input device index or name; None = default.
            room: Value for WindowResult.room.
            cpu_saver: Initial silence-gate state (settings "cpu_saver").
                Change later with ``listener.gate.enabled``.
            threshold: Silence-gate RMS threshold. With calibration on, it
                only applies until the first ~2 s have been measured.
            calibrate: Measure ~2 s of room noise at each start() and set the
                threshold from it (default). False = fixed ``threshold``.
            profile_store: Custom-sound store. If it has a
                ``match(embedding) -> list[CustomMatch]`` method (the real
                YamnetProfileStore does), it fills custom_matches on every
                active window. If it has ``add_background(embedding,
                source=)``, the calibration windows' embeddings (room noise)
                are added to its background bank.
        """
        self._on_window = on_window
        self.room = room
        self.calibrate = calibrate
        self.profile_store = profile_store
        self.gate = SilenceGate(threshold=threshold, enabled=cpu_saver)
        self.dropped_windows = 0
        self.error: Exception | None = None  # set if the model failed to load

        self._mic = MicCapture(self._on_capture, device=device)
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX_WINDOWS)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._model = None
        self._mapper = None

    # --- AudioListener protocol -------------------------------------------

    def start(self) -> None:
        """Open the mic and start delivering WindowResults in the background.

        Returns immediately (on the first start the model loads in the
        background; windows queue up meanwhile). Does nothing if already
        running. Raises RuntimeError if the microphone cannot be opened.
        """
        with self._lock:
            if self.is_running():
                return
            self.error = None
            self._queue = queue.Queue(maxsize=_QUEUE_MAX_WINDOWS)
            if self.calibrate:
                self.gate.start_calibration()
            self._thread = threading.Thread(target=self._run, name="YamnetListener", daemon=True)
            self._thread.start()
            try:
                self._mic.start()
            except RuntimeError:
                self._put(_STOP)
                self._thread.join()
                self._thread = None
                raise

    def stop(self) -> None:
        """Stop capture and release the mic.

        Blocks until ``on_window`` will not be called again (windows already
        queued are delivered first). Does nothing if not running.
        """
        with self._lock:
            thread, self._thread = self._thread, None
        self._mic.stop()  # after this, nothing new is queued
        if thread is not None:
            self._put(_STOP)
            if thread is not threading.current_thread():
                thread.join()

    def is_running(self) -> bool:
        """True while the mic is open and windows are being delivered."""
        return self._thread is not None and self._thread.is_alive() and self._mic.is_running()

    # --- extras -----------------------------------------------------------

    def stats(self) -> dict:
        """Silence-gate counts (see SilenceGate.stats) plus dropped windows."""
        return {**self.gate.stats(), "dropped_windows": self.dropped_windows,
                "mic_overflows": self._mic.overflows}

    # --- internals --------------------------------------------------------

    def _on_capture(self, samples: np.ndarray, timestamp: float) -> None:
        """MicCapture thread: cheap gate check, then hand off to inference."""
        decision = self.gate.check(samples)
        self._put((samples, timestamp, decision))

    def _put(self, item) -> None:
        while True:
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:  # drop the oldest window so we stay near real time
                    if self._queue.get_nowait() is not _STOP:
                        self.dropped_windows += 1
                except queue.Empty:
                    pass

    def _load_model(self) -> None:
        if self._model is None:
            from backend.ml.category_map import CategoryMapper
            from backend.ml.yamnet import YamnetModel

            self._model = YamnetModel.instance()
            self._mapper = CategoryMapper(self._model.class_names)

    def _run(self) -> None:
        """Inference thread."""
        try:
            self._load_model()
        except Exception as exc:
            log.exception("Could not load YAMNet; listener stopping")
            self.error = exc
            threading.Thread(target=self._mic.stop, daemon=True).start()
            return

        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            samples, timestamp, decision = item
            try:
                result = self._analyse(samples, timestamp, decision)
            except Exception:
                log.exception("Inference failed for one window")
                continue
            try:
                self._on_window(result)
            except Exception:
                log.exception("on_window callback raised")

    def _custom_matches(self, embedding: np.ndarray) -> list:
        match = getattr(self.profile_store, "match", None)
        if match is None:
            return []
        try:
            return list(match(embedding))
        except Exception:
            log.exception("Custom-sound matching failed; continuing without matches")
            return []

    def _add_background(self, embedding: np.ndarray) -> None:
        """Calibration windows are room noise: feed them to the store's background bank."""
        add = getattr(self.profile_store, "add_background", None)
        if add is None:
            return
        try:
            add(embedding, source="calibration")
        except Exception:
            log.exception("Could not add calibration window to the background bank")

    def _analyse(self, samples: np.ndarray, timestamp: float, decision) -> WindowResult:
        if decision.skipped:
            return WindowResult(
                timestamp=timestamp,
                rms=decision.rms,
                skipped=True,
                category_scores={},
                context_scores={},
                embedding=None,
                custom_matches=[],
                inference_ms=0.0,
                room=self.room,
            )
        out = self._model.infer(samples)
        if decision.calibrating:
            self._add_background(out.embedding)
        return WindowResult(
            timestamp=timestamp,
            rms=decision.rms,
            skipped=False,
            category_scores=self._mapper.category_scores(out.scores),
            context_scores=self._mapper.context_scores(out.scores),
            embedding=out.embedding.tolist(),
            custom_matches=self._custom_matches(out.embedding),
            inference_ms=out.inference_ms,
            room=self.room,
        )
