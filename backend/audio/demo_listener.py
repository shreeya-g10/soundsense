"""Demo for Part S4: the real AudioListener, printing each WindowResult compactly.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.audio.demo_listener                 # 30 s
    python -m backend.audio.demo_listener --seconds 0
    python -m backend.audio.demo_listener --no-calibrate --threshold 0.01
    python -m backend.audio.demo_listener --no-cpu-saver
    python -m backend.audio.demo_listener --log           # also write data/logs/listener-*.csv

Each active line shows the best category (or "—" if every category is below
LABEL_MIN) and the raw top-3 YAMNet classes. --log writes one CSV row per
active window: time, rms, top-3 YAMNet classes and every category score.
Audio is never saved; the CSV holds only class names and scores.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import TextIO

import numpy as np

from backend.audio.listener import YamnetListener
from backend.audio.silence import DEFAULT_THRESHOLD
from contracts.types import CATEGORIES, WindowResult

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "data" / "logs"  # gitignored (data/*)
LABEL_MIN = 0.15  # below this, no category is named: near-zero scores aren't a detection
TOP_K = 3

Top = list[tuple[str, float]]


def top_classes(scores: np.ndarray, names: list[str], k: int = TOP_K) -> Top:
    """The k highest-scoring YAMNet classes, best first."""
    idx = np.argsort(scores)[::-1][:k]
    return [(names[i], float(scores[i])) for i in idx]


class _RecordingModel:
    """Wraps YamnetModel and remembers the last output, for display only."""

    def __init__(self, model) -> None:
        self._model = model
        self.last = None

    def infer(self, samples: np.ndarray):
        self.last = self._model.infer(samples)
        return self.last

    def __getattr__(self, name):
        return getattr(self._model, name)


class DemoListener(YamnetListener):
    """YamnetListener that also hands the raw YAMNet top-3 to its callback.

    WindowResult and the category mapping are unchanged; ``on_result`` gets
    ``(result, top)`` where ``top`` is [] for skipped windows.
    """

    def __init__(self, on_result, **kwargs) -> None:
        super().__init__(self._deliver, **kwargs)
        self._on_result = on_result

    def _load_model(self) -> None:
        super()._load_model()
        if not isinstance(self._model, _RecordingModel):
            self._model = _RecordingModel(self._model)

    def _analyse(self, samples, timestamp, decision) -> WindowResult:
        self._model.last = None
        return super()._analyse(samples, timestamp, decision)

    def _deliver(self, w: WindowResult) -> None:
        last = self._model.last
        top = [] if w.skipped or last is None else top_classes(last.scores, self._model.class_names)
        self._on_result(w, top)


def fmt(w: WindowResult, top: Top) -> str:
    clock = time.strftime("%H:%M:%S", time.localtime(w.timestamp))
    if w.skipped:
        return f"{clock}  skipped  rms={w.rms:.4f}"
    ranked = sorted(w.category_scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, best_score), (second, second_score) = ranked[0], ranked[1]
    if best_score < LABEL_MIN:
        label = f"{'—':15s} {best_score:.2f}  {'':22s}"
    else:
        label = f"{best:15s} {best_score:.2f}  {f'(next {second} {second_score:.2f})':22s}"
    ctx = " ".join(f"{k[:2]}={v:.2f}" for k, v in w.context_scores.items())
    yam = ", ".join(f"{name} {score:.2f}" for name, score in top)
    return (
        f"{clock}  ACTIVE   rms={w.rms:.4f}  {label}  ctx[{ctx}]  {w.inference_ms:4.1f} ms"
        f"  yamnet: {yam}"
    )


class CsvLog:
    """One row per active window: time, rms, top-3 YAMNet classes, category scores."""

    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._writer = csv.writer(file)
        header = ["time", "rms"]
        for i in range(1, TOP_K + 1):
            header += [f"yamnet_{i}", f"yamnet_{i}_score"]
        self._writer.writerow(header + CATEGORIES)
        file.flush()

    def write(self, w: WindowResult, top: Top) -> None:
        if w.skipped:
            return
        row = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(w.timestamp)), f"{w.rms:.5f}"]
        for name, score in top + [("", float("nan"))] * (TOP_K - len(top)):
            row += [name, "" if np.isnan(score) else f"{score:.4f}"]
        row += [f"{w.category_scores.get(c, 0.0):.4f}" for c in CATEGORIES]
        self._writer.writerow(row)
        self._file.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print live WindowResults from the real listener.")
    parser.add_argument("--device", default=None, help="input device index or name")
    parser.add_argument("--seconds", type=float, default=30, help="duration; 0 = until Ctrl+C")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="silence gate threshold")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="use the fixed --threshold instead of calibrating from room noise")
    parser.add_argument("--no-cpu-saver", action="store_true", help="never skip windows")
    parser.add_argument("--log", action="store_true",
                        help=f"save each active window's top-3 YAMNet classes and category scores "
                             f"to a CSV in {LOG_DIR.relative_to(REPO_ROOT)}/")
    args = parser.parse_args()
    device = int(args.device) if isinstance(args.device, str) and args.device.isdigit() else args.device

    log_file = log = None
    if args.log:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"listener-{time.strftime('%Y%m%d-%H%M%S')}.csv"
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log = CsvLog(log_file)

    def on_result(w: WindowResult, top: Top) -> None:
        print(fmt(w, top), flush=True)
        if log is not None:
            log.write(w, top)

    listener = DemoListener(
        on_result,
        device=device,
        cpu_saver=not args.no_cpu_saver,
        threshold=args.threshold,
        calibrate=not args.no_calibrate,
    )
    try:
        try:
            listener.start()
        except RuntimeError as exc:
            raise SystemExit(f"{exc}\nCheck the terminal has microphone permission.")
        print("Listening (ctx: sp=speech mu=music te=television). Ctrl+C to stop.\n", flush=True)
        if log_file is not None:
            print(f"Logging active windows to {log_path}\n", flush=True)
        start = time.monotonic()
        try:
            while args.seconds <= 0 or time.monotonic() - start < args.seconds:
                if listener.error:
                    raise SystemExit(f"Listener failed: {listener.error}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
    finally:
        if log_file is not None:
            log_file.close()

    s = listener.stats()
    print(f"\n{s['total']} windows: {s['active']} active, {s['skipped']} skipped "
          f"(skip rate {s['skip_rate']:.0%}), dropped {s['dropped_windows']}, overflows {s['mic_overflows']}")
    if log_file is not None:
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
