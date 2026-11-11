"""Demo for Part S3: YAMNet on a WAV file, the live mic, or a timing benchmark.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.ml.demo_yamnet path/to/sound.wav     # classify a file
    python -m backend.ml.demo_yamnet --live                # mic + silence gate
    python -m backend.ml.demo_yamnet --live --calibrate --seconds 0
    python -m backend.ml.demo_yamnet --benchmark           # avg inference ms per window

The first run downloads YAMNet into data/models/ (git-ignored). Live audio is
never saved.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from backend.audio.capture import HOP_S, SAMPLE_RATE, WINDOW_SAMPLES, MicCapture, Windower
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from backend.ml.yamnet import YamnetModel, top_k


def _fmt_top(scores: np.ndarray, k: int = 3) -> str:
    return "  ".join(f"{name} {score:.2f}" for name, score in top_k(scores, k))


def _summary(times: list[float]) -> str:
    if not times:
        return "no windows ran the model"
    p95 = sorted(times)[max(0, round(0.95 * len(times)) - 1)]
    return (
        f"{len(times)} windows: mean {statistics.mean(times):.1f} ms, "
        f"median {statistics.median(times):.1f} ms, p95 {p95:.1f} ms, max {max(times):.1f} ms"
    )


def load_wav(path: str) -> np.ndarray:
    """Read any soundfile-supported file as 16 kHz mono float32."""
    import soundfile as sf
    import soxr

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE).astype(np.float32)
    return audio


def classify_file(path: str) -> None:
    audio = load_wav(path)
    model = YamnetModel.instance()
    if audio.size < WINDOW_SAMPLES:  # pad short clips to one full window
        audio = np.pad(audio, (0, WINDOW_SAMPLES - audio.size))
    print(f"{path}: {audio.size / SAMPLE_RATE:.2f} s\n")

    windower = Windower()
    all_scores, times = [], []
    for end, window in windower.feed(audio):
        out = model.infer(window)
        all_scores.append(out.scores)
        times.append(out.inference_ms)
        start_s = (end - WINDOW_SAMPLES) / SAMPLE_RATE
        print(f"{start_s:6.2f}s  {out.inference_ms:5.1f} ms  {_fmt_top(out.scores)}")

    print(f"\nWhole file: {_fmt_top(np.mean(all_scores, axis=0), 5)}")
    print(f"Inference: {_summary(times)}")


def live(args: argparse.Namespace) -> None:
    model = YamnetModel.instance()
    gate = SilenceGate(threshold=args.threshold)
    if args.calibrate:
        gate.start_calibration()
    times: list[float] = []

    def on_window(samples: np.ndarray, timestamp: float) -> None:
        decision = gate.check(samples)
        clock = time.strftime("%H:%M:%S", time.localtime(timestamp))
        if decision.skipped:
            print(f"{clock}  SKIPPED  rms={decision.rms:.4f}")
            return
        out = model.infer(samples)
        times.append(out.inference_ms)
        tag = "CALIB  " if decision.calibrating else "ACTIVE "
        print(f"{clock}  {tag}  rms={decision.rms:.4f}  {out.inference_ms:5.1f} ms  {_fmt_top(out.scores)}")

    mic = MicCapture(on_window, device=args.device)
    try:
        mic.start()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nCheck the terminal has microphone permission.")
    print(f"Listening ({HOP_S}s hop, gate threshold {gate.threshold:.4f}). Ctrl+C to stop.\n")
    start = time.monotonic()
    try:
        while mic.is_running() and (args.seconds <= 0 or time.monotonic() - start < args.seconds):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()

    s = gate.stats()
    print(f"\n{s['total']} windows: {s['active']} active, {s['skipped']} skipped (skip rate {s['skip_rate']:.0%})")
    print(f"Inference: {_summary(times)}")


def benchmark(n: int) -> None:
    t0 = time.perf_counter()
    model = YamnetModel.instance()
    print(f"Model load + warm-up: {(time.perf_counter() - t0) * 1000:.0f} ms")
    rng = np.random.default_rng(0)
    times = [model.infer((rng.standard_normal(WINDOW_SAMPLES) * 0.1).astype(np.float32)).inference_ms
             for _ in range(n)]
    print(f"Inference per 0.96 s window (CPU): {_summary(times)}")
    print(f"Real-time budget per window is {HOP_S * 1000:.0f} ms -> "
          f"{statistics.mean(times) / (HOP_S * 1000):.1%} of one core's time spent in YAMNet")


def main() -> None:
    parser = argparse.ArgumentParser(description="YAMNet demo (CPU).")
    parser.add_argument("wav", nargs="?", help="audio file to classify")
    parser.add_argument("--live", action="store_true", help="classify the microphone live")
    parser.add_argument("--benchmark", type=int, nargs="?", const=200, metavar="N",
                        help="time N windows of noise (default 200)")
    parser.add_argument("--device", default=None, help="input device index or name (live)")
    parser.add_argument("--seconds", type=float, default=15, help="live duration; 0 = until Ctrl+C")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="silence gate threshold")
    parser.add_argument("--calibrate", action="store_true", help="calibrate the silence gate first (live)")
    args = parser.parse_args()
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)

    if args.benchmark:
        benchmark(args.benchmark)
    elif args.live:
        live(args)
    elif args.wav:
        classify_file(args.wav)
    else:
        parser.error("give a WAV file, --live or --benchmark")


if __name__ == "__main__":
    main()
