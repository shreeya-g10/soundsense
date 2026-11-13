"""Experiment 8: CPU and RAM with the silence gate on vs off.

Owner: Shreeya (Role 1). Part S7.

A fixed 10-minute mixed stream (or --wav <recording>) goes through the app's
per-window pipeline (backend.audio.listener.analyse_window: silence gate,
YAMNet, category mapping) twice: gate on (cpu_saver, calibrated from the
first 2 s like the app) and gate off.

The default stream is deterministic (seed 0): quiet room tone (pink noise,
RMS 0.003) with clips dropped in after 2-15 s gaps. Clips come from the
Indian test set if present, else ESC-50, else synthetic tones.

CPU is measured as process CPU time (user + system, psutil) for the stream.
By default the stream is processed as fast as possible, so the main figure is
cpu_pct_realtime = CPU seconds / stream seconds x 100: the share of one CPU
core the app needs to keep up with live audio. --realtime paces windows at
the real 0.48 s hop instead (a 10-minute stream then takes 10 minutes per
setting) and cpu_pct_wall becomes the live CPU %. RAM is the process RSS
(start and peak, sampled during the run). Without psutil, the standard
library (os.times, resource) is used and ``measurement`` says so.

    python -m eval.model.exp8_resources                 # 10 min, ~1-2 min to run
    python -m eval.model.exp8_resources --minutes 1
    python -m eval.model.exp8_resources --wav data/indian/long_mixed.m4a --realtime

Writes exp8_resources.csv.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np

from backend.audio.capture import HOP_SAMPLES, SAMPLE_RATE, Windower
from backend.audio.silence import DEFAULT_THRESHOLD, SilenceGate
from eval.model import common as cm
from eval.model import datasets as ds

NAME = "exp8"
FIELDS = ["setting", "source", "stream_s", "event_fraction", "realtime", "windows", "inferred", "skipped",
          "skipped_pct", "avg_inference_ms", "p95_inference_ms", "cpu_s", "wall_s", "cpu_pct_realtime",
          "cpu_pct_wall", "rss_start_mb", "rss_peak_mb", "measurement"]
ROOM_RMS = 0.003


# --- the stream ------------------------------------------------------------------------

def synthetic_events(rng: np.random.Generator, n: int = 12) -> list[np.ndarray]:
    """Stand-in events when no dataset is available: beeps, chirps, noise bursts (1-4 s)."""
    out = []
    for i in range(n):
        dur = rng.uniform(1, 4)
        t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
        kind = i % 3
        if kind == 0:
            x = np.sin(2 * np.pi * rng.uniform(800, 3000) * t) * (np.sin(2 * np.pi * 3 * t) > 0)
        elif kind == 1:
            x = np.sin(2 * np.pi * (300 + 900 * t / dur) * t)
        else:
            x = rng.standard_normal(t.size)
        out.append((0.2 * x / (np.abs(x).max() or 1)).astype(np.float32))
    return out


def event_clips(paths: cm.DataPaths) -> tuple[str, list[ds.Clip]]:
    for source, get in ((cm.INDIAN, cm.get_indian_test), (cm.ESC50, cm.get_esc50)):
        try:
            return source, get(paths)
        except cm.DataMissing:
            continue
    return "synthetic", []


def build_stream(minutes: float, paths: cm.DataPaths, seed: int = 0) -> tuple[np.ndarray, str, float]:
    """(audio, source, fraction of the stream covered by events)."""
    from backend.custom.profile_store import pink_noise

    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * SAMPLE_RATE)
    stream = pink_noise(n, rng) * ROOM_RMS
    source, clips = event_clips(paths)
    synthetic = synthetic_events(rng) if not clips else None
    pos, covered = int(2.0 * SAMPLE_RATE), 0  # first 2 s: room only (gate calibration)
    while True:
        pos += int(rng.uniform(2, 15) * SAMPLE_RATE)
        if synthetic is not None:
            event = synthetic[int(rng.integers(len(synthetic)))]
        else:
            event = clips[int(rng.integers(len(clips)))].load()
        if pos + event.size > n:
            break
        stream[pos:pos + event.size] += event
        covered += event.size
        pos += event.size
    return np.clip(stream, -1, 1).astype(np.float32), source, covered / n


# --- measurement -----------------------------------------------------------------------

class Meter:
    """Process CPU seconds and RSS, via psutil or the standard library."""

    def __init__(self) -> None:
        try:
            import psutil

            self._proc = psutil.Process()
            self.kind = "psutil"
        except ImportError:
            self._proc = None
            self.kind = "stdlib (pip install psutil for RSS sampling)"

    def cpu_s(self) -> float:
        if self._proc is not None:
            t = self._proc.cpu_times()
            return t.user + t.system
        t = os.times()
        return t.user + t.system

    def rss_mb(self) -> float:
        if self._proc is not None:
            return self._proc.memory_info().rss / 1e6
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # peak only: bytes on macOS, KiB on Linux
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def process(stream: np.ndarray, gate_on: bool, extractor: cm.FeatureExtractor, meter: Meter,
            realtime: bool = False) -> dict:
    from backend.audio.listener import analyse_window

    extractor.load()
    gate = SilenceGate(threshold=DEFAULT_THRESHOLD, enabled=gate_on)
    gate.start_calibration()
    windower = Windower()
    times, windows, skipped = [], 0, 0
    rss_start = rss_peak = meter.rss_mb()
    cpu0, t0 = meter.cpu_s(), time.perf_counter()
    for i in range(0, stream.size, HOP_SAMPLES):  # mic-like blocks of one hop
        for end, w in windower.feed(stream[i:i + HOP_SAMPLES]):
            decision = gate.check(w)
            result = analyse_window(extractor.model, extractor.mapper, None, w, end / SAMPLE_RATE, decision, "Laptop")
            windows += 1
            if result.skipped:
                skipped += 1
            else:
                times.append(result.inference_ms)
            if windows % 25 == 0:
                rss_peak = max(rss_peak, meter.rss_mb())
            if realtime:
                delay = t0 + windows * HOP_SAMPLES / SAMPLE_RATE - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    cpu_s, wall_s = meter.cpu_s() - cpu0, time.perf_counter() - t0
    rss_peak = max(rss_peak, meter.rss_mb())
    stream_s = stream.size / SAMPLE_RATE
    return {"setting": "gate_on" if gate_on else "gate_off", "stream_s": stream_s, "realtime": realtime,
            "windows": windows, "inferred": len(times), "skipped": skipped,
            "skipped_pct": 100.0 * skipped / windows if windows else None,
            "avg_inference_ms": statistics.mean(times) if times else None,
            "p95_inference_ms": float(np.percentile(times, 95)) if times else None,
            "cpu_s": cpu_s, "wall_s": wall_s, "cpu_pct_realtime": 100.0 * cpu_s / stream_s,
            "cpu_pct_wall": 100.0 * cpu_s / wall_s if wall_s else None,
            "rss_start_mb": rss_start, "rss_peak_mb": rss_peak, "measurement": meter.kind}


def run(out_dir: Path = cm.RESULTS_DIR, paths: cm.DataPaths = cm.DataPaths(),
        extractor: cm.FeatureExtractor | None = None, minutes: float = 10.0, wav: Path | None = None,
        realtime: bool = False, seed: int = 0) -> list[Path]:
    extractor = extractor or cm.FeatureExtractor()
    if wav is not None:
        stream, source, fraction = ds.load_audio(wav), f"wav:{Path(wav).name}", None
    else:
        stream, source, fraction = build_stream(minutes, paths, seed)
        if source == "synthetic":
            print(f"[{NAME}] no Indian or ESC-50 clips found; using synthetic events", flush=True)
    print(f"[{NAME}] {stream.size / SAMPLE_RATE / 60:.1f} min stream from {source}"
          + (f", events cover {fraction:.0%}" if fraction is not None else ""), flush=True)
    extractor.load()
    for _ in range(3):  # warm-up so the first setting doesn't pay for graph tracing
        extractor.model.infer(stream[:15360] + 0.1)
    meter = Meter()
    rows = []
    for gate_on in (False, True):
        r = process(stream, gate_on, extractor, meter, realtime)
        r.update(source=source, event_fraction=fraction)
        rows.append(r)
        print(f"[{NAME}] {r['setting']}: skipped {r['skipped_pct']:.1f}%, avg inference "
              f"{r['avg_inference_ms'] or 0:.2f} ms, CPU {r['cpu_pct_realtime']:.1f}% of one core "
              f"(real-time equivalent), RSS peak {r['rss_peak_mb']:.0f} MB", flush=True)
    return [cm.write_csv(out_dir, "exp8_resources.csv", FIELDS, rows)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--wav", type=Path, help="use this recording instead of the generated stream")
    parser.add_argument("--realtime", action="store_true", help="pace at the real 0.48 s hop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=cm.RESULTS_DIR)
    parser.add_argument("--indian", type=Path, default=cm.DataPaths().indian, help="Indian data folder")
    parser.add_argument("--esc50", type=Path, default=cm.DataPaths().esc50, help="ESC-50 folder")
    args = parser.parse_args()
    paths = cm.DataPaths(indian=args.indian, esc50=args.esc50)
    for p in run(args.out, paths, minutes=args.minutes, wav=args.wav, realtime=args.realtime, seed=args.seed):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
