"""Demo for Parts S1-S2: live mic windows, loudness meter and silence gate.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.audio.demo_capture                  # 15 s, default mic, fixed threshold
    python -m backend.audio.demo_capture --calibrate      # measure ~2 s of room noise first
    python -m backend.audio.demo_capture --threshold 0.01
    python -m backend.audio.demo_capture --no-cpu-saver   # gate off: nothing is skipped
    python -m backend.audio.demo_capture --seconds 0      # until Ctrl+C
    python -m backend.audio.demo_capture --list-devices

Prints one line per 0.96 s window (every 0.48 s) marked CALIB, SKIPPED or
ACTIVE, then the CPU-saver statistics. Audio is never saved.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd

from backend.audio.capture import HOP_S, SAMPLE_RATE, WINDOW_SAMPLES, MicCapture
from backend.audio.silence import DEFAULT_MARGIN, DEFAULT_THRESHOLD, SilenceGate, rms_to_db

BAR_WIDTH = 30


def _device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live mic windows (nothing is recorded to disk).")
    parser.add_argument("--device", type=_device, default=None, help="input device index or name")
    parser.add_argument("--seconds", type=float, default=15, help="how long to run; 0 = until Ctrl+C")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"silence RMS threshold (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--calibrate", action="store_true", help="set the threshold from ~2 s of ambient noise")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help=f"calibration margin: threshold = ambient x margin (default {DEFAULT_MARGIN})")
    parser.add_argument("--no-cpu-saver", action="store_true", help="disable the gate (never skip)")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    gate = SilenceGate(threshold=args.threshold, enabled=not args.no_cpu_saver, margin=args.margin)
    if args.calibrate:
        gate.start_calibration()

    count = 0
    loudest = 0.0

    def show(samples: np.ndarray, timestamp: float) -> None:
        nonlocal count, loudest
        count += 1
        was_calibrating = gate.calibrating
        decision = gate.check(samples)
        loudest = max(loudest, decision.rms)

        db = rms_to_db(decision.rms)
        filled = int(np.clip((db + 60) / 60, 0, 1) * BAR_WIDTH)
        bar = "#" * filled + "." * (BAR_WIDTH - filled)
        label = "CALIB  " if decision.calibrating else ("SKIPPED" if decision.skipped else "ACTIVE ")
        print(
            f"#{count:4d}  {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
            f"  {label}  rms={decision.rms:.4f} ({db:6.1f} dB)  thr={gate.threshold:.4f}  {bar}"
        )
        if was_calibrating and not gate.calibrating:
            print(
                f"       calibrated: ambient rms={gate.ambient_level:.4f} x {gate.margin:g}"
                f" -> threshold {gate.threshold:.4f} ({rms_to_db(gate.threshold):.1f} dB)"
            )

    mic = MicCapture(show, device=args.device)
    try:
        mic.start()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nTry --list-devices, and check the terminal has microphone permission.")

    mode = "calibrating for ~2 s" if args.calibrate else f"threshold {gate.threshold:.4f}"
    print(
        f"Listening at {mic.device_rate:g} Hz -> {SAMPLE_RATE} Hz mono, {WINDOW_SAMPLES}-sample windows "
        f"every {HOP_S}s. CPU saver {'ON' if gate.enabled else 'OFF'} ({mode}). Ctrl+C to stop.\n"
    )
    start = time.monotonic()
    try:
        while mic.is_running() and (args.seconds <= 0 or time.monotonic() - start < args.seconds):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()

    s = gate.stats()
    print(
        f"\n{s['total']} windows: {s['active']} active, {s['skipped']} skipped "
        f"(skip rate {s['skip_rate']:.0%}; {s['below_threshold']} below threshold). "
        f"Overflows: {mic.overflows}, dropped blocks: {mic.dropped_blocks}."
    )
    if count and loudest == 0.0:
        print("Tip: every window was exactly silent; the terminal probably lacks microphone permission "
              "(System Settings > Privacy & Security > Microphone).")


if __name__ == "__main__":
    main()
