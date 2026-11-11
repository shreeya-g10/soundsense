"""Demo for Part S1: live microphone windows with a loudness meter.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.audio.demo_capture                 # 15 s, default mic
    python -m backend.audio.demo_capture --seconds 0     # until Ctrl+C
    python -m backend.audio.demo_capture --device 0
    python -m backend.audio.demo_capture --list-devices

Prints one line per 0.96 s window (every 0.48 s). Audio is never saved.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd

from backend.audio.capture import HOP_S, SAMPLE_RATE, WINDOW_SAMPLES, MicCapture, rms

BAR_WIDTH = 40


def _device(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live mic windows (nothing is recorded to disk).")
    parser.add_argument("--device", type=_device, default=None, help="input device index or name")
    parser.add_argument("--seconds", type=float, default=15, help="how long to run; 0 = until Ctrl+C")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    count = 0
    loudest = 0.0
    last_ts: float | None = None

    def show(samples: np.ndarray, timestamp: float) -> None:
        nonlocal count, loudest, last_ts
        count += 1
        level = rms(samples)
        loudest = max(loudest, level)
        peak = float(np.max(np.abs(samples)))
        gap = "  -  " if last_ts is None else f"{timestamp - last_ts:.3f}"
        last_ts = timestamp
        # Log-ish meter: -60 dBFS .. 0 dBFS
        db = 20 * np.log10(max(level, 1e-6))
        filled = int(np.clip((db + 60) / 60, 0, 1) * BAR_WIDTH)
        bar = "#" * filled + "." * (BAR_WIDTH - filled)
        print(
            f"#{count:4d}  {time.strftime('%H:%M:%S', time.localtime(timestamp))}"
            f"  +{gap}s  len={samples.size}  rms={level:.4f} ({db:6.1f} dB)  peak={peak:.3f}  {bar}"
        )

    mic = MicCapture(show, device=args.device)
    try:
        mic.start()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nTry --list-devices, and check the terminal has microphone permission.")

    print(
        f"Listening at {mic.device_rate:g} Hz -> {SAMPLE_RATE} Hz mono, "
        f"{WINDOW_SAMPLES}-sample windows every {HOP_S}s. Ctrl+C to stop.\n"
    )
    start = time.monotonic()
    try:
        while mic.is_running() and (args.seconds <= 0 or time.monotonic() - start < args.seconds):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()

    print(f"\n{count} windows. Overflows: {mic.overflows}, dropped blocks: {mic.dropped_blocks}.")
    if count and loudest == 0.0:
        print("Tip: if every rms is 0.0000, the terminal probably lacks microphone permission "
              "(System Settings > Privacy & Security > Microphone).")


if __name__ == "__main__":
    main()
