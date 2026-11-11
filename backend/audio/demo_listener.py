"""Demo for Part S4: the real AudioListener, printing each WindowResult compactly.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.audio.demo_listener                 # 30 s
    python -m backend.audio.demo_listener --calibrate --seconds 0
    python -m backend.audio.demo_listener --no-cpu-saver

Audio is never saved.
"""

from __future__ import annotations

import argparse
import time

from backend.audio.listener import YamnetListener
from backend.audio.silence import DEFAULT_THRESHOLD
from contracts.types import WindowResult


def fmt(w: WindowResult) -> str:
    clock = time.strftime("%H:%M:%S", time.localtime(w.timestamp))
    if w.skipped:
        return f"{clock}  skipped  rms={w.rms:.4f}"
    top = max(w.category_scores, key=w.category_scores.get)
    second = sorted(w.category_scores.items(), key=lambda kv: kv[1], reverse=True)[1]
    ctx = " ".join(f"{k[:2]}={v:.2f}" for k, v in w.context_scores.items())
    return (
        f"{clock}  ACTIVE   rms={w.rms:.4f}  {top:15s} {w.category_scores[top]:.2f}"
        f"  (next {second[0]} {second[1]:.2f})  ctx[{ctx}]  {w.inference_ms:4.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print live WindowResults from the real listener.")
    parser.add_argument("--device", default=None, help="input device index or name")
    parser.add_argument("--seconds", type=float, default=30, help="duration; 0 = until Ctrl+C")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="silence gate threshold")
    parser.add_argument("--calibrate", action="store_true", help="calibrate the silence gate from room noise")
    parser.add_argument("--no-cpu-saver", action="store_true", help="never skip windows")
    args = parser.parse_args()
    device = int(args.device) if isinstance(args.device, str) and args.device.isdigit() else args.device

    listener = YamnetListener(
        lambda w: print(fmt(w), flush=True),
        device=device,
        cpu_saver=not args.no_cpu_saver,
        threshold=args.threshold,
        calibrate=args.calibrate,
    )
    try:
        listener.start()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nCheck the terminal has microphone permission.")
    print("Listening (ctx: sp=speech mu=music te=television). Ctrl+C to stop.\n", flush=True)
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

    s = listener.stats()
    print(f"\n{s['total']} windows: {s['active']} active, {s['skipped']} skipped "
          f"(skip rate {s['skip_rate']:.0%}), dropped {s['dropped_windows']}, overflows {s['mic_overflows']}")


if __name__ == "__main__":
    main()
