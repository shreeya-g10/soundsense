"""Demo for Part S5: teach a custom sound, then live-test it.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.custom.demo_profiles                       # 5 x 3 s mic examples, 20 s live test
    python -m backend.custom.demo_profiles --name "My Kitchen Cooker" --category pressure_cooker
    python -m backend.custom.demo_profiles --from-wav sound.wav  # enrol from a file instead of the mic
    python -m backend.custom.demo_profiles --keep                # keep the profile afterwards

Only embeddings are stored (data/profiles/), never audio. Without --keep the
demo profile is deleted at the end.
"""

from __future__ import annotations

import argparse
import time

from backend.audio.listener import YamnetListener
from backend.custom.profile_store import DEFAULT_MATCH_THRESHOLD, DEFAULT_ROOT, YamnetProfileStore
from contracts.types import CATEGORIES, WindowResult


def countdown(label: str) -> None:
    print(f"{label} in 3...", end="", flush=True)
    for n in (2, 1):
        time.sleep(1)
        print(f" {n}...", end="", flush=True)
    time.sleep(1)
    print(" RECORDING", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a custom sound profile and live-test it.")
    parser.add_argument("--name", default="Demo sound")
    parser.add_argument("--category", default="doorbell", choices=CATEGORIES)
    parser.add_argument("--examples", type=int, default=5, help="number of mic recordings")
    parser.add_argument("--seconds", type=float, default=3, help="length of each recording")
    parser.add_argument("--from-wav", metavar="PATH", help="enrol from an audio file instead of the mic")
    parser.add_argument("--live", type=float, default=20, help="live-test seconds; 0 = until Ctrl+C")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="match threshold")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="profile folder")
    parser.add_argument("--keep", action="store_true", help="keep the profile after the demo")
    args = parser.parse_args()

    print("Loading YAMNet...", flush=True)
    store = YamnetProfileStore(args.root, threshold=args.threshold)
    store.embed_audio([0.0])  # load the model now, not during the first recording
    pid = store.create(args.name, args.category)
    print(f"Created profile {pid}: {args.name!r} ({args.category})\n")

    try:
        if args.from_wav:
            from backend.ml.demo_yamnet import load_wav

            n = store.add_example_from_audio(pid, load_wav(args.from_wav))
            print(f"Enrolled from {args.from_wav}: {n} examples")
        else:
            for i in range(1, args.examples + 1):
                countdown(f"Example {i}/{args.examples}: make the sound")
                n = store.add_example_from_mic(pid, args.seconds)
                print(f"  -> {n} examples so far (silent windows are dropped)\n")
        if next(p for p in store.list() if p["id"] == pid)["examples"] == 0:
            print("No sound was recorded (every window was silent). Make the sound louder or closer.")

        print(f"\nLive test for {args.live:g} s: make the sound again (and other sounds to compare). "
              f"Match threshold {store.threshold:.2f}.\n", flush=True)

        def show(w: WindowResult) -> None:
            clock = time.strftime("%H:%M:%S", time.localtime(w.timestamp))
            if w.skipped:
                print(f"{clock}  skipped", flush=True)
                return
            sim = store.similarity(pid, w.embedding)
            hit = next((m for m in w.custom_matches if m.profile_id == pid), None)
            bar = "#" * int(sim * 30)
            print(f"{clock}  similarity {sim:.2f}  {'MATCH' if hit else '     '}  {bar}", flush=True)

        listener = YamnetListener(show, profile_store=store)
        listener.start()
        start = time.monotonic()
        try:
            while args.live <= 0 or time.monotonic() - start < args.live:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
    finally:
        if args.keep:
            print(f"\nKept profile {pid} in {args.root}")
        else:
            store.delete(pid)
            print(f"\nDeleted demo profile {pid} (use --keep to keep it)")


if __name__ == "__main__":
    main()
