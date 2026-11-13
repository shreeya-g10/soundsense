"""Demo for Part S5: teach a custom sound, then live-test it.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.custom.demo_profiles                       # 5 x 3 s mic examples, 60 s live test
    python -m backend.custom.demo_profiles --name "My Kitchen Cooker" --category pressure_cooker --keep
    python -m backend.custom.demo_profiles --from-wav sound.wav  # enrol from a file instead of the mic
    python -m backend.custom.demo_profiles --seconds 0           # live test until Ctrl+C
    python -m backend.custom.demo_profiles --list                # saved profiles
    python -m backend.custom.demo_profiles --test "My Kitchen Cooker"   # live-test a saved profile (id or name)

Only embeddings are stored (data/profiles/), never audio. A profile created
by the demo is deleted at the end unless --keep is given; --test never
deletes the profile it tests.

Each live line shows the similarity to the profile and the top YAMNet class
of that window, e.g. "similarity 0.84  MATCH  [Steam 0.41]".
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from backend.audio.demo_listener import DemoListener, Top
from backend.custom.profile_store import DEFAULT_MATCH_THRESHOLD, DEFAULT_ROOT, YamnetProfileStore
from contracts.types import CATEGORIES, WindowResult


def countdown(label: str) -> None:
    print(f"{label} in 3...", end="", flush=True)
    for n in (2, 1):
        time.sleep(1)
        print(f" {n}...", end="", flush=True)
    time.sleep(1)
    print(" RECORDING", flush=True)


def resolve_profile(store: YamnetProfileStore, key: str) -> dict:
    """The saved profile whose id or name (case-insensitive) is ``key``.

    Raises:
        SystemExit: if no profile, or more than one profile by that name, matches.
    """
    profiles = store.list()
    by_id = [p for p in profiles if p["id"] == key]
    if by_id:
        return by_id[0]
    by_name = [p for p in profiles if p["name"].casefold() == key.casefold()]
    if len(by_name) == 1:
        return by_name[0]
    if by_name:
        ids = ", ".join(p["id"] for p in by_name)
        raise SystemExit(f"{len(by_name)} profiles are named {key!r} ({ids}); pass the id instead.")
    raise SystemExit(f"No saved profile with id or name {key!r}. Use --list to see them.")


def format_profiles(profiles: list[dict]) -> str:
    if not profiles:
        return "No saved profiles."
    rows = [("ID", "NAME", "CATEGORY", "EXAMPLES")] + [
        (p["id"], p["name"], p["category"], str(p["examples"])) for p in profiles
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)


def format_line(w: WindowResult, similarity: float | None, matched: bool, top: Top) -> str:
    clock = time.strftime("%H:%M:%S", time.localtime(w.timestamp))
    if w.skipped or similarity is None:
        return f"{clock}  skipped"
    yam = f"[{top[0][0]} {top[0][1]:.2f}]" if top else "[]"
    bar = "#" * int(max(similarity, 0.0) * 30)
    return f"{clock}  similarity {similarity:.2f} {'MATCH' if matched else '     '}  {yam:32s} {bar}"


@dataclass
class LiveSummary:
    """Counts over the active (non-skipped) windows of a live test."""

    similarities: list[float] = field(default_factory=list)
    matches: int = 0

    def add(self, similarity: float, matched: bool) -> None:
        self.similarities.append(similarity)
        self.matches += int(matched)

    def __str__(self) -> str:
        n = len(self.similarities)
        if n == 0:
            return "Summary: no windows tested (everything was skipped as silent)."
        return (
            f"Summary: {n} windows tested, {self.matches} matches ({self.matches / n:.0%}), "
            f"similarity max {max(self.similarities):.2f}, mean {sum(self.similarities) / n:.2f}"
        )


def live_test(store: YamnetProfileStore, pid: str, seconds: float) -> LiveSummary:
    summary = LiveSummary()

    def show(w: WindowResult, top: Top) -> None:
        if w.skipped:
            print(format_line(w, None, False, top), flush=True)
            return
        sim = store.similarity(pid, w.embedding)
        matched = any(m.profile_id == pid for m in w.custom_matches)
        summary.add(sim, matched)
        print(format_line(w, sim, matched, top), flush=True)

    listener = DemoListener(show, profile_store=store)
    try:
        listener.start()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\nCheck the terminal has microphone permission.")
    start = time.monotonic()
    try:
        while seconds <= 0 or time.monotonic() - start < seconds:
            if listener.error:
                raise SystemExit(f"Listener failed: {listener.error}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a custom sound profile and live-test it.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print saved profiles and exit")
    mode.add_argument("--test", metavar="ID_OR_NAME", help="skip recording; live-test a saved profile")
    parser.add_argument("--name", default="Demo sound")
    parser.add_argument("--category", default="doorbell", choices=CATEGORIES)
    parser.add_argument("--examples", type=int, default=5, help="number of mic recordings")
    parser.add_argument("--record-seconds", type=float, default=3, help="length of each recording")
    parser.add_argument("--from-wav", metavar="PATH", help="enrol from an audio file instead of the mic")
    parser.add_argument("--seconds", "--live", dest="seconds", type=float, default=60,
                        help="live-test length in seconds; 0 = until Ctrl+C (default 60)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD, help="match threshold")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="profile folder")
    parser.add_argument("--keep", action="store_true", help="keep the new profile after the demo")
    args = parser.parse_args()

    store = YamnetProfileStore(args.root, threshold=args.threshold)

    if args.list:
        print(format_profiles(store.list()))
        return

    created = None
    if args.test:
        profile = resolve_profile(store, args.test)
        pid = profile["id"]
        print(f"Testing saved profile {pid}: {profile['name']!r} ({profile['category']}, "
              f"{profile['examples']} examples). It will not be deleted.")
        if profile["examples"] == 0:
            print("This profile has no examples, so every similarity will be 0.")

    print("Loading YAMNet...", flush=True)
    store.embed_audio([0.0])  # load the model now, not during the first recording

    try:
        if not args.test:
            pid = created = store.create(args.name, args.category)
            print(f"Created profile {pid}: {args.name!r} ({args.category})")
            if args.keep:
                print("It will be kept (--keep).\n")
            else:
                print("It will be DELETED at the end; rerun with --keep to save it.\n")

            if args.from_wav:
                from backend.ml.demo_yamnet import load_wav

                n = store.add_example_from_audio(pid, load_wav(args.from_wav))
                print(f"Enrolled from {args.from_wav}: {n} examples")
            else:
                for i in range(1, args.examples + 1):
                    countdown(f"Example {i}/{args.examples}: make the sound")
                    n = store.add_example_from_mic(pid, args.record_seconds)
                    print(f"  -> {n} examples so far (silent windows are dropped)\n")
            if resolve_profile(store, pid)["examples"] == 0:
                print("No sound was recorded (every window was silent). Make the sound louder or closer.")

        length = "until Ctrl+C" if args.seconds <= 0 else f"for {args.seconds:g} s"
        print(f"\nLive test {length}: make the sound again (and other sounds to compare). "
              f"Match threshold {store.threshold:.2f}. The first ~2 s calibrate the silence gate.\n",
              flush=True)
        summary = live_test(store, pid, args.seconds)
        print(f"\n{summary}")
    finally:
        if created is not None:
            if args.keep:
                print(f"\nKept profile {created} in {args.root}")
            else:
                store.delete(created)
                print(f"\nDeleted demo profile {created} (use --keep to keep it)")


if __name__ == "__main__":
    main()
