"""Demo for Parts S5/S5f: teach a custom sound, then live-test it or test it on WAV files.

Owner: Shreeya (Role 1).

Run from the repo root:

    python -m backend.custom.demo_profiles                       # 5 x 3 s examples + 5 s room sound, 60 s live test
    python -m backend.custom.demo_profiles --name "My Kitchen Cooker" --category pressure_cooker --keep
    python -m backend.custom.demo_profiles --from-wav sound.wav  # enrol from a file instead of the mic
    python -m backend.custom.demo_profiles --seconds 0           # live test until Ctrl+C
    python -m backend.custom.demo_profiles --list                # saved profiles
    python -m backend.custom.demo_profiles --test "My Kitchen Cooker"   # live-test a saved profile (id or name)
    python -m backend.custom.demo_profiles --test "My Kitchen Cooker" --test-wav data/indian/test/
    python -m backend.custom.demo_profiles --record-test cooker  # record a test WAV (with consent)
    python -m backend.custom.demo_profiles --scoring raw --no-augment   # Experiment 6 comparisons

Enrolment: the examples are recorded into memory first, then 5 s of normal
room sound (no target sound). The room sound goes into the background bank,
and is the noise used to augment the examples (10 dB SNR copies); then the
examples are enrolled. Only embeddings are stored (data/profiles/).

A profile created by the demo is deleted at the end unless --keep is given;
--test never deletes the profile it tests.

--record-test is the ONE place SoundSense writes audio to disk: a
developer-only tool for the evaluation dataset in data/indian/test/<label>/
(git-ignored), used only with the consent of everyone audible. The running
app never calls it. (Pending a team-agreed exception to AGENTS.md rule 4.)
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.audio.capture import SAMPLE_RATE, WINDOW_SAMPLES, Windower
from backend.audio.demo_listener import DemoListener, Top
from backend.custom.profile_store import (
    DEFAULT_ROOT,
    REPO_ROOT,
    SCORING_CENTERED,
    SCORING_RAW,
    THRESHOLDS,
    ScoreDetail,
    YamnetProfileStore,
    record_from_mic,
)
from contracts.types import CATEGORIES, WindowResult

TEST_ROOT = REPO_ROOT / "data" / "indian" / "test"  # evaluation dataset, git-ignored (data/*)
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


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


def _score_text(similarity: float, matched: bool, top: Top, background: float | None) -> str:
    yam = f"[{top[0][0]} {top[0][1]:.2f}]" if top else "[]"
    bg = "" if background is None else f"bg {background:5.2f}"
    bar = "#" * int(max(similarity, 0.0) * 30)
    return f"similarity {similarity:.2f} {'MATCH' if matched else '     '}  {yam:32s} {bg:8s} {bar}"


def format_line(w: WindowResult, similarity: float | None, matched: bool, top: Top,
                background: float | None = None) -> str:
    clock = time.strftime("%H:%M:%S", time.localtime(w.timestamp))
    if w.skipped or similarity is None:
        return f"{clock}  skipped"
    return f"{clock}  {_score_text(similarity, matched, top, background)}"


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
        d = store.score(pid, w.embedding)
        matched = any(m.profile_id == pid for m in w.custom_matches)
        summary.add(d.similarity, matched)
        print(format_line(w, d.similarity, matched, top, d.background), flush=True)

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


# --- offline evaluation on WAV files (--test-wav) -----------------------------

def audio_paths(path: str | Path) -> list[Path]:
    """``path`` itself if it is a file, else every audio file under it (sorted)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"--test-wav: {path} does not exist")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())


@dataclass
class WindowEval:
    start_s: float
    detail: ScoreDetail
    top: Top


@dataclass
class FileEval:
    path: Path
    windows: list[WindowEval]
    total_windows: int

    @property
    def silent(self) -> int:
        return self.total_windows - len(self.windows)

    @property
    def max(self) -> float:
        return max((w.detail.similarity for w in self.windows), default=0.0)

    @property
    def mean(self) -> float:
        return float(np.mean([w.detail.similarity for w in self.windows])) if self.windows else 0.0

    @property
    def match_rate(self) -> float:
        return sum(w.detail.matched for w in self.windows) / len(self.windows) if self.windows else 0.0

    def summary(self) -> str:
        if not self.windows:
            return f"{self.path.name}: all {self.total_windows} windows silent"
        return (f"{self.path.name}: {len(self.windows)} windows ({self.silent} silent), "
                f"max {self.max:.2f}, mean {self.mean:.2f}, matched {self.match_rate:.0%}")


def evaluate_audio(store: YamnetProfileStore, pid: str, audio: np.ndarray, path: Path, model) -> FileEval:
    """Score every non-silent window of ``audio`` against profile ``pid``."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    padded = np.pad(audio, (0, max(0, WINDOW_SAMPLES - audio.size)))
    total = sum(1 for _ in Windower().feed(padded))
    names = getattr(model, "class_names", None)
    windows = []
    for start_s, w in store.active_windows(audio):
        out = model.infer(w)
        top: Top = []
        if names is not None and getattr(out, "scores", None) is not None:
            i = int(np.argmax(out.scores))
            top = [(names[i], float(out.scores[i]))]
        windows.append(WindowEval(start_s, store.score(pid, out.embedding), top))
    return FileEval(path, windows, total)


def evaluate_paths(store: YamnetProfileStore, pid: str, paths: list[Path], model, load=None,
                   echo=print) -> list[FileEval]:
    """Score each file, printing every window and a per-file summary."""
    if load is None:
        from backend.ml.demo_yamnet import load_wav as load
    results = []
    for path in paths:
        echo(f"\n{path}")
        result = evaluate_audio(store, pid, load(str(path)), path, model)
        for w in result.windows:
            d = w.detail
            echo(f"  {w.start_s:6.2f}s  {_score_text(d.similarity, d.matched, w.top, d.background)}")
        echo(f"  -> {result.summary()}")
        results.append(result)
    return results


def overall_summary(results: list[FileEval]) -> str:
    windows = [w for r in results for w in r.windows]
    if not windows:
        return f"Overall: {len(results)} files, no non-silent windows."
    sims = [w.detail.similarity for w in windows]
    matched_files = sum(r.match_rate > 0 for r in results)
    return (f"Overall: {len(results)} files ({matched_files} with a match), {len(windows)} windows, "
            f"max {max(sims):.2f}, mean {np.mean(sims):.2f}, "
            f"matched {sum(w.detail.matched for w in windows) / len(windows):.0%}")


# --- recording the evaluation dataset (--record-test) --------------------------

def safe_label(label: str) -> str:
    """Folder-safe label: letters, digits, '-' and '_' only."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip()).strip("_")
    if not clean:
        raise SystemExit(f"--record-test: label {label!r} has no usable characters")
    return clean


def clip_path(label: str, root: Path = TEST_ROOT, now: float | None = None) -> Path:
    label = safe_label(label)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return Path(root) / label / f"{label}-{stamp}.wav"


CONSENT_PROMPT = (
    "This saves a WAV to the SoundSense evaluation dataset on this computer (git-ignored).\n"
    "Only record with the consent of everyone who can be heard.\n"
    "Type 'yes' to record: "
)


def confirm_consent(ask=input) -> bool:
    try:
        return ask(CONSENT_PROMPT).strip().lower() == "yes"
    except EOFError:
        return False


def save_test_clip(audio: np.ndarray, path: Path) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, np.float32), SAMPLE_RATE, subtype="PCM_16")
    return path


def record_test(label: str, seconds: float, ask=input) -> Path | None:
    path = clip_path(label)
    print(f"Will record {seconds:g} s to {path}")
    if not confirm_consent(ask):
        print("Not recorded.")
        return None
    countdown("Make the sound")
    audio = record_from_mic(seconds)
    save_test_clip(audio, path)
    print(f"Saved {path} (rms {float(np.sqrt(np.mean(audio ** 2))):.4f})")
    return path


# --- main -----------------------------------------------------------------

def enrol(store: YamnetProfileStore, pid: str, args) -> None:
    """Examples into memory, then room sound, then enrol (so augmentation uses the room noise)."""
    if args.from_wav:
        from backend.ml.demo_yamnet import load_wav

        examples = [load_wav(args.from_wav)]
        print(f"Loaded {args.from_wav}")
    else:
        examples = []
        for i in range(1, args.examples + 1):
            countdown(f"Example {i}/{args.examples}: make the sound")
            examples.append(record_from_mic(args.record_seconds))
            n = len(store.active_windows(examples[-1]))
            print(f"  -> {n} non-silent windows (silent windows are dropped)\n")

    if args.room_seconds > 0:
        print(f"Now {args.room_seconds:g} s of NORMAL ROOM SOUND: stay quiet, don't make the target sound.")
        countdown("Room sound")
        added = store.add_background_from_audio(record_from_mic(args.room_seconds))
        counts = ", ".join(f"{k} {v}" for k, v in store.background_counts().items())
        print(f"  -> {added} room windows added to the background bank ({counts})\n")

    n = 0
    for audio in examples:
        n = store.add_example_from_audio(pid, audio)
    aug = " (+3 augmented copies each)" if store.augment else ""
    print(f"Enrolled: {n} examples{aug}")
    if n == 0:
        print("No sound was recorded (every window was silent). Make the sound louder or closer.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a custom sound profile and test it.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print saved profiles and exit")
    mode.add_argument("--test", metavar="ID_OR_NAME", help="skip recording; test a saved profile")
    mode.add_argument("--record-test", metavar="LABEL",
                      help=f"record a WAV into {TEST_ROOT.relative_to(REPO_ROOT)}/<LABEL>/ (asks for consent)")
    parser.add_argument("--name", default="Demo sound")
    parser.add_argument("--category", default="doorbell", choices=CATEGORIES)
    parser.add_argument("--examples", type=int, default=5, help="number of mic recordings")
    parser.add_argument("--record-seconds", type=float, default=3, help="length of each recording")
    parser.add_argument("--room-seconds", type=float, default=5,
                        help="normal room sound recorded for the background bank; 0 = skip")
    parser.add_argument("--from-wav", metavar="PATH", help="enrol from an audio file instead of the mic")
    parser.add_argument("--seconds", "--live", dest="seconds", type=float, default=60,
                        help="live-test length in seconds; 0 = until Ctrl+C (default 60)")
    parser.add_argument("--test-wav", metavar="FILE_OR_FOLDER",
                        help="score the profile against audio files instead of a live test")
    parser.add_argument("--clip-seconds", type=float, default=10, help="--record-test length (default 10)")
    parser.add_argument("--scoring", choices=[SCORING_CENTERED, SCORING_RAW], default=SCORING_CENTERED)
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"match threshold (default {THRESHOLDS[SCORING_CENTERED]} centered, "
                             f"{THRESHOLDS[SCORING_RAW]} raw)")
    parser.add_argument("--no-margin", action="store_true", help="don't require beating the background score")
    parser.add_argument("--no-augment", action="store_true", help="store recorded windows only")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="profile folder")
    parser.add_argument("--keep", action="store_true", help="keep the new profile after the demo")
    args = parser.parse_args()

    if args.record_test:
        record_test(args.record_test, args.clip_seconds)
        return

    store = YamnetProfileStore(args.root, threshold=args.threshold, scoring=args.scoring,
                               margin=not args.no_margin, augment=not args.no_augment)

    if args.list:
        print(format_profiles(store.list()))
        return

    wav_files = audio_paths(args.test_wav) if args.test_wav else None
    if wav_files == []:
        raise SystemExit(f"--test-wav: no audio files in {args.test_wav}")

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
            enrol(store, pid, args)

        rule = "" if args.no_margin else " and above the background score"
        print(f"\nScoring {store.scoring}, match = similarity >= {store.effective_threshold:.2f}{rule}. "
              f"Background bank: {store.background_size()} embeddings.")
        if wav_files is not None:
            results = evaluate_paths(store, pid, wav_files, store._get_model())
            print(f"\n{overall_summary(results)}")
        else:
            length = "until Ctrl+C" if args.seconds <= 0 else f"for {args.seconds:g} s"
            print(f"Live test {length}: make the sound again (and other sounds to compare). "
                  f"The first ~2 s calibrate the silence gate.\n", flush=True)
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
