# backend/audio/

**Owner:** Shreeya (Role 1)

Microphone capture (sounddevice) and audio preprocessing (Librosa). Audio stays in memory only — never written to disk.

## How to use

| Module | What it is | Use it for |
|---|---|---|
| `listener.py` | `YamnetListener`: the real `contracts.types.AudioListener` (mic → silence gate → YAMNet → category map → custom matching → `WindowResult`) | the app's laptop mic |
| `remote_input.py` | `YamnetRemoteInput`: the real `contracts.types.RemoteAudioInput` (phone frames → the same pipeline, per room) | the `/ws/listener` WebSocket (Phase 5) |
| `capture.py` | `MicCapture` (16 kHz mono windows on a thread), `Windower` (0.96 s windows every 0.48 s) | low-level capture; tests |
| `silence.py` | `SilenceGate` (CPU saver with auto-calibration), `rms`, `rms_to_db` | skipping silent windows; Experiment 8 counts |
| `demo_capture.py`, `demo_listener.py` | command-line demos (below) | checking a mic by hand |

**Wiring it into the backend.** This was checked against Vaidehi's `BackendRuntime`. `BackendRuntime` builds its engine itself, so build the runtime with the real profile store, then swap in the real inputs before `start()`:

```python
from backend.api.runtime import BackendRuntime
from backend.audio.listener import YamnetListener
from backend.audio.remote_input import YamnetRemoteInput
from backend.custom.profile_store import YamnetProfileStore

store = YamnetProfileStore()                      # data/profiles/ (embeddings only)
rt = BackendRuntime(profile_store=store)
rt.listener = YamnetListener(rt.engine.on_window, profile_store=store)        # cpu_saver=True, calibrate=True
rt.remote_audio = YamnetRemoteInput(rt.engine.on_window, profile_store=store)
```

- **Settings:** `rt.listener.gate.enabled = settings["cpu_saver"]` follows the CPU-saver setting.
- **Phone audio:** `rt.remote_audio.push(frame_bytes, room)` takes each `/ws/listener` frame.
- **Shutdown:** call `rt.remote_audio.close()` at shutdown.
- **Callbacks:** `on_window` is called from a worker thread and must return quickly. See `contracts/types.py`.

**Live check of a mic, nothing saved:**

```bash
python -m backend.audio.demo_capture --list-devices
python -m backend.audio.demo_capture --seconds 0          # RMS meter, CALIB / SKIPPED / ACTIVE per window
python -m backend.audio.demo_listener --seconds 0         # full pipeline: best category, top-3 YAMNet classes
python -m backend.audio.demo_listener --log               # also data/logs/listener-*.csv (top-3 classes + scores, no audio)
```

## Part S1: microphone capture (`capture.py`)

- `MicCapture(on_window, device=None)` opens the mic with sounddevice at 16 kHz mono float32 and calls `on_window(samples, timestamp)` on a background thread.
- Windows are 0.96 s (15 360 samples), one every 0.48 s (50% overlap), which is YAMNet's input size.
- If a mic can't run at 16 kHz, it captures at the device's own rate and resamples with `soxr` (installed by librosa).
- `Windower` does the window cutting with plain numpy and is unit-tested without a mic.

## Part S2: silence gate / CPU saver (`silence.py`)

- `SilenceGate(threshold=0.008, enabled=True, margin=2.0)`: `gate.check(samples)` returns `GateDecision(skipped, rms, calibrating)`. Below the threshold, the window is skipped and YAMNet won't run.
- `enabled` is the `cpu_saver` setting. When it's off, nothing is skipped, but quiet windows are still counted as `below_threshold`.
- Auto-calibration: `gate.start_calibration()` measures the next ~2 s (3 windows) of room noise and sets `threshold = ambient RMS x margin`, clamped to 0.0005–0.05. Calibration windows are never skipped. `gate.calibrate(clip)` does the same from an array.
- `gate.stats()` returns `total`, `skipped`, `active`, `below_threshold`, `skip_rate`, `threshold` and `enabled` for Experiment 8. `reset_stats()` clears the counts.

Demo (from the repo root, nothing is saved):

```bash
python -m backend.audio.demo_capture                # 15 s live meter
python -m backend.audio.demo_capture --list-devices
python -m backend.audio.demo_capture --device 0 --seconds 0   # until Ctrl+C
python -m backend.audio.demo_capture                           # S2: calibrates threshold from room noise
python -m backend.audio.demo_capture --no-calibrate --threshold 0.01
python -m backend.audio.demo_capture --no-cpu-saver            # S2: gate off, nothing skipped
```

Tests: `python -m pytest backend/tests/test_role1`

## Part S4: the real AudioListener (`listener.py`)

`YamnetListener(on_window, device=None, *, room="Laptop", cpu_saver=True, threshold=0.008, calibrate=True, profile_store=None)` implements `contracts.types.AudioListener`:

mic → `MicCapture` thread (windows + silence gate) → `YamnetListener` thread (YAMNet + category map) → `on_window(WindowResult)`

- Skipped windows follow the contract: empty dicts, `embedding=None`, `custom_matches=[]`, `inference_ms=0`.
- `start()` returns immediately. On the first start, YAMNet loads on the worker thread while windows queue up.
- Windows arrive in capture order. If inference falls ~15 s behind, the oldest windows are dropped and counted.
- `listener.gate.enabled` is the `cpu_saver` setting. `listener.stats()` returns the gate counts plus dropped windows and mic overflows.
- `profile_store`: if it has `match(embedding)` (the real `YamnetProfileStore` from S5 does), `custom_matches` is filled on every active window. If matching fails, the window is still delivered with `custom_matches=[]`.

Demo: `python -m backend.audio.demo_listener [--no-calibrate] [--threshold T] [--seconds 0] [--no-cpu-saver] [--log]`.
- **Each line** shows the best category, or `—` if every category is below 0.15. It also shows the context scores and YAMNet's raw top-3 classes.
- **`--log`** saves the top-3 classes and category scores of every active window to `data/logs/listener-<time>.csv` (git-ignored, no audio).
- **Calibration windows** (the first ~2 s after `start()`) are added to the profile store's background bank (`add_background`) when the store supports it.

## Part S6: RemoteAudioInput (`remote_input.py`, Phase 5)

`YamnetRemoteInput(on_window, *, profile_store=None, cpu_saver=True, threshold=0.008, calibrate=True)` implements `contracts.types.RemoteAudioInput`.
- **Calling it:** Vaidehi's `/ws/listener` handler calls `remote.push(frame_bytes, room)` once per binary frame. The frame is little-endian Float32 mono 16 kHz audio of any length that's a multiple of 4 bytes; anything else raises `ValueError`. `push()` only decodes the bytes and queues them, so it's safe inside an async handler.
- **Processing:** one "RemoteAudioInput" worker thread starts on the first push. It buffers each room separately, cuts 0.96 s windows every 0.48 s (50% overlap) and runs the same pipeline as `YamnetListener` (`listener.analyse_window`): silence gate, YAMNet, category mapping and custom matching. The results arrive at `on_window` as `WindowResult(room=room)`.
- **Silence gate per room:** each room has its own gate, calibrated from that room's first ~2 s. Use `remote.gate(room)` to reach it. Calibration windows also go into the background bank.
- **Timestamps:** a window's timestamp is the arrival time of the chunk that completed it, minus the audio that came after the window's end. It is kept monotonic per room.
- **Falling behind:** if analysis falls ~32 s behind (64 queued chunks), the oldest chunk is dropped (counted in `dropped_chunks`) and that room's buffer restarts, so no window spans a gap.
- **Stopping:** `remote.close()` blocks until queued chunks are analysed and `on_window` won't be called again. A later push starts a new worker. `remote.stats()` gives per-room gate counts and `dropped_chunks`. `remote.error` is set if YAMNet fails to load.
- **Privacy:** audio is processed in memory only and never saved.

The listener now passes each window's audio, in memory only, to `profile_store.match(embedding, samples=...)` when the store accepts it, so MFCC profiles can be matched.
