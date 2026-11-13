# backend/audio/

**Owner:** Shreeya (Role 1)

Microphone capture (sounddevice) and audio preprocessing (Librosa). Audio stays in memory only — never written to disk.

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

Demo: `python -m backend.audio.demo_listener [--no-calibrate] [--seconds 0] [--no-cpu-saver]`

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
