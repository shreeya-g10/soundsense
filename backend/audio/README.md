# backend/audio/

**Owner:** Shreeya (Role 1)

Microphone capture (sounddevice) and audio preprocessing (Librosa). Audio stays in memory only — never written to disk.

## Part S1: microphone capture (`capture.py`)

- `MicCapture(on_window, device=None)` opens the mic with sounddevice at 16 kHz mono float32 and calls `on_window(samples, timestamp)` on a background thread.
- Windows are 0.96 s (15 360 samples), one every 0.48 s (50% overlap), which is YAMNet's input size.
- If a mic can't run at 16 kHz, it captures at the device's own rate and resamples with `soxr` (installed by librosa).
- `Windower` does the window cutting with plain numpy and is unit-tested without a mic.

## Part S2: silence gate / CPU saver (`silence.py`)

- `SilenceGate(threshold=0.003, enabled=True, margin=2.0)`: `gate.check(samples)` returns `GateDecision(skipped, rms, calibrating)`. Below the threshold, the window is skipped and YAMNet won't run.
- `enabled` is the `cpu_saver` setting. When it's off, nothing is skipped, but quiet windows are still counted as `below_threshold`.
- Auto-calibration: `gate.start_calibration()` measures the next ~2 s (3 windows) of room noise and sets `threshold = ambient RMS x margin`, clamped to 0.0005–0.05. Calibration windows are never skipped. `gate.calibrate(clip)` does the same from an array.
- `gate.stats()` returns `total`, `skipped`, `active`, `below_threshold`, `skip_rate`, `threshold` and `enabled` for Experiment 8. `reset_stats()` clears the counts.

Demo (from the repo root, nothing is saved):

```bash
python -m backend.audio.demo_capture                # 15 s live meter
python -m backend.audio.demo_capture --list-devices
python -m backend.audio.demo_capture --device 0 --seconds 0   # until Ctrl+C
python -m backend.audio.demo_capture --calibrate               # S2: calibrate threshold from room noise
python -m backend.audio.demo_capture --threshold 0.01
python -m backend.audio.demo_capture --no-cpu-saver            # S2: gate off, nothing skipped
```

Tests: `python -m pytest backend/tests/test_role1`

## Part S4: the real AudioListener (`listener.py`)

`YamnetListener(on_window, device=None, *, room="Laptop", cpu_saver=True, threshold=0.003, calibrate=False, profile_store=None)` implements `contracts.types.AudioListener`:

mic → `MicCapture` thread (windows + silence gate) → `YamnetListener` thread (YAMNet + category map) → `on_window(WindowResult)`

- Skipped windows follow the contract: empty dicts, `embedding=None`, `custom_matches=[]`, `inference_ms=0`.
- `start()` returns immediately. On the first start, YAMNet loads on the worker thread while windows queue up.
- Windows arrive in capture order. If inference falls ~15 s behind, the oldest windows are dropped and counted.
- `listener.gate.enabled` is the `cpu_saver` setting. `listener.stats()` returns the gate counts plus dropped windows and mic overflows.
- `profile_store` is accepted now, but `custom_matches` stays `[]` until S5.

Demo: `python -m backend.audio.demo_listener [--calibrate] [--seconds 0] [--no-cpu-saver]`
