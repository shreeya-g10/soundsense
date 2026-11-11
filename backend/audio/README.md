# backend/audio/

**Owner:** Shreeya (Role 1)

Microphone capture (sounddevice) and audio preprocessing (Librosa). Audio stays in memory only — never written to disk.

## Part S1: microphone capture (`capture.py`)

- `MicCapture(on_window, device=None)` opens the mic with sounddevice at 16 kHz mono float32 and calls `on_window(samples, timestamp)` on a background thread.
- Windows are 0.96 s (15 360 samples), one every 0.48 s (50% overlap), which is YAMNet's input size.
- If a mic can't run at 16 kHz, it captures at the device's own rate and resamples with `soxr` (installed by librosa).
- `Windower` does the window cutting with plain numpy and is unit-tested without a mic.

Demo (from the repo root, nothing is saved):

```bash
python -m backend.audio.demo_capture                # 15 s live meter
python -m backend.audio.demo_capture --list-devices
python -m backend.audio.demo_capture --device 0 --seconds 0   # until Ctrl+C
```

Tests: `python -m pytest backend/tests/test_role1`
