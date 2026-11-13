# backend/tests/test_role1/

**Owner:** Shreeya (Role 1)

Tests for `backend/audio/`, `backend/ml/`, `backend/custom/` and `eval/model/`.

```bash
python -m pytest backend/tests/test_role1           # ~30 s on a laptop
python -m pytest backend/tests                      # role 1 + role 2
python -m pytest backend/tests/test_role1/test_profile_store.py -k feedback   # one file / topic
```

- **YAMNet:** tests that need YAMNet load it from `data/models/yamnet/`, downloading it on the first run. If it can't be loaded (offline on a fresh clone), they are **skipped**, not failed.
- **No mic, network or real data needed:** mics are faked (`FakeMic`, fake recorders), archives and datasets are built in temporary folders, and the experiment scripts run end to end on a synthetic mini-dataset (`run_all.smoke`).

| File | Covers |
|---|---|
| `test_capture.py` | `MicCapture` and `Windower`: 0.96 s windows, 0.48 s hop, resampling |
| `test_silence.py` | `SilenceGate`: threshold, calibration (default on), stats |
| `test_yamnet.py` | YAMNet loading, inference shapes, `top_k` |
| `test_listener.py` | category mapping vs the real class map, `YamnetListener` (WindowResult contract, threading, calibration → background bank), `demo_listener` output and CSV log |
| `test_remote_input.py` | `YamnetRemoteInput`: byte validation, per-room buffering and calibration, 50% overlap, timestamps, dropping, close |
| `test_profile_store.py` | `YamnetProfileStore`: contract, persistence, centered scoring, margin and negatives rules, augmentation, feedback, `is_negative`, MFCC baseline, `demo_profiles` helpers |
| `test_eval_datasets.py` | Drive naming guide, dataset loader, phone formats (m4a/mp3), ESC-50 downloader |
| `test_eval_metrics.py` | metric functions (confusion matrix, per-class P/R/F1, macro F1, detection, ROC AUC) |
| `test_eval_experiments.py` | Experiments 1, 4, 5 (TV-guard mixes), 6, 8 and noise robustness end to end; skipping missing data; every CSV column documented |
| `test_eval_supplementary.py` | `rangezip` (split zips, zip64, CRC), `mixer` SNR, FSD50K selection rules, Freesound licence filter, DEMAND download |
