# eval/model/

**Owner:** Shreeya (Role 1)

Model evaluation: accuracy, precision/recall, confusion matrices per sound class.

## Evaluation datasets

Everything under `data/` is git-ignored. Only the scripts in `eval/model/` are committed. Never commit audio.

### Our Indian recordings (`data/indian/`)

The folders match the team Drive folder exactly. Download the Drive folder and place it as `data/indian/`:

```
data/indian/
  profile/<category>/                     clips for teaching custom profiles (enrolment)
  test/<category>/                        test clips of a target sound
  test/hard_negatives/<type>/             sounds that must NOT alert
```

- `<category>` is a SoundSense category ID: `smoke_alarm`, `glass_break`, `pressure_cooker`, `baby_cry`, `door_knock`, `doorbell`, `dog_bark`, `appliance_beep` or `water_running`.
- `<type>` is a short lowercase name for a hard negative, e.g. `mixer_grinder`, `kettle`, `tv_ads`, `utensils`, `phone_ringtone`, `temple_bell`, `pressure_horn`.
- Optional: test clips may sit one level deeper by condition, as `test/<category>/<condition>/`. The loader accepts both. The condition is always read from the file name, and a mismatch with the folder is reported.

**File names:** `<category>_<name>_<condition>_<distance>_<n>.<ext>`. In `hard_negatives/<type>/` the name starts with `<type>` instead.

| Part | Rule | Examples |
|---|---|---|
| `category` / `type` | the folder name | `pressure_cooker`, `kettle` |
| `name` | what made the sound: brand, device or room, lowercase; may be left out | `hawkins`, `mainDoor`, `amma` |
| `condition` | `quiet` (nothing else on), `tv` (TV or music playing), `fan` (fan or AC on), `far` (another room) | `tv` |
| `distance` | metres from the phone to the source, then `m` | `1m`, `2.5m`, `6m` |
| `n` | take number for this combination, starting at 1 | `1`, `2` |

Examples: `pressure_cooker_hawkins_tv_3m_2.m4a`, `door_knock_main_quiet_1m_1.wav`, `baby_cry_quiet_1m_3.mp3` (no name), `hard_negatives/kettle/kettle_steel_fan_2m_1.m4a`. The last three parts are read from the right, so a name may contain underscores (`main_door`). `mainDoor` is easier to read, though.

**Recording:** phone recordings in m4a, mp3 or wav are all fine (also flac, ogg, aac, aiff, caf); the loader converts them to 16 kHz mono. Record only with the consent of everyone who can be heard. For `profile/`, record 3–5 clean takes of 3–5 s, at the distance you would normally be from the sound.

**ffmpeg** is needed to decode m4a/aac. The loader uses librosa, which falls back to audioread, which uses ffmpeg:

```bash
brew install ffmpeg                # macOS (Core Audio also decodes m4a on macOS, but install it anyway)
sudo apt install ffmpeg            # Ubuntu/Debian
winget install Gyan.FFmpeg         # Windows (then reopen the terminal)
```

WAV, FLAC, OGG and MP3 don't need ffmpeg; soundfile reads them.

**Check the folder** before evaluating. This lists every naming problem and prints a clip count per label and condition. Add `--decode` to also try decoding every file:

```bash
python -m eval.model.datasets
python -m eval.model.datasets --decode
```

### ESC-50 subset (`data/esc50/`)

```bash
python -m eval.model.download_esc50 --list    # the mapping only
python -m eval.model.download_esc50           # ~600 MB download; keeps 1240 clips (~550 MB), deletes the archive
python -m eval.model.download_esc50 --zip ~/Downloads/ESC-50-master.zip   # if you already have it
```

This downloads the official archive from https://github.com/karolpiczak/ESC-50 and keeps only these categories. It writes `data/esc50/audio/*.wav` and a filtered `data/esc50/meta/esc50.csv`, in the original format, which `build_background_bank.py` also reads.
- **Targets:** `crying_baby` → `baby_cry`, `dog` → `dog_bark`, `door_wood_knock` → `door_knock`, `glass_breaking` → `glass_break`, `pouring_water` → `water_running`. ESC-50 has nothing for `doorbell`, `pressure_cooker`, `smoke_alarm` or `appliance_beep`; those come from our recordings.
- **Negatives** (label `negative`, for false-alert tests): 26 categories.
  - Close to a target: `clock_alarm`, `siren`, `church_bells`, `door_wood_creaks`, `toilet_flush`, `water_drops`, `washing_machine`, `vacuum_cleaner`, `clock_tick`, `keyboard_typing`, `mouse_click`, `footsteps`, `can_opening`, `cat`.
  - Speech-like (ESC-50 has no plain speech): `laughing`, `coughing`, `sneezing`, `breathing`, `snoring`.
  - Background: `rain`, `wind`, `crackling_fire`, `drinking_sipping`, `brushing_teeth`, `car_horn`, `fireworks`.
  - `clock_alarm` is a negative on purpose. Generic "Alarm" is left unmapped so that alarm clocks don't raise smoke alarms (S4), and these clips test exactly that.

The list lives in `eval/model/datasets.py` (`ESC50_TO_CATEGORY`, `ESC50_NEGATIVES`). ESC-50 is licensed **CC BY-NC 3.0**: non-commercial use only, and cite Piczak, "ESC: Dataset for Environmental Sound Classification", ACM MM 2015.

### Loader (`eval/model/datasets.py`)

```python
from eval.model.datasets import iter_test_files, list_test_clips, list_profile_clips

for audio, label, condition, source in iter_test_files():   # every test file, decoded to 16 kHz mono float32
    ...  # label: category ID or "negative"; condition: quiet/tv/fan/far, "clean" for ESC-50; source: "indian" or "esc50"

clips = list_test_clips()           # metadata only (path, label, condition, source, negative_type, name, distance, take); clip.load() decodes
cooker = list_profile_clips(category="pressure_cooker")     # enrolment clips
```

- **Order:** `iter_test_files` yields our recordings first (`test/` and `hard_negatives/`), then ESC-50 if it has been downloaded.
- **Bad names:** a file whose name breaks the rule is still loaded, with condition `"unknown"`.
- **Unreadable files** are logged and skipped.

## Custom-sound threshold (Part S5f, Experiment 6)

`backend/custom/profile_store.py` scores custom sounds with `scoring="centered"` (default) or `"raw"`, and has a threshold for each: `RAW_MATCH_THRESHOLD = 0.80` (from S5) and `CENTERED_MATCH_THRESHOLD = 0.89`. While the background bank is empty, centered scoring is the same as raw, so the default threshold falls back to 0.80 (`effective_threshold`).

**How 0.89 was chosen (provisional).** We have no recordings of real household sounds yet, so the value was chosen with `tune_custom_threshold.py` on a **synthetic** set:

- 8 macOS system sounds (Glass, Ping, Hero, Submarine, Funk, Purr, Sosumi, Blow), each placed in a 3 s clip 4 times at gain 1.0, 0.5, 0.3 and 0.8, in noise at 25, 15, 20 and 10 dB SNR.
- 4 `say` voices (Daniel, Aman, Flo, Fred) × 4 sentences, with the same 4 gain/noise variants.
- 4 background clips of 5 s each: pink noise, brown noise, pink noise with 50 Hz hum, pink noise with fan tone, all at ~0.004 RMS.

Each label was enrolled on 2 files and tested against the other 2 (24 positive trials) and every other file (576 negative trials). A file counts as detected if any window matches. Best F1 per setting:

| Setting | Raw: best t, P / R / F1 | Centered: best t, P / R / F1 |
|---|---|---|
| margin + augmentation (default) | 0.84: 0.53 / 0.75 / 0.62 | **0.89: 0.68 / 0.79 / 0.73** |
| no margin | 0.84: 0.50 / 0.75 / 0.60 | 0.89: 0.68 / 0.79 / 0.73 |
| no augmentation | 0.80: 0.69 / 0.38 / 0.49 | 0.82: 0.29 / 0.75 / 0.42 |

Centered beat raw, and augmentation gave the biggest gain. The margin rule made no difference here because synthetic noise doesn't resemble any target. Real room sound (TV, talking, fans) is where the margin rule should help. Under centered scoring the precision drops fast below the best threshold (t = 0.85: P 0.29, R 0.92).

**Caveats.** The test variants use the same kinds of gain and noise changes as the augmentation does, so the augmentation gain is probably overstated. The hardest negatives were the other voices and chimes (max 0.92). **Re-tune on real data** with `--record-test` clips and a recording of the room:

```bash
python -m eval.model.tune_custom_threshold data/indian/test              # data/indian/test/<label>/*.wav + background/*.wav
python -m eval.model.tune_custom_threshold data/indian/test --no-margin --no-augment
```

Then update `CENTERED_MATCH_THRESHOLD` and this table.

### MFCC baseline threshold (Part S6)

`MFCC_MATCH_THRESHOLD = 0.991` (provisional) comes from the same synthetic set: `python -m eval.model.tune_custom_threshold <set> --method mfcc`. The tuner uses a 0.001 grid for MFCC, because MFCC cosines all sit close to 1.

| MFCC comparison | Best t | P / R / F1 |
|---|---|---|
| Full 40-number vector | 0.99 | 0.05 / 0.92 / 0.09: every pair ~1.00 |
| **Coefficient 0 ignored (default)** | **0.991** | **0.81 / 0.71 / 0.76** |
| Means only, coefficient 0 ignored | 0.99 | 0.21 / 0.83 / 0.33 |

Negatives still reach 0.99 at the 95th percentile, so the margin between a match and a non-match is a few thousandths. This makes the baseline brittle. Compare it with the embedding method in Experiment 6, and re-tune it on real data.

## Generic background bank (`build_background_bank.py`)

This script fills the `generic` source of `data/profiles/background.npz` with YAMNet embeddings of ESC-50 clips. It takes up to 8 clips per class and leaves out 11 classes that are close to a SoundSense category (dog, crying baby, door knock or creak, glass breaking, clock alarm, siren, church bells, water drops, pouring water, toilet flush). The bank is empty until you run it:

```bash
python -m eval.model.download_esc50          # once: ESC-50 subset into data/esc50/
python -m eval.model.build_background_bank    # reads data/esc50 by default
python -m eval.model.build_background_bank --clear
```

## Model experiments (Part S7)

One script per experiment, plus `run_all.py`. CSVs go to `eval/model/results/`, and they are committed so the report can chart them. If a dataset is missing, the script prints `[expN] SKIP <dataset>: <how to get it>` and carries on. Until the Indian recordings exist, everything runs on ESC-50.

```bash
python -m eval.model.download_esc50        # once
python -m eval.model.run_all               # every experiment, ~1-2 min on ESC-50 (laptop CPU)
python -m eval.model.run_all --smoke       # synthetic mini-dataset in a temp folder: checks every script end to end (~10 s)
python -m eval.model.exp1_yamnet --limit 100       # any one experiment; --limit N files for a quick look
```

The smoke test also runs in the test suite: `backend/tests/test_role1/test_eval_experiments.py`.

**Shared rules**
- **Windows:** 0.96 s every 0.48 s, as in the app. Silent windows (RMS < 0.008, the silence gate default) are dropped, because the app never classifies them; their count is in `silent_windows`.
- **Prediction:** the category with the highest YAMNet category score if it is ≥ 0.15 (`DETECTION_THRESHOLD`), else `none`. Negatives (ESC-50 negatives, `hard_negatives/`) have the true label `none`.
- **Window level vs clip level:** `level` = `window` scores every window. `clip` scores one prediction per file, like one alert: the category of the file's single highest-scoring window, or `none`.
- **Experiments 4 and 6:** profiles come from `data/indian/profile/` and are tested on `data/indian/test/`. A test file with the same path or the same bytes as a profile file is dropped (`removed_overlap`). Until those folders exist, an **ESC-50 stand-in** runs, reported as `dataset` = `esc50`: fold 1 targets build the profiles, fold 1 negatives fill the background bank, and folds 2–5 are the test set. Never mix the two datasets in one chart.
- **Missing values:** an empty cell means undefined. Precision is undefined when the class is never predicted, recall when it never occurs, and F1 when either is undefined. Macro F1 averages F1 over the classes that occur, counting a class that's never predicted as 0.
- **Numbers:** fractions are 0–1 with 4 decimals; `*_pct` columns are 0–100.

### Experiment 1: YAMNet alone (`exp1_yamnet.py`)

YAMNet's top category for each window, compared with the file's label, for ESC-50 and the Indian test set separately, and for the Indian set per condition.

#### `exp1_summary.csv`
One row per dataset × condition × level.

| Column | Meaning |
|---|---|
| `dataset` | `esc50` or `indian` |
| `condition` | `all`, or `quiet` / `tv` / `fan` / `far` (Indian only; ESC-50 is all `clean`) |
| `level` | `window` or `clip` |
| `threshold` | detection threshold used (0.15) |
| `n` | items scored (windows or clips) |
| `files` | files in the group |
| `silent_windows` | windows dropped as silent |
| `accuracy` | correct / n |
| `macro_f1` | mean F1 over the labels present |

#### `exp1_per_class.csv`
One row per dataset × condition × level × label. The labels are the 9 categories plus `none`.

| Column | Meaning |
|---|---|
| `dataset`, `condition`, `level`, `threshold` | as above |
| `label` | category ID or `none` |
| `support` | items whose true label is this |
| `tp`, `fp`, `fn` | true positives, false positives, false negatives |
| `precision`, `recall`, `f1` | per label (empty = undefined) |

#### `exp1_confusion.csv`
Long format, one row per cell, zeros included, for a heatmap.

| Column | Meaning |
|---|---|
| `dataset`, `condition`, `level`, `threshold` | as above |
| `true` | true label |
| `pred` | predicted label |
| `count` | items with this true and predicted label |

#### `exp1_threshold_sweep.csv`
The same evaluation for thresholds 0.05–0.95, to help map the settings' `sensitivity` to a threshold.

| Column | Meaning |
|---|---|
| `dataset`, `level` | as above (condition `all`) |
| `threshold` | detection threshold |
| `accuracy`, `macro_f1` | as above |

### Experiment 4: YAMNet + custom profiles (`exp4_custom.py`)

One embedding profile per category (centered scoring, default thresholds), built from profile files only. Three systems are compared on the same test windows:
- `yamnet`: Experiment 1 on these files.
- `custom_only`: the best matching profile's category, else `none`.
- `yamnet+custom`: the best matching profile's category if any profile matches, else YAMNet's prediction.

#### `exp4_summary.csv`

| Column | Meaning |
|---|---|
| `dataset` | `indian` or `esc50` (stand-in) |
| `system` | `yamnet`, `custom_only` or `yamnet+custom` |
| `condition`, `level`, `threshold`, `n`, `files`, `accuracy`, `macro_f1` | as in `exp1_summary.csv` |
| `profiles` | number of profiles (categories with profile files) |
| `profile_files` | files used to build them |
| `removed_overlap` | test files dropped because they were also profile files |
| `macro_f1_yamnet` | macro F1 of `yamnet` in the same group |
| `delta_macro_f1` | `macro_f1` − `macro_f1_yamnet` (positive = custom profiles help) |

#### `exp4_per_class.csv`

| Column | Meaning |
|---|---|
| `dataset`, `system`, `condition`, `level` | as above |
| `label`, `support`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1` | as in `exp1_per_class.csv` |

#### `exp4_confusion.csv`

| Column | Meaning |
|---|---|
| `dataset`, `system`, `condition`, `level` | as above |
| `true`, `pred`, `count` | as in `exp1_confusion.csv` |

### Experiment 6: embedding kNN vs MFCC (`exp6_methods.py`)

For each method, one profile per category is built from the same files. Every test window is then scored against every profile:
- **Target windows** (the profile's own category) give `tp` and `fn`. Recall is the hit rate.
- **Unrelated windows** (every other category, plus all negatives: `hard_negatives/` and ESC-50 negatives) give `fp`, the false matches, and `tn`.

At clip level, a clip counts as matched if any of its windows matched.

#### `exp6_per_profile.csv`
One row per dataset × method × category × level.

| Column | Meaning |
|---|---|
| `dataset` | `indian` or `esc50` (stand-in) |
| `method` | `embedding` (YAMNet embedding, centered, margin rule) or `mfcc` (40-number MFCC baseline) |
| `category` | the profile's category |
| `level` | `window` or `clip` |
| `threshold` | the method's match threshold (0.89 centered / 0.80 raw fallback, or 0.991 MFCC) |
| `n_target` | target items (`tp` + `fn`) |
| `n_unrelated` | unrelated items (`fp` + `tn`) |
| `tp`, `fn`, `fp`, `tn` | matched/missed targets, false/correct non-matches of unrelated items |
| `accuracy` | (`tp` + `tn`) / all |
| `precision`, `recall`, `f1` | detection metrics for this profile |
| `false_match_rate` | `fp` / `n_unrelated` |
| `fp_other_categories`, `n_other_categories` | false matches on, and number of, other categories' items |
| `fp_negatives`, `n_negatives` | false matches on, and number of, negative items |
| `mean_sim_target`, `mean_sim_unrelated` | mean similarity of target and unrelated items (at clip level, each clip's best window) |
| `auc` | ROC AUC of target vs unrelated similarities: the chance a random target item scores higher than a random unrelated one (0.5 = chance, 1 = perfect). It doesn't depend on the threshold, so it compares the methods fairly. |

#### `exp6_summary.csv`
One row per dataset × method × level.

| Column | Meaning |
|---|---|
| `dataset`, `method`, `level` | as above |
| `profiles` | number of profiles |
| `macro_f1` | mean F1 over the profiles |
| `mean_recall` | mean hit rate over the profiles |
| `mean_auc` | mean `auc` over the profiles |
| `false_match_rate` | all `fp` / all unrelated items, summed over the profiles |
| `false_match_rate_negatives` | the same, on negative items only |
| `tp`, `fn`, `fp`, `tn` | summed over the profiles |

### Experiment 8: CPU and RAM, silence gate on vs off (`exp8_resources.py`)

A fixed 10-minute stream runs through the app's per-window pipeline twice, with the gate off and then on. The gate is calibrated from the first 2 s, as in the app.
- **The stream:** seeded room tone (pink noise, RMS 0.003) with clips inserted after 2–15 s gaps. The clips come from the Indian test set, else ESC-50, else synthetic tones. `--wav <recording>` uses a real recording instead.
- **Separate process:** `run_all` runs Experiment 8 in a fresh Python process, so its RAM isn't inflated by the other experiments' caches.
- **CPU:** the stream is processed faster than real time, so CPU is reported as `cpu_pct_realtime`: CPU seconds per second of audio, i.e. the share of one core needed to keep up live. `--realtime` paces at the real 0.48 s hop, and then `cpu_pct_wall` is the live CPU %.

#### `exp8_resources.csv`
One row per setting.

| Column | Meaning |
|---|---|
| `setting` | `gate_on` or `gate_off` |
| `source` | where the events came from: `indian`, `esc50`, `synthetic` or `wav:<file>` |
| `stream_s` | stream length in seconds (600) |
| `event_fraction` | fraction of the stream covered by events (empty for `--wav`) |
| `realtime` | `True` if paced in real time |
| `windows` | windows processed |
| `inferred` | windows that ran YAMNet |
| `skipped` | windows skipped by the gate |
| `skipped_pct` | `skipped` / `windows` × 100 |
| `avg_inference_ms`, `p95_inference_ms` | YAMNet time per inferred window (mean, 95th percentile) |
| `cpu_s` | process CPU time (user + system) for the stream |
| `wall_s` | wall-clock time for the stream |
| `cpu_pct_realtime` | `cpu_s` / `stream_s` × 100: % of one core needed for live audio |
| `cpu_pct_wall` | `cpu_s` / `wall_s` × 100 (the live CPU % with `--realtime`) |
| `rss_start_mb`, `rss_peak_mb` | process memory (RSS) at the start and the peak during the run, in MB (includes TensorFlow) |
| `measurement` | `psutil`, or the standard-library fallback if psutil isn't installed |

### Results so far

Run on 2026-11-13 on **ESC-50 only**; the Indian recordings aren't available yet, so every `indian` row is missing (skipped). Re-run `python -m eval.model.run_all` once `data/indian/` is filled.

**Experiment 1** (YAMNet alone, ESC-50, threshold 0.15):
- **Overall:** window level macro F1 0.71 (accuracy 0.88, 8333 windows); clip level macro F1 0.68 (accuracy 0.77, 1240 clips).
- **Accuracy is inflated by negatives:** 26 of the 31 categories are negatives, so use macro F1.
- **Per category, window F1:** dog bark 0.81, baby cry 0.72, glass break 0.61, door knock 0.59, water running 0.55.
- **False alerts at clip level:** 26% of negative clips raise one, mostly glass_break (63), dog_bark (46), appliance_beep (46) and door_knock (35).
- **Threshold sweep:** at clip level macro F1 peaks at threshold 0.35 (0.74). At window level it peaks at 0.15.

**Experiment 4** (ESC-50 stand-in, 5 profiles from fold 1):
- **Window level:** yamnet+custom 0.731 vs YAMNet 0.711 (+0.020); custom only 0.596.
- **Clip level, which is closer to alerts:** custom only **0.790** vs YAMNet 0.676 (+0.114), while yamnet+custom gains only +0.007. The combined system still passes on YAMNet's false alerts for negatives whenever no profile matches.
- **Caveat:** the stand-in has profiles only for the 5 ESC-50 targets.

**Experiment 6** (ESC-50 stand-in):

| Method | Level | Macro F1 | Hit rate | False-match rate | AUC |
|---|---|---|---|---|---|
| embedding | window | 0.53 | 0.38 | 0.0005 | 0.98 |
| embedding | clip | 0.76 | 0.66 | 0.0019 | 1.00 |
| mfcc | window | 0.00 | 0.00 | 0.0001 | 0.57 |
| mfcc | clip | 0.00 | 0.00 | 0.0009 | 0.58 |

- **Embeddings:** they separate the sounds almost perfectly (AUC 0.98), but at the provisional 0.89 threshold they are strict: precision 0.92–0.99, recall 0.18–0.59.
- **MFCC:** it barely beats chance (AUC 0.57). Unrelated sounds score as high as the targets, so its 0.991 threshold, tuned on synthetic sounds, is never reached.
- **Both thresholds need re-tuning on the Indian set** (`tune_custom_threshold.py`).

**Experiment 8** (10-minute stream from ESC-50 clips, 36% events, Apple M2):
- **CPU:** the gate skips 71.6% of windows and cuts CPU from 1.3% to 0.4% of one core (real-time equivalent).
- **Inference:** ~2.6 ms per window either way.
- **RAM:** peak 760 MB in both settings. That is mostly TensorFlow and YAMNet, so the gate saves CPU, not memory.
