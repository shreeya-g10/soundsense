# eval/model/

**Owner:** Shreeya (Role 1)

Model evaluation: accuracy, precision/recall, confusion matrices per sound class.

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

## Generic background bank (`build_background_bank.py`)

This script fills the `generic` source of `data/profiles/background.npz` with YAMNet embeddings of ESC-50 clips. It takes up to 8 clips per class and leaves out 11 classes that are close to a SoundSense category (dog, crying baby, door knock or creak, glass breaking, clock alarm, siren, church bells, water drops, pouring water, toilet flush). The bank is empty until you run it:

```bash
python -m eval.model.build_background_bank --esc50 data/esc50/ESC-50-master   # unzipped under data/, git-ignored
python -m eval.model.build_background_bank --clear
```
