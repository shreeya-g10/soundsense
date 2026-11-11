# backend/ml/

**Owner:** Shreeya (Role 1)

YAMNet (TensorFlow CPU) inference and sound classification.

## Part S3: YAMNet inference (`yamnet.py`)

- `YamnetModel.instance()` loads YAMNet once per process (thread-safe) on CPU only. The first run downloads it from TensorFlow Hub into `data/models/yamnet/` (git-ignored); later runs load from there and work offline.
- `model.infer(samples)` takes 16 kHz mono float32. It returns `YamnetOutput(scores (521,), embedding (1024,), inference_ms)`, with scores and embedding averaged over YAMNet's frames (a 0.96 s window is one frame).
- `model.class_names` lists the 521 names from YAMNet's class-map CSV. `top_k(scores, k)` returns `[(name, score), ...]`, best first.

**Measured on this Mac (CPU):** model load and warm-up ~4 s. Inference is 2.6 ms per window when run back to back (300 windows) and ~11 ms per window live, at one call every 0.48 s.

Demo (from the repo root):

```bash
python -m backend.ml.demo_yamnet path/to/sound.wav       # classify a file (any format soundfile reads)
python -m backend.ml.demo_yamnet --live                  # mic + silence gate, top 3 per active window
python -m backend.ml.demo_yamnet --live --calibrate --seconds 0
python -m backend.ml.demo_yamnet --benchmark             # average inference time per window
```

## Part S4: category mapping (`category_map.py`)

- `CategoryMapper(class_names)` maps YAMNet's 521 scores to the 9 SoundSense categories. Each category's score is the **max** of its mapped classes. It also produces the `speech`, `music` and `television` context scores.
- Every mapped name is checked against the real class map when the mapper is built. A missing or misspelled name raises `ValueError`.
- `pressure_cooker` from YAMNet alone means generic whistling/steam/hiss. Custom profiles (S5) refine it to the user's own cooker.
