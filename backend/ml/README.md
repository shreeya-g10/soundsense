# backend/ml/

**Owner:** Shreeya (Role 1)

YAMNet (TensorFlow CPU) inference and sound classification.

## How to use

| Module | What it is | Use it for |
|---|---|---|
| `yamnet.py` | `YamnetModel.instance()`: YAMNet on CPU, loaded once per process. `infer(window)` gives `scores (521,)`, `embedding (1024,)` and `inference_ms`. `top_k(scores, k)` gives the k best classes. | anything that needs YAMNet |
| `category_map.py` | `CategoryMapper(class_names)`: `category_scores(scores)` (9 categories) and `context_scores(scores)` (speech, music, television). `CATEGORY_CLASSES` / `CONTEXT_CLASSES` hold the mapping. | turning YAMNet output into SoundSense categories |
| `demo_yamnet.py` | classify a file, the live mic, or benchmark inference; `load_wav(path)` reads any audio file at 16 kHz | checking YAMNet by hand |

```python
from backend.ml.yamnet import YamnetModel, top_k
from backend.ml.category_map import CategoryMapper

model = YamnetModel.instance()                 # first call: download to data/models/yamnet/ (~4 s), then offline
out = model.infer(window)                      # 0.96 s, 16 kHz mono float32
mapper = CategoryMapper(model.class_names)
mapper.category_scores(out.scores)             # {"smoke_alarm": 0.01, ..., "water_running": 0.0}
top_k(out.scores, 3)                           # [("Speech", 0.91), ...]
```

You normally don't call these directly: `YamnetListener` and `YamnetRemoteInput` do.

## Part S3: YAMNet inference (`yamnet.py`)

- `YamnetModel.instance()` loads YAMNet once per process (thread-safe) on CPU only. The first run downloads it from TensorFlow Hub into `data/models/yamnet/` (git-ignored); later runs load from there and work offline.
- `model.infer(samples)` takes 16 kHz mono float32. It returns `YamnetOutput(scores (521,), embedding (1024,), inference_ms)`, with scores and embedding averaged over YAMNet's frames (a 0.96 s window is one frame).
- `model.class_names` lists the 521 names from YAMNet's class-map CSV. `top_k(scores, k)` returns `[(name, score), ...]`, best first.

**Measured on this Mac (CPU):** model load and warm-up ~4 s. Inference is 2.6 ms per window when run back to back (300 windows) and ~11 ms per window live, at one call every 0.48 s.

Demo (from the repo root):

```bash
python -m backend.ml.demo_yamnet path/to/sound.wav       # classify a file (any format soundfile reads)
python -m backend.ml.demo_yamnet --live                  # mic + silence gate, top 3 per active window
python -m backend.ml.demo_yamnet --live --seconds 0
python -m backend.ml.demo_yamnet --benchmark             # average inference time per window
```

## Part S4: category mapping (`category_map.py`)

- `CategoryMapper(class_names)` maps YAMNet's 521 scores to the 9 SoundSense categories. Each category's score is the **max** of its mapped classes. It also produces the `speech`, `music` and `television` context scores.
- Every mapped name is checked against the real class map when the mapper is built. A missing or misspelled name raises `ValueError`.
- `pressure_cooker` from YAMNet alone means generic whistling/steam/hiss. Custom profiles (S5) refine it to the user's own cooker.
- `door_knock` maps to `Knock` and `Door` (S5d). In live tests, real knocks scored "Knock" ~0.21 and "Door" ~0.15. "Wood"/"Chop" also appeared but are not mapped, because they would cause kitchen false alerts.
- **Known confusions** are documented in `category_map.py`: generic "Alarm" is unmapped, so alarm clocks don't raise smoke alarms, and pressure-cooker steam scores as "Sink"/"Water tap" (water_running). Custom profiles are meant to resolve these.
