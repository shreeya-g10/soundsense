"""YAMNet sound classifier, CPU only.

Owner: Shreeya (Role 1). Part S3.

``YamnetModel.instance()`` loads YAMNet once per process (thread-safe lazy
singleton). The first run downloads it from TensorFlow Hub and copies it to
``data/models/yamnet/`` (git-ignored). Later runs load from there, so they
work offline.

    from backend.ml.yamnet import YamnetModel, top_k

    model = YamnetModel.instance()
    out = model.infer(window)          # float32, 16 kHz, mono, -1..1
    out.scores      # (521,) class scores, averaged over YAMNet's frames
    out.embedding   # (1024,) embedding, averaged over frames
    out.inference_ms
    top_k(out.scores, 3)  # [("Speech", 0.91), ...]
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# CPU only, and quiet TensorFlow's C++ logging. Must be set before TF is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"
MODEL_DIR = MODELS_DIR / "yamnet"
SAMPLE_RATE = 16_000
NUM_CLASSES = 521
EMBEDDING_SIZE = 1024
MIN_SAMPLES = 15_600  # YAMNet pads shorter input; one 0.975 s frame


@dataclass(frozen=True)
class YamnetOutput:
    """Result of YamnetModel.infer() for one window."""

    scores: np.ndarray  # (521,) float32, 0..1, mean over YAMNet frames
    embedding: np.ndarray  # (1024,) float32, mean over YAMNet frames
    inference_ms: float  # wall time of the model call


def _download(model_dir: Path) -> None:
    """Fetch YAMNet from TF Hub and store it as a plain SavedModel in model_dir."""
    cache = model_dir.parent / "tfhub_cache"
    os.environ["TFHUB_CACHE_DIR"] = str(cache)
    import tensorflow_hub as hub

    log.info("Downloading YAMNet from %s (first run only)", YAMNET_URL)
    resolved = Path(hub.resolve(YAMNET_URL))
    tmp = model_dir.with_name(model_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(resolved, tmp)
    tmp.rename(model_dir)  # atomic: a half-copied model is never used
    shutil.rmtree(cache, ignore_errors=True)
    log.info("YAMNet cached at %s", model_dir)


class YamnetModel:
    """YAMNet wrapper. Use ``YamnetModel.instance()``, not the constructor."""

    _instance: YamnetModel | None = None
    _lock = threading.Lock()
    load_count = 0  # how many times a model was actually loaded (tests check this)

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        if not (model_dir / "saved_model.pb").exists():
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            _download(model_dir)

        t0 = time.perf_counter()
        self._model = tf.saved_model.load(str(model_dir))
        self.class_names = _read_class_names(Path(self._model.class_map_path().numpy().decode()))
        if len(self.class_names) != NUM_CLASSES:
            raise RuntimeError(f"expected {NUM_CLASSES} classes, got {len(self.class_names)}")
        self._infer_lock = threading.Lock()
        self.infer(np.zeros(MIN_SAMPLES, dtype=np.float32))  # warm-up: first call builds the graph
        type(self).load_count += 1
        log.info("YAMNet loaded in %.0f ms", (time.perf_counter() - t0) * 1000)

    @classmethod
    def instance(cls) -> YamnetModel:
        """Return the shared model, loading it on first use (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def infer(self, samples: np.ndarray) -> YamnetOutput:
        """Run YAMNet on one window.

        Args:
            samples: 1-D float32 audio, 16 kHz mono, values -1..1. 0.96 s
                (15 360 samples) gives one YAMNet frame; longer input gives
                several frames, which are averaged.

        Raises:
            ValueError: if samples is not 1-D or is empty.
        """
        waveform = np.asarray(samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"expected non-empty 1-D audio, got shape {waveform.shape}")
        with self._infer_lock:
            t0 = time.perf_counter()
            scores, embeddings, _ = self._model(waveform)
            scores = scores.numpy()
            embeddings = embeddings.numpy()
            ms = (time.perf_counter() - t0) * 1000
        return YamnetOutput(
            scores=scores.mean(axis=0).astype(np.float32),
            embedding=embeddings.mean(axis=0).astype(np.float32),
            inference_ms=ms,
        )


def _read_class_names(csv_path: Path) -> list[str]:
    """Read display names from YAMNet's class map (index,mid,display_name)."""
    with open(csv_path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def class_names() -> list[str]:
    """The 521 YAMNet class names, in score order."""
    return YamnetModel.instance().class_names


def top_k(scores: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """The k highest-scoring classes as ``[(name, score), ...]``, best first."""
    names = class_names()
    scores = np.asarray(scores).ravel()
    if scores.size != len(names):
        raise ValueError(f"expected {len(names)} scores, got {scores.size}")
    idx = np.argsort(scores)[::-1][:k]
    return [(names[i], float(scores[i])) for i in idx]
