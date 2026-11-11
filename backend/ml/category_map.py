"""Map YAMNet's 521 classes onto SoundSense's 9 categories and 3 context scores.

Owner: Shreeya (Role 1). Part S4.

A category's score is the MAX of the scores of the YAMNet classes mapped to
it. Every name below is checked against YAMNet's real class map when a
CategoryMapper is built; a missing or misspelled name raises ValueError, so
a typo can never silently disable a category.

    mapper = CategoryMapper(model.class_names)
    mapper.category_scores(out.scores)  # {"smoke_alarm": 0.02, ...} all 9 categories
    mapper.context_scores(out.scores)   # {"speech": 0.9, "music": 0.0, "television": 0.0}
"""

from __future__ import annotations

import numpy as np

from contracts.types import (
    APPLIANCE_BEEP,
    BABY_CRY,
    CATEGORIES,
    CONTEXT_MUSIC,
    CONTEXT_SPEECH,
    CONTEXT_TELEVISION,
    CONTEXTS,
    DOG_BARK,
    DOOR_KNOCK,
    DOORBELL,
    GLASS_BREAK,
    PRESSURE_COOKER,
    SMOKE_ALARM,
    WATER_RUNNING,
)

# Category ID -> YAMNet display names (exact strings from yamnet_class_map.csv).
CATEGORY_CLASSES: dict[str, list[str]] = {
    SMOKE_ALARM: ["Smoke detector, smoke alarm", "Fire alarm"],
    GLASS_BREAK: ["Glass", "Shatter"],
    # YAMNet has no "pressure cooker" class. From YAMNet alone this category
    # is just generic whistling/steam/hiss (a kettle, a person whistling,
    # a hissing tap all score here). Custom profiles (S5), recorded from the
    # user's own cooker, refine it into "this is MY cooker".
    PRESSURE_COOKER: ["Whistling", "Whistle", "Steam", "Hiss"],
    BABY_CRY: ["Baby cry, infant cry"],
    DOOR_KNOCK: ["Knock"],
    DOORBELL: ["Doorbell", "Ding-dong"],
    DOG_BARK: ["Bark", "Dog"],
    APPLIANCE_BEEP: ["Beep, bleep", "Microwave oven"],
    WATER_RUNNING: ["Water tap, faucet", "Sink (filling or washing)"],
}

# Context key -> YAMNet display names (used by the TV guard, not alerts).
CONTEXT_CLASSES: dict[str, list[str]] = {
    CONTEXT_SPEECH: ["Speech"],
    CONTEXT_MUSIC: ["Music"],
    CONTEXT_TELEVISION: ["Television"],
}


class CategoryMapper:
    """Turns a (521,) YAMNet score vector into category and context scores."""

    def __init__(self, class_names: list[str]) -> None:
        """Resolve every mapped name to its index in ``class_names``.

        Raises:
            ValueError: if any mapped name is not in ``class_names``, or the
                mapping doesn't cover exactly CATEGORIES / CONTEXTS.
        """
        if set(CATEGORY_CLASSES) != set(CATEGORIES):
            raise ValueError("CATEGORY_CLASSES must cover exactly contracts.types.CATEGORIES")
        if set(CONTEXT_CLASSES) != set(CONTEXTS):
            raise ValueError("CONTEXT_CLASSES must cover exactly contracts.types.CONTEXTS")

        index = {name: i for i, name in enumerate(class_names)}
        missing = sorted(
            {n for names in (*CATEGORY_CLASSES.values(), *CONTEXT_CLASSES.values()) for n in names} - set(index)
        )
        if missing:
            raise ValueError(f"YAMNet class names not found in class map: {missing}")

        self.num_classes = len(class_names)
        self._cat_idx = {c: np.array([index[n] for n in CATEGORY_CLASSES[c]]) for c in CATEGORIES}
        self._ctx_idx = {c: np.array([index[n] for n in CONTEXT_CLASSES[c]]) for c in CONTEXTS}

    def _check(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores).ravel()
        if scores.size != self.num_classes:
            raise ValueError(f"expected {self.num_classes} scores, got {scores.size}")
        return scores

    def category_scores(self, scores: np.ndarray) -> dict[str, float]:
        """Score 0..1 for every category in CATEGORIES (max of its classes)."""
        scores = self._check(scores)
        return {c: float(scores[idx].max()) for c, idx in self._cat_idx.items()}

    def context_scores(self, scores: np.ndarray) -> dict[str, float]:
        """Score 0..1 for "speech", "music" and "television"."""
        scores = self._check(scores)
        return {c: float(scores[idx].max()) for c, idx in self._ctx_idx.items()}
