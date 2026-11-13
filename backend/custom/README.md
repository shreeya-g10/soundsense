# backend/custom/

**Owner:** Shreeya (Role 1)

User-taught custom sounds: embedding extraction and matching (scikit-learn). Embeddings are the only audio-derived data ever saved to disk.

## Part S5: real ProfileStore (`profile_store.py`)

`YamnetProfileStore(root="data/profiles", *, threshold=None, scoring="centered", margin=True, augment=True, top_k=3)` implements `contracts.types.ProfileStore`, raising the contract's `KeyError` and `ValueError`. It also has `match(embedding) -> list[CustomMatch]`, which `YamnetListener(profile_store=store)` calls on every active window, and `add_background(embedding, source)`, which the listener calls with each silence-gate calibration window.

- **Storage** (git-ignored, features only, never audio): `profiles.json` holds `{method, profiles: [{id, name, category, method}]}`. `<id>.npz` holds `positive`, `augmented` and `negative` arrays of shape (N, 1024) for embedding profiles, or `mfcc_positive`, `mfcc_augmented` and `mfcc_negative` arrays of shape (N, 40) for MFCC profiles. `category_negatives.npz` holds one (N, 1024) array per category of false alerts from the base model. `background.npz` holds the background bank, one array per source: `calibration`, `room` and `generic`. The newest 300 / 300 / 3000 are kept. Correct-alert feedback on base-model alerts goes to `feedback.npz` (`<category>_positive`). All writes are atomic (temp file + rename). Profiles saved before S5f load unchanged.
- **Recording:** `add_example_from_mic(id, seconds=3)` records into memory and cuts 0.96 s windows (0.48 s hop, like live). It drops silent windows and stores one embedding per window. It returns the example count (recorded windows only, unchanged if everything was silent).
- **Augmentation** (`augment=True`): for each recorded window, three more embeddings go in `augmented`: the window at −6 dB, at +6 dB, and mixed with background at 10 dB SNR. The noise is a random slice of the last room recording, which is held in memory only and never saved. If there is none, pink noise is used. Augmented copies are used for matching but don't count as examples.
- **Background bank:** `add_background_from_audio(audio)` adds the active windows of a "normal room sound" recording. The listener adds its calibration windows. `eval/model/build_background_bank.py` fills `generic` from ESC-50.
- **Scoring:** `"centered"` (the default) subtracts the background mean from every embedding, then L2-normalises, then takes the cosine. `"raw"` takes the plain cosine. With an empty bank the two are the same. A profile's similarity is the mean of its top-3 positive (recorded + augmented) examples.
- **Matching:** a window matches when its similarity reaches the threshold and, with `margin=True`, is also higher than its similarity to the background bank (top-3 mean, same scoring). `score(id, embedding)` returns `ScoreDetail(similarity, background, matched)`.
- **Thresholds:** 0.89 centered (provisional, from a synthetic set) and 0.80 raw. While the bank is empty, the default falls back to 0.80. An explicit `threshold=` is always used as given. See `eval/model/README.md` for how the value was chosen and how to re-tune it.

## Part S6: feedback and the MFCC baseline

**Feedback**, `add_feedback(embedding, category, profile_id, positive)`:
- **For a custom profile:** positive feedback appends the window to the profile's positives. Negative feedback appends it to the profile's negatives.
- **Negatives rule in `match()`:** a window no longer matches a profile when it is closer to the profile's negatives than to its positives (top-3 means, same scoring). `score()` reports this as `ScoreDetail.negative`.
- **For a base-model alert** (`profile_id=None`): negative feedback goes to `category_negatives.npz` (newest 500 per category). Positive feedback goes to `feedback.npz`.

**For the engine (Vaidehi), `store.is_negative(embedding, category) -> bool`:** before raising a base-model alert for `category`, skip it if this returns True.
- **Returns True** when the window's similarity to that category's false-alert embeddings (top-3 mean, same scoring as embedding profiles) reaches `effective_threshold` (0.89 centered, or 0.80 while the background bank is empty) and is higher than its similarity to the category's correct-alert feedback.
- **Returns False** for a category with no false-alert feedback.
- **Raises** `ValueError` for an unknown category.
- **Example:** `if store.is_negative(w.embedding, "smoke_alarm"): continue`. It isn't in `contracts/` yet; it's an extra method on `YamnetProfileStore`.

**MFCC baseline:**
- **Selecting it:** `set_method("mfcc")`, or `create(name, category, method="mfcc")`, builds new profiles from librosa MFCCs. Each window gives n_mfcc = 20 coefficients, with their mean and std over time, so a 40-number vector, stored as is. MFCC profiles need no YAMNet.
- **Methods are per profile:** each profile remembers the method it was built with and is always matched with it. `set_method` only affects profiles created afterwards, so switching never breaks existing profiles. Examples are stored per method, in separate npz arrays.
- **Matching:** the same top-3 kNN cosine, with no centering or background margin (the background bank holds embeddings). The threshold is 0.991, provisional (`eval/model/README.md`).
- **Coefficient 0 is ignored:** the cosine skips the mean and std of coefficient 0 (loudness). It is ~10× larger than the rest, so with it every pair of sounds scores ~1.00. `mfcc_drop_c0=False` keeps it, for comparison.
- **Audio for matching:** the listener passes the window's audio (in memory) as `match(embedding, samples=window)`, because MFCCs need it. The store remembers the MFCC of the last ~4096 matched windows in memory, so feedback, which only carries an embedding, can be stored for MFCC profiles. If a window is older than that, the feedback is kept as an embedding and a warning is logged.

Demo: `python -m backend.custom.demo_profiles [--name ... --category ...] [--from-wav file] [--keep] [--seconds 60]`.
- It records 5 × 3 s examples into memory (`--examples`, `--record-seconds`), then 5 s of normal room sound with no target (`--room-seconds`, 0 = skip). The room sound goes into the background bank and is used as augmentation noise. Then the examples are enrolled.
- It live-tests for `--seconds` (default 60; 0 = until Ctrl+C). Each line shows the similarity, MATCH, the window's top YAMNet class and the background score, e.g. `similarity 0.84 MATCH  [Steam 0.41]  bg 0.31`. A summary at the end shows windows tested, matches, and max/mean similarity.
- The new profile is deleted unless `--keep` is given.
- `--list` prints saved profiles (id, name, category, examples).
- `--test <id or name>` skips recording and tests a saved profile. It never deletes that profile.
- `--test-wav <file or folder>` scores the profile against audio files instead of a live test. It prints every window, then max, mean and % matched per file, then an overall line.
- `--record-test <label>` records a clip (`--clip-seconds`, default 10) to `data/indian/test/<label>/` (git-ignored) after a consent prompt. It is a developer tool for the evaluation dataset; the app never records audio to disk.
- `--scoring raw|centered`, `--threshold`, `--no-margin` and `--no-augment` are for Experiment 6 comparisons. `--method mfcc` builds the new profile with the MFCC baseline without changing the saved setting.
