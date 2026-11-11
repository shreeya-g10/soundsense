# backend/custom/

**Owner:** Shreeya (Role 1)

User-taught custom sounds: embedding extraction and matching (scikit-learn). Embeddings are the only audio-derived data ever saved to disk.

## Part S5: real ProfileStore (`profile_store.py`)

`YamnetProfileStore(root="data/profiles", *, threshold=0.80, top_k=3)` implements `contracts.types.ProfileStore`, raising the contract's `KeyError` and `ValueError`. It also has `match(embedding) -> list[CustomMatch]`, which `YamnetListener(profile_store=store)` calls on every active window.

- **Storage** (git-ignored, embeddings only, never audio): `profiles.json` holds `{method, profiles: [{id, name, category, method}]}`, and `<id>.npz` holds `positive` and `negative` arrays of shape (N, 1024). Feedback on base-model alerts goes to `feedback.npz`. All writes are atomic (temp file + rename).
- **Recording:** `add_example_from_mic(id, seconds=3)` records into memory, cuts 0.96 s windows (0.48 s hop, like live), drops silent windows and stores one embedding per window. It returns the total example count, unchanged if everything was silent.
- **Matching:** similarity is the mean cosine similarity of the 3 nearest positive examples. `match()` returns profiles at or above the threshold, best first. `test_from_mic()` returns the best window similarity.
- **Feedback:** positive feedback adds an example; negative feedback is stored as a counter-example (not used in matching yet).
- **Methods:** `set_method("mfcc")` is accepted and saved, but matching still uses embeddings until MFCC is implemented.
- **What 0.80 separates** (real YAMNet embeddings): enrolled speech vs the same file 0.92, another voice 0.84, chimes 0.13–0.55. It separates *kinds* of sound well. Two similar sounds can still match each other.

Demo: `python -m backend.custom.demo_profiles [--name ... --category ...] [--from-wav file] [--keep]`. It records 5 × 3 s examples, then live-tests and shows the similarity per window. The demo profile is deleted unless `--keep` is given.
