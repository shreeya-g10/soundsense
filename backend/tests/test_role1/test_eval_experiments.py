"""End-to-end smoke test of the experiment scripts (Part S7) and their skip behaviour.

The smoke test builds a synthetic mini-dataset in both layouts (Indian and
ESC-50), runs exp1/exp4/exp6/exp8 through run_all with real YAMNet, and
checks every CSV has the columns documented in eval/model/README.md.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from eval.model import common as cm
from eval.model import exp1_yamnet, exp4_custom, exp6_methods, exp8_resources, run_all

README = Path(__file__).resolve().parents[3] / "eval" / "model" / "README.md"


@pytest.fixture(scope="module")
def yamnet():
    try:
        from backend.ml.yamnet import YamnetModel

        YamnetModel.instance()
    except Exception as exc:
        pytest.skip(f"YAMNet unavailable: {exc}")


@pytest.fixture(scope="module")
def smoke_results(tmp_path_factory, yamnet):
    work = tmp_path_factory.mktemp("smoke")
    return work, run_all.smoke(work)


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_smoke_writes_every_documented_csv(smoke_results):
    _, written = smoke_results
    assert sorted(p.name for p in written) == sorted(run_all.EXPECTED_FIELDS)


def test_exp1_reports_both_datasets_and_every_indian_condition(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "exp1_summary.csv")
    seen = {(r["dataset"], r["condition"], r["level"]) for r in rows}
    for level in ("window", "clip"):
        assert ("esc50", "all", level) in seen
        for cond in ("all", "quiet", "tv", "fan", "music"):
            assert ("indian", cond, level) in seen
    conf = read(work / "results" / "exp1_confusion.csv")
    esc_clip = [r for r in conf if r["dataset"] == "esc50" and r["condition"] == "all" and r["level"] == "clip"]
    assert sum(int(r["count"]) for r in esc_clip) == 20  # every ESC-50 smoke file once
    sweep = read(work / "results" / "exp1_threshold_sweep.csv")
    assert {float(r["threshold"]) for r in sweep} >= {0.05, 0.5, 0.95}


def test_exp4_compares_systems_per_recorder_and_device(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "exp4_summary.csv")
    for dataset in ("indian", "esc50"):
        systems = {r["system"] for r in rows if r["dataset"] == dataset}
        assert systems == set(exp4_custom.SYSTEMS)
    indian = [r for r in rows if r["dataset"] == "indian"]
    assert {(r["recorder"], r["device"]) for r in indian} == {
        ("shreeya", "own"), ("shreeya", "other"), ("vaidehi", "own"), ("vaidehi", "other")}
    assert {(r["recorder"], r["device"]) for r in rows if r["dataset"] == "esc50"} == {("all", "all")}
    # Each person's own-device group is their own 9 test files (4 + 4 targets, 1 kettle).
    own = [r for r in indian if r["device"] == "own" and r["condition"] == "all" and r["level"] == "clip"]
    assert {r["files"] for r in own} == {"9"} and {r["profile_files"] for r in own} == {"4"}
    # The yamnet baseline doesn't depend on whose profiles are loaded.
    yam = {r["recorder"]: r for r in indian if r["system"] == "yamnet" and r["device"] == "own"
           and r["condition"] == "all" and r["level"] == "window"}
    assert float(yam["shreeya"]["delta_macro_f1"]) == 0.0


def test_splits_per_recorder_use_only_their_own_profile_clips(tmp_path):
    paths = run_all.make_smoke_data(tmp_path / "data")
    splits = {s.recorder: s for s in cm.custom_splits(cm.INDIAN, paths)}
    assert set(splits) == {"shreeya", "vaidehi"}
    sh, va = splits["shreeya"], splits["vaidehi"]
    assert {c.recorder for c in sh.profile} == {"shreeya"} and {c.recorder for c in va.profile} == {"vaidehi"}
    assert sh.categories == ["baby_cry", "door_knock"] and va.categories == ["baby_cry", "dog_bark"]
    assert [c.path for c in sh.test] == [c.path for c in va.test]  # both tested on ALL test clips
    assert {c.recorder for c in sh.test} == {"shreeya", "vaidehi"}
    # Room sound: their own home's background + the shared one, never the other person's.
    assert sorted(c.recorder or "shared" for c in sh.room) == ["shared", "shreeya"]
    assert sorted(c.recorder or "shared" for c in va.room) == ["shared", "vaidehi"]


def test_disjoint_drops_test_files_identical_to_profile_files(tmp_path):
    paths = run_all.make_smoke_data(tmp_path / "data")
    profile = cm.get_indian_profile(paths)
    copy = paths.test / "baby_cry" / "baby_cry_sneha_quiet_near_9.wav"
    copy.write_bytes(profile[0].path.read_bytes())  # same audio, different name
    for split in cm.custom_splits(cm.INDIAN, paths):
        assert split.removed == 1 and copy not in [c.path for c in split.test]
        assert not {c.path for c in profile} & {c.path for c in split.test}
    (esc,) = cm.custom_splits(cm.ESC50, paths)
    assert esc.recorder is None and esc.who == "all"
    assert {cm.esc50_fold(c) for c in esc.profile + esc.generic} == {1}
    assert {cm.esc50_fold(c) for c in esc.test} == {2}


def test_unnamed_profile_clips_are_left_out(tmp_path, capsys):
    paths = run_all.make_smoke_data(tmp_path / "data")
    import shutil

    shutil.copy(next((paths.profile / "baby_cry").iterdir()), paths.profile / "baby_cry" / "my_cry.wav")
    splits = cm.custom_splits(cm.INDIAN, paths)
    assert all("my_cry.wav" not in [c.path.name for c in s.profile] for s in splits)
    assert "1 profile clip(s) without a recorder name left out" in capsys.readouterr().out


def test_relation_and_device():
    from eval.model.datasets import Clip

    def clip(category, recorder):
        return Clip(Path("x.wav"), category or "negative", "quiet", "indian", category, None if category else "kettle",
                    recorder=recorder)

    assert cm.relation(clip("baby_cry", "shreeya"), "baby_cry", "shreeya") == cm.TARGET
    assert cm.relation(clip("baby_cry", "vaidehi"), "baby_cry", "shreeya") == cm.SAME_CATEGORY_OTHER_DEVICE
    assert cm.relation(clip("dog_bark", "shreeya"), "baby_cry", "shreeya") == cm.UNRELATED
    assert cm.relation(clip(None, "shreeya"), "baby_cry", "shreeya") == cm.UNRELATED
    assert cm.relation(clip("baby_cry", None), "baby_cry", None) == cm.TARGET  # ESC-50: no recorders
    assert cm.device(clip("baby_cry", "shreeya"), "shreeya") == "own"
    assert cm.device(clip("baby_cry", "vaidehi"), "shreeya") == "other"
    assert cm.device(clip("baby_cry", None), None) == "all"


def test_exp6_compares_methods_with_false_matches(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "exp6_per_profile.csv")
    assert {r["method"] for r in rows} == {"embedding", "mfcc"}
    indian = [r for r in rows if r["dataset"] == "indian" and r["level"] == "window"]
    assert {(r["recorder"], r["category"]) for r in indian} == {
        ("shreeya", "baby_cry"), ("shreeya", "door_knock"), ("vaidehi", "baby_cry"), ("vaidehi", "dog_bark")}
    assert all(int(r["n_negatives"]) > 0 for r in indian)  # hard negatives + ESC-50 negatives
    for r in indian:
        assert int(r["tp"]) + int(r["fn"]) == int(r["n_target"])
        assert int(r["fp"]) + int(r["tn"]) == int(r["n_unrelated"]) == int(r["n_other_categories"]) + int(r["n_negatives"])
    # Both recorded baby_cry: the other person's cries are reported separately, not as targets or false matches.
    for who, other in (("shreeya", "vaidehi"), ("vaidehi", "shreeya")):
        r = next(r for r in indian if r["recorder"] == who and r["category"] == "baby_cry" and r["method"] == "embedding")
        assert int(r["n_same_category_other_device"]) > 0 and r["same_category_other_device_match_rate"] != ""
        assert int(r["n_target"]) + int(r["n_same_category_other_device"]) > int(r["n_target"])
    solo = next(r for r in indian if r["category"] == "door_knock" and r["method"] == "embedding")
    assert int(solo["n_same_category_other_device"]) == 0 and solo["same_category_other_device_match_rate"] == ""
    summary = read(work / "results" / "exp6_summary.csv")
    keys = {(r["dataset"], r["recorder"], r["method"]) for r in summary}
    assert keys >= {("indian", "shreeya", "embedding"), ("indian", "vaidehi", "mfcc"), ("indian", "all", "mfcc"),
                    ("esc50", "all", "embedding")}


def test_exp8_gate_on_skips_and_saves_cpu(smoke_results):
    work, _ = smoke_results
    rows = {r["setting"]: r for r in read(work / "results" / "exp8_resources.csv")}
    on, off = rows["gate_on"], rows["gate_off"]
    assert float(off["skipped_pct"]) == 0.0 and float(on["skipped_pct"]) > 30
    assert int(on["inferred"]) < int(off["inferred"]) and on["windows"] == off["windows"]
    assert float(on["cpu_s"]) < float(off["cpu_s"])
    assert float(on["rss_peak_mb"]) > 0
    assert on["measurement"] == "psutil" or on["measurement"].startswith("stdlib")


def test_every_script_skips_missing_data_with_a_message(tmp_path, capsys):
    empty = cm.DataPaths(indian=tmp_path / "none", esc50=tmp_path / "none")
    ex = cm.FeatureExtractor(model=object(), mapper=object())  # never used: nothing to analyse
    out = tmp_path / "results"
    assert exp1_yamnet.run(out, empty, ex) == []
    assert exp4_custom.run(out, empty, ex) == []
    assert exp6_methods.run(out, empty, ex) == []
    text = capsys.readouterr().out
    for exp in ("exp1", "exp4", "exp6"):
        assert re.search(rf"\[{exp}\] SKIP esc50: ESC-50 not found .*download_esc50", text)
        assert re.search(rf"\[{exp}\] SKIP indian: No Indian ", text)
    assert not out.exists()


def test_exp8_falls_back_to_synthetic_events(tmp_path):
    empty = cm.DataPaths(indian=tmp_path / "none", esc50=tmp_path / "none")
    stream, source, fraction = exp8_resources.build_stream(0.5, empty)
    assert source == "synthetic" and stream.size == 30 * 16000 and 0 < fraction < 1
    again, _, _ = exp8_resources.build_stream(0.5, empty)
    np.testing.assert_array_equal(stream, again)  # fixed: same stream every run


def test_readme_documents_every_csv_column():
    text = README.read_text()
    for name, fields in run_all.EXPECTED_FIELDS.items():
        assert f"`{name}`" in text, name
        section = text.split(f"`{name}`", 1)[1].split("\n#", 1)[0]
        for f in fields:
            assert f"`{f}`" in section, f"{name}: column {f} not documented"


# --- S7d: noise robustness and TV-guard mixes ---------------------------------------

def test_noise_robustness_rows_for_every_snr_and_system(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "noise_summary.csv")
    esc = [r for r in rows if r["dataset"] == "esc50" and r["level"] == "clip"]
    assert {r["system"] for r in esc} == {"yamnet", "yamnet+custom"}
    seen = {(r["noise"], r["snr_db"]) for r in esc}
    assert ("clean", "") in seen
    for snr in ("20", "10", "5", "0"):
        assert {("DKITCHEN", snr), ("DLIVING", snr), ("all", snr)} <= seen
    for r in esc:  # every row is compared with the same system's clean F1
        clean = next(c for c in esc if c["noise"] == "clean" and c["system"] == r["system"])
        assert r["macro_f1_clean"] == clean["macro_f1"]
    fsd = {r["system"] for r in rows if r["dataset"] == "fsd50k"}
    assert fsd == {"yamnet"}  # no profile split for FSD50K
    indian = {(r["recorder"], r["system"]) for r in rows if r["dataset"] == "indian"}
    assert ("shreeya", "yamnet+custom") in indian and ("vaidehi", "yamnet+custom") in indian


def test_exp5_tvguard_mixes_have_window_scores_and_tv_only_rows(smoke_results):
    work, _ = smoke_results
    summary = read(work / "results" / "exp5_tvguard_summary.csv")
    tv_only = [r for r in summary if r["snr_db"] == "tv_only"]
    assert {r["background_type"] for r in tv_only} == {"speech", "music"}
    assert all(r["target_label"] == "none" and r["detection_rate"] == "" for r in tv_only)
    mixes = {(r["background_type"], r["snr_db"]) for r in summary if r["target_label"] != "none"}
    assert ("none", "clean") in mixes and {("speech", "0"), ("music", "20")} <= mixes
    windows = read(work / "results" / "exp5_tvguard_windows.csv")
    assert {"ctx_speech", "ctx_music", "ctx_television", "score_doorbell", "yamnet_pred"} <= set(windows[0])
    per_mix = {r["mix_id"] for r in windows}
    assert any(m.endswith(":speech:5") for m in per_mix) and any(m.endswith(":tv_only") for m in per_mix)
    assert all(0.0 <= float(r["ctx_speech"]) <= 1.0 for r in windows)


def test_noise_and_tvguard_skip_without_demand_or_fsd50k(tmp_path, capsys):
    from eval.model import exp5_tvguard, exp_noise

    empty = cm.DataPaths(indian=tmp_path / "x", esc50=tmp_path / "x", fsd50k=tmp_path / "x",
                         freesound=tmp_path / "x", demand=tmp_path / "x")
    ex = cm.FeatureExtractor(model=object(), mapper=object())
    assert exp_noise.run(tmp_path / "r", empty, ex) == []
    assert exp5_tvguard.run(tmp_path / "r", empty, ex) == []
    text = capsys.readouterr().out
    assert "[noise] SKIP all datasets: DEMAND noise not found" in text and "download_demand" in text
    assert "[exp5] SKIP TV-guard mixes: FSD50K Speech/Music clips not found" in text


def test_freesound_only_licensed_clips_reach_experiments(tmp_path):
    paths = run_all.make_smoke_data(tmp_path / "data")
    assert [c.path.name for c in cm.get_freesound(paths)] == ["bark_cc_by.wav"]  # the CC-BY-NC one is skipped
