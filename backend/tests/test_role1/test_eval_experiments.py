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
        for cond in ("all", "quiet", "tv", "fan", "far"):
            assert ("indian", cond, level) in seen
    conf = read(work / "results" / "exp1_confusion.csv")
    esc_clip = [r for r in conf if r["dataset"] == "esc50" and r["condition"] == "all" and r["level"] == "clip"]
    assert sum(int(r["count"]) for r in esc_clip) == 20  # every ESC-50 smoke file once
    sweep = read(work / "results" / "exp1_threshold_sweep.csv")
    assert {float(r["threshold"]) for r in sweep} >= {0.05, 0.5, 0.95}


def test_exp4_compares_systems_and_never_tests_on_profile_files(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "exp4_summary.csv")
    for dataset in ("indian", "esc50"):
        systems = {r["system"] for r in rows if r["dataset"] == dataset}
        assert systems == set(exp4_custom.SYSTEMS)
    yam = next(r for r in rows if r["dataset"] == "indian" and r["system"] == "yamnet"
               and r["condition"] == "all" and r["level"] == "window")
    exp1 = next(r for r in read(work / "results" / "exp1_summary.csv")
                if r["dataset"] == "indian" and r["condition"] == "all" and r["level"] == "window")
    assert yam["macro_f1"] == exp1["macro_f1"]  # the yamnet baseline IS Experiment 1 on the same files
    assert float(yam["delta_macro_f1"]) == 0.0


def test_disjoint_drops_test_files_identical_to_profile_files(tmp_path):
    paths = run_all.make_smoke_data(tmp_path / "data")
    profile = cm.get_indian_profile(paths)
    copy = paths.test / "baby_cry" / "baby_cry_copy_quiet_1m_1.wav"
    copy.write_bytes(profile[0].path.read_bytes())  # same audio, different name
    split = cm.custom_split(cm.INDIAN, paths)
    assert split.removed == 1 and copy not in [c.path for c in split.test]
    assert not {c.path for c in split.profile} & {c.path for c in split.test}
    esc = cm.custom_split(cm.ESC50, paths)
    assert {cm.esc50_fold(c) for c in esc.profile + esc.generic} == {1}
    assert {cm.esc50_fold(c) for c in esc.test} == {2}


def test_exp6_compares_methods_with_false_matches(smoke_results):
    work, _ = smoke_results
    rows = read(work / "results" / "exp6_per_profile.csv")
    assert {r["method"] for r in rows} == {"embedding", "mfcc"}
    indian = [r for r in rows if r["dataset"] == "indian" and r["level"] == "window"]
    assert all(int(r["n_negatives"]) > 0 for r in indian)  # hard negatives + ESC-50 negatives
    for r in indian:
        assert int(r["tp"]) + int(r["fn"]) == int(r["n_target"])
        assert int(r["fp"]) + int(r["tn"]) == int(r["n_unrelated"]) == int(r["n_other_categories"]) + int(r["n_negatives"])
    summary = read(work / "results" / "exp6_summary.csv")
    assert {(r["dataset"], r["method"]) for r in summary} >= {("indian", "embedding"), ("indian", "mfcc")}


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
