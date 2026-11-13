"""Tests for eval/model/datasets.py and eval/model/download_esc50.py (no network, no real data)."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
import soundfile as sf

from eval.model import datasets as ds
from eval.model import download_esc50 as dl

SR = 16000


def tone(seconds=1.0, sr=SR, freq=440.0, channels=1):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def write(path, seconds=1.0, sr=SR, channels=1, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone(seconds, sr, channels=channels), sr, **kw)
    return path


# --- naming rule (team Drive guide) -------------------------------------------------

def test_parse_name_drive_guide():
    p = ds.parse_name("pressure_cooker_shreeya_tv_near_2", "pressure_cooker")
    assert (p.prefix, p.name, p.condition, p.distance, p.take) == ("pressure_cooker", "shreeya", "tv", "near", 2)
    assert ds.parse_name("baby_cry_vaidehi_music_far_1", "baby_cry").distance == "far"
    assert ds.parse_name("door_knock_sneha_fan_3m_4", "door_knock").distance == "3m"  # metres also accepted
    assert ds.parse_name("door_knock_sneha_quiet_2.5m_1", "door_knock").distance == "2.5m"
    assert ds.parse_name("kettle_sneha_fan_near_1", "kettle").name == "sneha"
    p = ds.parse_name("Door_Knock_Shreeya_TV_Near_1", "door_knock")  # phones capitalise
    assert (p.name, p.condition, p.distance) == ("shreeya", "tv", "near")
    assert set(ds.CONDITIONS) == {"quiet", "tv", "fan", "music"}
    assert ds.RECORDERS == ("shreeya", "vaidehi", "sneha")


@pytest.mark.parametrize("stem, reason", [
    ("doorbell_shreeya_quiet_near_1", "should start with 'door_knock_'"),
    ("door_knock_quiet_near_1", "expected <name>_<condition>_<distance>_<n>"),  # no name
    ("door_knock_main_door_quiet_near_1", "expected <name>"),  # name must be one word (the person)
    ("door_knock_shreeya_far_near_1", "condition 'far'"),  # far is a distance now
    ("door_knock_shreeya_loud_near_1", "condition 'loud'"),
    ("door_knock_shreeya_quiet_middle_1", "distance 'middle'"),
    ("door_knock_shreeya_quiet_3_1", "distance '3'"),
    ("door_knock_shreeya_quiet_near_a", "take 'a'"),
    ("door_knock_shreeya_quiet_near_0", "take '0'"),
])
def test_parse_name_rejects_bad_names(stem, reason):
    with pytest.raises(ValueError, match=reason):
        ds.parse_name(stem, "door_knock")


# --- layout ------------------------------------------------------------------------

@pytest.fixture
def indian(tmp_path):
    root = tmp_path / "indian"
    write(root / "profile" / "pressure_cooker" / "pressure_cooker_shreeya_quiet_near_1.wav")
    write(root / "profile" / "doorbell" / "doorbell_vaidehi_quiet_near_1.wav")
    test = root / "test"
    write(test / "pressure_cooker" / "pressure_cooker_shreeya_tv_3m_2.wav", sr=44100, channels=2)  # phone-like
    write(test / "door_knock" / "music" / "door_knock_sneha_music_far_1.wav")  # optional condition folder
    write(test / "door_knock" / "door_knock_badname.wav")
    write(test / "hard_negatives" / "kettle" / "kettle_vaidehi_fan_near_1.flac", format="FLAC")
    write(test / "hard_negatives" / "tv_ads" / "tv_ads_amma_quiet_far_1.wav")  # unknown recorder
    write(test / "kitchen_misc" / "x.wav")  # not a category
    (test / "doorbell").mkdir(parents=True)
    (test / "doorbell" / "notes.txt").write_text("ignored")
    return root


def test_list_test_clips_labels_conditions_recorders(indian):
    problems: list[str] = []
    clips = ds.list_test_clips(indian / "test", esc50_root=None, problems=problems)
    by_name = {c.path.name: c for c in clips}
    assert set(by_name) == {"pressure_cooker_shreeya_tv_3m_2.wav", "door_knock_sneha_music_far_1.wav",
                            "door_knock_badname.wav", "kettle_vaidehi_fan_near_1.flac", "tv_ads_amma_quiet_far_1.wav"}
    c = by_name["pressure_cooker_shreeya_tv_3m_2.wav"]
    assert (c.label, c.condition, c.source, c.category, c.negative_type) == ("pressure_cooker", "tv", "indian", "pressure_cooker", None)
    assert (c.recorder, c.name, c.distance, c.take, c.split) == ("shreeya", "shreeya", "3m", 2, "test")
    k = by_name["kettle_vaidehi_fan_near_1.flac"]
    assert (k.label, k.condition, k.category, k.negative_type, k.recorder) == ("negative", "fan", None, "kettle", "vaidehi")
    d = by_name["door_knock_sneha_music_far_1.wav"]
    assert (d.condition, d.distance, d.recorder) == ("music", "far", "sneha")
    bad = by_name["door_knock_badname.wav"]
    assert bad.condition == "unknown" and bad.recorder is None
    assert by_name["tv_ads_amma_quiet_far_1.wav"].recorder == "amma"  # loaded, but reported:
    assert any("tv_ads_amma" in p and "'amma' is not one of shreeya, vaidehi, sneha" in p for p in problems)
    assert any("door_knock_badname" in p for p in problems)
    assert any("kitchen_misc" in p and "not a category" in p for p in problems)


def test_condition_folder_and_name_must_agree(tmp_path):
    root = tmp_path / "test"
    write(root / "doorbell" / "tv" / "doorbell_shreeya_quiet_near_1.wav")
    write(root / "doorbell" / "fan" / "doorbell_oops.wav")
    problems: list[str] = []
    clips = {c.path.name: c for c in ds.list_test_clips(root, esc50_root=None, problems=problems)}
    assert clips["doorbell_shreeya_quiet_near_1.wav"].condition == "quiet"  # the name wins...
    assert any("folder says 'tv'" in p for p in problems)  # ...and the mismatch is reported
    assert clips["doorbell_oops.wav"].condition == "fan"  # bad name: fall back to the folder


def test_list_profile_clips(indian):
    clips = ds.list_profile_clips(indian / "profile")
    assert sorted((c.category, c.recorder) for c in clips) == [("doorbell", "vaidehi"), ("pressure_cooker", "shreeya")]
    assert all(c.split == "profile" and c.source == "indian" for c in clips)
    (only,) = ds.list_profile_clips(indian / "profile", category="doorbell")
    assert only.path.name == "doorbell_vaidehi_quiet_near_1.wav"


def test_iter_test_files_yields_16k_mono_tuples(indian):
    items = list(ds.iter_test_files(indian / "test", esc50_root=None))
    assert len(items) == 5
    for audio, label, condition, source in items:
        assert audio.dtype == np.float32 and audio.ndim == 1
        assert audio.size == pytest.approx(SR, abs=50)  # every file is 1 s, incl. the 44.1 kHz stereo one
        assert np.abs(audio).max() <= 1.0 and source == "indian"
    labels = sorted(label for _, label, _, _ in items)
    assert labels == ["door_knock", "door_knock", "negative", "negative", "pressure_cooker"]


# --- audio decoding ------------------------------------------------------------------

def test_load_audio_converts_rate_and_channels(tmp_path):
    path = write(tmp_path / "x.wav", seconds=2.0, sr=48000, channels=2)
    audio = ds.load_audio(path)
    assert audio.shape == (2 * SR,) and audio.dtype == np.float32
    assert np.sqrt(np.mean(audio ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.02)


def test_load_audio_mp3(tmp_path):
    path = tmp_path / "x.mp3"
    try:
        sf.write(str(path), tone(1.0, 44100), 44100, format="MP3")
    except Exception as exc:
        pytest.skip(f"libsndfile can't write MP3 here: {exc}")
    assert ds.load_audio(path).size == pytest.approx(SR, abs=1200)  # mp3 encoder padding


def test_load_audio_m4a(tmp_path):
    src = write(tmp_path / "x.wav", seconds=1.0, sr=44100)
    m4a = tmp_path / "x.m4a"
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-loglevel", "error", "-i", str(src), "-c:a", "aac", str(m4a)]
    elif shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "m4af", "-d", "aac", str(src), str(m4a)]
    else:
        pytest.skip("no ffmpeg/afconvert to make an m4a")
    subprocess.run(cmd, check=True)
    audio = ds.load_audio(m4a)
    assert audio.ndim == 1 and audio.size == pytest.approx(SR, abs=2500)  # AAC priming/padding
    assert np.sqrt(np.mean(audio[2000:-2000] ** 2)) == pytest.approx(0.3 / np.sqrt(2), rel=0.1)


def test_undecodable_file_raises_and_is_skipped_by_iter(tmp_path):
    bad = tmp_path / "test" / "doorbell" / "doorbell_shreeya_quiet_near_1.m4a"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not audio at all")
    with pytest.raises(ds.AudioDecodeError, match="cannot decode"):
        ds.load_audio(bad)
    assert list(ds.iter_test_files(tmp_path / "test", esc50_root=None)) == []


# --- ESC-50 ---------------------------------------------------------------------------

ESC_FIELDS = ["filename", "fold", "target", "category", "esc10", "src_file", "take"]


def esc_rows():
    cats = ["crying_baby", "dog", "rain", "clock_alarm", "airplane", "chainsaw"]
    return [{"filename": f"1-{i}-A-{t}.wav", "fold": "1", "target": str(t), "category": c,
             "esc10": "False", "src_file": str(i), "take": "A" if i % 2 else "B"}
            for t, c in enumerate(cats) for i in range(2)]


def test_mapping_targets_are_real_categories_and_disjoint_from_negatives():
    from contracts.types import CATEGORIES

    assert set(ds.ESC50_TO_CATEGORY.values()) <= set(CATEGORIES)
    assert not set(ds.ESC50_TO_CATEGORY) & set(ds.ESC50_NEGATIVES)
    assert {"crying_baby", "door_wood_knock", "glass_breaking", "dog", "clock_alarm", "washing_machine",
            "rain"} <= ds.ESC50_KEEP
    assert ds.esc50_label("dog") == "dog_bark" and ds.esc50_label("clock_alarm") == "negative"


def test_esc50_clips_from_meta(tmp_path):
    root = tmp_path / "esc50"
    rows = [r for r in esc_rows() if r["category"] in ds.ESC50_KEEP]
    (root / "meta").mkdir(parents=True)
    with open(root / "meta" / "esc50.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ESC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    for r in rows[:-1]:  # the last file is missing on disk
        write(root / "audio" / r["filename"], seconds=5.0, sr=44100)
    problems: list[str] = []
    clips = ds.list_test_clips(tmp_path / "no_indian", esc50_root=root, problems=problems)
    assert len(clips) == len(rows) - 1 and any("missing" in p for p in problems)
    assert {(c.name, c.label, c.negative_type) for c in clips} == {
        ("crying_baby", "baby_cry", None), ("dog", "dog_bark", None),
        ("rain", "negative", "rain"), ("clock_alarm", "negative", "clock_alarm")}
    assert all(c.condition == "clean" and c.source == "esc50" and c.take in (1, 2) for c in clips)
    audio, label, condition, source = next(ds.iter_test_files(tmp_path / "no_indian", esc50_root=root))
    assert audio.size == 5 * SR and source == "esc50" and condition == "clean"


def fake_esc50_zip(path):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ESC_FIELDS)
    w.writeheader()
    w.writerows(esc_rows())
    wav = io.BytesIO()
    sf.write(wav, tone(0.1, 44100), 44100, format="WAV")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ESC-50-master/meta/esc50.csv", buf.getvalue())
        z.writestr("ESC-50-master/LICENSE", "CC BY-NC 3.0")
        for r in esc_rows():
            z.writestr(f"ESC-50-master/audio/{r['filename']}", wav.getvalue())
    return path


def test_extract_keeps_only_needed_categories(tmp_path):
    z = fake_esc50_zip(tmp_path / "esc.zip")
    out = tmp_path / "esc50"
    counts = dl.extract(z, out, ds.ESC50_KEEP)
    assert dict(counts) == {"crying_baby": 2, "dog": 2, "rain": 2, "clock_alarm": 2}
    assert len(list((out / "audio").glob("*.wav"))) == 8
    rows = list(csv.DictReader(open(out / "meta" / "esc50.csv")))
    assert list(rows[0]) == ESC_FIELDS and {r["category"] for r in rows} == set(counts)
    assert (out / "ESC50_LICENSE").read_text() == "CC BY-NC 3.0"
    assert len(ds.list_esc50_clips(out)) == 8  # the loader reads what the downloader wrote
    assert sum(dl.extract(z, tmp_path / "all", None).values()) == 12  # --all


def test_download_streams_to_file(tmp_path):
    src = fake_esc50_zip(tmp_path / "src.zip")
    dest = dl.download(src.as_uri(), tmp_path / "dl" / "ESC-50-master.zip")
    assert dest.read_bytes() == src.read_bytes() and not dest.with_name(dest.name + ".part").exists()


def test_main_with_zip_extracts_and_keeps_given_archive(tmp_path, monkeypatch, capsys):
    z = fake_esc50_zip(tmp_path / "esc.zip")
    monkeypatch.setattr("sys.argv", ["download_esc50", "--zip", str(z), "--out", str(tmp_path / "out")])
    dl.main()
    out = capsys.readouterr().out
    assert "crying_baby        -> baby_cry" in out and "clock_alarm" in out and "Kept 8 clips" in out
    assert z.exists()  # a user-supplied archive is never deleted


def test_profile_background_folder_is_room_sound(tmp_path):
    root = tmp_path / "profile"
    write(root / "doorbell" / "doorbell_shreeya_quiet_near_1.wav")
    write(root / "background" / "living room evening.wav")  # any name: shared room sound
    write(root / "background" / "background_vaidehi_fan_near_1.m4a.wav")  # not the rule: still shared
    write(root / "background" / "background_vaidehi_fan_near_2.wav")  # Vaidehi's home
    problems: list[str] = []
    clips = ds.list_profile_clips(root, problems=problems)
    assert problems == []  # background files never cause naming problems
    bg = {c.path.name: c for c in clips if c.label == "background"}
    assert len(bg) == 3 and all(c.category is None for c in bg.values())
    assert bg["background_vaidehi_fan_near_2.wav"].recorder == "vaidehi"
    assert bg["living room evening.wav"].recorder is None
    assert ds.list_profile_clips(root, category="doorbell")[0].category == "doorbell"
