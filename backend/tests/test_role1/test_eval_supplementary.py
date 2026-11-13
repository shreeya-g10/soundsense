"""Tests for the S7d supplementary datasets: rangezip, mixer, FSD50K selection, Freesound, DEMAND.

No network: archives are built locally and read through a file-backed fetcher.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.model import datasets as ds
from eval.model import download_demand, download_fsd50k, mixer
from eval.model.rangezip import RangeZip

SR = 16000


def file_fetcher(paths: list[Path], log: list | None = None):
    def fetch(part, start, end):
        if log is not None:
            log.append((part, start, end))
        with open(paths[part], "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)
    return fetch


def rangezip_for(paths: list[Path], log=None) -> RangeZip:
    return RangeZip([(str(p), p.stat().st_size) for p in paths], fetcher=file_fetcher(paths, log))


def payload(n: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# --- rangezip ---------------------------------------------------------------------

def test_rangezip_reads_selected_members_of_a_split_archive(tmp_path):
    if not shutil.which("zip"):
        pytest.skip("no zip command")
    src = tmp_path / "src" / "FSD50K.eval_audio"
    src.mkdir(parents=True)
    files = {f"{i}.wav": payload(30_000 + 7_000 * i, i) for i in range(12)}  # ~0.6 MB, incompressible
    files["quiet.wav"] = bytes(50_000)  # compresses well (deflate)
    for name, data in files.items():
        (src / name).write_bytes(data)
    subprocess.run(["zip", "-q", "-r", "-s", "128k", str(tmp_path / "eval.zip"), "FSD50K.eval_audio"],
                   cwd=tmp_path / "src", check=True)
    parts = sorted(tmp_path.glob("eval.z[0-9]*")) + [tmp_path / "eval.zip"]
    assert len(parts) >= 3  # really split

    log: list = []
    rz = rangezip_for(parts, log)
    entries = rz.entries()
    wavs = {Path(n).name: e for n, e in entries.items() if n.endswith(".wav")}
    assert set(wavs) == set(files) and len({e.disk for e in wavs.values()}) >= 3  # members on several parts
    straddle = [e for e in wavs.values()
                if rz.starts[e.disk] + e.offset + e.comp_size + 80 > rz.starts[e.disk] + rz.sizes[e.disk]]
    assert straddle, "test should include a member that crosses a part boundary"

    wanted = ["3.wav", "11.wav", "quiet.wav"] + [Path(straddle[0].name).name]
    log.clear()
    got = dict(rz.fetch([wavs[n] for n in wanted], max_gap=0))  # no grouping: one read per member
    assert {Path(n).name: d for n, d in got.items()} == {n: files[n] for n in wanted}
    fetched = sum(end - start + 1 for _, start, end in log)
    selected = sum(wavs[n].comp_size for n in wanted)
    assert fetched < selected + 120 * len(wanted)  # the members' bytes + their local headers, nothing else
    assert dict(rz.fetch([wavs[n] for n in wanted])) == got  # default grouping: same bytes out


def test_rangezip_stored_deflated_and_zip64_local_headers(tmp_path):
    path = tmp_path / "z64.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a/stored.bin", payload(5000, 1), compress_type=zipfile.ZIP_STORED)
        with z.open("a/deflated.bin", "w", force_zip64=True) as f:  # zip64 extra in the local header
            f.write(bytes(20000))
    rz = rangezip_for([path])
    e = rz.entries()
    assert e["a/stored.bin"].method == 0 and e["a/deflated.bin"].method == 8
    assert e["a/deflated.bin"].comp_size < 1000
    assert dict(rz.fetch(e.values())) == {"a/stored.bin": payload(5000, 1), "a/deflated.bin": bytes(20000)}
    log: list = []
    rz = rangezip_for([path], log)
    list(rz.fetch(rz.entries().values()))
    n_grouped = len(log) - 2  # minus the central-directory reads
    log.clear()
    list(rz.fetch(rz.entries().values(), max_gap=0, max_span=1))
    assert n_grouped == 1 and len(log) == 2  # nearby members share one request


def test_zip64_extra_field_overrides_32_bit_placeholders():
    import struct

    from eval.model.rangezip import _zip64_extra

    extra = struct.pack("<HHQQQI", 0x0001, 28, 6_000_000_000, 5_500_000_000, 4_400_000_000, 1)
    assert _zip64_extra(extra, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFF) == (
        6_000_000_000, 5_500_000_000, 4_400_000_000, 1)
    only_offset = struct.pack("<HHQ", 0x0001, 8, 4_400_000_000)
    assert _zip64_extra(only_offset, 10, 9, 0xFFFFFFFF, 0) == (10, 9, 4_400_000_000, 0)


def test_rangezip_detects_corruption(tmp_path):
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("x.bin", payload(4000, 2))
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF  # inside the member's data
    path.write_bytes(bytes(data))
    rz = rangezip_for([path])
    with pytest.raises(ValueError, match="CRC"):
        list(rz.fetch(rz.entries().values()))


# --- mixer --------------------------------------------------------------------------

def event_clip(seconds=5.0, event=(2.0, 3.0), amp=0.3):
    x = np.zeros(int(seconds * SR), np.float32)
    t = np.arange(int((event[1] - event[0]) * SR)) / SR
    x[int(event[0] * SR):int(event[1] * SR)] = amp * np.sin(2 * np.pi * 800 * t)
    return x


@pytest.mark.parametrize("snr", mixer.SNRS)
def test_mix_hits_the_active_part_snr(snr):
    clean = event_clip()
    noise = np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32) * 0.01
    mixed = mixer.mix(clean, noise, snr, np.random.default_rng(1))
    assert mixed.shape == clean.shape and mixed.dtype == np.float32
    assert mixer.measured_snr(clean, mixed) == pytest.approx(snr, abs=0.05)
    # The whole-clip SNR is far lower: SNR is NOT diluted by the clip's silence.
    whole = 20 * np.log10(mixer.rms(clean) / mixer.rms(mixed - clean))
    assert whole < snr - 3


def test_mix_is_seeded_loops_short_noise_and_clips():
    clean = event_clip(seconds=4.0)
    noise = np.random.default_rng(0).standard_normal(SR // 2).astype(np.float32)  # 0.5 s: looped
    a = mixer.mix(clean, noise, 5, np.random.default_rng(7))
    b = mixer.mix(clean, noise, 5, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, mixer.mix(clean, noise, 5, np.random.default_rng(8)))
    loud = mixer.mix(event_clip(amp=0.99), noise, 0, np.random.default_rng(0))
    assert np.abs(loud).max() <= 1.0
    with pytest.raises(ValueError, match="silent"):
        mixer.mix(clean, np.zeros(SR, np.float32), 5, np.random.default_rng(0))


def test_mix_skips_digitally_silent_stretches_of_the_noise():
    clean = event_clip(seconds=1.0, event=(0.2, 0.8))
    noise = np.zeros(6 * SR, np.float32)  # 4 s of silence, then 2 s of noise (like a long pause in speech)
    noise[-2 * SR:] = np.random.default_rng(0).standard_normal(2 * SR).astype(np.float32) * 0.1
    for seed in range(5):
        mixed = mixer.mix(clean, noise, 5, np.random.default_rng(seed))
        assert mixer.measured_snr(clean, mixed) == pytest.approx(5, abs=0.5)


def test_mix_uses_the_first_offset_when_the_noise_is_not_silent():
    clean = event_clip()
    noise = np.random.default_rng(3).standard_normal(4 * SR).astype(np.float32) * 0.001  # quiet, but noise
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    expected_seg = mixer.noise_segment(noise, clean.size, rng_b)
    mixed = mixer.mix(clean, noise, 10, rng_a)
    scale = mixer.active_rms(clean) / mixer.rms(expected_seg) / 10 ** 0.5
    np.testing.assert_allclose(mixed, np.clip(clean + expected_seg * scale, -1, 1), atol=1e-6)


# --- FSD50K selection ------------------------------------------------------------------

CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
BY = "http://creativecommons.org/licenses/by/3.0/"
NC = "http://creativecommons.org/licenses/by-nc/3.0/"
SAMPLING = "http://creativecommons.org/licenses/sampling+/1.0/"


def fake_fsd(rows):
    eval_rows = [{"fname": str(i), "labels": labels, "mids": ""} for i, (labels, _, _) in enumerate(rows)]
    info = {str(i): {"title": title, "tags": [], "license": lic, "uploader": "u"} for i, (_, lic, title) in enumerate(rows)}
    return eval_rows, info


def test_fsd50k_selection_rules():
    rows, info = fake_fsd([
        ("Doorbell,Domestic_sounds", CC0, "ding dong"),
        ("Bark,Dog,Animal", BY, "woof"),
        ("Crying_and_sobbing,Human_voice", CC0, "baby crying at night"),
        ("Crying_and_sobbing,Human_voice", CC0, "man sobbing"),  # not a baby
        ("Knock,Doorbell", CC0, "knock and ring"),  # two of our categories
        ("Shatter,Glass", SAMPLING, "smash"),  # licence not kept
        ("Glass", CC0, "clink"),  # Glass alone: not glass_break
        ("Sink_(filling_or_washing),Water", NC, "sink"),
        ("Speech,Male_speech_and_man_speaking", CC0, "talk"),
        ("Speech,Glass", CC0, "talk, clinking glasses"),  # TV background must not contain target-like sounds
        ("Music,Guitar", BY, "song"),
        ("Music", SAMPLING, "song nc"),
    ])
    picks, skipped = download_fsd50k.select(rows, info)
    got = {(p.role, p.our_label, p.fname) for p in picks}
    assert got == {("target", "doorbell", "0"), ("target", "dog_bark", "1"), ("target", "baby_cry", "2"),
                   ("target", "water_running", "7"), ("tv_background", "speech", "8"), ("tv_background", "music", "10")}
    assert skipped["maps to 2+ of our categories"] == 1 and skipped["licence Sampling+"] == 1
    assert skipped["Crying_and_sobbing without baby/infant in title or tags"] == 1
    lic = {p.fname: p.license for p in picks}
    assert lic["0"] == "CC0" and lic["1"] == "CC-BY" and lic["7"] == "CC-BY-NC"
    only_free, _ = download_fsd50k.select(rows, info, licenses=("CC0", "CC-BY"))
    assert "7" not in {p.fname for p in only_free}


def test_fsd50k_tv_backgrounds_are_capped_and_seeded():
    rows, info = fake_fsd([("Speech", CC0, "t")] * 50 + [("Music", CC0, "m")] * 50)
    a, _ = download_fsd50k.select(rows, info, tv_per_class=10)
    b, _ = download_fsd50k.select(rows, info, tv_per_class=10)
    assert [p.fname for p in a] == [p.fname for p in b]
    assert sum(p.our_label == "speech" for p in a) == 10 and sum(p.our_label == "music" for p in a) == 10


def test_fsd50k_clip_in_both_speech_and_music_is_used_once():
    rows, info = fake_fsd([("Speech,Music", CC0, "song with talking")] + [("Music", CC0, "m")] * 3)
    picks, skipped = download_fsd50k.select(rows, info, tv_per_class=10)
    assert [(p.fname, p.our_label) for p in picks if p.fname == "0"] == [("0", "speech")]
    assert len({p.fname for p in picks}) == len(picks)
    assert skipped["in both Speech and Music samples (kept once)"] == 1


def test_fsd50k_mapping_names_are_fsd50k_classes():
    assert set(download_fsd50k.FSD50K_TO_CATEGORY.values()) <= set(ds.CATEGORIES)
    assert {"smoke_alarm", "pressure_cooker", "television"} == set(download_fsd50k.NOT_IN_FSD50K)


def test_to_16k_wav_converts_44k_stereo():
    t = np.arange(44100) / 44100
    x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, 44100, format="WAV")
    audio, sr = sf.read(io.BytesIO(download_fsd50k.to_16k_wav(buf.getvalue())), dtype="float32")
    assert sr == 16000 and audio.ndim == 1 and audio.size == pytest.approx(16000, abs=2)


def test_list_fsd50k_clips_and_tv_backgrounds(tmp_path):
    root = tmp_path / "fsd50k"
    (root / "audio").mkdir(parents=True)
    rows = []
    for fid, role, label in [("1", "target", "doorbell"), ("2", "tv_background", "speech"),
                             ("3", "tv_background", "music"), ("4", "target", "dog_bark")]:
        if fid != "4":  # 4 is listed but missing
            sf.write(root / "audio" / f"{fid}.wav", np.zeros(SR, np.float32), SR)
        rows.append({"fname": fid, "file": f"audio/{fid}.wav", "role": role, "our_label": label,
                     "fsd50k_labels": "X", "license": "CC0", "license_url": "", "uploader": "u", "title": "t",
                     "freesound_url": ""})
    with open(root / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=download_fsd50k.METADATA_FIELDS)
        w.writeheader()
        w.writerows(rows)
    problems: list[str] = []
    clips = {c.path.stem: c for c in ds.list_fsd50k_clips(root, problems)}
    assert set(clips) == {"1", "2", "3"} and any("missing" in p for p in problems)
    assert (clips["1"].label, clips["1"].category, clips["1"].source) == ("doorbell", "doorbell", "fsd50k")
    assert (clips["2"].label, clips["2"].negative_type, clips["2"].category) == ("negative", "speech", None)
    bg = ds.list_tv_backgrounds(root)
    assert [c.path.stem for c in bg["speech"]] == ["2"] and [c.path.stem for c in bg["music"]] == ["3"]


# --- Freesound -------------------------------------------------------------------------

def test_freesound_template_and_licence_filter(tmp_path):
    root = tmp_path / "freesound"
    meta = ds.init_freesound(root)
    assert all((root / c).is_dir() for c in ds.CATEGORIES)
    assert meta.read_text().strip() == "file,author,url,license"
    ds.init_freesound(root)  # idempotent: never overwrites the CSV
    for name in ("cc0.wav", "by.wav", "nc.wav", "unlisted.wav"):
        sf.write(root / "doorbell" / name, np.zeros(SR, np.float32), SR)
    with open(meta, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doorbell/cc0.wav", "alice", "https://freesound.org/s/1/", "CC0"])
        w.writerow(["doorbell/by.wav", "bob", "https://freesound.org/s/2/", "https://creativecommons.org/licenses/by/4.0/"])
        w.writerow(["doorbell/nc.wav", "carol", "https://freesound.org/s/3/", "CC-BY-NC"])
    problems: list[str] = []
    clips = ds.list_freesound_clips(root, problems)
    assert sorted(c.path.name for c in clips) == ["by.wav", "cc0.wav"]
    assert all(c.source == "freesound" and c.category == "doorbell" and c.condition == "clean" for c in clips)
    assert {c.path.name: c.name for c in clips} == {"by.wav": "bob", "cc0.wav": "alice"}
    assert any("nc.wav" in p and "not CC0/CC-BY" in p for p in problems)
    assert any("unlisted.wav" in p and "no row" in p for p in problems)


# --- DEMAND ------------------------------------------------------------------------------

def test_download_demand_fetches_only_chosen_channels(tmp_path):
    zips = {}
    for env in ("DKITCHEN", "DLIVING", "DWASHING"):
        path = tmp_path / f"{env}_16k.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for ch in range(1, 17):
                buf = io.BytesIO()
                sf.write(buf, np.full(1600, ch / 100, np.float32), SR, format="WAV", subtype="PCM_16")
                z.writestr(f"{env}/ch{ch:02d}.wav", buf.getvalue())
        zips[path.name] = path
    out = tmp_path / "demand"
    written = download_demand.download(out, channels=(1, 9), sizes={k: p.stat().st_size for k, p in zips.items()},
                                       rangezip_factory=lambda key: rangezip_for([zips[key]]))
    assert sorted(p.relative_to(out).as_posix() for p in written) == [
        f"{env}/ch{c:02d}.wav" for env in ("DKITCHEN", "DLIVING", "DWASHING") for c in (1, 9)]
    noise = ds.list_demand_noise(out)
    assert list(noise) == ["DKITCHEN", "DLIVING", "DWASHING"] and all(len(v) == 2 for v in noise.values())
    audio = ds.load_audio(noise["DLIVING"][1])
    assert audio.mean() == pytest.approx(0.09, abs=1e-3)  # really channel 9
    assert "CC BY 4.0" in (out / "DEMAND_LICENSE_AND_ATTRIBUTION.txt").read_text()
    assert not list(out.rglob("*.zip"))
