"""Ground-truth tests: synthesize signals with known properties and check the tools report them."""

import asyncio

import numpy as np
import pytest
import soundfile as sf

from audio_mcp import server
from audio_mcp.audio_io import AudioError, load
from audio_mcp.synth import synthesize

SR = 44100


def write(path, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), x if x.ndim == 2 else x[:, None], sr, subtype="FLOAT")
    return str(path)


def sine(hz, seconds, amp=1.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * hz * t)


def issue_text(report):
    return " ".join(i["message"] for i in report["issues"])


def test_sine_levels_and_pitch(tmp_path):
    rep = server.audio_analyze(write(tmp_path / "a.wav", sine(440, 2.0, 0.5)))
    assert rep["levels"]["peak_dbfs"] == pytest.approx(-6.02, abs=0.05)
    assert rep["levels"]["rms_dbfs"] == pytest.approx(-9.03, abs=0.05)
    assert rep["pitch"]["median"]["note"] == "A4"
    assert abs(rep["pitch"]["median"]["hz"] - 440) < 1
    assert rep["spectrum"]["character"] == "tonal"
    assert rep["spectrum"]["spectral_peaks"][0]["note"] == "A4"
    assert rep["levels"]["clipped_samples"] == 0


def test_lufs_reference_tone(tmp_path):
    # EBU reference: a 997 Hz sine at 0 dBFS reads -3.01 LUFS; at -6 dBFS, -9.03.
    rep = server.audio_analyze(write(tmp_path / "ref.wav", sine(997, 5.0, 0.5)), pitch=False, rhythm=False)
    assert rep["levels"]["lufs_integrated"] == pytest.approx(-9.03, abs=0.2)


def test_chord_flagged_as_polyphonic(tmp_path):
    chord = sum(sine(f, 2.0, 0.2) for f in (349.2, 440.0, 523.3, 659.3))  # Fmaj7
    rep = server.audio_analyze(write(tmp_path / "chord.wav", chord), rhythm=False)
    assert rep["pitch"].get("polyphony_likely") is True
    assert not server.audio_analyze(write(tmp_path / "solo.wav", sine(440, 2.0, 0.5)), rhythm=False)["pitch"].get(
        "polyphony_likely"
    )


def test_clipping_detected(tmp_path):
    rep = server.audio_analyze(
        write(tmp_path / "clip.wav", np.clip(sine(220, 1.0, 2.0), -1, 1)), pitch=False, rhythm=False
    )
    assert rep["levels"]["clipped_samples"] > 100
    assert any(i["severity"] == "error" and "clipping" in i["message"].lower() for i in rep["issues"])


def test_click_train_tempo_and_onsets(tmp_path):
    x = np.zeros(int(8 * SR))
    for k in range(16):  # 120 BPM
        i = int(k * 0.5 * SR)
        x[i : i + 200] = np.hanning(200) * 0.9
    rep = server.audio_analyze(write(tmp_path / "clicks.wav", x), pitch=False)
    assert rep["rhythm"]["onset_count"] == 16
    assert rep["rhythm"]["tempo_bpm"] == pytest.approx(120, abs=2)
    assert rep["rhythm"]["first_onsets_s"][1] == pytest.approx(0.5, abs=0.03)


def test_out_of_phase_stereo(tmp_path):
    s = sine(300, 1.0, 0.5)
    rep = server.audio_analyze(write(tmp_path / "phase.wav", np.stack([s, -s], axis=1)), pitch=False, rhythm=False)
    assert rep["stereo"]["correlation"] == pytest.approx(-1.0, abs=0.01)
    assert "out of phase" in issue_text(rep)


def test_silence_and_edges(tmp_path):
    x = np.concatenate([np.zeros(SR), sine(500, 1.0, 0.5), np.zeros(SR // 2)])
    rep = server.audio_analyze(write(tmp_path / "pad.wav", x), pitch=False, rhythm=False)
    assert rep["silence"]["leading_s"] == pytest.approx(1.0, abs=0.02)
    assert rep["silence"]["trailing_s"] == pytest.approx(0.5, abs=0.02)
    assert "leading silence" in issue_text(rep)


def test_melody_transcription(tmp_path):
    seq = [["C4", 0.3], ["E4", 0.3], ["G4", 0.3], ["C5", 0.6]]
    out = tmp_path / "mel.wav"
    server.audio_synthesize({"layers": [{"wave": "triangle", "sequence": seq}]}, str(out))
    mel = server.audio_melody(str(out))
    assert [n["note"] for n in mel["notes"]] == ["C4", "E4", "G4", "C5"]
    assert mel["notes"][3]["dur_s"] == pytest.approx(0.6, abs=0.08)


def test_melody_keeps_repeated_notes(tmp_path):
    tune = [["E4", 0.25], ["D4", 0.25], ["C4", 0.25], ["D4", 0.25], ["E4", 0.25], ["E4", 0.25], ["E4", 0.5]]
    out = tmp_path / "mary.wav"
    server.audio_synthesize({"layers": [{"wave": "triangle", "sequence": tune}]}, str(out))
    assert [n["note"] for n in server.audio_melody(str(out))["notes"]] == [n for n, _ in tune]


def test_synth_sweep_glide(tmp_path):
    out = tmp_path / "sweep.wav"
    server.audio_synthesize({"layers": [{"wave": "sine", "freq": 200, "freq_end": 800, "duration": 1.0}]}, str(out))
    rep = server.audio_analyze(str(out), rhythm=False)
    assert rep["pitch"]["glide_semitones"] == pytest.approx(18, abs=4)  # quartile medians, so less than the full 24


def test_process_normalize_lufs_and_compare(tmp_path):
    src = write(tmp_path / "quiet.wav", sine(1000, 3.0, 0.05))
    res = server.audio_process(src, [{"op": "normalize_lufs", "lufs": -16}])
    assert res["after"]["lufs_integrated"] == pytest.approx(-16, abs=0.2)
    assert res["output"].endswith("quiet_processed.wav")
    cmp = server.audio_compare(src, res["output"])
    assert "same audio, only the level differs" in cmp["null_test"]["verdict"]
    assert cmp["metrics"]["levels.lufs_integrated"]["change"] > 10


def test_process_never_overwrites_by_default(tmp_path):
    src = write(tmp_path / "x.wav", sine(1000, 1.0, 0.5))
    with pytest.raises(AudioError, match="already exists"):
        server.audio_process(src, [{"op": "gain", "db": -3}], output=src)


def test_bad_operation_message(tmp_path):
    src = write(tmp_path / "x.wav", sine(1000, 1.0, 0.5))
    with pytest.raises(AudioError, match="Unknown op 'louder'"):
        server.audio_process(src, [{"op": "louder"}])
    with pytest.raises(AudioError, match="Bad parameters for 'gain'"):
        server.audio_process(src, [{"op": "gain", "decibels": 3}])


def test_loop_crossfade_removes_seam_click(tmp_path):
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(2 * SR))
    x = 0.8 * x / np.max(np.abs(x))
    src = write(tmp_path / "loop.wav", x)
    assert load(src) and server.audio_analyze(src, pitch=False, rhythm=False)["edges"]["loop_seam"]["likely_click"]
    res = server.audio_process(src, [{"op": "loop_crossfade", "seconds": 0.1}])
    assert not server.audio_analyze(res["output"], pitch=False, rhythm=False)["edges"]["loop_seam"]["likely_click"]


def test_eq_changes_only_its_band(tmp_path):
    rng = np.random.default_rng(2)
    src = write(tmp_path / "noise.wav", 0.2 * rng.standard_normal(3 * SR))
    res = server.audio_process(src, [{"op": "high_shelf", "hz": 6000, "db": -12}])
    cmp = server.audio_compare(src, res["output"])
    bands = cmp["band_level_change_db"]
    assert bands["brilliance"] < -8
    assert abs(bands["bass"]) < 0.5


def test_view_png(tmp_path):
    src = write(tmp_path / "v.wav", sine(440, 1.0, 0.5))
    img = server.audio_view(src, save_to=str(tmp_path / "v.png"), pitch_overlay=True)
    assert img.data[:4] == b"\x89PNG"
    assert (tmp_path / "v.png").stat().st_size > 10000


def test_timeline_streams(tmp_path):
    x = np.concatenate([sine(300, 2.0, 0.05), sine(300, 2.0, 0.8)])
    tl = server.audio_timeline(write(tmp_path / "t.wav", x), segments=4)
    rows = tl["segments"]
    assert len(rows) == 4
    assert rows[3]["rms_dbfs"] - rows[0]["rms_dbfs"] == pytest.approx(24.1, abs=0.5)


def test_info_directory(tmp_path):
    write(tmp_path / "one.wav", sine(440, 0.5))
    (tmp_path / "notes.txt").write_text("not audio")
    info = server.audio_info(str(tmp_path))
    assert [f["path"].endswith("one.wav") for f in info["files"]] == [True]


def test_short_clip_and_missing_file(tmp_path):
    rep = server.audio_analyze(write(tmp_path / "blip.wav", sine(2000, 0.05, 0.5)))
    assert rep["levels"]["lufs_integrated"] is None and "lufs_note" in rep["levels"]
    with pytest.raises(AudioError, match="File not found"):
        server.audio_analyze(str(tmp_path / "nope.wav"))


def test_synth_note_names_and_pan():
    x, sr = synthesize({"layers": [{"wave": "sine", "freq": "A4", "duration": 0.5, "pan": -1}]})
    assert x.shape[1] == 2
    assert np.max(np.abs(x[:, 1])) < 1e-3  # hard left


def test_limiter_holds_true_peak_ceiling(tmp_path):
    rng = np.random.default_rng(3)
    x = sine(60, 4.0, 0.6)
    for k in range(16):
        i = int(k * 0.25 * SR)
        x[i : i + 1500] += rng.standard_normal(1500) * np.exp(-np.arange(1500) / 200) * 2
    src = write(tmp_path / "hot.wav", 2 * x)
    res = server.audio_process(src, [{"op": "limit", "ceiling_db": -1}])
    rep = server.audio_analyze(res["output"], pitch=False, rhythm=False)
    assert rep["levels"]["true_peak_dbtp"] <= -0.98
    assert rep["levels"]["clipped_samples"] == 0


def test_normalize_lufs_with_limiter_reaches_target(tmp_path):
    rng = np.random.default_rng(4)
    x = 0.05 * rng.standard_normal(3 * SR)
    x[SR : SR + 50] = 0.9  # one big transient that would otherwise cap the gain
    src = write(tmp_path / "spiky.wav", x)
    capped = server.audio_process(src, [{"op": "normalize_lufs", "lufs": -14}], output=str(tmp_path / "capped.wav"))
    assert capped["after"]["lufs_integrated"] < -18
    limited = server.audio_process(
        src, [{"op": "normalize_lufs", "lufs": -14, "limit": True}], output=str(tmp_path / "lim.wav")
    )
    assert limited["after"]["lufs_integrated"] == pytest.approx(-14, abs=0.2)
    assert limited["after"]["peak_dbfs"] <= -1.0


def test_compressor_reduces_dynamic_range(tmp_path):
    x = np.concatenate([sine(440, 1.0, 0.03), sine(440, 1.0, 0.9)])
    src = write(tmp_path / "dyn.wav", x)
    res = server.audio_process(src, [{"op": "compress", "threshold_db": -30, "ratio": 4, "attack_ms": 5}])
    tl = server.audio_timeline(res["output"], segments=2)["segments"]
    assert tl[1]["rms_dbfs"] - tl[0]["rms_dbfs"] < 15  # was ~29.5 dB apart


def test_events_ground_truth(tmp_path):
    out = tmp_path / "seq.wav"
    spec = {
        "layers": [
            {
                "wave": "square",
                "freq": 1800,
                "freq_end": 200,
                "start": 0.1,
                "duration": 0.25,
                "envelope": {"attack": 0.002, "release": 0.05},
            },
            {"wave": "noise", "start": 0.6, "duration": 0.08, "envelope": {"attack": 0.001, "release": 0.06}},
            {"wave": "sine", "freq": "C5", "start": 1.0, "duration": 0.4},
        ]
    }
    server.audio_synthesize(spec, str(out))
    ev = server.audio_events(str(out))
    assert ev["event_count"] == 3
    starts = [e["start_s"] for e in ev["events"]]
    assert starts == pytest.approx([0.1, 0.6, 1.0], abs=0.03)
    zap, hit, tone = ev["events"]
    assert zap["glide_semitones"] < -24 and zap["attack_ms"] <= 10
    assert hit["character"] == "noisy"
    assert tone["pitch"]["note"] == "C5" and "glide_semitones" not in tone


def test_batch_survey_and_process(tmp_path):
    lib = tmp_path / "sfx"
    lib.mkdir()
    for i, amp in enumerate((0.3, 0.3, 0.3, 0.02)):
        write(lib / f"s{i}.wav", sine(500 + 100 * i, 1.0, amp))
    write(lib / "odd_rate.wav", sine(500, 1.0, 0.3, sr=22050), sr=22050)
    survey = server.audio_batch_survey(str(lib))
    text = " ".join(survey["findings"])
    assert "s3.wav" in text and "quieter" in text
    assert "Mixed sample rate" in text
    ops = survey["suggested_fixes"][0]["audio_batch_process_operations"]
    res = server.audio_batch_process(str(lib), ops, str(tmp_path / "out"))
    assert res["processed"] == 5 and not res["errors"]
    after = server.audio_batch_survey(str(tmp_path / "out"))
    assert after["stats"]["spread_lufs_p10_p90"] < 0.5
    with pytest.raises(AudioError, match="contains the input files"):
        server.audio_batch_process(str(lib), ops, str(lib))


def test_hum_detected_and_fix_suggested(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(3 * SR) / SR
    x = sine(700, 3.0, 0.2) * (np.sin(2 * np.pi * 2 * t) > 0)  # gated tone, so there are quiet gaps
    x += 0.02 * np.sin(2 * np.pi * 60 * t) + 0.006 * np.sin(2 * np.pi * 180 * t) + 0.002 * rng.standard_normal(len(t))
    rep = server.audio_analyze(write(tmp_path / "hum.wav", x), rhythm=False)
    hum = [i for i in rep["issues"] if "Mains hum at 60 Hz" in i["message"]]
    assert hum and '"op":"eq","hz":180' in hum[0]["message"]


def test_denoise_lowers_noise_floor_keeps_signal(tmp_path):
    rng = np.random.default_rng(6)
    tone = sine(440, 4.0, 0.3)
    tone[:SR] = 0  # first second is noise only
    x = tone + 0.01 * rng.standard_normal(len(tone))
    src = write(tmp_path / "noisy.wav", x)
    res = server.audio_process(src, [{"op": "denoise", "reduction_db": 18, "noise_start": 0, "noise_end": 0.9}])
    assert "noise ~-40" in res["operations"][0]  # 0.01 RMS white noise = -40 dBFS
    before = server.audio_timeline(src, segments=4)["segments"]
    after = server.audio_timeline(res["output"], segments=4)["segments"]
    assert after[0]["rms_dbfs"] < before[0]["rms_dbfs"] - 12  # noise-only section much quieter
    assert after[2]["rms_dbfs"] == pytest.approx(before[2]["rms_dbfs"], abs=0.5)  # tone section preserved


def test_trim_ignores_noise_spikes(tmp_path):
    rng = np.random.default_rng(7)
    x = np.concatenate([np.zeros(SR), sine(440, 1.0, 0.5)]) + 0.0035 * rng.standard_normal(2 * SR)  # hiss ~-49 dBFS
    res = server.audio_process(
        write(tmp_path / "lead.wav", x), [{"op": "trim_silence", "threshold_db": -45, "pad_ms": 0}]
    )
    assert res["after"]["duration_s"] == pytest.approx(1.0, abs=0.04)


def test_transcribe_windows_tts(tmp_path):
    """Ground truth from the Windows speech synthesizer; skipped where it or faster-whisper is unavailable."""
    import shutil
    import subprocess
    import sys

    pytest.importorskip("ctranslate2")
    if sys.platform != "win32" or not shutil.which("powershell"):
        pytest.skip("needs Windows text-to-speech")
    wav = tmp_path / "speech.wav"
    script = (
        "Add-Type -AssemblyName System.Speech; $v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$v.SetOutputToWaveFile('{wav}'); $v.Speak('Green bottles are standing on the old stone wall.'); $v.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)
    out = server.audio_transcribe(str(wav), model="base", language="en")
    assert "green bottles are standing" in out["text"].lower().replace(",", "")
    assert out["words"][0]["start_s"] < 0.5


def test_mcp_tools_registered():
    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "audio_info",
        "audio_analyze",
        "audio_timeline",
        "audio_view",
        "audio_melody",
        "audio_events",
        "audio_transcribe",
        "audio_batch_survey",
        "audio_batch_process",
        "audio_compare",
        "audio_process",
        "audio_synthesize",
        "audio_check_delivery",
        "audio_check_game_project",
    }
    assert all(t.description for t in tools)


def test_compare_silence_and_offsets(tmp_path):
    silent = write(tmp_path / "silent.wav", np.zeros(SR))
    assert "both are silent" in server.audio_compare(silent, silent)["null_test"]["verdict"]
    tt = np.arange(2 * SR) / SR
    x = 0.4 * np.sin(2 * np.pi * (200 * tt + 300 * tt**2))  # chirp: no repeating period, so one true alignment
    a = write(tmp_path / "a.wav", x)
    b = write(tmp_path / "b.wav", np.concatenate([np.zeros(441), x]))  # B starts 10 ms later
    nt = server.audio_compare(a, b)["null_test"]
    assert nt["offset_ms"] == pytest.approx(10.0, abs=0.1)
    assert nt["verdict"].startswith("identical")
    # Musical material is nearly periodic; normalised correlation must still find the true offset, not ~0.
    tone = sine(220, 2.0, 0.3) + sine(330, 2.0, 0.2) + 0.01 * np.random.default_rng(9).standard_normal(2 * SR)
    a = write(tmp_path / "ta.wav", tone)
    b = write(tmp_path / "tb.wav", np.concatenate([np.zeros(2205), tone]))  # 50 ms late
    assert server.audio_compare(a, b)["null_test"]["offset_ms"] == pytest.approx(50.0, abs=0.1)


def test_events_on_continuous_melody(tmp_path):
    """No quiet gaps between notes: events must still be found (split at the note attacks)."""
    tune = [["C4", 0.3], ["E4", 0.3], ["G4", 0.3], ["E4", 0.3], ["C4", 0.6]]
    out = tmp_path / "legato.wav"
    server.audio_synthesize(
        {
            "layers": [
                {"wave": "sine", "freq": 220, "duration": 1.8, "gain_db": -12},
                {"wave": "triangle", "sequence": tune},
            ]
        },
        str(out),
    )
    ev = server.audio_events(str(out))
    assert ev["event_count"] == len(tune)
    assert [e["start_s"] for e in ev["events"]] == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.2], abs=0.04)
    silent = write(tmp_path / "silent.wav", np.zeros(SR))
    assert server.audio_events(silent)["event_count"] == 0


@pytest.mark.parametrize("hz", [82.41, 220.0, 440.0, 997.0, 1760.0])
def test_pitch_accuracy_within_one_cent(hz):
    """Found while building the reference-signal pack: interpolating YIN's normalised curve read 1760 Hz 10 cents sharp."""
    from audio_mcp import pitch

    sr = 48000
    times, f0, _ = pitch.track(sine(hz, 2.0, 0.5, sr=sr), sr)
    assert abs(1200 * np.log2(np.nanmedian(f0) / hz)) < 1.0


@pytest.mark.parametrize("wave", ["noise", "pink", "brown"])
def test_coloured_noise_is_noise_like(tmp_path, wave):
    """Whole-spectrum flatness called pink 'mixed' and brown 'tonal'; per-octave flatness gets every colour right."""
    out = tmp_path / f"{wave}.wav"
    server.audio_synthesize({"seed": 1, "layers": [{"wave": wave, "duration": 3}]}, str(out))
    assert server.audio_analyze(str(out), pitch=False, rhythm=False)["spectrum"]["character"] == "noise-like"


def test_tones_stay_tonal(tmp_path):
    out = tmp_path / "saw.wav"
    server.audio_synthesize({"layers": [{"wave": "saw", "freq": 110, "duration": 2}]}, str(out))
    assert server.audio_analyze(str(out), pitch=False, rhythm=False)["spectrum"]["character"] == "tonal"


def test_uncorrelated_stereo_is_wide_not_a_phase_error(tmp_path):
    """Independent channels have correlation ~0 (often slightly negative); that's wide stereo, not a fault."""
    out = tmp_path / "wide.wav"
    server.audio_synthesize(
        {
            "seed": 5,
            "layers": [{"wave": "noise", "duration": 4, "pan": -1}, {"wave": "noise", "duration": 4, "pan": 1}],
        },
        str(out),
    )
    rep = server.audio_analyze(str(out), pitch=False, rhythm=False)
    assert abs(rep["stereo"]["correlation"]) < 0.05
    assert rep["stereo"]["width"] == "wide"
    assert "out of phase" not in issue_text(rep)


def test_clicks_count_as_events_including_one_at_the_very_end(tmp_path):
    """A 4 ms click is an event, and a click in the final frames is an onset (mirror of the t=0 edge case)."""
    seq = [["x", 0.004], ["rest", 0.496]] * 7 + [["x", 0.004]]  # 8 clicks, the last one ends the file
    out = tmp_path / "clicks.wav"
    server.audio_synthesize(
        {
            "seed": 6,
            "layers": [
                {"wave": "noise", "highpass": 1000, "sequence": seq, "envelope": {"attack": 0.0005, "release": 0.003}}
            ],
        },
        str(out),
    )
    assert server.audio_analyze(str(out), pitch=False)["rhythm"]["onset_count"] == 8
    assert server.audio_events(str(out))["event_count"] == 8


def test_test_signal_waves_are_exact():
    x, sr = synthesize(
        {"sample_rate": 48000, "normalize_peak_db": None, "layers": [{"wave": "impulse", "duration": 1}]}
    )
    assert np.flatnonzero(x[:, 0]).tolist() == [0] and x[0, 0] == 1.0
    x, _ = synthesize(
        {"sample_rate": 48000, "normalize_peak_db": None, "layers": [{"wave": "impulse", "freq": 4, "duration": 1}]}
    )
    assert np.flatnonzero(x[:, 0]).tolist() == [0, 12000, 24000, 36000]
    x, _ = synthesize({"normalize_peak_db": None, "layers": [{"wave": "dc", "duration": 0.1, "gain_db": -20}]})
    assert np.allclose(x, 0.1)
    x, _ = synthesize({"normalize_peak_db": None, "layers": [{"wave": "silence", "duration": 0.5}]})
    assert not x.any()


def test_oscillator_phase_and_intersample_peak(tmp_path):
    """fs/4 sine at 45 degrees: samples sit at +-0.707 of the true peak, so true peak is 3.01 dB above sample peak.
    Short fades matter: an abruptly cut sine really does overshoot further (+0.1 dB) at its edges."""
    out = tmp_path / "isp.wav"
    spec = {
        "sample_rate": 48000,
        "normalize_peak_db": -1,
        "layers": [
            {"wave": "sine", "freq": 12000, "phase": 45, "duration": 1, "envelope": {"attack": 0.01, "release": 0.01}}
        ],
    }
    server.audio_synthesize(spec, str(out))
    lv = server.audio_analyze(str(out), pitch=False, rhythm=False)["levels"]
    assert lv["peak_dbfs"] == pytest.approx(-1.0, abs=0.01)
    assert lv["true_peak_dbtp"] == pytest.approx(2.01, abs=0.05)


def test_polarity_and_normalize_rms(tmp_path):
    out = tmp_path / "st.wav"
    server.audio_synthesize({"stereo": True, "layers": [{"wave": "sine", "freq": 300, "duration": 2}]}, str(out))
    res = server.audio_process(str(out), [{"op": "polarity", "channel": 1}, {"op": "normalize_rms", "db": -20}])
    rep = server.audio_analyze(res["output"], pitch=False, rhythm=False)
    assert rep["stereo"]["correlation"] == pytest.approx(-1.0, abs=0.001)
    assert rep["levels"]["rms_dbfs"] == pytest.approx(-20.0, abs=0.01)
    with pytest.raises(AudioError, match="channel must be between 0 and 1"):
        server.audio_process(str(out), [{"op": "polarity", "channel": 2}])


def test_short_sounds_normalize_by_momentary_loudness(tmp_path):
    """Integrated LUFS is undefined under 0.4 s; momentary max lets short effects be matched to each other."""
    blip = write(tmp_path / "blip.wav", sine(1000, 0.1, 0.9))
    with pytest.raises(AudioError, match="momentary_max"):
        server.audio_process(blip, [{"op": "normalize_lufs", "lufs": -20}], output=str(tmp_path / "x.wav"))
    res = server.audio_process(blip, [{"op": "normalize_lufs", "lufs": -20, "measure": "momentary_max"}])
    assert server.audio_analyze(res["output"], rhythm=False)["levels"]["lufs_momentary_max"] == pytest.approx(
        -20, abs=0.1
    )


def test_normalize_to_true_peak(tmp_path):
    out = tmp_path / "isp.wav"
    spec = {
        "sample_rate": 48000,
        "layers": [
            {"wave": "sine", "freq": 12000, "phase": 45, "duration": 1, "envelope": {"attack": 0.01, "release": 0.01}}
        ],
    }
    server.audio_synthesize(spec, str(out))
    res = server.audio_process(str(out), [{"op": "normalize_peak", "db": -1, "true_peak": True}])
    lv = server.audio_analyze(res["output"], pitch=False, rhythm=False)["levels"]
    assert lv["true_peak_dbtp"] == pytest.approx(-1.0, abs=0.05)
    assert lv["peak_dbfs"] < -3.5  # sample peak sits ~3 dB lower for this signal


def test_single_tonal_peak_on_a_mains_harmonic_is_not_hum(tmp_path):
    """A note at 250 Hz (5 x 50) in non-tonal material used to be reported as 50 Hz hum."""
    rng = np.random.default_rng(11)
    x = sine(250, 2.0, 0.3) + 0.05 * rng.standard_normal(2 * SR)
    rep = server.audio_analyze(write(tmp_path / "note.wav", x), rhythm=False)
    assert "hum" not in issue_text(rep)


@pytest.mark.parametrize("wave", ["pink", "brown"])
def test_coloured_noise_has_no_sub_audio_energy(wave):
    x, sr = synthesize({"sample_rate": 48000, "seed": 3, "layers": [{"wave": wave, "duration": 0.2}]})
    spec = np.abs(np.fft.rfft(x[:, 0])) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    assert spec[f < 15].sum() / spec.sum() < 0.01
    assert abs(float(x.mean())) < 0.01
