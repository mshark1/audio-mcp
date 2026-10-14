"""audio_check_delivery against constructed signals: known passes, known failures per criterion, and the fix loop
(apply the returned operations with audio_process, re-check, expect a pass)."""

import numpy as np
import pytest
import soundfile as sf

from audio_mcp import analysis, delivery, server
from audio_mcp.audio_io import AudioError

SR = 44100
RNG = np.random.default_rng(7)


def write(path, x, sr=SR):
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), x if x.ndim == 2 else x[:, None], sr, subtype="FLOAT")
    return str(path)


def db(v):
    return 10 ** (v / 20)


def speechlike(seconds, sr=SR, lead=1.5, tail=1.5, phrase=1.2, pause=0.8):
    """Harmonic 'voice' (f0 ~120 Hz) in 4 Hz syllables, grouped into phrases with pauses; 0 dB RMS while talking."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ph = 2 * np.pi * np.cumsum(120 + 15 * np.sin(2 * np.pi * 0.7 * t)) / sr
    voice = sum(np.sin(k * ph) / k for k in range(1, 15))
    env = (0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)) ** 2
    gate = np.zeros(n)
    s = lead
    while s + phrase <= seconds - tail:
        gate[int(s * sr) : int((s + phrase) * sr)] = 1
        s += phrase + pause
    x = voice * env * gate
    return x / np.sqrt(np.mean(x[gate > 0] ** 2))


def noise(n, level_db):
    return db(level_db) * RNG.standard_normal(n)


def check(path, spec, **kw):
    return server.audio_check_delivery(path, spec, **kw)


def status(rep, name):
    return next(c["status"] for c in rep["checks"] if c["check"] == name)


def failing(rep):
    return {c["check"] for c in rep["checks"] if c["status"] == "fail"}


def apply_fix(path, rep, tmp_path, name="fixed.wav"):
    ops = rep["fix"]["operations"]
    assert ops, rep["fix"]
    return server.audio_process(path, ops, output=str(tmp_path / name), overwrite=True)["output"]


def fix_loop(path, spec, tmp_path, expect=("pass", "pass with warnings")):
    rep = check(path, spec)
    assert rep["verdict"] == "fail" or rep["verdict"] == "pass with warnings", rep
    assert rep["fix"]["verified"] is True, rep["fix"]
    after = check(apply_fix(path, rep, tmp_path), spec)
    assert after["verdict"] in expect, after["checks"]
    assert after["verdict"] == rep["fix"]["expected_verdict"]
    return rep, after


def op_index(ops, name):
    return [o["op"] for o in ops].index(name)


# ---------------------------------------------------------------- helpers and the spec registry


def test_windowed_rms_matches_direct_computation():
    x = RNG.standard_normal((SR * 3, 2)).astype(np.float32) * np.linspace(0.01, 1, SR * 3)[:, None]
    got = analysis.windowed_rms_db(x, SR, 0.5, 0.05)
    hop, k = int(0.05 * SR), 10
    direct = [10 * np.log10(np.mean(x[i * hop : i * hop + k * hop].astype(np.float64) ** 2)) for i in range(len(got))]
    assert len(got) == (len(x) // hop) - k + 1
    assert np.allclose(got, direct, atol=1e-6)
    assert analysis.windowed_rms_db(x[:100], SR).shape == (1,)  # shorter than a window: one whole-signal value


def test_spec_list_and_unknown_spec(tmp_path):
    listed = check("unused.wav", "list")["specs"]
    assert (
        set(listed)
        == set(delivery.SPECS)
        >= {
            "apple_podcasts",
            "spotify",
            "youtube",
            "ebu_r128",
            "atsc_a85",
            "acx_audiobook",
            "game_sfx",
        }
    )
    assert "not an official spec" in listed["game_sfx"]
    with pytest.raises(AudioError, match="Unknown delivery spec 'netflix'.*ebu_r128: European broadcast"):
        check(write(tmp_path / "a.wav", np.zeros(SR)), "netflix")


def test_every_spec_cites_dated_sources():
    for name, spec in delivery.SPECS.items():
        assert spec["notes"], name
        for s in spec["sources"]:
            assert s["url"].startswith("https://") and s["accessed"] == delivery.SOURCES_CHECKED
        if spec["official"]:
            assert spec["sources"], name


# ---------------------------------------------------------------- loudness specs

LOUDNESS = {  # spec: (target, true-peak limit)
    "apple_podcasts": (-16, -1),
    "spotify": (-14, -1),
    "youtube": (-14, -1),
    "ebu_r128": (-23, -1),
    "atsc_a85": (-24, -2),
}


@pytest.mark.parametrize("spec", list(LOUDNESS))
def test_loudness_specs_pass_on_target_and_fix_quiet_files(tmp_path, spec):
    target, tp_max = LOUDNESS[spec]
    x = speechlike(8.0, sr=48000)
    stereo = np.stack([x, x], 1) * db(-30)
    quiet = write(tmp_path / "quiet.wav", stereo, sr=48000)
    rep, after = fix_loop(quiet, spec, tmp_path)
    assert status(rep, "integrated_loudness") == ("warn" if spec in ("spotify", "youtube") else "fail")
    assert rep["fix"]["operations"][-1]["op"] == "normalize_lufs"
    assert rep["fix"]["operations"][-1]["ceiling_db"] <= tp_max
    assert after["verdict"] == "pass"
    lufs = float(next(c for c in after["checks"] if c["check"] == "integrated_loudness")["measured"].split()[0])
    assert lufs == pytest.approx(target, abs=0.2)
    assert after["fix"] is None
    assert after["spec"]["name"] == spec and after["spec"]["sources"]


def test_hot_peaks_fail_true_peak_and_the_limiter_fixes_them(tmp_path):
    x = speechlike(8.0) * db(-20)
    x[int(2.0 * SR) :: SR // 3] = 0.99  # isolated spikes: loudness about right, true peak far too high
    path = write(tmp_path / "spiky.wav", np.stack([x, x], 1))
    rep = check(path, "ebu_r128")
    assert "true_peak" in failing(rep)
    rep, after = fix_loop(path, "ebu_r128", tmp_path)
    assert after["verdict"] == "pass"


def test_atsc_tolerance_and_ceiling(tmp_path):
    x = speechlike(6.0) * db(-30)
    st = np.stack([x, x], 1)
    base = write(tmp_path / "base.wav", st)
    at = server.audio_process(base, [{"op": "normalize_lufs", "lufs": -25.5}], output=str(tmp_path / "a.wav"))
    assert check(at["output"], "atsc_a85")["verdict"] == "pass"  # -25.5 is inside +/-2
    assert "integrated_loudness" in failing(check(at["output"], "ebu_r128"))  # but not inside R 128's 0.2


def test_r128_live_tolerance_is_a_warning(tmp_path):
    x = speechlike(6.0) * db(-30)
    base = write(tmp_path / "base.wav", np.stack([x, x], 1))
    out = server.audio_process(base, [{"op": "normalize_lufs", "lufs": -23.6}], output=str(tmp_path / "a.wav"))
    rep = check(out["output"], "ebu_r128")
    assert rep["verdict"] == "pass with warnings" and status(rep, "integrated_loudness") == "warn"


def test_spotify_stricter_ceiling_for_loud_masters(tmp_path):
    t = np.arange(4 * SR) / SR
    x = 0.8 * np.sin(2 * np.pi * 997 * t)  # -1.9 dBFS peak, about -1.9 LUFS as stereo: louder than -14
    path = write(tmp_path / "loud.wav", np.stack([x, x], 1))
    rep = check(path, "spotify")
    tp = next(c for c in rep["checks"] if c["check"] == "true_peak")
    assert tp["required"] == "<= -2 dBTP" and tp["status"] == "fail"
    assert status(rep, "integrated_loudness") == "warn"  # loud is a warning: Spotify turns it down
    fix_loop(path, "spotify", tmp_path)


def test_apple_mono_and_noisy_speech_fix_order(tmp_path):
    x = speechlike(12.0) * db(-26) + noise(int(12.0 * SR), -56)
    path = write(tmp_path / "noisy_mono.wav", x)
    rep, after = fix_loop(path, "apple_podcasts", tmp_path)
    assert status(rep, "channels") == "warn" and status(rep, "background_noise") == "warn"
    ops = rep["fix"]["operations"]
    assert op_index(ops, "to_stereo") < op_index(ops, "denoise") < op_index(ops, "normalize_lufs")
    assert op_index(ops, "highpass") < op_index(ops, "normalize_lufs")
    assert any("audio_transcribe" in s for s in rep["fix"]["steps"])
    assert after["channels"] == 2 and after["verdict"] == "pass"


def test_short_clip_fails_loudness_specs_cleanly(tmp_path):
    t = np.arange(int(0.2 * SR)) / SR
    rep = check(write(tmp_path / "short.wav", 0.5 * np.sin(2 * np.pi * 440 * t)), "ebu_r128")
    assert rep["verdict"] == "fail" and "integrated_loudness" in failing(rep)
    assert rep["fix"]["operations"] == [] and rep["fix"]["cannot_fix"]


# ---------------------------------------------------------------- ACX


def acx_file(tmp_path, name, level_db=-20.0, noise_db=-70.0, seconds=12.0, sr=SR, **kw):
    x = speechlike(seconds, sr=sr, **kw) * db(level_db + 3)  # RMS while talking is ~3 dB above whole-file RMS
    x = x + db(noise_db) * np.random.default_rng(3).standard_normal(len(x))
    return write(tmp_path / name, x, sr=sr)


def test_acx_good_file_passes_with_only_the_mp3_warning(tmp_path):
    rep = check(acx_file(tmp_path, "good.wav"), "acx_audiobook")
    assert failing(rep) == set(), rep["checks"]
    assert rep["verdict"] == "pass with warnings"
    assert status(rep, "format") == "warn"  # WAV master; ACX wants 192 kbps+ CBR MP3
    for name in ("rms", "peak", "noise_floor", "room_tone_head", "room_tone_tail", "sample_rate", "length"):
        assert status(rep, name) == "pass", name
    assert rep["fix"]["operations"] == []


def test_acx_measures_noise_floor_as_quietest_half_second(tmp_path):
    n = 10 * SR
    hum = db(-65) * np.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)  # a steady -65 dB RMS background
    x = speechlike(10.0, lead=2.0) * db(-20) + hum
    rep = check(write(tmp_path / "nf.wav", x), "acx_audiobook")
    nf = next(c for c in rep["checks"] if c["check"] == "noise_floor")
    assert float(nf["measured"].split()[0]) == pytest.approx(-65.0, abs=0.2)
    head = next(c for c in rep["checks"] if c["check"] == "room_tone_head")
    assert float(head["measured"].split()[0]) == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize(
    "case,kwargs,fails",
    [
        ("quiet", {"level_db": -30.0, "noise_db": -80.0}, {"rms"}),
        ("noisy", {"level_db": -28.0, "noise_db": -56.0}, {"rms", "noise_floor"}),
        ("long_head", {"lead": 7.0, "seconds": 16.0}, {"room_tone_head"}),
        ("rate", {"sr": 48000}, {"sample_rate"}),
    ],
)
def test_acx_failures_and_fix_loop(tmp_path, case, kwargs, fails):
    path = acx_file(tmp_path, f"{case}.wav", **kwargs)
    rep = check(path, "acx_audiobook")
    assert failing(rep) == fails, rep["checks"]
    rep, after = fix_loop(path, "acx_audiobook", tmp_path, expect=("pass with warnings",))
    assert failing(after) == set()
    ops = rep["fix"]["operations"]
    if "noise_floor" in fails:
        assert op_index(ops, "highpass") < op_index(ops, "denoise") < op_index(ops, "normalize_rms")
        assert op_index(ops, "normalize_rms") < op_index(ops, "limit")
        assert any("audio_transcribe" in s for s in rep["fix"]["steps"])
    if case == "long_head":
        assert ops[0]["op"] == "crop"


def test_acx_hot_peaks(tmp_path):
    x = speechlike(12.0) * db(-17)
    x[int(3 * SR) :: SR] = 0.95
    path = write(tmp_path / "hot.wav", x)
    rep = check(path, "acx_audiobook")
    assert "peak" in failing(rep)
    fix_loop(path, "acx_audiobook", tmp_path, expect=("pass with warnings",))


def test_acx_noise_too_high_to_fix_is_said_not_hidden(tmp_path):
    path = acx_file(tmp_path, "very_noisy.wav", level_db=-24.0, noise_db=-38.0)
    rep = check(path, "acx_audiobook")
    assert "noise_floor" in failing(rep)
    assert any("Noise floor" in c for c in rep["fix"]["cannot_fix"])
    assert not any(o["op"] == "denoise" for o in rep["fix"]["operations"])
    assert rep["fix"]["verified"] is False and "noise_floor" in rep["fix"]["verification"]


# ---------------------------------------------------------------- game SFX


def blip(seconds=0.3, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * 600 * t) * np.sin(np.pi * t / t[-1]) ** 2


def test_sfx_good_passes_and_is_labelled_unofficial(tmp_path):
    rep = check(write(tmp_path / "good.wav", blip()), "game_sfx")
    assert rep["verdict"] == "pass", rep["checks"]
    assert rep["spec"]["status"] == "recommended practice, not an official spec"


@pytest.mark.parametrize(
    "case,make,fails",
    [
        ("lead", lambda b: np.concatenate([np.zeros(int(0.05 * SR)), b]), {"leading_silence"}),
        ("end_click", lambda b: b[: int(0.2 * SR)], {"click_at_end"}),
        ("dc", lambda b: b + 0.02, {"dc_offset", "click_at_start", "click_at_end"}),
        ("quiet", lambda b: b * db(-20), {"loudness"}),
        ("loud", lambda b: np.tanh(8 * b) * 0.99, {"loudness", "true_peak"}),
    ],
)
def test_sfx_failures_and_fix_loop(tmp_path, case, make, fails):
    path = write(tmp_path / f"{case}.wav", make(blip()))
    rep = check(path, "game_sfx")
    assert failing(rep) == fails, rep["checks"]
    rep, after = fix_loop(path, "game_sfx", tmp_path, expect=("pass",))


def test_sfx_hot_zap_with_dc_and_lead_needs_simulation_to_fix(tmp_path):
    # DC offset hides the 50 ms of leading silence until remove_dc runs; the simulated re-check finds it.
    t = np.arange(int(0.35 * SR)) / SR
    zap = 0.98 * np.tanh(4 * np.sin(2 * np.pi * (900 * t - 1000 * t**2))) * np.exp(-3 * t)
    path = write(tmp_path / "zap.wav", np.concatenate([np.zeros(int(0.05 * SR)), zap]) + 0.01)
    rep, after = fix_loop(path, "game_sfx", tmp_path, expect=("pass",))
    assert [o["op"] for o in rep["fix"]["operations"]][0] == "crop"


def test_sfx_clipping_is_reported_as_unfixable(tmp_path):
    t = np.arange(int(0.3 * SR)) / SR
    x = np.clip(1.6 * np.sin(2 * np.pi * 300 * t), -1, 1) * np.sin(np.pi * t / t[-1])
    rep = check(write(tmp_path / "clip.wav", x), "game_sfx")
    assert "clipping" in failing(rep)
    assert rep["fix"]["cannot_fix"] and rep["fix"]["verified"] is False


def test_section_check_reports_range(tmp_path):
    path = write(tmp_path / "two.wav", np.concatenate([blip(), np.zeros(SR), blip()]))
    rep = check(path, "game_sfx", start=1.3)
    assert rep["range_s"][0] == pytest.approx(1.3)


def test_heavy_hum_is_flagged_not_silently_amplified(tmp_path):
    """Found reviewing the merge: on a voice recording with loud 60 Hz hum and hiss, the pauses sat only ~12 dB below
    the speech, so the noise advisory (meant to skip continuous music) stayed silent and the fix just raised the
    hum by 16 dB. Low contrast must still warn, and the fix must include the speech-safe high-pass but no denoise."""
    sr = 44100
    t_ = np.arange(6 * sr) / sr
    rng = np.random.default_rng(3)
    speechy = 0.25 * np.sin(2 * np.pi * 220 * t_) * (np.sin(2 * np.pi * 0.5 * t_) > 0)  # bursts with pauses
    hum = (
        0.08 * np.sin(2 * np.pi * 60 * t_) + 0.03 * np.sin(2 * np.pi * 180 * t_) + 0.004 * rng.standard_normal(len(t_))
    )
    path = tmp_path / "hummy.wav"
    sf.write(str(path), np.stack([0.25 * (speechy + hum)] * 2, axis=1).astype(np.float32), sr)
    rep = server.audio_check_delivery(str(path), "apple_podcasts")
    noise = next(c for c in rep["checks"] if c["check"] == "background_noise")
    assert noise["status"] == "warn" and "hum" in noise["note"]
    ops = [o["op"] for o in rep["fix"]["operations"]]
    assert "highpass" in ops and "denoise" not in ops
    assert ops.index("highpass") < ops.index("normalize_lufs")
