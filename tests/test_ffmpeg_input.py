"""FFmpeg input: formats libsndfile can't read (M4A/AAC, video, WMA) are decoded through FFmpeg when it's present.

The plumbing tests use a fake ffmpeg (a Python script) so they run everywhere. The real round trip at the end
needs an actual ffmpeg and runs in CI on Linux."""

import json
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from audio_mcp import audio_io, server
from audio_mcp.audio_io import AudioError

SR = 48000
FAKE = textwrap.dedent(
    """
    import json, os, sys, time
    import numpy as np
    import soundfile as sf

    args = sys.argv[1:]
    log = os.environ.get("FAKE_FFMPEG_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(args) + "\\n")
    mode = os.environ.get("FAKE_FFMPEG_MODE", "ok")
    if "-show_entries" in args:  # acting as ffprobe
        if mode == "noprobe":
            sys.exit(1)
        print(json.dumps({"streams": [{"codec_name": "aac", "sample_rate": "48000", "channels": 2}],
                          "format": {"duration": "1.000000"}}))
        sys.exit(0)
    if mode == "sleep":
        time.sleep(4)
    if mode == "fail":
        sys.stderr.write("[mov,mp4] moov atom not found\\nfile:x.m4a: Invalid data found when processing input\\n")
        sys.exit(1)
    inp = args[args.index("-i") + 1]
    assert inp.startswith("file:"), inp
    assert os.path.isfile(inp[5:]), inp
    out = args[-1][5:] if args[-1].startswith("file:") else args[-1]
    t = np.arange(48000) / 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(out, np.stack([tone, 0.25 * tone], axis=1).astype(np.float32), 48000, subtype="FLOAT")
    """
)


def make_exe(directory, name):
    """A runnable `name` (ffmpeg/ffprobe) that runs FAKE with this Python."""
    script = directory / "fake_ffmpeg.py"
    script.write_text(FAKE, encoding="utf-8")
    if os.name == "nt":
        exe = directory / f"{name}.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        exe = directory / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


@pytest.fixture(autouse=True)
def fresh_cache():
    audio_io.clear_ffmpeg_cache()
    yield
    audio_io.clear_ffmpeg_cache()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = make_exe(bindir, "ffmpeg")
    log = tmp_path / "calls.log"
    monkeypatch.setenv("AUDIO_MCP_FFMPEG", str(exe))
    monkeypatch.delenv("AUDIO_MCP_FFPROBE", raising=False)
    monkeypatch.setenv("FAKE_FFMPEG_LOG", str(log))
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "ok")

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    media = tmp_path / "media"
    media.mkdir()
    m4a = media / "voice memo.m4a"
    m4a.write_bytes(b"\x00\x00\x00\x20ftypM4A not really")  # the fake doesn't read it
    return {"bindir": bindir, "exe": exe, "calls": calls, "m4a": m4a, "dir": media}


def decode_calls(calls):
    return [c for c in calls if "-show_entries" not in c]


def test_tools_work_on_decoded_file_and_decode_once(fake, tmp_path):
    p = str(fake["m4a"])
    rep = server.audio_analyze(p)
    assert rep["format"] == "M4A"  # no ffprobe next to the fake: codec unknown, container from the extension
    assert rep["duration_s"] == pytest.approx(1.0, abs=1e-3)
    assert rep["channels"] == 2 and rep["sample_rate"] == SR
    assert rep["pitch"]["median"]["note"] == "A4"
    assert rep["levels"]["peak_dbfs"] == pytest.approx(20 * np.log10(0.5), abs=0.1)
    tl = server.audio_timeline(p, segments=4)
    assert len(tl["segments"]) == 4 and tl["duration_s"] == pytest.approx(1.0, abs=1e-3)
    server.audio_view(p, kind="waveform")
    server.audio_events(p)
    server.audio_melody(p)
    server.audio_compare(p, p)
    part = server.audio_analyze(p, start=0.25, end=0.75, pitch=False, rhythm=False)
    assert part["duration_s"] == pytest.approx(0.5, abs=1e-3)
    out = server.audio_process(p, [{"op": "gain", "db": -6}])
    assert out["output"].endswith("voice memo_processed.wav")  # outputs stay in writable formats
    assert sf.info(out["output"]).channels == 2
    info = server.audio_info(p)
    assert info["decoded_via"] == "ffmpeg" and info["format"] == "M4A" and info["sample_rate"] == SR

    calls = decode_calls(fake["calls"]())
    assert len(calls) == 1, "the cached decode must be reused across tool calls"
    args = calls[0]
    assert args[args.index("-i") + 1] == f"file:{fake['m4a']}"
    assert args[args.index("-protocol_whitelist") + 1] == "file"
    assert "-nostdin" in args and args[args.index("-v") + 1] == "error"
    assert "hls" not in args[args.index("-format_whitelist") + 1].split(",")


def test_changed_file_is_decoded_again(fake):
    p = str(fake["m4a"])
    server.audio_analyze(p, pitch=False, rhythm=False)
    fake["m4a"].write_bytes(b"different contents, different size")
    server.audio_analyze(p, pitch=False, rhythm=False)
    assert len(decode_calls(fake["calls"]())) == 2


def test_cache_is_bounded(fake, monkeypatch):
    monkeypatch.setattr(audio_io, "FFMPEG_CACHE_MAX_FILES", 2)
    files = []
    for i in range(3):
        f = fake["dir"] / f"clip{i}.mp4"
        f.write_bytes(b"x" * (i + 1))
        files.append(f)
        server.audio_analyze(str(f), pitch=False, rhythm=False)
    assert len(audio_io._cache) == 2
    assert len(list(audio_io._get_cache_dir().glob("*.wav"))) == 2
    server.audio_analyze(str(files[0]), pitch=False, rhythm=False)  # evicted, so decoded again
    assert len(decode_calls(fake["calls"]())) == 4


def test_ffprobe_reports_codec_without_decoding(fake):
    make_exe(fake["bindir"], "ffprobe")  # found next to ffmpeg
    info = server.audio_info(str(fake["m4a"]))
    assert info["subtype"] == "AAC" and info["decoded_via"] == "ffmpeg"
    assert info["duration_s"] == 1.0 and info["channels"] == 2
    assert decode_calls(fake["calls"]()) == [], "audio_info should not decode when ffprobe can read the header"
    probe = fake["calls"]()[0]
    assert probe[-1] == f"file:{fake['m4a']}" and probe[probe.index("-protocol_whitelist") + 1] == "file"
    assert server.audio_analyze(str(fake["m4a"]), pitch=False, rhythm=False)["format"] == "M4A AAC"


def test_ffprobe_failure_falls_back_to_decoding(fake, monkeypatch):
    make_exe(fake["bindir"], "ffprobe")
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "noprobe")
    info = server.audio_info(str(fake["m4a"]))
    assert info["sample_rate"] == SR and info["duration_s"] == 1.0 and info["subtype"] == ""


def test_listing_and_batch_include_ffmpeg_formats(fake):
    sf.write(str(fake["dir"] / "a.wav"), np.zeros(SR // 10, dtype=np.float32), SR)
    listing = server.audio_info(str(fake["dir"]))
    assert len(listing["files"]) == 2 and "needs_ffmpeg" not in listing
    survey = server.audio_batch_survey(str(fake["dir"]))
    assert survey["file_count"] == 2


def test_ffmpeg_failure_is_a_clean_error(fake, monkeypatch):
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "fail")
    with pytest.raises(AudioError, match=r"Could not decode 'voice memo.m4a': FFmpeg failed \(.*moov atom"):
        server.audio_analyze(str(fake["m4a"]))
    assert list(audio_io._get_cache_dir().glob("*.wav")) == []


def test_ffmpeg_timeout_is_a_clean_error(fake, monkeypatch):
    monkeypatch.setenv("FAKE_FFMPEG_MODE", "sleep")
    monkeypatch.setenv("AUDIO_MCP_FFMPEG_TIMEOUT", "0.5")
    with pytest.raises(AudioError, match="AUDIO_MCP_FFMPEG_TIMEOUT"):
        server.audio_timeline(str(fake["m4a"]))


def test_decode_beyond_cache_limit_is_refused_not_truncated(fake, monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_FFMPEG_CACHE_MB", "0.1")  # the fake writes ~384 KB
    with pytest.raises(AudioError, match="AUDIO_MCP_FFMPEG_CACHE_MB"):
        server.audio_analyze(str(fake["m4a"]))


def test_sandbox_applies_before_ffmpeg_runs(fake, monkeypatch, tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_DIRS", str(inside))
    with pytest.raises(AudioError, match="not allowed"):
        server.audio_analyze(str(fake["m4a"]))
    assert fake["calls"]() == []


def test_missing_ffmpeg_gives_install_hints(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_MCP_FFMPEG", raising=False)
    monkeypatch.setattr(audio_io.shutil, "which", lambda *a, **k: None)
    m4a = tmp_path / "talk.m4a"
    m4a.write_bytes(b"\x00" * 64)
    for call in (lambda: server.audio_analyze(str(m4a)), lambda: server.audio_info(str(m4a))):
        with pytest.raises(AudioError) as e:
            call()
        msg = str(e.value)
        assert "M4A" in msg and "FFmpeg" in msg
        assert "winget" in msg and "brew install ffmpeg" in msg and "apt install ffmpeg" in msg
    listing = server.audio_info(str(tmp_path))
    assert listing["files"] == [] and listing["needs_ffmpeg"]["count"] == 1
    with pytest.raises(AudioError, match="No audio files found"):  # batch only picks up what it can open
        server.audio_batch_survey(str(tmp_path))


def test_bad_ffmpeg_setting_is_named(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_MCP_FFMPEG", str(tmp_path / "nope" / "ffmpeg.exe"))
    junk = tmp_path / "x.wav"
    junk.write_bytes(b"not audio")
    with pytest.raises(AudioError, match=r"Could not decode 'x.wav'.*AUDIO_MCP_FFMPEG is set to"):
        server.audio_analyze(str(junk))


def test_ffmpeg_used_when_libsndfile_cant_read(fake):
    weird = fake["dir"] / "odd.wav"
    weird.write_bytes(b"RIFF....WAVE but not one libsndfile understands")
    assert server.audio_analyze(str(weird), pitch=False, rhythm=False)["duration_s"] == pytest.approx(1.0, abs=1e-3)
    assert len(decode_calls(fake["calls"]())) == 1


# --- real FFmpeg ------------------------------------------------------------------------------------------------

REAL = shutil.which("ffmpeg")


@pytest.mark.skipif(REAL is None, reason="needs ffmpeg on PATH")
@pytest.mark.parametrize("ext", [".m4a", ".wma", ".webm", ".mp4"])
def test_real_ffmpeg_round_trip(tmp_path, monkeypatch, ext):
    monkeypatch.delenv("AUDIO_MCP_FFMPEG", raising=False)
    monkeypatch.delenv("AUDIO_MCP_FFPROBE", raising=False)
    t = np.arange(3 * SR) / SR
    tone = (10 ** (-12 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # -12 dBFS RMS
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), np.stack([tone, tone], axis=1), SR, subtype="PCM_24")
    src = tmp_path / f"tone{ext}"
    cmd = [REAL, "-nostdin", "-v", "error", "-y"]
    if ext == ".mp4":  # a video file with an audio track
        cmd += ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=3"]
    cmd += ["-i", str(wav)]
    if ext == ".mp4":
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-shortest"]
    cmd += ["-b:a", "160k", str(src)]
    made = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if made.returncode != 0:
        pytest.skip(f"this ffmpeg build can't write {ext}: {made.stderr[-200:]}")

    ref = server.audio_analyze(str(wav))
    rep = server.audio_analyze(str(src))
    assert rep["format"].startswith(ext[1:].upper())
    assert rep["duration_s"] == pytest.approx(3.0, abs=0.1)
    assert rep["pitch"]["median"]["note"] == "A4"
    assert rep["pitch"]["median"]["hz"] == pytest.approx(440, abs=2)
    assert rep["levels"]["rms_dbfs"] == pytest.approx(ref["levels"]["rms_dbfs"], abs=1.0)
    assert rep["levels"]["lufs_integrated"] == pytest.approx(ref["levels"]["lufs_integrated"], abs=1.0)
    info = server.audio_info(str(src))
    assert info["decoded_via"] == "ffmpeg" and info["channels"] == 2
    assert info["duration_s"] == pytest.approx(3.0, abs=0.15)
    if shutil.which("ffprobe"):
        assert info["subtype"]  # codec name, e.g. AAC
    tl = server.audio_timeline(str(src), segments=6)
    assert len(tl["segments"]) == 6
    out = server.audio_process(str(src), [{"op": "gain", "db": -6}])
    assert out["output"].endswith(".wav")
    assert server.audio_compare(str(src), out["output"])["null_test"]
