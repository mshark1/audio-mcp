"""Contract: every tool, on any input, either succeeds or raises AudioError with a message. Never anything else."""

import itertools

import numpy as np
import pytest
import soundfile as sf

from audio_mcp import server
from audio_mcp.audio_io import AudioError

SR = 44100


def signal_for(kind, n, ch):
    rng = np.random.default_rng(0)
    t = np.arange(n) / SR
    base = {
        "silence": np.zeros(n),
        "dc": np.full(n, 0.5),
        "noise": 0.3 * rng.standard_normal(n),
        "square": np.sign(np.sin(2 * np.pi * 110 * t)),
        "sine": 0.5 * np.sin(2 * np.pi * 440 * t),
    }[kind]
    return np.repeat(base[:, None], ch, axis=1) * np.linspace(1.0, 0.8, ch)[None, :]


FORMATS = [("wav", "PCM_16"), ("wav", "PCM_24"), ("wav", "FLOAT"), ("flac", "PCM_16"), ("ogg", "VORBIS")]
LENGTHS = [1, 64, int(0.3 * SR), 3 * SR]
KINDS = ["silence", "dc", "noise", "square", "sine"]


def call_all(path, tmp_path):
    p = str(path)
    server.audio_info(p)
    server.audio_analyze(p)
    server.audio_timeline(p)
    server.audio_view(p)
    server.audio_melody(p)
    server.audio_events(p)
    server.audio_compare(p, p)
    server.audio_process(
        p,
        [{"op": "normalize_peak"}, {"op": "fade_out", "seconds": 0.01}],
        output=str(tmp_path / "o.wav"),
        overwrite=True,
    )


def check(fn):
    try:
        fn()
    except AudioError as e:
        assert str(e).strip(), "AudioError without a message"
        assert "Internal error" not in str(e), f"unexpected exception (a bug): {e}"


@pytest.mark.parametrize("kind,n", list(itertools.product(KINDS, LENGTHS)))
def test_mono_contents_and_lengths(tmp_path, kind, n):
    path = tmp_path / "x.wav"
    sf.write(str(path), signal_for(kind, n, 1).astype(np.float32), SR, subtype="PCM_24")
    check(lambda: call_all(path, tmp_path))


@pytest.mark.parametrize("ext,subtype,ch", [(e, s, c) for (e, s), c in itertools.product(FORMATS, (1, 2, 6))])
def test_formats_and_channel_counts(tmp_path, ext, subtype, ch):
    if ext == "ogg" and ch == 6:
        pytest.skip("vorbis channel mapping not needed")
    path = tmp_path / f"x.{ext}"
    sf.write(str(path), signal_for("sine", 2 * SR, ch).astype(np.float32), SR, subtype=subtype)
    check(lambda: call_all(path, tmp_path))


@pytest.mark.parametrize("sr", [8000, 22050, 48000, 96000])
def test_sample_rates(tmp_path, sr):
    path = tmp_path / "x.wav"
    t = np.arange(2 * sr) / sr
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    call_all(path, tmp_path)  # ordinary audio at any common rate must fully succeed
    assert server.audio_analyze(str(path))["pitch"]["median"]["note"] == "A4"


def test_non_finite_samples_are_rejected_clearly(tmp_path):
    x = (0.5 * np.sin(np.arange(SR) / 10)).astype(np.float32)
    x[100] = np.nan
    x[200] = np.inf
    path = tmp_path / "nan.wav"
    sf.write(str(path), x, SR, subtype="FLOAT")
    with pytest.raises(AudioError, match="NaN|infinite"):
        server.audio_analyze(str(path))


def test_not_audio_and_bad_ranges(tmp_path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not really a wav file")
    with pytest.raises(AudioError, match="Could not decode"):
        server.audio_analyze(str(junk))
    ok = tmp_path / "ok.wav"
    sf.write(str(ok), np.zeros(SR, dtype=np.float32), SR)
    for start, end in [(-1, None), (5, None), (0.5, 0.2), (0.2, 0.2)]:
        with pytest.raises(AudioError):
            server.audio_analyze(str(ok), start=start, end=end)


@pytest.mark.parametrize(
    "ops",
    [
        [{"op": "crop", "start": 5}],
        [{"op": "highpass", "hz": 30000}],
        [{"op": "resample", "rate": -1}],
        [{"op": "loop_crossfade", "seconds": 10}],
        [{"op": "fade_in", "seconds": 0.1, "curve": "wobbly"}],
        [{"op": "mix", "path": "does_not_exist.wav"}],
        [{"op": "compress", "ratio": 0.5}],
        [{"op": "normalize_peak"}],  # on silence
        "not a list",
        [],
        [{"no_op_key": 1}],
    ],
)
def test_bad_operations_fail_cleanly(tmp_path, ops):
    path = tmp_path / "s.wav"
    sf.write(str(path), np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(AudioError) as e:
        server.audio_process(str(path), ops, output=str(tmp_path / "o.wav"))
    assert "Internal error" not in str(e.value)


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"layers": []},
        {"layers": [{"wave": "kazoo", "freq": 440, "duration": 1}]},
        {"layers": [{"wave": "sine", "duration": 1}]},
        {"layers": [{"wave": "sine", "freq": "H9", "duration": 1}]},
        {"layers": [{"wave": "sine", "freq": 440}]},
        {"layers": [{"wave": "sine", "freq": 440, "duration": 1e6}]},
        {"sample_rate": 10, "layers": [{"wave": "sine", "freq": 440, "duration": 1}]},
        {"layers": [{"wave": "sine", "freq": -5, "duration": 1}]},
        {"layers": [{"wave": "sine", "freq": 440, "duration": 1, "lowpass": 1e6}]},
    ],
)
def test_bad_synth_specs_fail_cleanly(tmp_path, spec):
    with pytest.raises(AudioError) as e:
        server.audio_synthesize(spec, str(tmp_path / "o.wav"))
    assert "Internal error" not in str(e.value)


def test_unexpected_errors_are_reported_not_hidden(monkeypatch, tmp_path):
    from audio_mcp import analysis

    def boom(*a, **k):
        raise ZeroDivisionError("simulated bug")

    monkeypatch.setattr(analysis, "analyze", boom)
    path = tmp_path / "x.wav"
    sf.write(str(path), np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(AudioError, match="Internal error in audio_analyze: ZeroDivisionError: simulated bug"):
        server.audio_analyze(str(path))


def test_allowed_dirs_sandbox(tmp_path, monkeypatch):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    for d in (inside, outside):
        sf.write(str(d / "x.wav"), np.zeros(SR, dtype=np.float32), SR)
    monkeypatch.setenv("AUDIO_MCP_ALLOWED_DIRS", str(inside))
    server.audio_info(str(inside / "x.wav"))  # allowed
    for call in (
        lambda: server.audio_info(str(outside / "x.wav")),
        lambda: server.audio_analyze(str(inside / ".." / "outside" / "x.wav")),  # traversal
        lambda: server.audio_process(str(inside / "x.wav"), [{"op": "gain", "db": 1}], output=str(outside / "y.wav")),
        lambda: server.audio_process(str(inside / "x.wav"), [{"op": "mix", "path": str(outside / "x.wav")}]),
        lambda: server.audio_view(str(inside / "x.wav"), save_to=str(outside / "v.png")),
        lambda: server.audio_batch_survey(str(outside)),
    ):
        with pytest.raises(AudioError, match="not allowed"):
            call()
    assert not (outside / "y.wav").exists() and not (outside / "v.png").exists()
