"""Our chunked BS.1770 meter must agree with pyloudnorm, an independent reference implementation.

We compare against pyloudnorm's "DeMan" filters, which implement the standard's K-weighting exactly. Its default
filters use approximate parameters (1500 Hz shelf instead of 1681.97 Hz) and drift by up to 0.2 LU at low sample
rates, which is why the EBU reference-tone check below runs at every rate."""

import numpy as np
import pytest

from audio_mcp import loudness

pyloudnorm = pytest.importorskip("pyloudnorm")


def signals():
    rng = np.random.default_rng(0)
    for sr in (8000, 22050, 44100, 48000, 96000):
        t = np.arange(int(6.3 * sr)) / sr
        yield f"sine-{sr}", sr, 0.5 * np.sin(2 * np.pi * 997 * t)
        burst = 0.3 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 0.5 * t) > 0)  # gating matters here
        yield f"bursts-{sr}", sr, np.stack([burst, 0.5 * burst], axis=1)
        yield f"bass-{sr}", sr, 0.4 * np.sin(2 * np.pi * 50 * t)  # K-weighting high-pass matters here


@pytest.mark.parametrize("name,sr,x", list(signals()), ids=lambda v: v if isinstance(v, str) else "")
def test_matches_pyloudnorm(name, sr, x):
    ours = loudness.integrated_loudness(x, sr)
    ref = pyloudnorm.Meter(sr, filter_class="DeMan").integrated_loudness(x)
    assert ours == pytest.approx(ref, abs=0.01)


@pytest.mark.parametrize("sr", [8000, 16000, 22050, 44100, 48000, 96000])
def test_ebu_reference_tone(sr):
    """A full-scale 997 Hz sine reads -3.01 LUFS (EBU Tech 3341)."""
    t = np.arange(10 * sr) / sr
    assert loudness.integrated_loudness(np.sin(2 * np.pi * 997 * t), sr) == pytest.approx(-3.01, abs=0.05)


def test_chunk_boundaries_do_not_matter(monkeypatch):
    sr = 44100
    x = np.random.default_rng(1).standard_normal((5 * sr, 2)) * 0.1
    whole = loudness.integrated_loudness(x, sr)
    monkeypatch.setattr(loudness, "CHUNK", 12345)
    assert loudness.integrated_loudness(x, sr) == pytest.approx(whole, abs=1e-6)


def test_too_short_and_silent():
    assert loudness.integrated_loudness(np.zeros(1000), 44100) is None
    assert loudness.integrated_loudness(np.zeros(44100), 44100) == float("-inf")


def test_chunked_resample_matches_whole(monkeypatch):
    """dsp.resample streams long signals; the result must equal one-shot polyphase resampling."""
    from scipy import signal

    from audio_mcp import dsp

    monkeypatch.setattr(dsp, "CHUNK", 5000)
    x = np.random.default_rng(2).standard_normal(100_003).astype(np.float32)
    for sr_from, sr_to in [(44100, 16000), (48000, 16000), (22050, 44100), (96000, 44100)]:
        g = np.gcd(sr_from, sr_to)
        ref = signal.resample_poly(x, sr_to // g, sr_from // g)
        out = dsp.resample(x, sr_from, sr_to)
        assert len(out) == len(ref)
        assert np.max(np.abs(out - ref)) < 1e-4


def test_frame_rms_chunked_matches_direct(monkeypatch):
    from audio_mcp import dsp

    monkeypatch.setattr(dsp, "CHUNK", 777)
    x = np.random.default_rng(3).standard_normal(50_000)
    frame, hop = 441, 220
    direct = np.sqrt([np.mean(x[i : i + frame] ** 2) for i in range(0, len(x) - frame + 1, hop)])
    assert np.allclose(dsp.frame_rms(x, frame, hop), direct)
