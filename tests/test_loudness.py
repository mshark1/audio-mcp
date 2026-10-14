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


def test_momentary_max_matches_integrated_for_steady_tone():
    """For a steady signal every 400 ms window has the same loudness, so M-max equals integrated loudness."""
    sr = 48000
    x = 0.3 * np.sin(2 * np.pi * 997 * np.arange(5 * sr) / sr)
    assert loudness.momentary_max(x, sr) == pytest.approx(loudness.integrated_loudness(x, sr), abs=0.05)


def test_momentary_max_finds_the_loud_part_and_handles_short_clips():
    sr = 48000
    quiet = 0.01 * np.sin(2 * np.pi * 997 * np.arange(3 * sr) / sr)
    loud = 0.5 * np.sin(2 * np.pi * 997 * np.arange(sr) / sr)
    x = np.concatenate([quiet, loud, quiet])
    assert loudness.momentary_max(x, sr) == pytest.approx(loudness.integrated_loudness(loud, sr), abs=0.1)
    # 100 ms tone: one 400 ms window with 300 ms of silence -> 10*log10(0.25) = -6.02 LU below the steady tone
    short = loud[: sr // 10]
    assert loudness.momentary_max(short, sr) == pytest.approx(loudness.integrated_loudness(loud, sr) - 6.02, abs=0.15)
    assert loudness.momentary_max(np.zeros(100), sr) == float("-inf")


def test_momentary_max_against_pyloudnorm_block_loudness():
    """Cross-check: the loudest 400 ms window must equal pyloudnorm's loudness of exactly that single block."""
    sr = 48000
    rng = np.random.default_rng(4)
    x = np.zeros(4 * sr)
    block = slice(sr, sr + int(0.4 * sr))  # aligned to our 10 ms steps; silence before it, so filter state matches
    x[block] = 0.2 * rng.standard_normal(int(0.4 * sr))
    ref = pyloudnorm.Meter(sr, filter_class="DeMan", block_size=0.4).integrated_loudness(x[block])
    assert loudness.momentary_max(x, sr) == pytest.approx(ref, abs=0.05)


@pytest.mark.parametrize("seconds", [3, 6, 12])
def test_true_peak_correct_across_block_boundaries(seconds):
    """Found by the perception benchmark: steady tones longer than one 2^18-sample block read +1.08 dB high because
    the maximum included the filter ringing in each block's padded margins."""
    from scipy import signal as sps

    from audio_mcp.dsp import amp_db, true_peak

    sr = 48000
    x = (0.1 * np.sin(2 * np.pi * 997 * np.arange(seconds * sr) / sr)).astype(np.float32)[:, None]
    assert float(amp_db(true_peak(x, sr))) == pytest.approx(-20.0, abs=0.02)
    # and against a whole-signal oversampling reference on a real transient
    rng = np.random.default_rng(5)
    y = (0.3 * rng.standard_normal(seconds * sr)).astype(np.float32)
    y[min(1 << 18, len(y) - 10)] = 0.99  # on the first block boundary when the signal is long enough
    ref = float(np.max(np.abs(sps.resample_poly(y, 4, 1))))
    assert true_peak(y[:, None], sr) == pytest.approx(ref, rel=1e-3)
