"""Shared DSP helpers. Signals are float arrays nominally in [-1, 1]."""

from __future__ import annotations

import math
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

AMP_FLOOR = 1e-6  # -120 dB
POW_FLOOR = 1e-12  # -120 dB
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SPARK_CHARS = "▁▂▃▄▅▆▇█"
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def amp_db(x):
    """Linear amplitude to dB, floored at -120 dB."""
    return 20.0 * np.log10(np.maximum(np.abs(x), AMP_FLOOR))


def pow_db(p):
    """Power (mean square) to dB, floored at -120 dB."""
    return 10.0 * np.log10(np.maximum(p, POW_FLOOR))


def r(x, nd: int = 2):
    """Round for JSON output; non-finite values become None."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd) if nd > 0 else int(round(x))


CHUNK = 1 << 20


def frame_rms(x, frame: int, hop: int) -> np.ndarray:
    """RMS of each frame. Works chunk by chunk with a running sum, so memory stays small on long signals."""
    x = np.asarray(x)
    frame = max(1, int(frame))
    hop = max(1, int(hop))
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n_frames = 1 + (len(x) - frame) // hop
    out = np.empty(n_frames)
    per = max(1, CHUNK // hop)
    for f0 in range(0, n_frames, per):
        f1 = min(n_frames, f0 + per)
        seg = x[f0 * hop : (f1 - 1) * hop + frame].astype(np.float64)
        cs = np.concatenate(([0.0], np.cumsum(seg * seg)))
        starts = np.arange(f1 - f0) * hop
        out[f0:f1] = np.sqrt(np.maximum(cs[starts + frame] - cs[starts], 0.0) / frame)
    return out


def stft_chunks(x, n_fft: int = 2048, hop: int = 512, chunk: int = 1024):
    """Yield (first_frame_index, magnitudes[frames, bins]) so long files never need a full STFT in memory."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    window = np.hanning(n_fft).astype(np.float32)
    frames = sliding_window_view(x, n_fft)[::hop]
    for i in range(0, len(frames), chunk):
        yield i, np.abs(np.fft.rfft(frames[i : i + chunk] * window, axis=1))


def resample(x, sr_from: int, sr_to: int, axis: int = 0) -> np.ndarray:
    """Polyphase resampling. Long signals go through in aligned, overlapping chunks: identical output, small memory."""
    x = np.asarray(x)
    if int(sr_from) == int(sr_to):
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    if axis != 0 or len(x) <= 4 * CHUNK:
        return signal.resample_poly(x, up, down, axis=axis).astype(np.float32)
    # Chunk starts and padding are multiples of `down`, so each chunk's output lands on exact output samples;
    # the padding covers resample_poly's filter half-length (10 * max(up, down) taps at the upsampled rate).
    pad = down * math.ceil((10 * max(up, down) / up + 8) / down)
    step = down * max(1, CHUNK // down)
    out = []
    for a in range(0, len(x), step):
        lo, hi = max(0, a - pad), min(len(x), a + step + pad)
        y = signal.resample_poly(x[lo:hi], up, down, axis=0)
        first = (a - lo) * up // down
        n_out = math.ceil(min(step, len(x) - a) * up / down)
        out.append(y[first : first + n_out].astype(np.float32))
    return np.concatenate(out)


def true_peak(samples: np.ndarray, sr: int) -> float:
    """Inter-sample peak estimated by 4x oversampling (ITU-R BS.1770 style)."""
    if sr >= 176400:
        return float(np.max(np.abs(samples)))
    peak, block, pad = 0.0, 1 << 18, 64
    n = samples.shape[0]
    for start in range(0, n, block):
        lo, hi = max(0, start - pad), min(n, start + block + pad)
        up = signal.resample_poly(samples[lo:hi], 4, 1, axis=0)
        # The padding gives the filter context; only this block's own samples count. Including the padded margins
        # measured the filter's ringing where the chunk cuts the waveform (+1.1 dB on steady tones > 5.5 s).
        core = up[(start - lo) * 4 : (min(n, start + block) - lo) * 4]
        peak = max(peak, float(np.max(np.abs(core))))
    return peak


def octave_flatness(power, freqs) -> np.ndarray:
    """Spectral flatness (0 = pure tones, ~0.56 = noise) measured within each octave from 62.5 Hz to 16 kHz and
    energy-weighted. Unlike whole-spectrum flatness it doesn't mistake coloured noise (pink, brown, which slope
    downward) for tonal sound. `power` is (bins,) or (frames, bins)."""
    p = np.atleast_2d(power)
    edges = 62.5 * 2.0 ** np.arange(9)
    total = np.zeros(len(p))
    weight = np.zeros(len(p))
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        if band.sum() < 4:
            continue
        pb = p[:, band] + 1e-20
        energy = pb.sum(axis=1)
        total += energy * np.exp(np.mean(np.log(pb), axis=1)) / np.mean(pb, axis=1)
        weight += energy
    out = total / np.maximum(weight, 1e-30)
    return out if np.ndim(power) == 2 else out[0]


def runs(mask) -> list[tuple[int, int]]:
    """[start, end) index pairs for each run of True values."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def parabolic(y, i: int) -> tuple[float, float]:
    """Refine the extremum at index i with a parabola through its neighbours."""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    d = a - 2.0 * b + c
    if d == 0:
        return float(i), b
    p = 0.5 * (a - c) / d
    return i + p, b - 0.25 * (a - c) * p


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(np.asarray(f, dtype=np.float64) / 440.0)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def midi_to_name(m) -> str:
    m = int(round(float(m)))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def describe_hz(f) -> dict:
    """{'hz': 440.0, 'note': 'A4', 'cents': 0} - nearest equal-tempered note and deviation."""
    midi = float(hz_to_midi(f))
    n = round(midi)
    return {"hz": r(f, 1), "note": midi_to_name(n), "cents": int(round((midi - n) * 100))}


def parse_freq(value) -> float:
    """Accept a frequency in Hz or a note name such as 'A4', 'C#3', 'Bb2'."""
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"frequency must be positive, got {value}")
        return float(value)
    s = str(value).strip()
    try:
        return parse_freq(float(s))
    except ValueError:
        pass
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"'{value}' is not a frequency in Hz or a note name like 'A4' or 'C#3'")
    letter, accidental, octave = m.groups()
    pc = NOTE_NAMES.index(letter.upper()) + {"#": 1, "b": -1, "": 0}[accidental]
    return float(midi_to_hz(pc + (int(octave) + 1) * 12))


def sparkline(values, lo: float | None = None, hi: float | None = None) -> str:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return ""
    lo = float(np.min(v)) if lo is None else lo
    hi = float(np.max(v)) if hi is None else hi
    if hi - lo < 1e-9:
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(v)
    idx = np.clip(np.round((v - lo) / (hi - lo) * (len(SPARK_CHARS) - 1)), 0, len(SPARK_CHARS) - 1)
    return "".join(SPARK_CHARS[int(i)] for i in idx)
