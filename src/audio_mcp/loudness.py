"""Integrated loudness per ITU-R BS.1770-4 / EBU R128, computed in chunks so memory stays flat.

K-weighting coefficients follow libebur128's sample-rate-independent formulas. Verified against pyloudnorm in
tests/test_loudness.py.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

CHUNK = 1 << 20
ABSOLUTE_GATE = -70.0
RELATIVE_GATE = -10.0


def k_weighting_sos(sr: int) -> np.ndarray:
    # Stage 1: high shelf (+4 dB above ~1.7 kHz), modelling the head.
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / sr)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0,
        2 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        1.0,
        2 * (k * k - 1) / a0,
        (1 - k / q + k * k) / a0,
    ]
    # Stage 2: high pass at ~38 Hz (RLB weighting).
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def channel_weights(channels: int) -> np.ndarray:
    if channels == 6:  # 5.1 in the usual L R C LFE Ls Rs order: LFE excluded, surrounds +1.5 dB
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.array([1.0, 1.0, 1.0, 1.41, 1.41] + [1.0] * max(0, channels - 5))[:channels]


def integrated_loudness(x: np.ndarray, sr: int) -> float | None:
    """Gated integrated loudness in LUFS; None if shorter than one 400 ms block; -inf if everything is gated."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    step = int(round(0.1 * sr))  # blocks are 400 ms with 75% overlap, i.e. four 100 ms steps
    if len(x) < 4 * step:
        return None
    sos = k_weighting_sos(sr)
    zi = np.zeros((sos.shape[0], 2, x.shape[1]))
    chunk = (CHUNK // step) * step
    steps = []
    carry = np.zeros((0, x.shape[1]))
    for a in range(0, len(x), chunk):
        y, zi = signal.sosfilt(sos, x[a : a + chunk].astype(np.float64), axis=0, zi=zi)
        y = np.concatenate([carry, y])
        whole = (len(y) // step) * step
        steps.append(np.square(y[:whole]).reshape(-1, step, x.shape[1]).sum(axis=1))
        carry = y[whole:]
    energy = np.concatenate(steps)  # per 100 ms step, per channel
    blocks = (energy[:-3] + energy[1:-2] + energy[2:-1] + energy[3:]) / (4 * step)
    power = blocks @ channel_weights(x.shape[1])
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(power)
    gated = power[lk > ABSOLUTE_GATE]
    if gated.size == 0:
        return float("-inf")
    relative = -0.691 + 10 * np.log10(gated.mean()) + RELATIVE_GATE
    gated = power[(lk > ABSOLUTE_GATE) & (lk > relative)]
    return float(-0.691 + 10 * np.log10(gated.mean()))


def momentary_max(x: np.ndarray, sr: int) -> float | None:
    """Maximum momentary loudness (EBU R128 'M': K-weighted, 400 ms sliding window, here in 10 ms steps), in LUFS.

    Defined for any length: a clip shorter than 400 ms is measured as one window padded with silence, as a meter
    would see it. Useful for short sound effects, where integrated loudness (which needs >= 400 ms and gating) is
    undefined. Returns None for empty input and -inf for silence."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) == 0:
        return None
    step = max(1, int(round(0.01 * sr)))
    window_steps = 40
    min_len = window_steps * step
    if len(x) < min_len:
        x = np.concatenate([x, np.zeros((min_len - len(x), x.shape[1]), dtype=x.dtype)])
    sos = k_weighting_sos(sr)
    zi = np.zeros((sos.shape[0], 2, x.shape[1]))
    chunk = (CHUNK // step) * step
    weights = channel_weights(x.shape[1])
    energy, carry = [], np.zeros((0, x.shape[1]))
    for a in range(0, len(x), chunk):
        y, zi = signal.sosfilt(sos, x[a : a + chunk].astype(np.float64), axis=0, zi=zi)
        y = np.concatenate([carry, y])
        whole = (len(y) // step) * step
        energy.append((np.square(y[:whole]).reshape(-1, step, x.shape[1]).sum(axis=1)) @ weights)
        carry = y[whole:]
    e = np.concatenate(energy)
    cs = np.concatenate(([0.0], np.cumsum(e)))
    windows = (cs[window_steps:] - cs[:-window_steps]) / (window_steps * step)
    peak = float(windows.max()) if len(windows) else 0.0
    return -0.691 + 10 * math.log10(peak) if peak > 0 else float("-inf")
