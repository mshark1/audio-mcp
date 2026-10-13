"""Dynamics processing: a look-ahead true-peak limiter and a feed-forward compressor.

Gains are computed at a control rate (one value per BLOCK samples) so the recursive attack/release smoothing
can run as a plain Python loop, then interpolated back to audio rate.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .dsp import amp_db, pow_db

BLOCK = 32


def _blocks(n: int) -> int:
    return -(-n // BLOCK)


def _block_peaks(x: np.ndarray, oversample: bool) -> np.ndarray:
    """Max |sample| per block across channels; with oversample, inter-sample (true) peaks are included."""
    n = len(x)
    nb = _blocks(n)
    peaks = np.zeros(nb)
    chunk_blocks = 4096
    pad = 64
    for b0 in range(0, nb, chunk_blocks):
        b1 = min(nb, b0 + chunk_blocks)
        s0, s1 = b0 * BLOCK, min(n, b1 * BLOCK)
        if oversample:
            lo = max(0, s0 - pad)
            up = np.abs(signal.resample_poly(x[lo : min(n, s1 + pad)], 4, 1, axis=0)).max(axis=1)
            seg = up[(s0 - lo) * 4 : (s0 - lo) * 4 + (s1 - s0) * 4]
            width = 4 * BLOCK
        else:
            seg = np.abs(x[s0:s1]).max(axis=1)
            width = BLOCK
        seg = np.pad(seg, (0, (b1 - b0) * width - len(seg)))
        peaks[b0:b1] = seg.reshape(b1 - b0, width).max(axis=1)
    return peaks


def _apply_gain(x: np.ndarray, block_gain: np.ndarray, in_db: bool = False) -> np.ndarray:
    """Interpolate per-block gains to per-sample and apply them, chunk by chunk, in float32."""
    centers = (np.arange(len(block_gain)) + 0.5) * BLOCK
    y = np.empty(x.shape, dtype=np.float32)
    step = 1 << 20
    for a in range(0, len(x), step):
        g = np.interp(np.arange(a, min(len(x), a + step)), centers, block_gain)
        if in_db:
            g = 10 ** (g / 20)
        y[a : a + step] = x[a : a + step] * g.astype(np.float32)[:, None]
    return y


def limit(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    release_ms: float = 60.0,
    lookahead_ms: float = 5.0,
    true_peak: bool = True,
):
    """Returns (limited, stats). Never lets a (true) peak exceed the ceiling."""
    ceiling = 10 ** (ceiling_db / 20)
    peaks = _block_peaks(x, oversample=true_peak and sr < 176400)
    need = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    la = max(1, int(round(lookahead_ms / 1000 * sr / BLOCK)))
    # Gain must already be down `la` blocks before a peak and stay down `la` after it.
    need = ndimage.minimum_filter1d(need, size=2 * la + 1, mode="nearest")
    rel = np.exp(-BLOCK / (sr * release_ms / 1000))
    g = np.empty_like(need)
    cur = 1.0
    for i, t in enumerate(need):
        cur = min(t, 1.0 - (1.0 - cur) * rel)  # drop at once to the requirement, recover exponentially
        g[i] = cur
    # Average over half the look-ahead window: smooths the attack and stays below every requirement
    # (each averaged value comes from within `la` blocks, whose minimum filter already covers this block).
    g = ndimage.uniform_filter1d(g, size=la + 1, mode="nearest")
    g = ndimage.minimum_filter1d(g, size=3, mode="nearest")  # keep interpolation between block centres safe
    y = _apply_gain(x, g)
    np.clip(y, -ceiling, ceiling, out=y)  # safety net; normally a no-op
    reduction = -amp_db(g)
    return y, {
        "max_reduction_db": round(float(reduction.max()), 2),
        "limiting_percent": round(100 * float(np.mean(reduction > 0.1)), 1),
    }


def compress(
    x: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    knee_db: float = 6.0,
    makeup_db: float = 0.0,
):
    """Feed-forward RMS compressor with a soft knee; channels are linked. Returns (compressed, stats)."""
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    n = len(x)
    nb = _blocks(n)
    sq = np.pad(np.mean(np.square(x, dtype=np.float64), axis=1), (0, nb * BLOCK - n))
    level = pow_db(sq.reshape(nb, BLOCK).mean(axis=1))
    over = level - threshold_db
    slope = 1.0 / ratio - 1.0
    k = max(knee_db, 1e-6)
    target = np.where(over <= -k / 2, 0.0, np.where(over >= k / 2, slope * over, slope * (over + k / 2) ** 2 / (2 * k)))
    att = np.exp(-BLOCK / (sr * max(attack_ms, 0.1) / 1000))
    rel = np.exp(-BLOCK / (sr * max(release_ms, 0.1) / 1000))
    gr = np.empty(nb)
    cur = 0.0
    for i, t in enumerate(target):
        coef = att if t < cur else rel
        cur = t + coef * (cur - t)
        gr[i] = cur
    y = _apply_gain(x, gr + makeup_db, in_db=True)
    active = gr < -0.1
    return y, {
        "max_reduction_db": round(float(-gr.min()), 2),
        "avg_reduction_when_active_db": round(float(-gr[active].mean()), 2) if active.any() else 0.0,
        "compressing_percent": round(100 * float(active.mean()), 1),
    }
