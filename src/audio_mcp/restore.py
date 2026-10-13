"""Noise reduction by spectral gating, with a noise profile learned from the quietest frames or a given range."""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

N_FFT = 2048
HOP = 512
CHUNK_S = 20.0
OVERLAP_S = 0.5


def _stft(x, sr):
    return signal.stft(x, sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="even", padded=True)[2]


def noise_profile(
    mono: np.ndarray, sr: int, start: float | None = None, end: float | None = None, quantile: float = 0.1
) -> np.ndarray:
    """Mean power per frequency bin of the noise: either the given range, or the quietest `quantile` of frames."""
    if start is not None or end is not None:
        a = int((start or 0.0) * sr)
        b = len(mono) if end is None else int(end * sr)
        seg = mono[a:b]
        if len(seg) < N_FFT:
            raise ValueError(f"The noise range {start}..{end}s is shorter than {N_FFT / sr * 1000:.0f} ms.")
        return np.mean(np.abs(_stft(seg, sr)) ** 2, axis=1)
    # Sample up to ~60 s spread over the file to keep this cheap on long recordings.
    step = max(1, len(mono) // int(60 * sr)) if len(mono) > 60 * sr else 1
    pieces = [mono[i : i + int(sr)] for i in range(0, len(mono), int(sr) * step)][:60] if step > 1 else [mono]
    power = np.concatenate([np.abs(_stft(p, sr)) ** 2 for p in pieces if len(p) >= N_FFT], axis=1)
    energy = power.sum(axis=0)
    quiet = energy <= np.quantile(energy, quantile)
    return power[:, quiet].mean(axis=1)


def _gate(x: np.ndarray, sr: int, noise_pow: np.ndarray, reduction_db: float, sensitivity: float) -> np.ndarray:
    Z = _stft(x, sr)
    # Smooth the power over neighbouring bins and frames first: raw noise power fluctuates so much that many
    # noise-only bins would otherwise poke above the profile and pass through ("musical noise").
    P = ndimage.uniform_filter(np.abs(Z) ** 2, size=(3, 5), mode="nearest")
    floor = 10 ** (-reduction_db / 20)
    # Inside sound: gentle power-subtraction gain per bin, so quiet consonants (f, s, th) survive; noise under
    # the signal is largely masked anyway. `sensitivity` scales the noise estimate.
    gain = np.sqrt(np.maximum(1.0 - sensitivity * noise_pow[:, None] / np.maximum(P, 1e-20), 0.0))
    gain = ndimage.uniform_filter(gain, size=(1, 3), mode="nearest")
    # Between sounds (pauses), where noise is most audible: full reduction. Frames count as sound when their
    # energy is 3 dB above the noise profile; a 150 ms hold on both sides protects word onsets and tails.
    active = P.sum(axis=0) > 2.0 * sensitivity * noise_pow.sum()
    hold = max(1, int(round(0.15 * sr / HOP)))
    active = ndimage.binary_dilation(active, structure=np.ones(2 * hold + 1, dtype=bool))
    gate = ndimage.uniform_filter1d(np.where(active, 1.0, 0.0), size=5, mode="nearest")  # ~60 ms fades
    gain = np.maximum(gain * gate[None, :], floor)
    y = signal.istft(Z * gain, sr, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=True)[1]
    return y[: len(x)]


def denoise(
    x: np.ndarray,
    sr: int,
    reduction_db: float = 12.0,
    sensitivity: float = 1.0,
    noise_start: float | None = None,
    noise_end: float | None = None,
):
    """Returns (cleaned [frames, channels], stats). Processes in overlapping chunks so memory stays bounded."""
    mono = x.mean(axis=1)
    noise_pow = noise_profile(mono, sr, noise_start, noise_end)
    chunk, ov = int(CHUNK_S * sr), int(OVERLAP_S * sr)
    out = np.zeros(x.shape, dtype=np.float32)
    weight = np.zeros(len(x), dtype=np.float32)
    fade = np.linspace(0.0, 1.0, ov) if ov else np.zeros(0)
    for a in range(0, len(x), chunk):
        lo, hi = max(0, a - ov), min(len(x), a + chunk + ov)
        w = np.ones(hi - lo)
        if lo > 0:
            w[: min(ov, len(w))] = fade[: min(ov, len(w))]
        if hi < len(x):
            w[-ov:] = np.minimum(w[-ov:], fade[::-1])
        for c in range(x.shape[1]):
            out[lo:hi, c] += _gate(x[lo:hi, c], sr, noise_pow, reduction_db, sensitivity) * w
        weight[lo:hi] += w
    out /= np.maximum(weight, 1e-9)[:, None]
    # scipy's STFT divides by sum(window); undo that before applying Parseval to get the noise's mean square.
    w = signal.get_window("hann", N_FFT)
    noise_ms = float(noise_pow.sum()) * 2 * w.sum() ** 2 / (N_FFT * np.sum(w**2))
    noise_db = 10 * np.log10(max(noise_ms, 1e-12))
    return out.astype(np.float32), {"estimated_noise_dbfs": round(noise_db, 1)}
