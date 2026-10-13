"""Waveform and spectrogram images, laid out for a vision model to read."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from . import pitch as pitch_mod  # noqa: E402
from .audio_io import Audio, AudioError  # noqa: E402
from .dsp import amp_db, pow_db, stft_chunks  # noqa: E402

KINDS = ("both", "waveform", "spectrogram")
FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def _hz_label(v, _pos=None):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def _waveform(ax, audio: Audio):
    sr, t0 = audio.sr, audio.offset
    chans = audio.samples if audio.channels <= 2 else audio.mono[:, None]
    labels = ["L", "R"] if chans.shape[1] == 2 else ["mono"]
    colors = ["#1f77b4", "#ff7f0e"]
    n = len(chans)
    for c in range(chans.shape[1]):
        y = chans[:, c]
        if n <= 4000:
            ax.plot(t0 + np.arange(n) / sr, y, lw=0.8, color=colors[c], label=labels[c])
        else:
            edges = np.linspace(0, n, 2401).astype(int)[:-1]
            ax.fill_between(
                t0 + edges / sr,
                np.minimum.reduceat(y, edges),
                np.maximum.reduceat(y, edges),
                step="post",
                lw=0,
                alpha=0.6,
                color=colors[c],
                label=labels[c],
            )
    for lim in (-1, 1):
        ax.axhline(lim, color="red", lw=0.6, ls="--")
    ax.set_ylim(-1.08, 1.08)
    ax.set_ylabel("amplitude")
    ax.grid(alpha=0.3)
    if chans.shape[1] == 2:
        ax.legend(loc="upper right", fontsize=8)


def _column_power(x, sr: int, nperseg: int, max_columns: int = 1600):
    """Power spectrogram with at most `max_columns` time columns. Each column averages every STFT frame in its
    time span, so on long files nothing between columns is skipped. Streams, so memory stays small."""
    hop = nperseg // 8 if len(x) < 200 * nperseg else nperseg // 2
    n_frames = 1 + max(0, len(x) - nperseg) // hop
    cols = min(max_columns, n_frames)
    col_of_frame = (np.arange(n_frames) * cols) // n_frames
    S = np.zeros((nperseg // 2 + 1, cols))
    counts = np.bincount(col_of_frame, minlength=cols)
    for i0, mag in stft_chunks(x, nperseg, hop):
        idx = col_of_frame[i0 : i0 + len(mag)]
        np.add.at(S.T, idx, mag.astype(np.float64) ** 2)
    S /= np.maximum(counts, 1)[None, :]
    edges = np.searchsorted(col_of_frame, np.arange(cols))
    t = (edges * hop + (counts * hop) / 2 + nperseg / 2) / sr
    return S, t, np.fft.rfftfreq(nperseg, 1 / sr)


def _spectrogram(ax, audio: Audio, freq_scale: str, max_freq, pitch_overlay: bool):
    sr, x = audio.sr, audio.mono
    nperseg = 2048 if sr <= 48000 else 4096
    while nperseg > 64 and nperseg > len(x):
        nperseg //= 2
    S, t, f = _column_power(x, sr, nperseg)
    S_db = pow_db(S)
    vmax = float(S_db.max())
    f, S_db = f[1:], S_db[1:]
    top = min(float(max_freq) if max_freq else sr / 2, sr / 2)
    mesh = ax.pcolormesh(
        t + audio.offset, f, S_db, shading="auto", cmap="magma", vmin=vmax - 90, vmax=vmax, rasterized=True
    )
    if freq_scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(max(20.0, float(f[0])), top)
        ax.yaxis.set_major_locator(FixedLocator([v for v in FREQ_TICKS if v <= top]))
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(FuncFormatter(_hz_label))
    ax.set_ylabel("frequency (Hz)")
    if pitch_overlay:
        times, f0, _ = pitch_mod.track(x, sr)
        ax.plot(times + audio.offset, f0, color="cyan", lw=1.2, label="pitch (f0)")
        ax.legend(loc="upper right", fontsize=8)
    return mesh


def render(
    audio: Audio, kind: str = "both", freq_scale: str = "log", max_freq=None, pitch_overlay: bool = False
) -> bytes:
    if kind not in KINDS:
        raise AudioError(f"kind must be one of {KINDS}.")
    if kind != "waveform" and audio.frames < 256:
        raise AudioError(f"Only {audio.frames} samples: too short for a spectrogram. Use kind='waveform'.")
    if freq_scale not in ("log", "linear"):
        raise AudioError("freq_scale must be 'log' or 'linear'.")
    if kind == "both":
        fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2.3]})
    elif kind == "waveform":
        fig, ax_w = plt.subplots(figsize=(12, 3.5))
        ax_s = None
    else:
        fig, ax_s = plt.subplots(figsize=(12, 5))
        ax_w = None
    try:
        if ax_w is not None:
            _waveform(ax_w, audio)
        if ax_s is not None:
            mesh = _spectrogram(ax_s, audio, freq_scale, max_freq, pitch_overlay)
            fig.colorbar(
                mesh, ax=[a for a in (ax_w, ax_s) if a is not None], pad=0.01, fraction=0.03, label="dB (relative)"
            )
        bottom = ax_s if ax_s is not None else ax_w
        bottom.set_xlabel("time (s)")
        bottom.set_xlim(audio.offset, audio.offset + audio.duration)
        name = audio.path.name if audio.path else "audio"
        peak = float(amp_db(np.max(np.abs(audio.samples))))
        fig.suptitle(
            f"{name}  |  {audio.offset:.2f}-{audio.offset + audio.duration:.2f}s  |  {audio.sr} Hz, {audio.channels} ch  |  peak {peak:+.1f} dBFS",
            fontsize=11,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)
