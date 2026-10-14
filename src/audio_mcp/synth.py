"""A small declarative synthesizer for sound effects, test tones and sketches of melodies."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .audio_io import AudioError
from .dsp import amp_db, parse_freq

WAVES = ("sine", "square", "saw", "triangle", "noise", "pink", "brown", "impulse", "dc", "silence")
NOISES = ("noise", "pink", "brown")
TEST_SIGNALS = ("impulse", "dc", "silence")  # exact signals: no default fade unless an envelope is given
MAX_DURATION = 600.0
MAX_HARMONICS = 300

SPEC_HELP = """spec = {
  "sample_rate": 44100, "duration": <seconds, default: end of last layer>, "normalize_peak_db": -1 (null to skip),
  "layers": [{
    "wave": "sine|square|saw|triangle|noise|pink|brown|impulse|dc|silence",
        # pink/brown are generated circularly, so they loop seamlessly as-is.
        # impulse: one full-scale sample at the layer start, or a train with "freq" impulses per second.
        # dc: a constant offset (level set by gain_db; 0 dB = +1.0). silence: exact zeros.
    "freq": 440 or "A4", "freq_end": <optional sweep target>, "sweep": "exp|linear", "phase": <start phase, degrees>,
    "start": 0, "duration": <seconds>, "gain_db": 0, "pan": -1..1,
    "envelope": {"attack": 0.005, "decay": 0, "sustain": 1, "release": 0.01},
    "vibrato": {"rate_hz": 5, "depth_cents": 20}, "tremolo": {"rate_hz": 6, "depth": 0.5},
    "lowpass": <hz>, "highpass": <hz>,
    "sequence": [["C4", 0.25], ["rest", 0.25], ["E4", 0.5]]   # optional: one note per item, played back to back
  }]
}"""


def _envelope(n, sr, env):
    env = env or {}
    a = int(float(env.get("attack", 0.005)) * sr)
    d = int(float(env.get("decay", 0.0)) * sr)
    rl = int(float(env.get("release", 0.01)) * sr)
    s = float(env.get("sustain", 1.0))
    if a + d + rl > n:
        k = n / max(1, a + d + rl)
        a, d, rl = int(a * k), int(d * k), int(rl * k)
    hold = n - a - d - rl
    return np.concatenate(
        [
            np.linspace(0, 1, a, endpoint=False),
            np.linspace(1, s, d, endpoint=False),
            np.full(hold, s),
            np.linspace(s, 0, rl),
        ]
    )[:n]


def _colored_noise(n, rng, exponent, sr):
    """1/f^exponent noise shaped in the frequency domain (so it loops seamlessly), with nothing below 20 Hz:
    unbounded low-frequency energy would show up as DC drift and rumble, which real audio doesn't have."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    shape = np.zeros_like(f)
    audible = f >= 20.0
    shape[audible] = (f[audible] / 20.0) ** -exponent
    y = np.fft.irfft(spec * shape, n)
    return y / (np.max(np.abs(y)) or 1.0)


def _oscillator(wave, freq, sr, rng, phase_deg=0.0, n=None):
    n = len(freq) if freq is not None else n
    if wave == "silence":
        return np.zeros(n)
    if wave == "dc":
        return np.ones(n)
    if wave == "impulse":
        out = np.zeros(n)
        if freq is None:
            out[0] = 1.0
        else:  # an impulse at the start of every period
            cycles = np.floor(np.concatenate(([0.0], np.cumsum(freq[:-1]) / sr)) + 1e-9)
            out[np.flatnonzero(np.diff(np.concatenate(([-1.0], cycles))) > 0)] = 1.0
        return out
    if wave == "noise":
        return np.clip(rng.standard_normal(n) / 3.0, -1, 1)
    if wave == "pink":
        return _colored_noise(n, rng, 0.5, sr)
    if wave == "brown":
        return _colored_noise(n, rng, 1.0, sr)
    # Start exactly at phase_deg on the first sample (cumulative phase of the preceding samples only).
    phase = 2 * np.pi * np.concatenate(([0.0], np.cumsum(freq[:-1]))) / sr + np.deg2rad(phase_deg)
    if wave == "sine":
        return np.sin(phase)
    nyq = sr / 2
    kmax = int(min(MAX_HARMONICS, nyq / max(float(freq.min()), 1.0)))
    out = np.zeros(n)
    for k in range(1, kmax + 1):
        if wave in ("square", "triangle") and k % 2 == 0:
            continue
        if wave == "triangle":
            amp = (-1) ** ((k - 1) // 2) / k**2
        else:
            amp = 1.0 / k
        out += amp * np.sin(k * phase) * (k * freq < nyq)  # band-limited: drop harmonics above Nyquist
    scale = {"saw": 2 / np.pi, "square": 4 / np.pi, "triangle": 8 / np.pi**2}[wave]
    return out * scale


def _freq(value):
    try:
        return parse_freq(value)
    except ValueError as e:
        raise AudioError(str(e)) from None


def _render_layer(layer, sr, rng):
    wave = layer.get("wave", "sine")
    if wave not in WAVES:
        raise AudioError(f"Unknown wave '{wave}'. Use one of: {', '.join(WAVES)}.")
    dur = float(layer["duration"])
    n = int(round(dur * sr))
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / sr
    if wave in NOISES or wave in ("dc", "silence") or (wave == "impulse" and "freq" not in layer):
        freq = np.ones(n) if wave != "impulse" else None
    else:
        if "freq" not in layer:
            raise AudioError(f"Layer with wave '{wave}' needs a 'freq' (Hz or note name like 'A4').")
        f0 = _freq(layer["freq"])
        if "freq_end" in layer and layer["freq_end"] is not None:
            f1 = _freq(layer["freq_end"])
            if layer.get("sweep", "exp") == "linear":
                freq = f0 + (f1 - f0) * t / dur
            else:
                freq = f0 * (f1 / f0) ** (t / dur)
        else:
            freq = np.full(n, f0)
        vib = layer.get("vibrato")
        if vib:
            freq = freq * 2 ** (
                float(vib.get("depth_cents", 20)) / 1200 * np.sin(2 * np.pi * float(vib.get("rate_hz", 5)) * t)
            )
    y = _oscillator(wave, freq, sr, rng, float(layer.get("phase", 0.0)), n)
    for key, btype in (("highpass", "highpass"), ("lowpass", "lowpass")):
        if layer.get(key):
            hz = float(layer[key])
            if not 0 < hz < sr / 2:
                raise AudioError(f"{key} {hz} Hz must be below Nyquist ({sr / 2:g} Hz).")
            y = signal.sosfilt(signal.butter(2, hz, btype, fs=sr, output="sos"), y)
    trem = layer.get("tremolo")
    if trem:
        depth = float(trem.get("depth", 0.5))
        y = y * (1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * float(trem.get("rate_hz", 6)) * t))
    if wave not in TEST_SIGNALS or "envelope" in layer:
        y = y * _envelope(n, sr, layer.get("envelope"))
    return y * 10 ** (float(layer.get("gain_db", 0)) / 20)


def _expand_sequences(layers):
    out = []
    for layer in layers:
        seq = layer.get("sequence")
        if not seq:
            out.append(layer)
            continue
        t = float(layer.get("start", 0.0))
        for item in seq:
            if isinstance(item, dict):
                note, dur = item.get("freq"), float(item["dur"])
            else:
                note, dur = item[0], float(item[1])
            if note not in (None, "rest", "r", "-"):
                sub = {k: v for k, v in layer.items() if k != "sequence"}
                sub.update({"freq": note, "start": t, "duration": dur})
                sub.setdefault(
                    "envelope", {"attack": 0.01, "decay": 0.05, "sustain": 0.7, "release": min(0.05, dur / 3)}
                )
                out.append(sub)
            t += dur
    return out


def synthesize(spec: dict):
    """Render a spec to (samples[frames, channels], sample_rate)."""
    if not isinstance(spec, dict) or not spec.get("layers"):
        raise AudioError("spec needs a non-empty 'layers' list.\n" + SPEC_HELP)
    sr = int(spec.get("sample_rate", 44100))
    if not 8000 <= sr <= 192000:
        raise AudioError("sample_rate must be between 8000 and 192000.")
    rng = np.random.default_rng(spec.get("seed", 0))
    layers = [dict(layer) for layer in _expand_sequences(spec["layers"])]
    total = spec.get("duration")
    for layer in layers:
        start = float(layer.get("start", 0.0))
        if layer.get("duration") is None:
            if total is None:
                raise AudioError("Give each layer a 'duration', or set the overall 'duration'.")
            layer["duration"] = float(total) - start
    end = max(float(layer.get("start", 0)) + float(layer["duration"]) for layer in layers)
    total = float(total) if total is not None else end
    if not 0 < total <= MAX_DURATION:
        raise AudioError(f"duration must be between 0 and {MAX_DURATION:g} seconds.")
    stereo = bool(spec.get("stereo")) or any("pan" in layer for layer in layers)
    out = np.zeros((int(round(total * sr)), 2 if stereo else 1))
    for layer in layers:
        y = _render_layer(layer, sr, rng)
        a = int(round(float(layer.get("start", 0.0)) * sr))
        b = min(len(out), a + len(y))
        if b <= a:
            continue
        if stereo:
            theta = (float(layer.get("pan", 0.0)) + 1) * np.pi / 4
            out[a:b, 0] += y[: b - a] * np.cos(theta)
            out[a:b, 1] += y[: b - a] * np.sin(theta)
        else:
            out[a:b, 0] += y[: b - a]
    target = spec.get("normalize_peak_db", -1.0)
    peak = float(np.max(np.abs(out)))
    if target is not None and peak > 0:
        out *= 10 ** ((float(target) - float(amp_db(peak))) / 20)
    return out.astype(np.float32), sr
