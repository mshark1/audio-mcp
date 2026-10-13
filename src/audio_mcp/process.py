"""Non-destructive editing: an ordered list of operations, always written to a new file by default."""

from __future__ import annotations

import inspect

import numpy as np
from scipy import signal

from . import dynamics, loudness, restore
from .audio_io import Audio, AudioError, load
from .dsp import amp_db, frame_rms, resample

OPS: dict[str, callable] = {}


def op(fn):
    OPS[fn.__name__.lstrip("_")] = fn
    return fn


def describe_ops() -> dict:
    """op name -> 'param=default, ...: description' for error messages and docs."""
    out = {}
    for name, fn in OPS.items():
        params = [
            p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}"
            for p in list(inspect.signature(fn).parameters.values())[2:]
        ]
        out[name] = f"({', '.join(params)}) {inspect.getdoc(fn) or ''}".strip()
    return out


def _seconds(sr, s):
    return max(0, int(round(float(s) * sr)))


def _check_hz(sr, hz):
    if not 0 < hz < sr / 2:
        raise AudioError(f"Frequency {hz} Hz must be between 0 and Nyquist ({sr / 2:g} Hz).")


def _lufs(x, sr):
    value = loudness.integrated_loudness(x, sr)
    if value is None:
        raise AudioError("LUFS needs at least 0.4 s of audio; use normalize_peak for short clips.")
    return value


@op
def _gain(x, sr, db):
    """Change level by `db` decibels."""
    return x * 10 ** (db / 20), sr, f"gain {db:+g} dB"


@op
def _normalize_peak(x, sr, db=-1.0):
    """Scale so the sample peak sits at `db` dBFS."""
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        raise AudioError("Can't normalize silence.")
    g = db - float(amp_db(peak))
    return x * 10 ** (g / 20), sr, f"normalize_peak to {db:g} dBFS ({g:+.2f} dB)"


@op
def _normalize_lufs(x, sr, lufs=-16.0, ceiling_db=-1.0, limit=False):
    """Scale to an integrated loudness target. Peaks never exceed `ceiling_db`: by default the gain is capped;
    with limit=true a true-peak limiter holds the peaks down so the loudness target is reached."""
    current = _lufs(x, sr)
    if not np.isfinite(current):
        raise AudioError("Loudness is undefined (audio is silent).")
    g = lufs - current
    peak_db = float(amp_db(np.max(np.abs(x))))
    if peak_db + g <= ceiling_db:
        return x * 10 ** (g / 20), sr, f"normalize_lufs {current:.1f} -> {lufs:g} LUFS ({g:+.2f} dB)"
    if not limit:
        g = ceiling_db - peak_db
        return (
            x * 10 ** (g / 20),
            sr,
            (
                f"normalize_lufs: target {lufs:g} LUFS would push the peak above {ceiling_db:g} dBFS; "
                f"gain capped at {g:+.2f} dB (about {current + g:.1f} LUFS). Pass limit=true to reach the target with a limiter."
            ),
        )
    for _ in range(4):  # limiting lowers loudness a little, so converge on the target
        y, stats = dynamics.limit(x * 10 ** (g / 20), sr, ceiling_db)
        reached = _lufs(y, sr)
        if abs(reached - lufs) < 0.1:
            break
        g += lufs - reached
    note = f"normalize_lufs {current:.1f} -> {reached:.1f} LUFS ({g:+.2f} dB gain, limiter up to {stats['max_reduction_db']} dB)"
    if stats["max_reduction_db"] > 6:
        note += "; heavy limiting, which can sound squashed - consider a lower target or compress first"
    return y, sr, note


@op
def _limit(x, sr, ceiling_db=-1.0, release_ms=60.0, lookahead_ms=5.0):
    """Look-ahead true-peak limiter: no peak (including inter-sample peaks) exceeds `ceiling_db`."""
    y, stats = dynamics.limit(x, sr, ceiling_db, release_ms, lookahead_ms)
    return (
        y,
        sr,
        f"limit at {ceiling_db:g} dBTP (max reduction {stats['max_reduction_db']} dB, active {stats['limiting_percent']}% of the time)",
    )


@op
def _denoise(x, sr, reduction_db=12.0, sensitivity=1.0, noise_start=None, noise_end=None):
    """Reduce steady background noise (hiss, hum, fans, room tone) by up to `reduction_db`. The noise profile is
    learned from the quietest parts, or from noise_start..noise_end seconds if given (best: a stretch of noise only).
    Raise `sensitivity` (e.g. 1.5) to remove more; too high dulls quiet consonants (f, s, th) - verify speech with audio_transcribe."""
    try:
        y, stats = restore.denoise(x, sr, reduction_db, sensitivity, noise_start, noise_end)
    except ValueError as e:
        raise AudioError(str(e)) from None
    src = f"{noise_start}-{noise_end}s" if noise_start is not None or noise_end is not None else "quietest frames"
    return y, sr, f"denoise up to {reduction_db:g} dB (noise ~{stats['estimated_noise_dbfs']} dBFS, profile from {src})"


@op
def _compress(x, sr, threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=120.0, knee_db=6.0, makeup_db=0.0):
    """Reduce dynamic range: levels above `threshold_db` are scaled down by `ratio` (soft knee, linked channels)."""
    try:
        y, stats = dynamics.compress(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_db)
    except ValueError as e:
        raise AudioError(str(e)) from None
    return (
        y,
        sr,
        (
            f"compress {ratio:g}:1 above {threshold_db:g} dBFS (max reduction {stats['max_reduction_db']} dB, "
            f"avg {stats['avg_reduction_when_active_db']} dB, active {stats['compressing_percent']}%), makeup {makeup_db:+g} dB"
        ),
    )


@op
def _trim_silence(x, sr, threshold_db=-60.0, pad_ms=10.0):
    """Remove leading and trailing audio quieter than `threshold_db`, keeping `pad_ms` of margin."""
    hop = max(1, int(0.005 * sr))
    # 30 ms frames: long enough that random noise spikes don't count as sound, short enough to keep attacks.
    active = np.flatnonzero(amp_db(frame_rms(x.mean(axis=1), 6 * hop, hop)) > threshold_db)
    if active.size == 0:
        raise AudioError(f"Everything is below {threshold_db} dBFS; nothing would remain.")
    pad = _seconds(sr, pad_ms / 1000)
    a = max(0, active[0] * hop - pad)
    b = min(len(x), active[-1] * hop + 6 * hop + pad)
    return x[a:b], sr, f"trim_silence removed {a / sr:.3f}s at start, {(len(x) - b) / sr:.3f}s at end"


def _fade_curve(n, curve):
    t = np.linspace(0.0, 1.0, n)
    if curve == "linear":
        return t
    if curve == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if curve == "equal_power":
        return np.sin(0.5 * np.pi * t)
    raise AudioError(f"Unknown fade curve '{curve}'. Use linear, cosine or equal_power.")


@op
def _fade_in(x, sr, seconds=0.01, curve="cosine"):
    """Fade in over `seconds`."""
    n = min(len(x), _seconds(sr, seconds))
    x = x.copy()
    x[:n] *= _fade_curve(n, curve)[:, None]
    return x, sr, f"fade_in {n / sr:.3f}s ({curve})"


@op
def _fade_out(x, sr, seconds=0.01, curve="cosine"):
    """Fade out over `seconds`."""
    n = min(len(x), _seconds(sr, seconds))
    x = x.copy()
    if n:
        x[-n:] *= _fade_curve(n, curve)[::-1, None]
    return x, sr, f"fade_out {n / sr:.3f}s ({curve})"


@op
def _crop(x, sr, start=0.0, end=None):
    """Keep only start..end seconds (relative to the audio at this point in the chain)."""
    a = _seconds(sr, start)
    b = len(x) if end is None else min(len(x), _seconds(sr, end))
    if b <= a:
        raise AudioError(f"crop range {start}..{end}s is empty (audio is {len(x) / sr:.3f}s).")
    return x[a:b], sr, f"crop {a / sr:.3f}-{b / sr:.3f}s"


@op
def _pad(x, sr, start=0.0, end=0.0):
    """Add silence before and/or after."""
    return np.pad(x, ((_seconds(sr, start), _seconds(sr, end)), (0, 0))), sr, f"pad {start}s before, {end}s after"


@op
def _highpass(x, sr, hz, order=4):
    """Remove content below `hz` (zero-phase Butterworth)."""
    _check_hz(sr, hz)
    sos = signal.butter(order, hz, "highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32), sr, f"highpass {hz:g} Hz"


@op
def _lowpass(x, sr, hz, order=4):
    """Remove content above `hz` (zero-phase Butterworth)."""
    _check_hz(sr, hz)
    sos = signal.butter(order, hz, "lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32), sr, f"lowpass {hz:g} Hz"


def _biquad(kind, sr, hz, db, q):
    """RBJ audio-EQ-cookbook coefficients."""
    A = 10 ** (db / 40)
    w0 = 2 * np.pi * hz / sr
    cw, sw = np.cos(w0), np.sin(w0)
    if kind == "peak":
        alpha = sw / (2 * q)
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        alpha = sw / 2 * np.sqrt(2)  # shelf slope S = 1
        sa = 2 * np.sqrt(A) * alpha
        if kind == "low":
            b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
            a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
        else:
            b = [
                A * ((A + 1) + (A - 1) * cw + sa),
                -2 * A * ((A - 1) + (A + 1) * cw),
                A * ((A + 1) + (A - 1) * cw - sa),
            ]
            a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return np.array(b) / a[0], np.array(a) / a[0]


@op
def _eq(x, sr, hz, db, q=1.0):
    """Peaking EQ: boost/cut `db` around `hz` with bandwidth `q` (higher = narrower)."""
    _check_hz(sr, hz)
    b, a = _biquad("peak", sr, hz, db, q)
    return signal.lfilter(b, a, x, axis=0).astype(np.float32), sr, f"eq {db:+g} dB at {hz:g} Hz (Q {q:g})"


@op
def _low_shelf(x, sr, hz, db):
    """Boost/cut everything below `hz`."""
    _check_hz(sr, hz)
    b, a = _biquad("low", sr, hz, db, 0.707)
    return signal.lfilter(b, a, x, axis=0).astype(np.float32), sr, f"low_shelf {db:+g} dB below {hz:g} Hz"


@op
def _high_shelf(x, sr, hz, db):
    """Boost/cut everything above `hz`."""
    _check_hz(sr, hz)
    b, a = _biquad("high", sr, hz, db, 0.707)
    return signal.lfilter(b, a, x, axis=0).astype(np.float32), sr, f"high_shelf {db:+g} dB above {hz:g} Hz"


@op
def _remove_dc(x, sr):
    """Subtract each channel's mean."""
    return x - x.mean(axis=0, keepdims=True), sr, "remove_dc"


@op
def _to_mono(x, sr):
    """Average all channels into one."""
    return x.mean(axis=1, keepdims=True), sr, "to_mono"


@op
def _to_stereo(x, sr):
    """Duplicate a mono signal to two channels."""
    if x.shape[1] != 1:
        raise AudioError("to_stereo expects mono input.")
    return np.repeat(x, 2, axis=1), sr, "to_stereo"


@op
def _resample(x, sr, rate):
    """Change the sample rate to `rate` Hz."""
    if not 1000 <= rate <= 384000:
        raise AudioError(f"rate must be between 1000 and 384000 Hz, got {rate}.")
    return resample(x, sr, int(rate)), int(rate), f"resample {sr} -> {int(rate)} Hz"


@op
def _reverse(x, sr):
    """Play backwards."""
    return x[::-1].copy(), sr, "reverse"


@op
def _loop_crossfade(x, sr, seconds=0.05):
    """Make a seamless loop: blend the tail into the head (output gets `seconds` shorter)."""
    n = _seconds(sr, seconds)
    if n < 1 or 2 * n >= len(x):
        raise AudioError(f"loop_crossfade needs 0 < seconds < half the length ({len(x) / sr / 2:.3f}s).")
    t = np.linspace(0.0, 1.0, n)[:, None]
    blend = x[-n:] * np.cos(0.5 * np.pi * t) + x[:n] * np.sin(0.5 * np.pi * t)
    return np.concatenate([blend, x[n:-n]]), sr, f"loop_crossfade {n / sr:.3f}s"


def _load_other(path, sr, channels):
    other = load(path)
    y = resample(other.samples, other.sr, sr)
    if y.shape[1] != channels:
        y = np.repeat(y.mean(axis=1, keepdims=True), channels, axis=1)
    return y


@op
def _mix(x, sr, path, at=0.0, gain_db=0.0):
    """Layer another file on top, starting at `at` seconds (extends the length if needed)."""
    y = _load_other(path, sr, x.shape[1]) * 10 ** (gain_db / 20)
    a = _seconds(sr, at)
    out = np.zeros((max(len(x), a + len(y)), x.shape[1]), dtype=np.float32)
    out[: len(x)] += x
    out[a : a + len(y)] += y
    return out, sr, f"mix {path} at {at}s ({gain_db:+g} dB)"


@op
def _append(x, sr, path, gap=0.0):
    """Append another file after `gap` seconds of silence."""
    y = _load_other(path, sr, x.shape[1])
    gap_n = np.zeros((_seconds(sr, gap), x.shape[1]), dtype=np.float32)
    return np.concatenate([x, gap_n, y]), sr, f"append {path}"


def apply_chain(audio: Audio, operations: list[dict]):
    if not isinstance(operations, list) or not operations:
        raise AudioError('operations must be a non-empty list, e.g. [{"op": "normalize_peak", "db": -1}].')
    x = audio.samples.astype(np.float32, copy=False)
    sr = audio.sr
    log = []
    for i, spec in enumerate(operations, 1):
        if not isinstance(spec, dict) or "op" not in spec:
            raise AudioError(f'Operation #{i} must be an object like {{"op": "gain", "db": -3}}.')
        name = spec["op"]
        if name not in OPS:
            raise AudioError(f"Unknown op '{name}'. Available ops: {describe_ops()}")
        params = {k: v for k, v in spec.items() if k != "op"}
        try:
            inspect.signature(OPS[name]).bind(x, sr, **params)
        except TypeError as e:
            raise AudioError(f"Bad parameters for '{name}' ({e}). Signature: {name}{describe_ops()[name]}") from None
        x, sr, note = OPS[name](x, sr, **params)
        x = np.asarray(x, dtype=np.float32)
        if len(x) == 0:
            raise AudioError(f"Operation #{i} ('{name}') left no audio.")
        log.append(note)
    return x, sr, log
