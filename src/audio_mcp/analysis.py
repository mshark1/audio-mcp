"""Measurements: levels, loudness, stereo image, spectrum, silence, rhythm, edges, and a plain-language issue list."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from . import loudness
from . import pitch as pitch_mod
from .audio_io import Audio, AudioError, open_source, resolve_path
from .dsp import (
    amp_db,
    describe_hz,
    frame_rms,
    octave_flatness,
    parabolic,
    pow_db,
    r,
    runs,
    sparkline,
    stft_chunks,
    true_peak,
)

BANDS = (
    ("sub", 20, 60),
    ("bass", 60, 250),
    ("low_mid", 250, 500),
    ("mid", 500, 2000),
    ("high_mid", 2000, 4000),
    ("presence", 4000, 6000),
    ("brilliance", 6000, 20000),
)
SILENCE_DB = -60.0
PHASE_PROBLEM_CORRELATION = -0.3  # below this, mono playback loses substantial level
LOUDNESS_TARGETS = "typical targets: -14 LUFS music streaming, -16 podcasts/voice, -23 broadcast (EBU R128)"


# ---------------------------------------------------------------- spectral profile


@dataclass
class SpectralProfile:
    freqs: np.ndarray
    power: np.ndarray  # mean power per bin; sums to the signal's mean square
    flatness: np.ndarray  # per frame
    frame_db: np.ndarray  # per frame level


def spectral_profile(mono, sr: int, n_fft: int = 8192, hop: int = 2048) -> SpectralProfile:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    window = np.hanning(n_fft)
    scale = 2.0 / (n_fft * np.sum(window**2))
    total = np.zeros(len(freqs))
    count = 0
    flat, level = [], []
    for _, mag in stft_chunks(mono, n_fft, hop):
        p = mag.astype(np.float64) ** 2 * scale
        total += p.sum(axis=0)
        count += len(p)
        level.append(pow_db(p.sum(axis=1)))
        flat.append(octave_flatness(p, freqs))
    return SpectralProfile(freqs, total / max(count, 1), np.concatenate(flat), np.concatenate(level))


def band_levels_db(profile: SpectralProfile) -> dict:
    f, p = profile.freqs, profile.power
    return {name: r(pow_db(p[(f >= lo) & (f < hi)].sum()), 1) for name, lo, hi in BANDS}


def spectrum_stats(profile: SpectralProfile, duration: float) -> dict:
    f, p = profile.freqs, profile.power
    total = p.sum()
    if total <= 1e-14:
        return {"comment": "silent"}
    mag = np.sqrt(p)
    centroid = float(np.sum(f * mag) / np.sum(mag))
    rolloff = float(f[min(np.searchsorted(np.cumsum(p) / total, 0.85), len(f) - 1)])
    active = profile.frame_db > max(float(profile.frame_db.max()) - 50.0, -70.0)
    flatness = float(np.median(profile.flatness[active])) if active.any() else 0.0

    pdb = pow_db(p)
    smooth = ndimage.uniform_filter1d(pdb, 5)
    above = np.flatnonzero(smooth > smooth.max() - 70.0)
    upper = float(f[above[-1]]) if above.size else 0.0

    peaks, props = signal.find_peaks(pdb, prominence=10.0, distance=3)
    order = np.argsort(pdb[peaks])[::-1][:6]
    df = f[1] - f[0]
    tonal_peaks = []
    for k in sorted(peaks[order]):
        pos, val = parabolic(pdb, int(k))
        hz = pos * df
        if hz < 20:
            continue
        entry = describe_hz(hz)
        entry["level_db"] = r(val - pdb.max(), 1)
        tonal_peaks.append(entry)

    out = {
        "centroid_hz": r(centroid, 0),
        "rolloff_85_hz": r(rolloff, 0),
        "flatness": r(flatness, 3),
        "brightness": _brightness_word(centroid),
        "character": "noise-like" if flatness > 0.35 else "mixed tonal/noise" if flatness > 0.08 else "tonal",
        "band_energy_percent": {name: r(100 * p[(f >= lo) & (f < hi)].sum() / total, 1) for name, lo, hi in BANDS},
        "below_20hz_percent": r(100 * p[f < 20].sum() / total, 1),
        "content_upper_limit_hz": r(upper, 0),
        "spectral_peaks": tonal_peaks,
    }
    return out


def mains_hum(peaks: list[dict], noise_floor_dbfs, character) -> tuple[int, list[float]] | None:
    """Detect 50/60 Hz hum: needs a peak at the mains frequency or a harmonic, plus supporting evidence (two or
    more harmonics, a raised noise floor, or non-tonal material where a steady low tone is unexpected)."""
    for base in (60, 50):
        found = [p["hz"] for p in peaks if any(abs(p["hz"] - k * base) < 1.5 for k in range(1, 7))]
        # Real hum always shows the mains fundamental or its 2nd harmonic; one peak that merely lands on
        # e.g. 250 Hz (5 x 50) is just a note.
        if not any(abs(h - k * base) < 1.5 for h in found for k in (1, 2)):
            continue
        noisy_floor = noise_floor_dbfs is not None and noise_floor_dbfs > -55
        if len(found) >= 2 or noisy_floor or character != "tonal":
            return base, [float(round(h / base) * base) for h in found]
    return None


def likely_polyphonic(peaks: list[dict], within_db: float = 12.0) -> bool:
    """True if the strong peaks can't be explained as harmonics of one fundamental (tried down to base/4)."""
    hz = sorted(p["hz"] for p in peaks if p["level_db"] >= -within_db)
    if len(hz) < 2:
        return False
    for k in range(1, 5):
        f0 = hz[0] / k
        if all(abs(h / f0 - round(h / f0)) < 0.1 for h in hz):
            return False
    return True


def _brightness_word(centroid: float) -> str:
    for limit, word in ((400, "very dark"), (1000, "dark/warm"), (2500, "balanced"), (5000, "bright")):
        if centroid < limit:
            return word
    return "very bright/harsh"


# ---------------------------------------------------------------- levels


CHUNK = 1 << 20  # samples per chunk for whole-signal statistics; keeps temporary arrays small


def _chunks(n: int, extra: int = 0):
    """(start, stop, stop_with_lookahead) for chunked passes that need `extra` samples beyond each chunk."""
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        yield a, b, min(n, b + extra)


def level_stats(audio: Audio) -> dict:
    x, sr = audio.samples, audio.sr
    n = len(x)
    peak, sumsq, colsum = 0.0, 0.0, np.zeros(x.shape[1])
    clipped = []  # first index of each pair of consecutive full-scale samples
    for a, b, c in _chunks(n, extra=1):
        blk = x[a:c]
        mag = np.abs(blk)
        peak = max(peak, float(mag[: b - a].max()))
        sumsq += float(np.sum(np.square(blk[: b - a], dtype=np.float64)))
        colsum += blk[: b - a].sum(axis=0, dtype=np.float64)
        full = (mag >= 0.999).any(axis=1)
        clipped.append(a + np.flatnonzero(full[:-1] & full[1:])[: b - a])
    clipped_idx = np.concatenate(clipped)
    clipped_idx = clipped_idx[clipped_idx < n - 1]
    events = [g for g in np.split(clipped_idx, np.flatnonzero(np.diff(clipped_idx) > sr * 0.01) + 1) if g.size]
    flat_top = 0
    if 0.1 < peak < 0.999:  # clipped earlier, then turned down: runs of 3 samples stuck at the peak
        for a, b, c in _chunks(n, extra=2):
            near = (np.abs(x[a:c]) >= peak * 0.9995).any(axis=1)
            flat_top += int(np.count_nonzero((near[2:] & near[1:-1] & near[:-2])[: b - a]))
    ms = sumsq / (n * x.shape[1])
    lufs = loudness.integrated_loudness(x, sr)
    rms_db = float(pow_db(ms))
    out = {
        "peak_dbfs": r(amp_db(peak), 2),
        "true_peak_dbtp": r(amp_db(true_peak(x, sr)), 2),
        "rms_dbfs": r(rms_db, 2),
        "lufs_integrated": r(lufs, 2),
        "lufs_momentary_max": r(loudness.momentary_max(x, sr), 2),
        "crest_factor_db": r(float(amp_db(peak)) - rms_db, 2),
        "dc_offset": r(float(np.max(np.abs(colsum / n))), 4),
        "clipped_samples": int(clipped_idx.size),
        "clip_events": len(events),
        "flat_top_samples": flat_top,
    }
    if events:
        out["first_clips_s"] = [r(g[0] / sr + audio.offset, 3) for g in events[:5]]
    if lufs is None:
        out["lufs_note"] = (
            "Integrated LUFS needs at least 0.4 s of audio; for short clips compare lufs_momentary_max instead."
        )
    return out


def loudness_shape(mono, sr: int, segments: int = 40) -> dict:
    n = min(segments, max(1, len(mono) // max(1, int(0.01 * sr))))
    edges = np.linspace(0, len(mono), n + 1).astype(int)
    lv = np.array([pow_db(np.mean(np.square(mono[a:b], dtype=np.float64))) for a, b in zip(edges[:-1], edges[1:])])
    top = float(lv.max())
    return {
        "sparkline": sparkline(np.maximum(lv, top - 40), top - 40, top),
        "segment_s": r(len(mono) / sr / n, 3),
        "range_db": [r(max(lv.min(), -120), 1), r(top, 1)],
        "note": "each character is one segment's RMS level; scale spans the top 40 dB",
    }


def silence_stats(mono, sr: int, offset: float = 0.0) -> dict:
    hop = max(1, int(0.01 * sr))
    lv = amp_db(frame_rms(mono, hop, hop))
    active = lv > SILENCE_DB
    dur = len(mono) / sr
    if not active.any():
        return {"all_silent": True, "threshold_dbfs": SILENCE_DB}
    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))
    inner = lv[first : last + 1]
    gaps = [(a + first, b + first) for a, b in runs(inner <= SILENCE_DB) if (b - a) * hop / sr >= 0.3]
    return {
        "threshold_dbfs": SILENCE_DB,
        "leading_s": r(first * hop / sr, 3),
        "trailing_s": r(max(0.0, dur - (last + 1) * hop / sr), 3),
        "silent_fraction": r(1.0 - active.mean(), 3),
        "gap_count": len(gaps),
        "gaps_s": [[r(a * hop / sr + offset, 2), r(b * hop / sr + offset, 2)] for a, b in gaps[:10]],
        "noise_floor_dbfs": r(np.percentile(inner, 5), 1),
        "loud_level_dbfs": r(np.percentile(inner, 95), 1),
    }


def stereo_stats(x: np.ndarray) -> dict:
    eL = eR = eLR = 0.0
    max_diff = 0.0
    for a, b, _ in _chunks(len(x)):
        L = x[a:b, 0].astype(np.float64)
        R = x[a:b, 1].astype(np.float64)
        eL += float(L @ L)
        eR += float(R @ R)
        eLR += float(L @ R)
        max_diff = max(max_diff, float(np.max(np.abs(L - R))))
    eM = (eL + eR + 2 * eLR) / 4  # energy of (L+R)/2
    eS = (eL + eR - 2 * eLR) / 4  # energy of (L-R)/2
    corr = eLR / np.sqrt(eL * eR) if eL > 0 and eR > 0 else None
    if max_diff < 1e-5:
        width = "mono (identical channels)"
    elif corr is None:
        width = "one channel silent"
    elif corr < PHASE_PROBLEM_CORRELATION:
        width = "phase problem (negative correlation)"
    else:
        # Near-zero correlation is normal for wide material (stereo reverb, spaced mics, independent noise).
        width = "narrow" if corr > 0.9 else "moderate" if corr > 0.5 else "wide"
    return {
        "correlation": r(corr, 3),
        "width": width,
        "side_to_mid_db": r(pow_db(eS) - pow_db(eM), 1),
        "balance_db": r(pow_db(eL) - pow_db(eR), 2),
        "mono_sum_change_db": r(pow_db(eM) - pow_db((eL + eR) / 2), 2),
    }


# ---------------------------------------------------------------- rhythm


def onset_envelope(mono, sr: int):
    n_fft = 2048 if sr <= 48000 else 4096
    hop = n_fft // 4
    norm = np.hanning(n_fft).sum() / 2
    # Centre frames on their timestamps and start from silence so a sound at t=0 counts as an onset.
    padded = np.pad(np.asarray(mono, dtype=np.float32), (n_fft // 2, n_fft // 2))
    prev, flux = np.zeros(n_fft // 2 + 1), []
    for _, mag in stft_chunks(padded, n_fft, hop):
        lm = np.log1p(100.0 * mag / norm)
        flux.append(np.maximum(np.diff(np.vstack([prev[None, :], lm]), axis=0), 0.0).mean(axis=1))
        prev = lm[-1]
    env = np.concatenate(flux)
    times = (np.arange(len(env)) + 0.5) * hop / sr  # a window registers an onset before its centre reaches it
    times[0] = 0.0
    return env, times, sr / hop


def pick_onsets(env, fps: float) -> np.ndarray:
    spread = float(np.percentile(env, 99) - np.median(env))
    if spread <= 1e-4:
        return np.array([], dtype=int)
    local = ndimage.median_filter(env, size=max(3, int(0.5 * fps)) | 1, mode="constant", cval=0.0)
    # find_peaks never reports the first or last sample, so pad both ends with zeros to catch a sound that starts
    # or ends the file.
    padded = np.concatenate(([0.0], env, [0.0]))
    height = np.concatenate(([np.inf], local + 0.15 * spread, [np.inf]))
    peaks, _ = signal.find_peaks(padded, height=height, distance=max(1, int(0.05 * fps)))
    return peaks - 1


def estimate_tempo(env, fps: float):
    e = env - env.mean()
    n = len(e)
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * n)) ** 2)[:n]
    lag_min, lag_max = int(fps * 60 / 200), min(int(fps * 60 / 50) + 1, n // 2)
    if ac[0] <= 0 or lag_max <= lag_min + 2:
        return None, 0.0
    lags = np.arange(lag_min, lag_max)
    bpm = 60 * fps / lags
    score = ac[lags] * np.exp(
        -0.5 * (np.log2(bpm / 120) / 0.9) ** 2
    )  # mild prior toward ~120 BPM resolves octave ambiguity
    best = lag_min + int(np.argmax(score))
    lag, _ = parabolic(ac, best)
    return 60 * fps / lag, float(ac[best] / ac[0])


def rhythm_stats(mono, sr: int, offset: float = 0.0) -> dict:
    env, times, fps = onset_envelope(mono, sr)
    onsets = times[pick_onsets(env, fps)] + offset
    dur = len(mono) / sr
    out = {
        "onset_count": int(len(onsets)),
        "onsets_per_s": r(len(onsets) / dur, 2),
        "first_onsets_s": [r(t, 3) for t in onsets[:16]],
    }
    if len(onsets) >= 4:
        ioi = np.diff(onsets)
        cv = float(np.std(ioi) / np.mean(ioi))
        out["timing"] = "very regular" if cv < 0.05 else "regular" if cv < 0.2 else "irregular"
    if dur >= 4.0 and len(onsets) >= 4:
        bpm, conf = estimate_tempo(env, fps)
        if bpm:
            out["tempo_bpm"] = r(bpm, 1)
            out["tempo_confidence"] = r(conf, 2)
            if conf < 0.2:
                out["tempo_note"] = "weak periodicity; tempo is a guess (half/double tempo errors are common)"
    return out


# ---------------------------------------------------------------- edges / loop seam


def edge_stats(audio: Audio) -> dict:
    x, sr = audio.samples, audio.sr
    mono = audio.mono
    n = max(1, min(int(0.05 * sr), audio.frames // 4))
    # Typical sample-to-sample step; on long files estimate it from 64 evenly spaced windows.
    if len(mono) > 4 * CHUNK:
        starts = np.linspace(0, len(mono) - 65536, 64).astype(int)
        steps = np.concatenate([np.abs(np.diff(mono[s0 : s0 + 65536])) for s0 in starts])
    else:
        steps = np.abs(np.diff(mono))
    typical = float(np.percentile(steps, 99.9)) if len(steps) > 10 else 0.0
    jump = float(np.max(np.abs(x[-1] - x[0])))
    head = pow_db(np.mean(np.square(x[:n], dtype=np.float64)))
    tail = pow_db(np.mean(np.square(x[-n:], dtype=np.float64)))
    return {
        "start_amplitude": r(np.max(np.abs(x[0])), 4),
        "end_amplitude": r(np.max(np.abs(x[-1])), 4),
        "loop_seam": {
            "sample_jump": r(jump, 4),
            "typical_step": r(typical, 4),
            "level_jump_db": r(head - tail, 1),
            "likely_click": bool(jump > max(3 * typical, 0.01)),
        },
    }


# ---------------------------------------------------------------- issues


def find_issues(rep: dict) -> list[dict]:
    issues = []

    def add(severity, message):
        issues.append({"severity": severity, "message": message})

    lv = rep["levels"]
    if lv["clipped_samples"]:
        where = ", ".join(f"{t}s" for t in lv.get("first_clips_s", []))
        add(
            "error",
            f"Digital clipping: {lv['clipped_samples']} clipped samples in {lv['clip_events']} places (first at {where}).",
        )
    elif lv["flat_top_samples"] > 10:
        add(
            "warning",
            f"{lv['flat_top_samples']} flat-topped samples at the peak level: looks clipped earlier and then turned down.",
        )
    if not lv["clipped_samples"] and lv["true_peak_dbtp"] is not None and lv["true_peak_dbtp"] > -1.0:
        add(
            "warning",
            f"True peak {lv['true_peak_dbtp']:+.1f} dBTP; above -1 dBTP risks clipping after MP3/AAC encoding.",
        )
    lufs = lv["lufs_integrated"]
    if lufs is not None and lufs < -35:
        add("warning", f"Very quiet: {lufs} LUFS integrated ({LOUDNESS_TARGETS}).")
    elif lufs is not None and lufs > -8:
        add(
            "warning", f"Extremely loud: {lufs} LUFS; likely heavily limited, and streaming services will turn it down."
        )
    if lv["dc_offset"] > 0.005:
        add("warning", f"DC offset of {lv['dc_offset']}; fix with the remove_dc operation.")

    si = rep.get("silence", {})
    if si.get("all_silent"):
        add("error", f"The audio is silent (nothing above {SILENCE_DB} dBFS).")
    else:
        for key, label in (("leading_s", "leading"), ("trailing_s", "trailing")):
            if si.get(key, 0) >= 0.5:
                add("info", f"{si[key]}s of {label} silence (trim_silence removes it).")
        if si.get("noise_floor_dbfs", -120) > -50 and si.get("loud_level_dbfs", 0) - si.get("noise_floor_dbfs", 0) < 40:
            add(
                "info",
                f"Noise floor around {si['noise_floor_dbfs']} dBFS in the quietest parts; hiss or background noise may be audible.",
            )

    st = rep.get("stereo")
    if st:
        if st["correlation"] is not None and st["correlation"] < PHASE_PROBLEM_CORRELATION:
            add(
                "error",
                f"Channels are out of phase (correlation {st['correlation']}); summed to mono the level changes by "
                f"{st['mono_sum_change_db']} dB, so it will thin out or vanish on mono speakers.",
            )
        if st["width"].startswith("mono"):
            add("info", "Both channels are identical; the file could be saved as mono at half the size.")
        if abs(st["balance_db"] or 0) > 3:
            side = "left" if st["balance_db"] > 0 else "right"
            add("warning", f"Unbalanced stereo: {side} channel is {abs(st['balance_db'])} dB louder.")

    sp = rep.get("spectrum", {})
    if sp.get("below_20hz_percent", 0) > 5:
        add("warning", f"{sp['below_20hz_percent']}% of the energy is below 20 Hz (rumble/DC); try highpass at ~30 Hz.")
    hum = mains_hum(sp.get("spectral_peaks", []), si.get("noise_floor_dbfs"), sp.get("character"))
    if hum:
        base, found = hum
        notches = ", ".join(f'{{"op":"eq","hz":{h:g},"db":-30,"q":30}}' for h in found)
        add(
            "warning",
            f"Mains hum at {base} Hz (peaks at {', '.join(f'{h:g}' for h in found)} Hz). Fix: notch each with {notches}; "
            "for voice, highpass at ~80-100 Hz also removes the fundamental.",
        )
    upper = sp.get("content_upper_limit_hz")
    if (
        upper
        and rep["sample_rate"] >= 44100
        and 1000 < upper < 15500
        and sp.get("brightness") not in ("very dark", "dark/warm")
    ):
        add("info", f"Almost no content above {upper / 1000:.1f} kHz; could be a low-bitrate or upsampled source.")

    edges = rep.get("edges", {})
    if rep["duration_s"] > 0.05:
        if edges.get("start_amplitude", 0) > 0.05:
            add("info", "Starts abruptly at a non-zero sample; may click on playback (fade_in 5 ms fixes it).")
        if edges.get("end_amplitude", 0) > 0.05:
            add("info", "Ends abruptly at a non-zero sample; may click (fade_out fixes it).")
    return issues


# ---------------------------------------------------------------- entry points


def analyze_with_profile(audio: Audio, pitch: bool = True, rhythm: bool = True):
    mono = audio.mono
    profile = spectral_profile(mono, audio.sr)
    rep = {
        "file": str(audio.path) if audio.path else None,
        "format": f"{audio.format} {audio.subtype}".strip(),
        "sample_rate": audio.sr,
        "channels": audio.channels,
        "duration_s": r(audio.duration, 3),
    }
    if audio.offset:
        rep["range_s"] = [r(audio.offset, 3), r(audio.offset + audio.duration, 3)]
    rep["levels"] = level_stats(audio)
    rep["loudness_shape"] = loudness_shape(mono, audio.sr)
    rep["silence"] = silence_stats(mono, audio.sr, audio.offset)
    if audio.channels == 2:
        rep["stereo"] = stereo_stats(audio.samples)
    rep["spectrum"] = spectrum_stats(profile, audio.duration)
    if pitch:
        rep["pitch"] = pitch_mod.summary(*pitch_mod.track(mono, audio.sr))
        if likely_polyphonic(rep["spectrum"].get("spectral_peaks", [])):
            rep["pitch"]["polyphony_likely"] = True
            rep["pitch"]["comment"] = (
                "Strong spectral peaks don't fit one harmonic series, so this is probably a chord or mix. "
                "The pitch values describe the combined periodicity, not a melody; see spectrum.spectral_peaks for the notes."
            )
    if rhythm:
        rep["rhythm"] = rhythm_stats(mono, audio.sr, audio.offset)
    rep["edges"] = edge_stats(audio)
    rep["issues"] = find_issues(rep)
    return rep, profile


def analyze(audio: Audio, pitch: bool = True, rhythm: bool = True) -> dict:
    return analyze_with_profile(audio, pitch, rhythm)[0]


def quick_levels(samples: np.ndarray, sr: int) -> dict:
    peak = max((float(np.max(np.abs(samples[a:b]))) for a, b, _ in _chunks(len(samples))), default=0.0)
    return {
        "duration_s": r(len(samples) / sr, 3),
        "sample_rate": sr,
        "channels": samples.shape[1],
        "peak_dbfs": r(amp_db(peak), 2),
        "lufs_integrated": r(loudness.integrated_loudness(samples, sr), 2),
        "lufs_momentary_max": r(loudness.momentary_max(samples, sr), 2),
    }


def timeline(path, segments: int = 24, start: float | None = None, end: float | None = None) -> dict:
    """Streams the file, so it works on recordings of any length."""
    p = resolve_path(path)
    src = open_source(p)  # FFmpeg-decoded formats stream from their cached WAV
    info = src.info
    sr = info.samplerate
    s0 = int(round((start or 0.0) * sr))
    s1 = info.frames if end is None else min(info.frames, int(round(end * sr)))
    if s1 <= s0:
        raise AudioError(f"The range {start}..{end}s is empty (file is {info.duration:.3f}s).")
    n = s1 - s0
    segments = int(max(1, min(segments, 200, n // 256)))
    seg_len = n / segments
    block = int(max(256, min(4096, seg_len)))
    window = np.hanning(block)
    freqs = np.fft.rfftfreq(block, 1.0 / sr)
    sumsq = np.zeros(segments)
    count = np.zeros(segments)
    peak = np.zeros(segments)
    cnum = np.zeros(segments)
    cden = np.zeros(segments)
    pos = s0
    bounds = s0 + np.round(np.arange(1, segments) * seg_len).astype(np.int64)  # first sample of segments 1..n-1
    for blk in sf.blocks(str(src.read_path), blocksize=block, start=s0, stop=s1, dtype="float32", always_2d=True):
        # Split the block at segment boundaries so levels never leak into a neighbouring segment.
        i = 0
        while i < len(blk):
            seg = int(np.searchsorted(bounds, pos + i, side="right"))
            stop = len(blk) if seg >= len(bounds) else min(len(blk), int(bounds[seg] - pos))
            part = blk[i:stop]
            sumsq[seg] += float(np.sum(np.square(part, dtype=np.float64))) / part.shape[1]
            count[seg] += len(part)
            peak[seg] = max(peak[seg], float(np.max(np.abs(part))))
            i = stop
        seg = int(np.searchsorted(bounds, pos + len(blk) // 2, side="right"))  # brightness: block's majority segment
        pos += len(blk)
        m = blk.mean(axis=1)
        if len(m) == block:
            mag = np.abs(np.fft.rfft(m * window))
            if mag.sum() > 0:
                e = float(np.mean(m.astype(np.float64) ** 2))
                cnum[seg] += e * float(np.sum(freqs * mag) / mag.sum())
                cden[seg] += e
    rms_db = pow_db(sumsq / np.maximum(count, 1))
    rows = []
    for i in range(segments):
        t0 = (s0 + i * seg_len) / sr
        rows.append(
            {
                "start_s": r(t0, 2),
                "end_s": r(t0 + seg_len / sr, 2),
                "rms_dbfs": r(rms_db[i], 1),
                "peak_dbfs": r(amp_db(peak[i]), 1),
                "centroid_hz": r(cnum[i] / cden[i], 0) if cden[i] > 0 else None,
            }
        )
    top = float(rms_db.max())
    return {
        "file": str(p),
        "duration_s": r(info.duration, 3),
        "segment_s": r(seg_len / sr, 3),
        "loudness_sparkline": sparkline(np.maximum(rms_db, top - 40), top - 40, top),
        "loudest_segment_start_s": rows[int(np.argmax(rms_db))]["start_s"],
        "quietest_segment_start_s": rows[int(np.argmin(rms_db))]["start_s"],
        "segments": rows,
    }
