"""Monophonic pitch tracking (YIN) and note segmentation."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .audio_io import AudioError
from .dsp import (
    amp_db,
    describe_hz,
    frame_rms,
    hz_to_midi,
    midi_to_hz,
    midi_to_name,
    parabolic,
    r,
    resample,
)

PITCH_SR = 16000
HOP = 160  # 10 ms at PITCH_SR
HOP_S = HOP / PITCH_SR
CHUNK = 2048


def track(mono, sr: int, fmin: float = 50.0, fmax: float = 2000.0, threshold: float = 0.15):
    """YIN f0 track. Returns (times_s, f0_hz with NaN where unvoiced, confidence 0..1)."""
    if not 20 <= fmin < fmax:
        raise AudioError(f"Need 20 <= fmin < fmax (got fmin={fmin}, fmax={fmax}).")
    x = resample(np.asarray(mono, dtype=np.float32), sr, PITCH_SR)
    max_tau = int(PITCH_SR / fmin)
    min_tau = max(2, int(PITCH_SR / min(fmax, PITCH_SR / 4)))
    W = max_tau  # integration window
    L = W + max_tau
    if len(x) < L:
        x = np.pad(x, (0, L - len(x)))
    frames = sliding_window_view(x, L)[::HOP]
    n = len(frames)
    level = amp_db(frame_rms(x, W, HOP))[:n]
    gate = max(float(level.max()) - 40.0, -70.0)
    nfft = 1 << (L - 1).bit_length()
    taus = np.arange(max_tau + 1)
    f0 = np.full(n, np.nan)
    conf = np.zeros(n)

    for c0 in range(0, n, CHUNK):
        fr = frames[c0 : c0 + CHUNK].astype(np.float64)
        spec = np.fft.rfft(fr, nfft) * np.conj(np.fft.rfft(fr[:, :W], nfft))
        cross = np.fft.irfft(spec, nfft)[:, : max_tau + 1]
        cs = np.zeros((len(fr), L + 1))
        cs[:, 1:] = np.cumsum(fr * fr, axis=1)
        e2 = cs[:, taus + W] - cs[:, taus]
        d = np.maximum(cs[:, W][:, None] + e2 - 2.0 * cross, 0.0)
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(np.cumsum(d[:, 1:], axis=1), 1e-12)

        for i in range(len(fr)):
            if level[c0 + i] < gate:
                continue
            row = cmnd[i]
            below = np.flatnonzero(row[min_tau:] < threshold)
            if below.size == 0:
                conf[c0 + i] = max(0.0, 1.0 - float(row[min_tau:].min()))
                continue
            t = min_tau + int(below[0])
            while t + 1 <= max_tau and row[t + 1] < row[t]:
                t += 1
            # Refine the period on the raw difference function: interpolating the normalised curve is biased
            # sharp when the period spans few samples (+10 cents at 1.7 kHz); the raw curve stays within ~0.6.
            tau, _ = parabolic(d[i], t)
            val = row[t]
            f0[c0 + i] = PITCH_SR / tau
            conf[c0 + i] = float(np.clip(1.0 - val, 0.0, 1.0))

    times = (np.arange(n) * HOP + L / 2) / PITCH_SR
    return times, f0, conf


def summary(times, f0, conf) -> dict:
    voiced = np.isfinite(f0)
    out = {"voiced_fraction": r(voiced.mean() if len(f0) else 0.0, 2)}
    if voiced.sum() < 5:
        out["comment"] = "No stable pitch: the sound is unpitched, noisy, polyphonic or too short."
        return out
    fv = f0[voiced]
    med = float(np.median(fv))
    lo, hi = np.percentile(fv, [5, 95])
    out.update(
        {
            "median": describe_hz(med),
            "range": [describe_hz(lo), describe_hz(hi)],
            "range_semitones": r(12 * np.log2(hi / lo), 1),
            "spread_cents": r(np.std(1200 * np.log2(fv / med)), 0),
            "mean_confidence": r(conf[voiced].mean(), 2),
        }
    )
    k = max(1, len(fv) // 4)
    glide = 12 * np.log2(np.median(fv[-k:]) / np.median(fv[:k]))
    if abs(glide) >= 1:
        out["glide_semitones"] = r(glide, 1)
    if out["voiced_fraction"] < 0.3:
        out["comment"] = (
            "Pitch found in only part of the audio; treat as approximate (polyphonic or noisy material confuses a monophonic tracker)."
        )
    return out


def segment_notes(times, f0, conf, min_dur: float = 0.06, max_gap_frames: int = 2, breaks=()) -> list[dict]:
    """Group the f0 track into discrete notes. Short unvoiced gaps inside a note are tolerated.
    `breaks` are frame indices of detected attacks: a new note starts there even if the pitch is unchanged."""
    breaks = set(int(b) for b in breaks)
    min_frames = max(1, int(round(min_dur / HOP_S)))
    midi = hz_to_midi(np.where(np.isfinite(f0), f0, 1.0))
    voiced = np.isfinite(f0)
    notes: list[dict] = []
    cur = None

    def finish(c, end_idx):
        dur = (end_idx - c["i0"]) * HOP_S
        if dur < min_dur:
            return
        mv = float(np.median(c["vals"]))
        nm = int(round(mv))
        notes.append(
            {
                "start_s": r(times[c["i0"]] - HOP_S / 2, 3),
                "dur_s": r(dur, 3),
                "note": midi_to_name(nm),
                "midi": nm,
                "cents": int(round((mv - nm) * 100)),
                "hz": r(midi_to_hz(mv), 1),
                "confidence": r(np.mean(c["conf"]), 2),
            }
        )

    for i in range(len(f0)):
        if not voiced[i]:
            if cur is not None:
                cur["gap"] += 1
                if cur["gap"] > max_gap_frames:
                    finish(cur, i - cur["gap"] + 1)
                    cur = None
            continue
        m = midi[i]
        reattack = i in breaks and cur is not None and i - cur["i0"] >= min_frames
        if cur is not None and not reattack and abs(m - cur["sum"] / len(cur["vals"])) < 0.6:
            cur["vals"].append(m)
            cur["conf"].append(conf[i])
            cur["sum"] += m
            cur["gap"] = 0
            continue
        if cur is not None:
            finish(cur, i - cur["gap"])
        cur = {"i0": i, "vals": [m], "conf": [conf[i]], "sum": m, "gap": 0}
    if cur is not None:
        finish(cur, len(f0) - cur["gap"])
    return notes


def melody(mono, sr: int, offset: float = 0.0, fmin: float = 50.0, fmax: float = 2000.0, max_notes: int = 300) -> dict:
    from .analysis import onset_envelope, pick_onsets  # local import: analysis imports this module

    times, f0, conf = track(mono, sr, fmin, fmax)
    env, onset_times, fps = onset_envelope(mono, sr)
    attacks = onset_times[pick_onsets(env, fps)]
    breaks = np.searchsorted(times, attacks)  # first pitch frame at or after each attack
    times = times + offset
    notes = segment_notes(times, f0, conf, breaks=breaks)
    out = {"pitch": summary(times, f0, conf), "note_count": len(notes)}
    if notes:
        out["as_text"] = " ".join(f"{n['note']}({n['dur_s']:g})" for n in notes[:max_notes])
    out["notes"] = notes[:max_notes]
    if len(notes) > max_notes:
        out["truncated"] = f"showing first {max_notes} of {len(notes)} notes; pass start/end for later sections"
    return out
