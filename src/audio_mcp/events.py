"""Event segmentation: split audio into distinct sounds and describe each one."""

from __future__ import annotations

import numpy as np

from . import pitch as pitch_mod
from .analysis import _brightness_word, onset_envelope, pick_onsets
from .audio_io import Audio
from .dsp import amp_db, describe_hz, frame_rms, pow_db, r, runs

HOP_S = 0.01


def _describe(x: np.ndarray, sr: int, f0_times, f0, a: int, b: int, offset: float, lv: np.ndarray, hop: int) -> dict:
    seg = x[a:b]
    n_fft = 1 << int(np.ceil(np.log2(max(len(seg), 64))))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mag = np.sqrt(spec)
    centroid = float(np.sum(freqs * mag) / max(np.sum(mag), 1e-12))
    band = (freqs >= 50) & (freqs <= min(16000, sr / 2))
    pb = spec[band] + 1e-20
    flatness = float(np.exp(np.mean(np.log(pb))) / np.mean(pb))
    fl, fr = a // hop, max(a // hop + 1, b // hop)
    env = lv[fl:fr]
    attack_i = int(np.argmax(env >= env.max() - 3.0))  # time to come within 3 dB of the peak
    start, end = a / sr + offset, b / sr + offset
    ev = {
        "start_s": r(start, 3),
        "end_s": r(end, 3),
        "dur_s": r(end - start, 3),
        "peak_dbfs": r(amp_db(np.max(np.abs(seg))), 1),
        "rms_dbfs": r(pow_db(np.mean(seg.astype(np.float64) ** 2)), 1),
        "attack_ms": int(round(attack_i * HOP_S * 1000)),
        "character": "noisy" if flatness > 0.35 else "mixed" if flatness > 0.08 else "tonal",
        "brightness": _brightness_word(centroid),
        "centroid_hz": r(centroid, 0),
    }
    sel = (f0_times >= start) & (f0_times < end)
    voiced = f0[sel][np.isfinite(f0[sel])]
    if sel.any() and len(voiced) >= max(3, 0.3 * sel.sum()):
        ev["pitch"] = describe_hz(float(np.median(voiced)))
        k = max(1, min(5, len(voiced) // 8))  # edges of the voiced region, a few frames each
        f_start, f_end = float(np.median(voiced[:k])), float(np.median(voiced[-k:]))
        glide = 12 * np.log2(f_end / f_start)
        if abs(glide) >= 1:
            ev["glide_semitones"] = r(glide, 1)
            ev["pitch_start"], ev["pitch_end"] = describe_hz(f_start), describe_hz(f_end)
    return ev


def _line(ev: dict) -> str:
    parts = [f"{ev['start_s']:.2f}-{ev['end_s']:.2f}s", f"{ev['rms_dbfs']:.0f}dB", ev["character"], ev["brightness"]]
    if "glide_semitones" in ev:
        parts.append(f"glide {ev['pitch_start']['note']}->{ev['pitch_end']['note']} ({ev['glide_semitones']:+g}st)")
    elif "pitch" in ev:
        parts.append(ev["pitch"]["note"])
    if ev["attack_ms"] <= 10:
        parts.append("sharp attack")
    elif ev["attack_ms"] >= 100:
        parts.append(f"slow attack {ev['attack_ms']}ms")
    return " | ".join(parts)


def detect_events(
    audio: Audio,
    threshold_db: float | None = None,
    split_on_onsets: bool = True,
    min_gap_s: float = 0.05,
    min_dur_s: float = 0.03,
    max_events: int = 200,
) -> dict:
    """Find sound events with hysteresis on the level envelope, optionally split at onsets."""
    x, sr = audio.mono, audio.sr
    hop = max(1, int(HOP_S * sr))
    lv = amp_db(frame_rms(x, 2 * hop, hop))
    floor = float(np.percentile(lv, 10))
    top = float(np.percentile(lv, 99.5))
    if threshold_db is not None:
        on = threshold_db
    else:
        # 12 dB above the background, but never more than 6 dB below the loudest part: continuous material with
        # no quiet gaps (music, a held drone) becomes one region that onsets then split into notes.
        on = min(max(floor + 12.0, top - 40.0), top - 6.0)
    off = on - 6.0
    if top < on or top <= -90.0:
        return {
            "event_count": 0,
            "threshold_dbfs": r(on, 1),
            "comment": "Silent." if top <= -90.0 else "Nothing rises above the detection threshold.",
            "as_text": [],
            "events": [],
        }

    regions = [(a, b) for a, b in runs(lv > off) if lv[a:b].max() > on]
    merged: list[list[int]] = []
    gap = int(round(min_gap_s / HOP_S))
    for a, b in regions:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    if split_on_onsets:
        env, times, fps = onset_envelope(x, sr)
        onset_frames = np.round(times[pick_onsets(env, fps)] / HOP_S).astype(int)
        min_len = max(1, int(round(min_dur_s / HOP_S)))
        split = []
        for a, b in merged:
            cuts = [a] + [int(o) for o in onset_frames if a + min_len <= o <= b - min_len] + [b]
            cuts = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] >= min_len or c == b]
            split.extend(zip(cuts[:-1], cuts[1:]))
        merged = [list(s) for s in split if s[1] > s[0]]

    merged = [(a, b) for a, b in merged if (b - a) * HOP_S >= min_dur_s]
    f0_times, f0, _ = pitch_mod.track(x, sr)
    f0_times = f0_times + audio.offset
    events = []
    for i, (a, b) in enumerate(merged[:max_events]):
        end = min(len(x), b * hop + hop)  # frames are 2 hops long, so extend the last one
        if i + 1 < len(merged):
            end = min(end, merged[i + 1][0] * hop)
        events.append(_describe(x, sr, f0_times, f0, a * hop, end, audio.offset, lv, hop))

    out = {
        "event_count": len(merged),
        "threshold_dbfs": r(on, 1),
        "background_level_dbfs": r(floor, 1),
        "sounding_fraction": r(sum(e["dur_s"] for e in events) / audio.duration, 2),
    }
    if len(events) >= 3:
        starts = np.array([e["start_s"] for e in events])
        ioi = np.diff(starts)
        cv = float(np.std(ioi) / np.mean(ioi)) if np.mean(ioi) > 0 else 0.0
        out["spacing"] = {
            "mean_s": r(np.mean(ioi), 3),
            "pattern": "very regular" if cv < 0.05 else "regular" if cv < 0.2 else "irregular",
        }
        notes = [e["pitch"]["note"] for e in events if "pitch" in e]
        if len(notes) >= 3:
            out["pitched_sequence"] = " ".join(notes[:64])
    out["as_text"] = [_line(e) for e in events[:60]]
    out["events"] = events
    if len(merged) > max_events:
        out["truncated"] = (
            f"described the first {max_events} of {len(merged)} events; pass start/end for later sections"
        )
    return out
