"""Delivery checks: measure a file against a named delivery specification and plan an ordered fix.

Every number below comes from the source cited next to it (accessed on SOURCES_CHECKED). Where a source is a
recommendation rather than a hard requirement, or where we could only find third-party descriptions, the spec's
`notes` say so and the affected checks report "warn" rather than "fail".

The fix is planned from the measurements, then (for files up to SIMULATE_MAX_S) applied in memory and re-checked,
so the caller learns whether it actually works on this file instead of trusting a recipe.
"""

from __future__ import annotations

import math

import numpy as np

from . import analysis, process
from .audio_io import Audio, AudioError
from .dsp import r

SOURCES_CHECKED = "2026-10-13"
SIMULATE_MAX_S = 600.0  # longer files get an unverified fix (applying it in memory would cost too much)
TP_MARGIN = 0.3  # dB below a true-peak limit for the limiter ceiling: room for file quantization and meter spread
LEAD_RELATIVE_DB = -50.0  # SFX leading silence: samples more than 50 dB below the clip's peak count as silence
NOISE_WINDOW_S = 0.5  # ACX-style noise floor: RMS of the quietest 0.5 s
MAX_SPEECH_SAFE_DENOISE_DB = 20.0  # beyond this the spectral gate audibly damages quiet speech sounds


def _src(url, what):
    return {"url": url, "what": what, "accessed": SOURCES_CHECKED}


# ---------------------------------------------------------------- specifications
#
# Loudness specs share one checker; `loudness_fail` says whether missing the window is a "fail" (stated requirement)
# or a "warn" (the platform normalizes anyway, or the number is not officially published).

SPECS: dict[str, dict] = {
    "apple_podcasts": {
        "title": "Apple Podcasts",
        "description": "Podcast episodes: -16 LKFS +/-1 dB integrated, true peak <= -1 dBFS (BS.1770-5).",
        "official": True,
        "family": "loudness",
        # https://podcasters.apple.com/support/audio-requirements: "the overall loudness remains around -16 dB LKFS,
        # with a +/- 1 dB tolerance" and "the true-peak value doesn't exceed -1 dB FS", per ITU-R BS.1770-5.
        # Same page: WAV/FLAC minimum 44.1 kHz, "Single-channel audio will not be accepted for WAV or FLAC files."
        "target": -16.0,
        "tolerance": 1.0,
        "loudness_fail": True,
        "true_peak_max": -1.0,
        "min_sample_rate": 44100,
        "no_mono_lossless": True,
        "speech": True,
        "sources": [_src("https://podcasters.apple.com/support/audio-requirements", "loudness, true peak, formats")],
        "notes": [
            "Apple measures with BS.1770-5; this tool implements BS.1770-4 gating, which gives the same result for "
            "mono and stereo files.",
            "A mono file measures about 3 LU quieter than the same audio as dual-mono stereo, so convert first "
            "(the fix does) and then normalize.",
        ],
    },
    "spotify": {
        "title": "Spotify (music mastering guidance)",
        "description": "Spotify for Artists mastering guidance: -14 LUFS integrated, true peak below -1 dBTP "
        "(below -2 dBTP if louder than -14 LUFS). Loudness is normalized on playback.",
        "official": True,
        "family": "loudness",
        # https://support.spotify.com/artists/article/loudness-normalization/: "Target the loudness level of your
        # master at -14dB integrated LUFS", "Keep it below -1dB TP (True Peak) max", and "If your master is louder
        # than -14dB integrated LUFS, keep True Peak below -2dB to avoid extra distortion."
        "target": -14.0,
        "tolerance": 1.0,
        "loudness_fail": False,
        "true_peak_max": -1.0,
        "true_peak_max_if_louder": -2.0,
        "speech": False,
        "sources": [_src("https://support.spotify.com/artists/article/loudness-normalization/", "mastering guidance")],
        "notes": [
            "This is Spotify's guidance for music. It does not cover podcasts, and we found no separate official "
            "podcast loudness spec from Spotify.",
            "Spotify normalizes playback (louder masters are turned down), so loudness off target is a warning; "
            "the true-peak guidance is treated as the requirement.",
        ],
    },
    "youtube": {
        "title": "YouTube (loudness reference, not a published spec)",
        "description": "YouTube normalization reference of about -14 LUFS (reported by third parties, not published "
        "by YouTube); true peak <= -1 dBTP as good practice; 48 kHz recommended.",
        "official": False,
        "status": "platform loudness reference; not a published delivery spec",
        "family": "loudness",
        # YouTube's own upload page recommends 48 kHz audio but says nothing about loudness:
        # https://support.google.com/youtube/answer/1722171. The -14 LUFS playback reference is from third-party
        # measurements (e.g. https://meterplugs.com/blog/2019/09/18/youtube-changes-loudness-reference-to-14-lufs.html).
        "target": -14.0,
        "tolerance": 1.0,
        "loudness_fail": False,
        "true_peak_max": -1.0,
        "true_peak_fail": False,
        "recommended_sample_rate": 48000,
        "speech": False,
        "sources": [
            _src("https://support.google.com/youtube/answer/1722171", "recommended upload settings (48 kHz)"),
            _src(
                "https://meterplugs.com/blog/2019/09/18/youtube-changes-loudness-reference-to-14-lufs.html",
                "third-party report of the -14 LUFS playback reference",
            ),
        ],
        "notes": [
            "YouTube normalizes playback loudness and does not publish a delivery spec; this check never fails a "
            "file, it only warns.",
            "Third-party reports say YouTube turns loud uploads down but does not turn quiet ones up.",
        ],
    },
    "ebu_r128": {
        "title": "EBU R 128 (broadcast)",
        "description": "European broadcast: -23.0 LUFS integrated (+/-1.0 LU only where not practically achievable, "
        "e.g. live), true peak <= -1 dBTP.",
        "official": True,
        "family": "loudness",
        # https://tech.ebu.ch/docs/r/r128.pdf (R 128 v5.0, November 2023): "normalised to a Target Level of -23.0
        # LUFS. Where attaining the Target Level is not achievable practically (for example, live programmes), a
        # tolerance of +/-1.0 LU is permitted"; "a tolerance of +/-0.2 LU is allowed [in QC] to take account of
        # measurement errors"; "True Peak Level of a programme shall not exceed -1 dBTP".
        "target": -23.0,
        "tolerance": 0.2,
        "warn_tolerance": 1.0,
        "loudness_fail": True,
        "true_peak_max": -1.0,
        "speech": False,
        "sources": [
            _src("https://tech.ebu.ch/docs/r/r128.pdf", "R 128 v5.0 (2023): target, tolerances, true peak"),
            _src("https://tech.ebu.ch/publications/r128", "version history"),
        ],
        "notes": [
            "R 128 v3 (2014) allowed +/-0.5 LU; v4 (2020) and v5 (2023) ask for the target itself, with +/-0.2 LU "
            "for measurement error in QC and +/-1.0 LU only for live programmes. We pass within 0.2 LU and warn "
            "within 1.0 LU.",
            "Loudness Range (LRA) is informational in R 128 and is not measured here.",
            "Individual broadcasters publish their own delivery specs on top of R 128; check those too.",
        ],
    },
    "atsc_a85": {
        "title": "ATSC A/85 (US broadcast)",
        "description": "US television: -24 LKFS integrated (+/-2 dB measurement variation acceptable), true peak "
        "below -2 dBTP.",
        "official": True,
        "family": "loudness",
        # https://www.atsc.org/wp-content/uploads/2025/06/A85-2013-with-Corrigendum-No-1.pdf, section 6: "the Target
        # Loudness value should be -24 LKFS. Minor measurement variations of up to approximately +/-2 dB about this
        # value are anticipated ... and are acceptable"; "Content loudness should not be targeted to the high or low
        # side of this range"; "The true-peak level should be kept below -2 dB TP".
        "target": -24.0,
        "tolerance": 2.0,
        "loudness_fail": True,
        "true_peak_max": -2.0,
        "speech": False,
        "sources": [
            _src(
                "https://www.atsc.org/wp-content/uploads/2025/06/A85-2013-with-Corrigendum-No-1.pdf",
                "A/85:2013 section 6: target loudness and true peak",
            ),
            _src(
                "https://www.sportsvideo.org/2026/07/16/atsc-releases-first-major-revision-to-a-85-audio-loudness-"
                "recommended-practice-since-2013/",
                "news of the A/85:2026-07 revision (numbers not reviewed)",
            ),
        ],
        "notes": [
            "Numbers are from A/85:2013 (the version the FCC's CALM Act rules reference). ATSC approved a revised "
            "A/85 in July 2026 whose text we could not review; check it if your contract cites it.",
            "The +/-2 dB is described as measurement variation, not a target window: aim for -24 (the fix does).",
        ],
    },
    "acx_audiobook": {
        "title": "ACX / Audible audiobook",
        "description": "Audiobook chapters: RMS -23 to -18 dB, peaks below -3 dB, noise floor below -60 dB RMS, "
        "1-5 s room tone at head and tail (max 5 s), 44.1 kHz 192 kbps+ CBR MP3, max 120 min.",
        "official": True,
        "family": "acx",
        # https://help.acx.com/s/article/acx-audio-submission-requirements (last updated Apr 15, 2026): "Each file
        # needs to fall between ... -23dB and -18dB RMS"; "peak values no higher than -3dB"; "a noise floor no
        # higher than -60dB RMS"; "between 1 and 5 seconds of room tone at the beginning and end of each file";
        # "Room tone spacing must not exceed 5 seconds"; "192 kbps or higher CBR, 44.1kHz MP3"; "no longer than 120
        # minutes"; "all mono or all stereo".
        "rms_range": (-23.0, -18.0),
        "rms_target": -20.0,
        "peak_max": -3.0,
        "noise_max": -60.0,
        "room_tone_min": 1.0,
        "room_tone_max": 5.0,
        "sample_rate": 44100,
        "max_minutes": 120.0,
        "speech": True,
        "sources": [
            _src("https://help.acx.com/s/article/acx-audio-submission-requirements", "all audio requirements"),
        ],
        "notes": [
            "ACX does not publish how it measures. We follow the ACX Check plugin's widely used reading: RMS is "
            "unweighted over the whole file; noise floor is the RMS of the quietest 0.5 s; peak is the sample peak. "
            "ACX's own analyzer may differ by a dB or so, so leave margin (the fix aims for -20 dB RMS, down to -22 when that keeps the noise floor passing).",
            "ACX also requires a human narrator, one chapter per file, credits and consistent sound across files; "
            "those can't be checked from one file's audio.",
        ],
    },
    "game_sfx": {
        "title": "Game sound effects (recommended practice, not an official spec)",
        "description": "Recommended practice, not an official spec: one-shot SFX with true peak <= -1 dBTP, no "
        "clipping, <= 10 ms leading silence, no clicks at start/end, no DC offset, momentary-max loudness "
        "-18 +/-4 LUFS.",
        "official": False,
        "status": "recommended practice, not an official spec",
        "family": "sfx",
        "true_peak_max": -1.0,
        "lead_max_ms": 10.0,
        "lead_threshold_db": LEAD_RELATIVE_DB,  # relative to the peak, so it does not depend on the level
        "edge_max": 0.01,  # -40 dBFS: a jump from silence to this is an audible tick on quiet playback
        "dc_max": 0.005,  # same threshold audio_analyze reports
        "momentary_target": -18.0,
        "momentary_tolerance": 4.0,
        "speech": False,
        "sources": [],
        "notes": [
            "House recommendation of this tool, not a published standard. Integrated loudness is undefined or "
            "unstable for short clips, so loudness uses the maximum momentary loudness (400 ms window, EBU R 128 "
            "'M'). -18 LUFS +/-4 keeps a library consistent with headroom; set your own target per game.",
            "Loops are not one-shots: the click checks assume silence at the edges. For loops use audio_analyze's "
            "loop_seam instead.",
        ],
    },
}


def spec_list() -> dict:
    return {"specs": {name: s["description"] for name, s in SPECS.items()}}


def get_spec(name: str) -> dict:
    key = str(name).strip().lower()
    if key not in SPECS:
        listing = " ".join(f"{n}: {s['description']}" for n, s in SPECS.items())
        raise AudioError(f"Unknown delivery spec '{name}'. Available: {listing}")
    return SPECS[key]


# ---------------------------------------------------------------- measurement


def measure(audio: Audio) -> dict:
    x, sr = audio.samples, audio.sr
    lv = analysis.level_stats(audio)
    win = analysis.windowed_rms_db(x, sr, NOISE_WINDOW_S, 0.05)
    dc = _channel_means(x)
    peak = float(lv["peak_dbfs"]) if lv["peak_dbfs"] is not None else -120.0
    lead = _leading_silence(x, sr, max(peak + LEAD_RELATIVE_DB, -100.0), dc)
    return {
        "sr": sr,
        "channels": audio.channels,
        "duration": audio.duration,
        "format": audio.format,
        "subtype": audio.subtype,
        "lufs": lv["lufs_integrated"],
        "lufs_m": lv["lufs_momentary_max"],
        "true_peak": lv["true_peak_dbtp"],
        "peak": lv["peak_dbfs"],
        "rms": lv["rms_dbfs"],
        "dc": lv["dc_offset"],
        "clipped": lv["clipped_samples"],
        "flat_top": lv["flat_top_samples"],
        "noise": float(win.min()),
        "loud": float(np.percentile(win, 95)),
        "noise_measurable": audio.duration >= NOISE_WINDOW_S,
        "lead_s": lead,
        "start_amp": float(np.max(np.abs(x[0]))),
        "end_amp": float(np.max(np.abs(x[-1]))),
        "silent": lv["lufs_momentary_max"] is None or float(np.max(np.abs(x))) < 1e-6,
    }


def _channel_means(x) -> np.ndarray:
    total = np.zeros(x.shape[1])
    for a in range(0, len(x), analysis.CHUNK):
        total += x[a : a + analysis.CHUNK].sum(axis=0, dtype=np.float64)
    return total / max(1, len(x))


def _leading_silence(x, sr, threshold_dbfs, dc) -> float | None:
    """Seconds before the first sample that departs from the DC offset by more than threshold_dbfs (None if none
    does). Ignoring DC means an offset doesn't hide leading silence (remove_dc would reveal it)."""
    thr = 10 ** (threshold_dbfs / 20)
    for a in range(0, len(x), analysis.CHUNK):
        hit = np.flatnonzero((np.abs(x[a : a + analysis.CHUNK] - dc) > thr).any(axis=1))
        if hit.size:
            return (a + int(hit[0])) / sr
    return None


def room_tone(x, sr) -> dict:
    """Head and tail room tone: time before the first and after the last stretch of speech. Speech frames are 20 ms
    frames well above the quietest 0.5 s (>= 12 dB) and within 30 dB of the loud frames; at least 60 ms in a row,
    so a click or breath doesn't count. Approximate to about +/-50 ms."""
    frames = analysis.windowed_rms_db(x, sr, 0.02, 0.02)
    noise = float(analysis.windowed_rms_db(x, sr, NOISE_WINDOW_S, 0.05).min())
    loud = float(np.percentile(frames, 95)) if len(frames) else noise
    if len(frames) < 3 or loud - noise < 15:
        return {"reliable": False}
    active = frames > max(noise + 12, loud - 30)
    sustained = active[:-2] & active[1:-1] & active[2:]
    idx = np.flatnonzero(sustained)
    if idx.size == 0:
        return {"reliable": False}
    first = idx[0] * 0.02
    last = (idx[-1] + 3) * 0.02
    dur = len(x) / sr
    return {"reliable": True, "head_s": first, "tail_s": max(0.0, dur - last), "speech_end_s": last}


# ---------------------------------------------------------------- checks


def _check(name, required, measured, status, note):
    return {"check": name, "required": required, "measured": measured, "status": status, "note": note}


def _loudness_checks(m, spec, rt=None) -> list[dict]:
    out = []
    t, tol = spec["target"], spec["tolerance"]
    warn_tol = spec.get("warn_tolerance")
    hard = spec["loudness_fail"]
    lufs = m["lufs"]
    req = f"{t:g} LUFS +/-{tol:g}" + (f" (+/-{warn_tol:g} for live)" if warn_tol else "")
    if m["silent"] or lufs is None and m["duration"] >= 0.4:
        out.append(
            _check(
                "integrated_loudness",
                req,
                None,
                "fail",
                "Silent (or entirely below the -70 LUFS gate); loudness is undefined.",
            )
        )
    elif lufs is None:
        out.append(
            _check(
                "integrated_loudness",
                req,
                None,
                "fail",
                "Shorter than 0.4 s, so integrated loudness can't be measured; this spec is for programmes.",
            )
        )
    else:
        dev = lufs - t
        if abs(dev) <= tol + 1e-9:
            status, note = "pass", "On target."
        elif warn_tol and abs(dev) <= warn_tol:
            status, note = "warn", f"{dev:+.1f} LU off target; this much is permitted only for live programmes."
        else:
            status = "fail" if hard else "warn"
            note = f"{abs(dev):.1f} LU {'louder' if dev > 0 else 'quieter'} than the {t:g} LUFS target."
            if not hard:
                note += (
                    " Playback normalization will turn it down."
                    if dev > 0
                    else " Normalization may not turn it up fully, so it can sound quieter than other content."
                )
        out.append(_check("integrated_loudness", req, f"{lufs} LUFS", status, note))

    tp_max = spec["true_peak_max"]
    if lufs is not None and spec.get("true_peak_max_if_louder") is not None and lufs > spec["target"]:
        tp_max = spec["true_peak_max_if_louder"]
    tp = m["true_peak"]
    if tp is None:
        out.append(_check("true_peak", f"<= {tp_max:g} dBTP", None, "pass", "Silent."))
    else:
        ok = tp <= tp_max
        status = "pass" if ok else ("fail" if spec.get("true_peak_fail", True) else "warn")
        note = "Within the ceiling." if ok else f"{tp - tp_max:.1f} dB over the ceiling; lossy encoding may clip."
        if tp_max != spec["true_peak_max"]:
            note += " (Stricter ceiling applies because the file is louder than the target.)"
        out.append(_check("true_peak", f"<= {tp_max:g} dBTP", f"{tp} dBTP", status, note))

    out.append(_clip_check(m, "warn"))
    if spec.get("min_sample_rate") and m["sr"] < spec["min_sample_rate"]:
        out.append(
            _check("sample_rate", f">= {spec['min_sample_rate']} Hz", m["sr"], "fail", "Below the minimum rate.")
        )
    if spec.get("recommended_sample_rate") and m["sr"] != spec["recommended_sample_rate"]:
        out.append(
            _check(
                "sample_rate",
                f"{spec['recommended_sample_rate']} Hz recommended",
                m["sr"],
                "warn",
                "Not the recommended rate; the platform will resample.",
            )
        )
    if spec.get("no_mono_lossless") and m["channels"] == 1 and m["format"] in ("WAV", "FLAC"):
        out.append(
            _check(
                "channels",
                "not mono for WAV/FLAC",
                "mono",
                "warn",
                "Apple's audio requirements page says single-channel WAV/FLAC is not accepted (mono MP3 is). "
                "Convert to stereo or deliver MP3/AAC.",
            )
        )
    if spec.get("speech") and m["noise_measurable"] and not m["silent"] and m["lufs"] is not None:
        after = m["noise"] + (spec["target"] - m["lufs"])
        if after > -50 and m["loud"] - m["noise"] > 15:
            out.append(
                _check(
                    "background_noise",
                    "advisory: quiet parts below -50 dBFS after normalization",
                    f"{r(after, 1)} dBFS",
                    "warn",
                    "Raising the loudness raises the noise in the pauses too; the fix reduces it first. If the file "
                    "has no pauses (continuous music), this is the quietest passage, not noise: ignore it.",
                )
            )
    return out


def _clip_check(m, fail_status):
    if m["clipped"]:
        return _check(
            "clipping",
            "none",
            f"{m['clipped']} clipped samples",
            fail_status,
            "Clipped at 0 dBFS. Gain changes and limiting can't undo clipping already in the waveform.",
        )
    if m["flat_top"] > 10:
        return _check(
            "clipping",
            "none",
            f"{m['flat_top']} flat-topped samples",
            fail_status,
            "Flat-topped peaks: clipped earlier and then turned down. This is baked in.",
        )
    return _check("clipping", "none", "none", "pass", "No clipping found.")


def _acx_checks(m, spec, rt) -> list[dict]:
    out = []
    lo, hi = spec["rms_range"]
    rms = m["rms"]
    if m["silent"] or rms is None:
        out.append(_check("rms", f"{lo:g} to {hi:g} dB RMS", None, "fail", "The audio is silent."))
    else:
        ok = lo <= rms <= hi
        out.append(
            _check(
                "rms",
                f"{lo:g} to {hi:g} dB RMS",
                f"{rms} dBFS",
                "pass" if ok else "fail",
                "Unweighted RMS over the whole file, all channels together (ACX Check's reading)."
                + ("" if ok else f" {'Too quiet' if rms < lo else 'Too loud'}."),
            )
        )
    pk, pmax = m["peak"], spec["peak_max"]
    if pk is not None and pk > pmax:
        out.append(_check("peak", f"<= {pmax:g} dBFS", f"{pk} dBFS", "fail", "Sample peak above the ceiling."))
    elif m["true_peak"] is not None and m["true_peak"] > pmax:
        out.append(
            _check(
                "peak",
                f"<= {pmax:g} dBFS",
                f"{pk} dBFS (true peak {m['true_peak']} dBTP)",
                "warn",
                "Sample peaks pass, but inter-sample peaks exceed -3 dB and MP3 encoding can push the sample peak "
                "over. ACX does not say which peak it measures.",
            )
        )
    else:
        out.append(_check("peak", f"<= {pmax:g} dBFS", f"{pk} dBFS", "pass", "Sample and true peak within -3 dB."))
    if not m["noise_measurable"]:
        out.append(
            _check("noise_floor", "<= -60 dB RMS", None, "fail", "Shorter than 0.5 s; the noise floor can't be read.")
        )
    else:
        ok = m["noise"] <= spec["noise_max"]
        out.append(
            _check(
                "noise_floor",
                f"<= {spec['noise_max']:g} dB RMS",
                f"{r(m['noise'], 1)} dBFS",
                "pass" if ok else "fail",
                "RMS of the quietest 0.5 s (ACX Check's method; ACX's own analyzer may read slightly differently). "
                "Needs at least 0.5 s of clean room tone somewhere in the file.",
            )
        )
    if not rt.get("reliable"):
        out.append(
            _check(
                "room_tone",
                f"{spec['room_tone_min']:g}-{spec['room_tone_max']:g} s at head and tail",
                None,
                "warn",
                "Couldn't separate speech from background (too noisy or no speech), so room tone wasn't measured.",
            )
        )
    else:
        for key, label in (("head_s", "head"), ("tail_s", "tail")):
            v = rt[key]
            if v > spec["room_tone_max"]:
                status, note = "fail", f"More than {spec['room_tone_max']:g} s of room tone at the {label}."
            elif v < spec["room_tone_min"]:
                status = "warn"
                note = (
                    f"ACX recommends at least {spec['room_tone_min']:g} s at the {label}. Add real room tone in "
                    "your editor; padding with digital silence (pad) is not room tone."
                )
            else:
                status, note = "pass", "Within the recommended range."
            out.append(
                _check(
                    f"room_tone_{label}",
                    f"{spec['room_tone_min']:g}-{spec['room_tone_max']:g} s",
                    f"{r(v, 2)} s",
                    status,
                    note + " Measured from where speech starts/ends (about +/-50 ms).",
                )
            )
    out.append(
        _check(
            "sample_rate",
            f"{spec['sample_rate']} Hz",
            m["sr"],
            "pass" if m["sr"] == spec["sample_rate"] else "fail",
            "ACX wants 44.1 kHz MP3." if m["sr"] != spec["sample_rate"] else "OK.",
        )
    )
    if m["format"] == "MP3":
        out.append(
            _check(
                "format",
                "MP3, CBR, >= 192 kbps",
                "MP3",
                "info",
                "Bitrate and CBR can't be read here; check them in your encoder settings.",
            )
        )
    else:
        out.append(
            _check(
                "format",
                "MP3, CBR, >= 192 kbps",
                f"{m['format']} {m['subtype']}".strip(),
                "warn",
                "Fine as a master. Export the final upload as a 192 kbps+ CBR MP3 (e.g. ffmpeg -c:a libmp3lame "
                "-b:a 192k); re-check the MP3, since encoding changes peaks slightly.",
            )
        )
    minutes = m["duration"] / 60
    out.append(
        _check(
            "length",
            f"<= {spec['max_minutes']:g} min",
            f"{r(minutes, 1)} min",
            "pass" if minutes <= spec["max_minutes"] else "fail",
            "Split into shorter files (one chapter per file)." if minutes > spec["max_minutes"] else "OK.",
        )
    )
    out.append(_clip_check(m, "warn"))
    return out


def _sfx_checks(m, spec, rt=None) -> list[dict]:
    out = []
    tp = m["true_peak"]
    if tp is not None:
        ok = tp <= spec["true_peak_max"]
        out.append(
            _check(
                "true_peak",
                f"<= {spec['true_peak_max']:g} dBTP",
                f"{tp} dBTP",
                "pass" if ok else "fail",
                "OK." if ok else "Inter-sample peaks can clip in the engine's mixer or after compression.",
            )
        )
    out.append(_clip_check(m, "fail"))
    if m["silent"] or m["lead_s"] is None:
        t, tol = spec["momentary_target"], spec["momentary_tolerance"]
        out.append(_check("loudness", f"momentary max {t:g} +/-{tol:g} LUFS", None, "fail", "The audio is silent."))
        return out
    lead_ms = m["lead_s"] * 1000
    out.append(
        _check(
            "leading_silence",
            f"<= {spec['lead_max_ms']:g} ms",
            f"{r(lead_ms, 1)} ms",
            "pass" if lead_ms <= spec["lead_max_ms"] else "fail",
            f"Time before the sound first rises above {-spec['lead_threshold_db']:g} dB below its peak (ignoring DC "
            "offset); extra "
            "silence delays the sound after its trigger.",
        )
    )
    for key, label in (("start_amp", "start"), ("end_amp", "end")):
        v = m[key]
        ok = v <= spec["edge_max"]
        out.append(
            _check(
                f"click_at_{label}",
                f"|first/last sample| <= {spec['edge_max']:g} (-40 dBFS)",
                r(v, 4),
                "pass" if ok else "fail",
                "OK." if ok else f"The sound {'starts' if label == 'start' else 'ends'} abruptly; it can click.",
            )
        )
    ok = m["dc"] <= spec["dc_max"]
    out.append(
        _check(
            "dc_offset",
            f"<= {spec['dc_max']:g}",
            m["dc"],
            "pass" if ok else "fail",
            "OK." if ok else "DC offset wastes headroom and thumps when the sound starts or stops.",
        )
    )
    t, tol = spec["momentary_target"], spec["momentary_tolerance"]
    v = m["lufs_m"]
    ok = abs(v - t) <= tol
    out.append(
        _check(
            "loudness",
            f"momentary max {t:g} +/-{tol:g} LUFS",
            f"{v} LUFS",
            "pass" if ok else "fail",
            "Loudest 400 ms window (integrated loudness is undefined for short clips)."
            + ("" if ok else f" {'Too loud' if v > t else 'Too quiet'}."),
        )
    )
    return out


CHECKERS = {"loudness": _loudness_checks, "acx": _acx_checks, "sfx": _sfx_checks}


def run_checks(audio: Audio, spec: dict) -> tuple[dict, list[dict]]:
    m = measure(audio)
    rt = room_tone(audio.samples, audio.sr) if spec["family"] == "acx" else {}
    m["room_tone"] = rt
    return m, CHECKERS[spec["family"]](m, spec, rt)


def verdict(checks: list[dict]) -> str:
    statuses = {c["status"] for c in checks}
    if "fail" in statuses:
        return "fail"
    return "pass with warnings" if "warn" in statuses else "pass"


# ---------------------------------------------------------------- fix planning


def _status(checks, name):
    for c in checks:
        if c["check"] == name:
            return c["status"]
    return None


def _bad(checks, *names):
    return any(_status(checks, n) in ("fail", "warn") for n in names)


def _plan_loudness(m, spec, checks, adjust):
    ops, why, cannot = [], [], []
    if m["silent"] or m["lufs"] is None:
        cannot.append("Loudness can't be measured (silent, or shorter than 0.4 s), so it can't be normalized.")
        return ops, why, cannot
    if _status(checks, "clipping") == "warn":
        cannot.append(
            "Clipping is baked into the waveform; normalizing makes it pass the meters but it still sounds "
            "distorted. Re-export from an unclipped source if you can."
        )
    if spec.get("no_mono_lossless") and _status(checks, "channels") == "warn":
        ops.append({"op": "to_stereo"})
        why.append("to_stereo first: Apple rejects mono WAV/FLAC, and loudness must be measured on the final layout.")
    rate = None
    if _status(checks, "sample_rate") == "fail":
        rate = spec["min_sample_rate"]
    elif _status(checks, "sample_rate") == "warn":
        rate = spec["recommended_sample_rate"]
    if rate:
        ops.append({"op": "resample", "rate": rate})
        why.append(f"resample to {rate} Hz before any level change, so peaks are measured at the final rate.")
    if _status(checks, "background_noise") == "warn":
        after = m["noise"] + (spec["target"] - m["lufs"])
        red = float(min(MAX_SPEECH_SAFE_DENOISE_DB, max(6, math.ceil(after + 53 + adjust.get("noise", 0)))))
        ops += [{"op": "highpass", "hz": 80}, {"op": "denoise", "reduction_db": red}]
        why.append(
            "highpass 80 Hz and denoise BEFORE raising loudness, because the gain raises the noise floor too "
            "(assumes speech: drop both for music). Verify the words with audio_transcribe afterwards."
        )
    needs_level = _bad(checks, "integrated_loudness", "true_peak") or bool(ops)
    if needs_level:
        tp_max = spec["true_peak_max"]
        ceiling = round(tp_max - TP_MARGIN + adjust.get("tp", 0), 2)  # the result sits at the target, not louder
        ops.append({"op": "normalize_lufs", "lufs": spec["target"], "ceiling_db": ceiling, "limit": True})
        why.append(
            f"normalize_lufs to {spec['target']:g} LUFS last, with a true-peak limiter at {ceiling:g} dBTP "
            f"({TP_MARGIN:g} dB below the {tp_max:g} dBTP limit, for file and meter tolerances)."
        )
    return ops, why, cannot


def _plan_acx(m, spec, checks, adjust):
    ops, why, cannot = [], [], []
    if m["silent"]:
        return ops, why, ["The file is silent."]
    if _status(checks, "length") == "fail":
        cannot.append("Longer than 120 minutes: split it into chapters (crop) before delivery.")
    if not m["noise_measurable"]:
        cannot.append("Shorter than 0.5 s: too short for an audiobook file.")
        return ops, why, cannot
    if _status(checks, "clipping") == "warn":
        cannot.append("Clipping is baked in; ACX listeners and QC will hear the distortion. Re-record if you can.")
    rt = m["room_tone"]
    keep = 2.0  # seconds of room tone kept when trimming: inside ACX's recommended 1-5 s
    if rt.get("reliable") and (rt["head_s"] > spec["room_tone_max"] or rt["tail_s"] > spec["room_tone_max"]):
        start = max(0.0, rt["head_s"] - keep) if rt["head_s"] > spec["room_tone_max"] else 0.0
        end = min(m["duration"], rt["speech_end_s"] + keep) if rt["tail_s"] > spec["room_tone_max"] else None
        crop = {"op": "crop", "start": round(start, 3)}
        if end is not None:
            crop["end"] = round(end, 3)
        ops.append(crop)
        why.append(f"crop first, keeping {keep:g} s of the existing room tone at the long end(s).")
    if m["sr"] != spec["sample_rate"]:
        ops.append({"op": "resample", "rate": spec["sample_rate"]})
        why.append("resample to 44.1 kHz before level changes, so peaks are measured at the final rate.")
    rms_bad = _status(checks, "rms") == "fail"
    peak_bad = _status(checks, "peak") in ("fail", "warn")
    noise_bad = _status(checks, "noise_floor") == "fail"
    target = spec["rms_target"] + adjust.get("rms", 0.0)
    gain = target - m["rms"] if (rms_bad or peak_bad or noise_bad) else 0.0
    noise_after = m["noise"] + gain
    if noise_after > spec["noise_max"] - 3:
        # Aim lower in the RMS window to need less noise reduction, but keep 1 dB inside the window.
        lower = max(spec["rms_range"][0] + 1.0, target - (noise_after - (spec["noise_max"] - 3)))
        if lower < target:
            gain -= target - lower
            target = lower
            noise_after = m["noise"] + gain
    need = noise_after - (spec["noise_max"] - 3) + adjust.get("noise", 0.0)
    if need > 0:
        red = float(max(6, math.ceil(need)))
        if red > MAX_SPEECH_SAFE_DENOISE_DB:
            cannot.append(
                f"Noise floor: after setting the level the room noise would sit near {noise_after:.0f} dB RMS and "
                f"need about {red:.0f} dB of reduction to pass; that much denoising audibly damages speech. "
                "Re-record in a quieter space, or try a dedicated restoration tool and verify with audio_transcribe."
            )
        else:
            ops += [{"op": "highpass", "hz": 80}, {"op": "denoise", "reduction_db": red}]
            why.append(
                f"highpass 80 Hz and denoise ({red:g} dB) BEFORE raising the level, because gain raises the noise "
                "floor by the same amount. Verify the words with audio_transcribe afterwards (denoising can blur "
                "quiet consonants)."
            )
    if rms_bad or peak_bad or any(o["op"] == "denoise" for o in ops):
        ceiling = round(spec["peak_max"] - 0.5 + adjust.get("tp", 0.0), 2)
        ops.append({"op": "normalize_rms", "db": round(target, 2)})
        ops.append({"op": "limit", "ceiling_db": ceiling})
        why.append(
            f"normalize_rms to {target:g} dB (inside the -23..-18 window with margin), then a true-peak limiter at "
            f"{ceiling:g} dB so peaks stay under -3 dB even after MP3 encoding."
        )
    return ops, why, cannot


def _plan_sfx(m, spec, checks, adjust):
    ops, why, cannot = [], [], []
    if m["silent"] or m["lead_s"] is None:
        return ops, why, ["The file is silent."]
    if _status(checks, "clipping") == "fail":
        cannot.append(
            "Clipping is baked in: turning it down passes the meters, but the flat-topped peaks remain (the check "
            "will still flag them). Re-export from an unclipped source."
        )
    lead_fail = _status(checks, "leading_silence") == "fail"
    if lead_fail or adjust.get("lead"):
        # adjust["lead"]: silence that only shows once DC is removed and the gain is set (found by simulation)
        start = (max(0.0, m["lead_s"] - 0.001) if lead_fail else 0.0) + adjust.get("lead", 0.0)
        ops.append({"op": "crop", "start": round(start, 4)})
        why.append("crop the leading silence first (keeping 1 ms before the sound).")
    if _status(checks, "dc_offset") == "fail":
        ops.append({"op": "remove_dc"})
        why.append("remove_dc before fades and gain, so the edges settle at zero.")
    if _status(checks, "click_at_start") == "fail" or adjust.get("fade_in"):
        ops.append({"op": "fade_in", "seconds": 0.002})
        why.append("fade_in 2 ms to remove the click at the start (barely softens the attack).")
    if _status(checks, "click_at_end") == "fail" or adjust.get("fade_out"):
        ops.append({"op": "fade_out", "seconds": 0.01})
        why.append("fade_out 10 ms to remove the click at the end.")
    if _bad(checks, "loudness", "true_peak"):
        ceiling = round(spec["true_peak_max"] - TP_MARGIN + adjust.get("tp", 0.0), 2)
        ops.append(
            {
                "op": "normalize_lufs",
                "lufs": spec["momentary_target"],
                "measure": "momentary_max",
                "ceiling_db": ceiling,
                "limit": True,
            }
        )
        why.append(
            f"normalize_lufs (momentary_max) to {spec['momentary_target']:g} LUFS last, limiter at {ceiling:g} dBTP."
        )
    return ops, why, cannot


PLANNERS = {"loudness": _plan_loudness, "acx": _plan_acx, "sfx": _plan_sfx}


def _refine(adjust: dict, after_m: dict, after_checks: list[dict], spec: dict) -> bool:
    """Nudge the plan from a simulated result. Returns False if there is nothing sensible to change."""
    changed = False
    if _status(after_checks, "true_peak") == "fail" or _status(after_checks, "peak") in ("fail", "warn"):
        adjust["tp"] = adjust.get("tp", 0.0) - 0.5
        changed = True
    if spec["family"] == "acx":
        rms = after_m["rms"]
        lo, hi = spec["rms_range"]
        if rms is not None and not lo + 0.5 <= rms <= hi - 0.5:
            adjust["rms"] = adjust.get("rms", 0.0) + (spec["rms_target"] - rms)
            changed = True
        if _status(after_checks, "noise_floor") == "fail":
            adjust["noise"] = adjust.get("noise", 0.0) + (after_m["noise"] - spec["noise_max"]) + 2
            changed = True
    if spec["family"] == "loudness" and _status(after_checks, "background_noise") == "warn":
        adjust["noise"] = adjust.get("noise", 0.0) + 3
        changed = True
    if spec["family"] == "sfx":
        if _status(after_checks, "leading_silence") == "fail" and after_m["lead_s"]:
            adjust["lead"] = adjust.get("lead", 0.0) + max(0.0, after_m["lead_s"] - 0.001)
            changed = True
        if _status(after_checks, "click_at_start") == "fail" and not adjust.get("fade_in"):
            adjust["fade_in"] = changed = True
        if _status(after_checks, "click_at_end") == "fail" and not adjust.get("fade_out"):
            adjust["fade_out"] = changed = True
    return changed


def plan_fix(audio: Audio, spec: dict, m: dict, checks: list[dict]) -> dict:
    adjust: dict = {}
    planner = PLANNERS[spec["family"]]
    ops, why, cannot = planner(m, spec, checks, adjust)
    fix = {"operations": ops, "steps": why, "cannot_fix": cannot}
    if not ops:
        fix["verified"] = False
        fix["verification"] = (
            "No automatic fix is possible for the failing checks; see cannot_fix."
            if cannot
            else "Nothing to process: the remaining warnings need a decision or an export step (see each note)."
        )
        return fix
    if audio.duration > SIMULATE_MAX_S:
        fix["verified"] = None
        fix["verification"] = (
            f"Not simulated (file longer than {SIMULATE_MAX_S:g} s). Apply it with audio_process and run "
            "audio_check_delivery on the output."
        )
        return fix
    result = None
    for _attempt in range(3):
        try:
            y, sr, _log = process.apply_chain(audio, ops)
        except AudioError as e:
            fix["verified"] = False
            fix["verification"] = f"The planned operations failed in simulation: {e}"
            return fix
        sim = Audio(y, sr, None, audio.format, audio.subtype)
        after_m, after_checks = run_checks(sim, spec)
        result = verdict(after_checks)
        if result != "fail":
            break
        if not _refine(adjust, after_m, after_checks, spec):
            break
        ops, why, cannot = planner(m, spec, checks, adjust)
        fix.update(operations=ops, steps=why, cannot_fix=cannot)
    still = [c["check"] for c in after_checks if c["status"] == "fail"]
    fix["verified"] = result != "fail"
    fix["expected_verdict"] = result
    if result == "fail":
        fix["verification"] = f"Simulated on this file: still fails {', '.join(still)} after these operations. " + (
            "See cannot_fix." if cannot else "Fix those by hand."
        )
    else:
        fix["verification"] = f"Simulated on this file: the result would be '{result}'."
        if cannot:
            fix["verification"] += " The meters would pass, but the cannot_fix problems remain audible."
    return fix


# ---------------------------------------------------------------- entry point


def check_delivery(audio: Audio, spec_name: str) -> dict:
    spec = get_spec(spec_name)
    m, checks = run_checks(audio, spec)
    v = verdict(checks)
    out = {
        "verdict": v,
        "file": str(audio.path) if audio.path else None,
        "format": f"{audio.format} {audio.subtype}".strip(),
        "sample_rate": audio.sr,
        "channels": audio.channels,
        "duration_s": r(audio.duration, 3),
    }
    if audio.offset:
        out["range_s"] = [r(audio.offset, 3), r(audio.offset + audio.duration, 3)]
    out["checks"] = checks
    if v == "pass":
        out["fix"] = None
    else:
        fix = plan_fix(audio, spec, m, checks)
        fix["how"] = "audio_process(path, operations) writes a new file; then run audio_check_delivery on it." + (
            " Only part of the file was checked; the operations apply to the whole file." if audio.offset else ""
        )
        if not fix["operations"]:
            fix["how"] = "Nothing to apply automatically."
        out["fix"] = fix
    out["spec"] = {
        "name": spec_name.strip().lower(),
        "title": spec["title"],
        "status": spec.get("status", "official"),
        "sources": spec["sources"],
        "notes": spec["notes"],
    }
    return out
