"""A/B comparison: metric deltas, per-band level changes, spectral similarity and a null test."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .analysis import analyze_with_profile, band_levels_db
from .audio_io import load
from .dsp import pow_db, r, resample

METRICS = (
    ("duration_s",),
    ("levels", "lufs_integrated"),
    ("levels", "peak_dbfs"),
    ("levels", "true_peak_dbtp"),
    ("levels", "rms_dbfs"),
    ("levels", "crest_factor_db"),
    ("silence", "noise_floor_dbfs"),
    ("spectrum", "centroid_hz"),
    ("spectrum", "rolloff_85_hz"),
    ("spectrum", "flatness"),
    ("stereo", "correlation"),
    ("rhythm", "tempo_bpm"),
)


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _log_spectrum(profile, grid):
    return np.interp(grid, profile.freqs, pow_db(profile.power))


def null_test(xa: np.ndarray, xb: np.ndarray, sr: int) -> dict:
    """Align B to A (both mono at rate `sr`), subtract, and report how much is left. Sums are taken chunk by
    chunk, so long files never need full-length float64 copies."""
    n = min(len(xa), len(xb), 20 * sr)
    max_lag = min(sr, n // 2)
    a64, b64 = xa[:n].astype(np.float64), xb[:n].astype(np.float64)
    corr = signal.correlate(b64, a64, mode="full", method="fft")
    lags = signal.correlation_lags(n, n, mode="full")
    keep = np.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    # Normalise by the energy in each lag's overlap; raw correlation favours small lags (more overlap) and can
    # pick the wrong period on music or tones.
    ca = np.concatenate(([0.0], np.cumsum(a64 * a64)))
    cb = np.concatenate(([0.0], np.cumsum(b64 * b64)))
    ea = ca[np.minimum(n, n - lags)] - ca[np.maximum(0, -lags)]
    eb = cb[np.minimum(n, n + lags)] - cb[np.maximum(0, lags)]
    norm = corr / np.sqrt(np.maximum(ea * eb, 1e-30))
    i = int(np.argmax(np.abs(norm)))
    if corr[i] == 0:  # no correlation at all (e.g. silence): don't invent an offset
        lag, sign = 0, 1.0
    else:
        lag = int(lags[i])
        sign = -1.0 if corr[i] < 0 else 1.0
    if lag >= 0:
        xb = xb[lag:]
    else:
        xa = xa[-lag:]
    m = min(len(xa), len(xb))
    xa, xb = xa[:m], xb[:m]
    saa = sbb = sab = 0.0
    for c in range(0, m, 1 << 20):
        a = xa[c : c + (1 << 20)].astype(np.float64)
        b = sign * xb[c : c + (1 << 20)].astype(np.float64)
        saa += float(a @ a)
        sbb += float(b @ b)
        sab += float(a @ b)
    ref = pow_db(saa / m)
    resid = pow_db(max(saa + sbb - 2 * sab, 0.0) / m) - ref
    g = sab / sbb if sbb > 0 else 1.0  # least-squares gain that best matches B to A
    resid_g = pow_db(max(saa - 2 * g * sab + g * g * sbb, 0.0) / m) - ref
    if saa == 0 or sbb == 0:
        silent = "both are silent" if saa == sbb else ("A is silent" if saa == 0 else "B is silent")
        return {"verdict": f"no comparison possible: {silent}", "offset_ms": 0.0}
    if resid < -60:
        verdict = "identical (nulls to below -60 dB)"
    elif resid_g < -60:
        verdict = f"same audio, only the level differs (B is {r(-20 * np.log10(abs(g)), 2)} dB relative to A)"
    elif resid_g < -20:
        verdict = "same source with small changes"
    elif resid_g < -6:
        verdict = "related but clearly changed"
    else:
        verdict = "different audio"
    return {
        "offset_ms": r(lag / sr * 1000, 2),
        "polarity_inverted": sign < 0,
        "residual_db": r(resid, 1),
        "residual_after_gain_match_db": r(resid_g, 1),
        "verdict": verdict,
        "note": "residual is the difference signal's level relative to A (mono, after time alignment)",
    }


def _analyze_file(path, sr_out: int | None = None):
    """Analyze one file and keep only what the comparison needs, so two long files are never in memory together."""
    audio = load(path)
    rep, profile = analyze_with_profile(audio)
    mono = resample(audio.mono, audio.sr, sr_out or audio.sr)
    return rep, profile, mono, audio.sr, str(audio.path), audio.duration


def compare(path_a, path_b) -> dict:
    ra, pa, mono_a, sr_a, name_a, dur_a = _analyze_file(path_a)
    rb, pb, mono_b, sr_b, name_b, dur_b = _analyze_file(path_b, sr_out=sr_a)
    metrics = {}
    for path in METRICS:
        va, vb = _get(ra, path), _get(rb, path)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            metrics[".".join(path)] = {"a": va, "b": vb, "change": r(vb - va, 3)}
    ma, mb = _get(ra, ("pitch", "median", "hz")), _get(rb, ("pitch", "median", "hz"))
    steady = all((_get(rep, ("pitch", "spread_cents")) or 1e9) < 150 for rep in (ra, rb))
    if ma and mb and steady:  # a median pitch is meaningless for glides and melodies
        metrics["pitch.median_hz"] = {"a": ma, "b": mb, "change_cents": r(1200 * np.log2(mb / ma), 0)}

    la, lb = band_levels_db(pa), band_levels_db(pb)
    bands = {k: r(lb[k] - la[k], 1) for k in la if la[k] is not None and lb[k] is not None and max(la[k], lb[k]) > -100}

    hi = min(16000, sr_a / 2, sr_b / 2)
    grid = np.geomspace(40, hi, 96)
    sa, sb = _log_spectrum(pa, grid), _log_spectrum(pb, grid)
    similarity = float(np.corrcoef(sa - sa.mean(), sb - sb.mean())[0, 1]) if np.std(sa) > 0 and np.std(sb) > 0 else None

    summary = []
    d = metrics.get("levels.lufs_integrated")
    if d and abs(d["change"]) >= 0.5:
        summary.append(
            f"B is {abs(d['change']):.1f} LU {'louder' if d['change'] > 0 else 'quieter'} (integrated loudness)."
        )
    d = metrics.get("duration_s")
    if d and abs(d["change"]) >= 0.01:
        summary.append(f"B is {abs(d['change']):.3f}s {'longer' if d['change'] > 0 else 'shorter'}.")
    d = metrics.get("spectrum.centroid_hz")
    if d and d["a"] and abs(d["change"]) / d["a"] > 0.15:
        summary.append(
            f"B is {'brighter' if d['change'] > 0 else 'darker'} (spectral centroid {d['a']:.0f} -> {d['b']:.0f} Hz)."
        )
    lufs_change = (metrics.get("levels.lufs_integrated") or {}).get("change") or 0
    for band, change in bands.items():
        if abs(change - lufs_change) >= 3:
            summary.append(f"{band} band {change:+.1f} dB (vs {lufs_change:+.1f} overall).")
    d = metrics.get("pitch.median_hz")
    if d and abs(d["change_cents"]) >= 20:
        summary.append(f"Pitch moved {d['change_cents']:+.0f} cents.")
    issues_a = {i["message"] for i in ra["issues"]}
    issues_b = {i["message"] for i in rb["issues"]}
    out = {
        "a": name_a,
        "b": name_b,
        "summary": summary or ["No large differences in level, length, brightness or band balance."],
        "metrics": metrics,
        "band_level_change_db": bands,
        "spectral_shape_similarity": r(similarity, 3),
        "issues_only_in_a": sorted(issues_a - issues_b),
        "issues_only_in_b": sorted(issues_b - issues_a),
    }
    if dur_a >= 0.05 and dur_b >= 0.05:
        out["null_test"] = null_test(mono_a, mono_b, sr_a)
    return out
