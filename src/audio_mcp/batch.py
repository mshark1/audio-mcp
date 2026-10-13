"""Folder-level work: survey many files for consistency, and apply one edit chain to many files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .analysis import analyze, quick_levels
from .audio_io import AUDIO_EXTENSIONS, AudioError, load, resolve_path, save
from .dsp import r
from .process import apply_chain

MAX_FILES = 500
FULL_ANALYSIS_LIMIT_S = 600.0


def collect(target: str | list[str], pattern: str = "*") -> list[Path]:
    """Files from a directory (recursive glob) or an explicit list of paths."""
    if isinstance(target, list):
        paths = [resolve_path(p) for p in target]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise AudioError(f"Files not found: {missing[:10]}")
    else:
        d = resolve_path(target)
        if not d.is_dir():
            raise AudioError(f"Not a directory: {d}. Pass a folder or a list of file paths.")
        paths = sorted(p for p in d.rglob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
    if not paths:
        raise AudioError(f"No audio files found ({', '.join(sorted(AUDIO_EXTENSIONS))}).")
    if len(paths) > MAX_FILES:
        raise AudioError(
            f"{len(paths)} files is more than the {MAX_FILES} limit; narrow it with `pattern` or a subfolder."
        )
    return paths


def survey(target: str | list[str], pattern: str = "*") -> dict:
    """Analyze every file and report how consistent the set is, with outliers and suggested fixes."""
    rows, errors = [], []
    for p in collect(target, pattern):
        try:
            audio = load(p, end=FULL_ANALYSIS_LIMIT_S)
            rep = analyze(audio, pitch=False, rhythm=False)
        except AudioError as e:
            errors.append(f"{p.name}: {e}")
            continue
        lv, si, sp = rep["levels"], rep["silence"], rep["spectrum"]
        rows.append(
            {
                "file": str(p),
                "name": p.name,
                "duration_s": rep["duration_s"],
                "sample_rate": rep["sample_rate"],
                "channels": rep["channels"],
                "format": rep["format"],
                "lufs": lv["lufs_integrated"],
                "rms_dbfs": lv["rms_dbfs"],
                "true_peak_dbtp": lv["true_peak_dbtp"],
                "leading_silence_s": si.get("leading_s"),
                "trailing_silence_s": si.get("trailing_s"),
                "centroid_hz": sp.get("centroid_hz"),
                "issues": [f"{i['severity']}: {i['message']}" for i in rep["issues"] if i["severity"] != "info"],
            }
        )
    if not rows:
        raise AudioError("No file could be analyzed: " + "; ".join(errors[:5]))

    findings, suggestions = [], []
    # Loudness: use LUFS where defined, otherwise RMS (short one-shots).
    use_lufs = sum(row["lufs"] is not None for row in rows) >= 0.8 * len(rows)
    key = "lufs" if use_lufs else "rms_dbfs"
    vals = np.array([row[key] for row in rows if row[key] is not None])
    med = float(np.median(vals))
    spread = float(np.percentile(vals, 90) - np.percentile(vals, 10)) if len(vals) > 1 else 0.0
    unit = "LUFS" if use_lufs else "dBFS RMS"
    loud = [(row["name"], row[key]) for row in rows if row[key] is not None and abs(row[key] - med) > 6]
    for name, v in loud:
        findings.append(
            f"{name} is {abs(v - med):.1f} dB {'louder' if v > med else 'quieter'} than the set median ({v:.1f} vs {med:.1f} {unit})."
        )
    if spread > 6 or loud:
        op = (
            {"op": "normalize_lufs", "lufs": round(med), "limit": True}
            if use_lufs
            else {"op": "normalize_peak", "db": -1}
        )
        suggestions.append(
            {
                "why": f"loudness spread is {spread:.1f} dB (10th-90th percentile)",
                "audio_batch_process_operations": [op],
            }
        )

    for field, label in (("sample_rate", "sample rate"), ("channels", "channel count"), ("format", "format")):
        counts = Counter(row[field] for row in rows)
        if len(counts) > 1:
            common = counts.most_common(1)[0][0]
            odd = [row["name"] for row in rows if row[field] != common]
            findings.append(
                f"Mixed {label}: most files are {common}; different: {odd[:10]}{' ...' if len(odd) > 10 else ''}."
            )
            if field == "sample_rate":
                suggestions.append(
                    {
                        "why": "mixed sample rates",
                        "audio_batch_process_operations": [{"op": "resample", "rate": common}],
                    }
                )

    padded = [row["name"] for row in rows if (row["leading_silence_s"] or 0) > 0.05]
    if padded and len(padded) < len(rows):
        findings.append(f"{len(padded)} file(s) start with silence (>50 ms) while others don't: {padded[:10]}.")
        suggestions.append(
            {
                "why": "inconsistent leading silence (delays playback of one-shots)",
                "audio_batch_process_operations": [{"op": "trim_silence"}],
            }
        )
    problem = [row["name"] for row in rows if row["issues"]]
    if problem:
        findings.append(f"{len(problem)} file(s) have warnings or errors (see each file's `issues`): {problem[:10]}.")

    return {
        "file_count": len(rows),
        "findings": findings
        or ["The set is consistent in loudness, sample rate, channels, format and leading silence."],
        "suggested_fixes": suggestions,
        "stats": {
            f"median_{key}": r(med, 1),
            f"spread_{key}_p10_p90": r(spread, 1),
            "total_duration_s": r(sum(row["duration_s"] for row in rows), 1),
            "sample_rates": dict(Counter(row["sample_rate"] for row in rows)),
        },
        "files": rows,
        **({"unreadable": errors} if errors else {}),
    }


def process_many(
    target: str | list[str], operations: list[dict], output_dir: str, pattern: str = "*", overwrite: bool = False
) -> dict:
    """Apply one edit chain to many files, writing results (same names) into output_dir."""
    paths = collect(target, pattern)
    out_dir = resolve_path(output_dir)
    if any(p.parent == out_dir for p in paths) and not overwrite:
        raise AudioError(
            "output_dir contains the input files; choose a different folder (or pass overwrite=true to replace them)."
        )
    base = resolve_path(target) if isinstance(target, str) else None
    results, errors = [], []
    for p in paths:
        try:
            audio = load(p)
            x, sr, log = apply_chain(audio, operations)
            rel = p.relative_to(base) if base else Path(p.name)
            dest = out_dir / rel
            if dest.suffix.lower() not in (".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3"):
                dest = dest.with_suffix(".wav")
            written, warnings = save(x, sr, dest, overwrite=overwrite)
            after = quick_levels(x, sr)
            results.append(
                {
                    "file": p.name,
                    "output": str(written),
                    "lufs": after["lufs_integrated"],
                    "peak_dbfs": after["peak_dbfs"],
                    "notes": log,
                    "warnings": warnings,
                }
            )
        except AudioError as e:
            errors.append(f"{p.name}: {e}")
    return {
        "processed": len(results),
        "failed": len(errors),
        "output_dir": str(out_dir),
        "results": results,
        "errors": errors,
    }
