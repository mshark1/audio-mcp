"""Loading and saving audio files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp.server.mcpserver.exceptions import ToolError

from .dsp import amp_db

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aif", ".aiff", ".aifc", ".w64", ".caf"}
DEFAULT_SUBTYPES = {
    ".wav": "PCM_24",
    ".flac": "PCM_24",
    ".aif": "PCM_24",
    ".aiff": "PCM_24",
    ".ogg": "VORBIS",
    ".mp3": "MPEG_LAYER_III",
}
FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}


def max_seconds() -> float:
    return float(os.environ.get("AUDIO_MCP_MAX_SECONDS", 1800))


class AudioError(ToolError):
    """An error whose message is written for the caller. As a ToolError, the MCP SDK passes it to the client
    instead of replacing it with a generic 'Error executing tool'."""


@dataclass
class Audio:
    samples: np.ndarray  # (frames, channels) float32
    sr: int
    path: Path | None = None
    format: str = ""
    subtype: str = ""
    offset: float = 0.0  # start time of `samples` within the source file, seconds

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def frames(self) -> int:
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        return self.frames / self.sr

    @cached_property
    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=1) if self.channels > 1 else self.samples[:, 0]


def allowed_dirs() -> list[Path]:
    """Directories from AUDIO_MCP_ALLOWED_DIRS (os.pathsep-separated). Empty means unrestricted."""
    raw = os.environ.get("AUDIO_MCP_ALLOWED_DIRS", "")
    return [Path(d).expanduser().resolve() for d in raw.split(os.pathsep) if d.strip()]


def resolve_path(path: str | os.PathLike) -> Path:
    """Absolute, symlink-resolved path. Every file the tools read or write passes through here, so this is also
    where the optional AUDIO_MCP_ALLOWED_DIRS sandbox is enforced (after resolving, so links can't escape it)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path(os.environ.get("AUDIO_MCP_ROOT", os.getcwd())) / p
    p = p.resolve()
    roots = allowed_dirs()
    if roots and not any(p == root or root in p.parents for root in roots):
        raise AudioError(
            f"Access to {p} is not allowed. This server is restricted to: {', '.join(map(str, roots))} "
            "(AUDIO_MCP_ALLOWED_DIRS)."
        )
    return p


def _info(p: Path):
    if not p.is_file():
        raise AudioError(f"File not found: {p}")
    try:
        return sf.info(str(p))
    except Exception as e:
        raise AudioError(
            f"Could not decode '{p.name}' ({e}). Supported: WAV, FLAC, OGG/Vorbis, Opus, MP3, AIFF, CAF. "
            "Convert other formats (M4A/AAC, WMA, video files) to WAV first, e.g. `ffmpeg -i in.m4a out.wav`."
        ) from None


def file_info(path) -> dict:
    p = resolve_path(path)
    info = _info(p)
    return {
        "path": str(p),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_s": round(info.duration, 3),
        "size_bytes": p.stat().st_size,
    }


def list_audio(directory, limit: int = 200) -> dict:
    d = resolve_path(directory)
    if not d.is_dir():
        raise AudioError(f"Not a directory: {d}")
    files, errors = [], []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTENSIONS or not p.is_file():
            continue
        if len(files) >= limit:
            return {"directory": str(d), "files": files, "truncated": True}
        try:
            info = file_info(p)
            files.append({k: info[k] for k in ("path", "sample_rate", "channels", "duration_s")})
        except AudioError as e:
            errors.append(str(e))
    out = {"directory": str(d), "files": files, "truncated": False}
    if errors:
        out["unreadable"] = errors[:20]
    return out


def load(path, start: float | None = None, end: float | None = None) -> Audio:
    p = resolve_path(path)
    info = _info(p)
    sr, total = info.samplerate, info.duration
    s = 0.0 if start is None else float(start)
    e = total if end is None else min(float(end), total)
    if s < 0 or s >= total:
        raise AudioError(f"start={s}s is outside the file (duration {total:.3f}s).")
    if e <= s:
        raise AudioError(f"end ({e}s) must be after start ({s}s).")
    if e - s > max_seconds():
        raise AudioError(
            f"The selection is {e - s:.0f}s long; in-memory analysis is limited to {max_seconds():.0f}s. "
            "Pass start/end to work on a section, or use audio_timeline for a whole-file overview (it streams). "
            "The limit can be raised with the AUDIO_MCP_MAX_SECONDS environment variable."
        )
    stop = None if end is None else int(round(e * sr))
    data, sr = sf.read(str(p), start=int(round(s * sr)), stop=stop, dtype="float32", always_2d=True)
    if len(data) == 0:
        raise AudioError("The selected range contains no samples.")
    bad = ~np.isfinite(data)
    if bad.any():
        first = int(np.flatnonzero(bad.any(axis=1))[0])
        raise AudioError(
            f"The audio contains {int(bad.sum())} NaN or infinite sample values (first at {s + first / sr:.3f}s). "
            "The file is corrupt or was written by software that overflowed; re-export it from the source."
        )
    return Audio(data, sr, p, info.format, info.subtype, s)


def default_output(source: Path, tag: str) -> Path:
    suffix = source.suffix.lower() if source.suffix.lower() in DEFAULT_SUBTYPES else ".wav"
    candidate = source.with_name(f"{source.stem}_{tag}{suffix}")
    n = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}_{tag}{n}{suffix}")
        n += 1
    return candidate


def save(
    samples: np.ndarray, sr: int, path, overwrite: bool = False, subtype: str | None = None
) -> tuple[Path, list[str]]:
    p = resolve_path(path)
    ext = p.suffix.lower()
    if ext not in DEFAULT_SUBTYPES:
        raise AudioError(f"Can't write '{ext or p.name}'. Use one of: {', '.join(sorted(DEFAULT_SUBTYPES))}.")
    if p.exists() and not overwrite:
        raise AudioError(f"{p} already exists. Choose another output path or pass overwrite=true.")
    subtype = (subtype or DEFAULT_SUBTYPES[ext]).upper()
    warnings = []
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0 and subtype not in FLOAT_SUBTYPES:
        warnings.append(
            f"Peak was {float(amp_db(peak)):+.2f} dBFS; samples above 0 dBFS were clipped when writing {subtype}. "
            "Lower the gain, or write a WAV with subtype FLOAT to keep them."
        )
        samples = np.clip(samples, -1.0, 1.0)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(p), samples, sr, subtype=subtype)
    except Exception as e:
        raise AudioError(f"Could not write {p.name} as {subtype}: {e}") from None
    return p, warnings
