"""Loading and saving audio files."""

from __future__ import annotations

import atexit
import contextlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
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

# Containers libsndfile can't read. They're decoded with FFmpeg when it's installed.
FFMPEG_EXTENSIONS = {
    ".m4a", ".m4b", ".mp4", ".m4v", ".aac", ".mov", ".3gp", ".webm", ".mkv", ".mka",
    ".wma", ".wmv", ".asf", ".avi", ".flv", ".ac3", ".amr",
}  # fmt: skip
NATIVE_FORMATS = "WAV, FLAC, OGG/Vorbis, Opus, MP3, AIFF, CAF"
FFMPEG_INSTALL_HINT = (
    "Install FFmpeg and restart the MCP server: Windows `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), "
    "macOS `brew install ffmpeg`, Debian/Ubuntu `sudo apt install ffmpeg`. Or set AUDIO_MCP_FFMPEG to the path of "
    "the ffmpeg executable."
)
# Demuxers FFmpeg may use on user files. Playlist-style formats (hls, concat, ...) are left out so a file can't make
# FFmpeg open other files, and -protocol_whitelist file keeps it off the network.
FFMPEG_DEMUXERS = (
    "mov,matroska,webm,asf,avi,flv,aac,ac3,eac3,amr,wav,w64,mp3,flac,ogg,aiff,caf,mpeg,mpegts,wv,ape,dts,"
    "truehd,mlp,tta,au,voc,rm,dsf,loas,latm"
)
FFMPEG_CACHE_MAX_FILES = 16


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


# ---------------------------------------------------------------------------------------------------------------
# FFmpeg input: formats libsndfile can't read are decoded once to a temporary float WAV and cached for the process.


def _executable(env_var: str) -> str | None:
    value = os.environ.get(env_var, "").strip()
    if not value:
        return None
    p = Path(value).expanduser()
    return str(p) if p.is_file() else shutil.which(value)


def ffmpeg_path() -> str | None:
    """AUDIO_MCP_FFMPEG if set, else `ffmpeg` on PATH. Looked up on every call."""
    if os.environ.get("AUDIO_MCP_FFMPEG", "").strip():
        return _executable("AUDIO_MCP_FFMPEG")
    return shutil.which("ffmpeg")


def ffprobe_path(ffmpeg: str) -> str | None:
    """AUDIO_MCP_FFPROBE, else ffprobe next to ffmpeg, else (when ffmpeg came from PATH) ffprobe on PATH."""
    if os.environ.get("AUDIO_MCP_FFPROBE", "").strip():
        return _executable("AUDIO_MCP_FFPROBE")
    f = Path(ffmpeg)
    sibling = f.with_name("ffprobe" + f.suffix)
    if sibling.is_file():
        return str(sibling)
    return None if os.environ.get("AUDIO_MCP_FFMPEG", "").strip() else shutil.which("ffprobe")


def readable_extensions() -> set[str]:
    """Extensions the tools can open right now (FFmpeg formats only when FFmpeg is available)."""
    return AUDIO_EXTENSIONS | FFMPEG_EXTENSIONS if ffmpeg_path() else set(AUDIO_EXTENSIONS)


def _ffmpeg_timeout() -> float:
    return float(os.environ.get("AUDIO_MCP_FFMPEG_TIMEOUT", 300))


def _cache_limit_bytes() -> int:
    return int(float(os.environ.get("AUDIO_MCP_FFMPEG_CACHE_MB", 2048)) * 2**20)


@dataclass
class _Decoded:
    path: Path
    size: int
    codec: str


_cache: OrderedDict[tuple, _Decoded] = OrderedDict()
_cache_lock = threading.Lock()
_cache_dir: Path | None = None
_counter = itertools.count()


def _get_cache_dir() -> Path:
    global _cache_dir
    if _cache_dir is None or not _cache_dir.is_dir():
        _cache_dir = Path(tempfile.mkdtemp(prefix="audio-mcp-ffmpeg-"))
        atexit.register(shutil.rmtree, _cache_dir, ignore_errors=True)
    return _cache_dir


def _unlink(p: Path):
    with contextlib.suppress(OSError):  # still open somewhere (Windows); the atexit cleanup removes it later
        p.unlink(missing_ok=True)


def clear_ffmpeg_cache():
    with _cache_lock:
        for entry in _cache.values():
            _unlink(entry.path)
        _cache.clear()


def _evict():
    """Drop least recently used decodes beyond the count and size caps. The newest entry always stays."""
    limit = _cache_limit_bytes()
    while len(_cache) > 1 and (len(_cache) > FFMPEG_CACHE_MAX_FILES or sum(e.size for e in _cache.values()) > limit):
        _, old = _cache.popitem(last=False)
        _unlink(old.path)


def _run(cmd: list[str], timeout: float) -> tuple[int, str, bytes]:
    """Run without a shell. stderr goes to a file rather than a pipe, so a timeout can't then hang waiting for a
    grandchild process that still holds the pipe open."""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, timeout=timeout)
        err.seek(0)
        return proc.returncode, err.read().decode("utf-8", "replace"), proc.stdout


def _tail(text: str, n: int = 300) -> str:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    msg = " | ".join(lines[-3:])
    return msg[-n:] if msg else "no error message"


def _probe(p: Path, ffmpeg: str) -> dict:
    """Codec and first-audio-stream facts from ffprobe, if installed. Optional: returns {} on any problem."""
    probe = ffprobe_path(ffmpeg)
    if not probe:
        return {}
    cmd = [
        probe, "-v", "error", "-protocol_whitelist", "file", "-format_whitelist", FFMPEG_DEMUXERS,
        "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of", "json", f"file:{p}",
    ]  # fmt: skip
    try:
        code, _, out = _run(cmd, timeout=min(30.0, _ffmpeg_timeout()))
        if code != 0:
            return {}
        data = json.loads(out.decode("utf-8", "replace"))
        streams = data.get("streams") or []
        if not streams:
            return {}
        st, fmt = streams[0], data.get("format") or {}
        dur = st.get("duration") or fmt.get("duration")
        return {
            "codec": str(st.get("codec_name") or ""),
            "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None,
            "channels": int(st["channels"]) if st.get("channels") else None,
            "duration": float(dur) if dur not in (None, "N/A") else None,
        }
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def _decode(p: Path, ffmpeg: str) -> _Decoded:
    """Decode p (which has already passed resolve_path) to a float WAV, cached per (path, size, mtime)."""
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    with _cache_lock:  # also serializes decodes, so two calls never decode the same file at once
        hit = _cache.get(key)
        if hit is not None and hit.path.is_file():
            _cache.move_to_end(key)
            return hit
        out = _get_cache_dir() / f"{next(_counter)}.wav"
        limit = _cache_limit_bytes()
        cmd = [
            ffmpeg, "-nostdin", "-hide_banner", "-v", "error",
            "-protocol_whitelist", "file", "-format_whitelist", FFMPEG_DEMUXERS,
            "-i", f"file:{p}",  # "file:" so a name like "http:..." or "pipe:..." is never taken as a protocol
            "-map", "0:a:0", "-c:a", "pcm_f32le", "-rf64", "auto", "-fs", str(limit),
            "-f", "wav", "-y", f"file:{out}",
        ]  # fmt: skip
        timeout = _ffmpeg_timeout()
        try:
            code, err, _ = _run(cmd, timeout)
        except subprocess.TimeoutExpired:
            _unlink(out)
            raise AudioError(
                f"FFmpeg took longer than {timeout:g}s to decode '{p.name}' and was stopped. Convert it to WAV or "
                "FLAC yourself, or raise the limit with the AUDIO_MCP_FFMPEG_TIMEOUT environment variable (seconds)."
            ) from None
        except OSError as e:
            raise AudioError(
                f"Could not run FFmpeg ({ffmpeg}): {e}. Check AUDIO_MCP_FFMPEG or reinstall FFmpeg."
            ) from None
        size = out.stat().st_size if out.is_file() else 0
        if code != 0 or size <= 44:
            _unlink(out)
            raise AudioError(
                f"Could not decode '{p.name}': FFmpeg failed ({_tail(err)}). The file may be corrupt, have no audio "
                "track, be DRM-protected, or use a container this server doesn't open."
            )
        if size >= 0.99 * limit:  # -fs stopped the output: don't silently analyze a truncated file
            _unlink(out)
            raise AudioError(
                f"'{p.name}' decodes to more than {limit / 2**20:.0f} MB, the FFmpeg cache limit. Convert it to FLAC "
                "yourself, or raise AUDIO_MCP_FFMPEG_CACHE_MB."
            )
        entry = _Decoded(out, size, _probe(p, ffmpeg).get("codec", ""))
        _cache[key] = entry
        _evict()
        return entry


@dataclass
class Source:
    """A user file and what soundfile should read for it: the file itself, or its FFmpeg-decoded WAV."""

    path: Path
    read_path: Path
    info: object  # soundfile info of read_path
    format: str
    subtype: str
    via_ffmpeg: bool = False


def _label(p: Path) -> str:
    return p.suffix[1:].upper() or "this file"


def _undecodable(p: Path, native_error: Exception | None) -> AudioError:
    """The error for a file libsndfile can't read when FFmpeg isn't available."""
    env = os.environ.get("AUDIO_MCP_FFMPEG", "").strip()
    missing = (
        f"AUDIO_MCP_FFMPEG is set to '{env}', which is not an executable file."
        if env
        else "FFmpeg was not found on PATH."
    )
    if native_error is None:
        return AudioError(
            f"Could not decode '{p.name}': {_label(p)} files are read through FFmpeg, and {missing} "
            f"{FFMPEG_INSTALL_HINT} Without FFmpeg, supported formats are {NATIVE_FORMATS}."
        )
    return AudioError(
        f"Could not decode '{p.name}' ({native_error}). Supported: {NATIVE_FORMATS}. Other formats (M4A/AAC, WMA, "
        f"MP4/MOV/WebM video) open automatically when FFmpeg is installed; {missing} {FFMPEG_INSTALL_HINT}"
    )


def _native_info(p: Path):
    """(soundfile info, None) if libsndfile reads p, else (None, error). Known FFmpeg-only containers skip it."""
    if not p.is_file():
        raise AudioError(f"File not found: {p}")
    if p.suffix.lower() in FFMPEG_EXTENSIONS:
        return None, None
    try:
        return sf.info(str(p)), None
    except Exception as e:
        return None, e


def open_source(p: Path) -> Source:
    """How to read p (which has already passed resolve_path): directly, or decoded by FFmpeg if libsndfile can't."""
    info, err = _native_info(p)
    if info is not None:
        return Source(p, p, info, info.format, info.subtype)
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise _undecodable(p, err)
    dec = _decode(p, ffmpeg)
    try:
        info = sf.info(str(dec.path))
    except Exception as e:
        raise AudioError(f"Could not read FFmpeg's output for '{p.name}': {e}") from None
    return Source(p, dec.path, info, _label(p), dec.codec.upper(), via_ffmpeg=True)


def file_info(path) -> dict:
    p = resolve_path(path)
    info, err = _native_info(p)
    if info is not None:
        fmt, subtype, sr, ch, dur = info.format, info.subtype, info.samplerate, info.channels, info.duration
    else:
        ffmpeg = ffmpeg_path()
        if ffmpeg is None:
            raise _undecodable(p, err)
        probe = _probe(p, ffmpeg)  # fast: ffprobe reads headers only
        if probe.get("sample_rate") and probe.get("channels") and probe.get("duration") is not None:
            subtype, sr, ch, dur = probe["codec"].upper(), probe["sample_rate"], probe["channels"], probe["duration"]
        else:  # no ffprobe: decode (cached, so the next tool call on this file reuses it)
            src = open_source(p)
            subtype, sr, ch, dur = src.subtype, src.info.samplerate, src.info.channels, src.info.duration
        fmt = _label(p)
    out = {
        "path": str(p),
        "format": fmt,
        "subtype": subtype,
        "sample_rate": sr,
        "channels": ch,
        "duration_s": round(dur, 3),
        "size_bytes": p.stat().st_size,
    }
    if info is None:
        out["decoded_via"] = "ffmpeg"
    return out


def list_audio(directory, limit: int = 200) -> dict:
    d = resolve_path(directory)
    if not d.is_dir():
        raise AudioError(f"Not a directory: {d}")
    have_ffmpeg = ffmpeg_path() is not None
    files, errors, need_ffmpeg = [], [], []
    for p in sorted(d.rglob("*")):
        ext = p.suffix.lower()
        if (ext not in AUDIO_EXTENSIONS and ext not in FFMPEG_EXTENSIONS) or not p.is_file():
            continue
        if ext in FFMPEG_EXTENSIONS and not have_ffmpeg:
            need_ffmpeg.append(str(p))
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
    if need_ffmpeg:
        out["needs_ffmpeg"] = {
            "count": len(need_ffmpeg),
            "files": need_ffmpeg[:20],
            "hint": f"These formats open automatically once FFmpeg is installed. {FFMPEG_INSTALL_HINT}",
        }
    return out


def load(path, start: float | None = None, end: float | None = None) -> Audio:
    p = resolve_path(path)
    src = open_source(p)
    sr, total = src.info.samplerate, src.info.duration
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
    data, sr = sf.read(str(src.read_path), start=int(round(s * sr)), stop=stop, dtype="float32", always_2d=True)
    if len(data) == 0:
        raise AudioError("The selected range contains no samples.")
    bad = ~np.isfinite(data)
    if bad.any():
        first = int(np.flatnonzero(bad.any(axis=1))[0])
        raise AudioError(
            f"The audio contains {int(bad.sum())} NaN or infinite sample values (first at {s + first / sr:.3f}s). "
            "The file is corrupt or was written by software that overflowed; re-export it from the source."
        )
    return Audio(data, sr, p, src.format, src.subtype, s)


def default_output(source: Path, tag: str) -> Path:
    suffix = source.suffix.lower() if source.suffix.lower() in DEFAULT_SUBTYPES else ".wav"
    candidate = source.with_name(f"{source.stem}_{tag}{suffix}")
    n = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}_{tag}{n}{suffix}")
        n += 1
    return candidate


def write_chunked(path, samples: np.ndarray, sr: int, subtype: str, frames: int = 65536) -> None:
    """Write in blocks. libsndfile's Vorbis encoder overflows the 1 MB Windows stack when handed about a minute of
    audio in one call (a hard process crash, not an exception); blocks of 65536 frames are safe at any length."""
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with sf.SoundFile(str(path), "w", sr, channels, subtype=subtype) as f:
        for i in range(0, len(samples), frames):
            f.write(samples[i : i + frames])


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
        write_chunked(p, samples, sr, subtype)
    except Exception as e:
        raise AudioError(f"Could not write {p.name} as {subtype}: {e}") from None
    return p, warnings
