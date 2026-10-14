"""Audit the audio of a Godot 4 project: inventory, import settings, loudness per category, per-file problems and
broken or unused res:// references.

Godot 4 import format (sources accessed 2026-10-13):
- Every imported asset gets an INI-style ``<file>.import`` sidecar next to it ([remap] with importer/type/uid/path/
  valid, [deps], [params] with the importer options). Imported data lives in the hidden ``res://.godot/imported/``.
  https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/import_process.html
- WAV (``importer="wav"``): ``edit/loop_mode`` int enum "Detect From WAV,Disabled,Forward,Ping-Pong,Backward"
  (default 0), ``edit/loop_begin`` / ``edit/loop_end`` in samples (end -1 = end of file), ``edit/trim`` (cuts the
  ends below -50 dB, 500-sample fades), ``edit/normalize`` (peak to 0 dB), ``force/mono``, ``force/max_rate`` +
  ``force/max_rate_hz``, ``force/8_bit``, ``compress/mode``. With loop_mode 0 the importer loops only if the file
  has a RIFF ``smpl`` chunk with a loop of type 0/1/2 (forward/ping-pong/backward); otherwise looping is disabled.
  https://raw.githubusercontent.com/godotengine/godot/master/editor/import/resource_importer_wav.cpp
  https://raw.githubusercontent.com/godotengine/godot/4.0-stable/editor/import/resource_importer_wav.cpp
  https://docs.godotengine.org/en/stable/classes/class_resourceimporterwav.html
  (4.2's docs describe loop_begin/end in seconds, 4.3+ in samples; we only use them for the seam position and accept
  either reading being slightly off, see `_wav_loop_region`.)
- Ogg Vorbis (``importer="oggvorbisstr"``) and MP3 (``importer="mp3"``): ``loop`` bool (default false),
  ``loop_offset`` float seconds, plus ``bpm``, ``beat_count``, ``bar_beats``.
  https://raw.githubusercontent.com/godotengine/godot/master/modules/vorbis/resource_importer_ogg_vorbis.cpp
  https://raw.githubusercontent.com/godotengine/godot/4.3-stable/modules/minimp3/resource_importer_mp3.cpp
  https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_audio_samples.html
- Folders containing a ``.gdignore`` file are not imported.
  https://docs.godotengine.org/en/stable/tutorials/best_practices/project_organization.html
Unknown keys are ignored and unparseable values are treated as "unknown", never as an error.
"""

from __future__ import annotations

import fnmatch
import os
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .analysis import analyze, edge_stats
from .audio_io import Audio, AudioError, _info, load, resolve_path
from .dsp import r

AUDIO_EXT = (".wav", ".ogg", ".mp3")  # what Godot 4 imports as AudioStream
TEXT_EXT = (".tscn", ".tres", ".gd", ".cs")
CATEGORIES = ("ui", "sfx", "voice", "music", "ambience", "uncategorized")
LONG_FORM = {"music", "ambience"}  # expected to loop; compared by integrated loudness
SYNONYMS = {
    "music": {"music", "musics", "bgm", "bgms", "soundtrack", "soundtracks", "ost", "song", "songs"},
    "ambience": {
        "ambience",
        "ambiences",
        "ambiance",
        "ambient",
        "ambients",
        "amb",
        "atmos",
        "atmosphere",
        "atmospheres",
    },
    "voice": {
        "voice",
        "voices",
        "vo",
        "vox",
        "dialogue",
        "dialogues",
        "dialog",
        "dialogs",
        "speech",
        "barks",
        "narration",
    },
    "ui": {"ui", "gui", "hud", "menu", "menus", "interface"},
    "sfx": {"sfx", "fx", "soundfx", "effects", "se"},
}
MAX_INVENTORY = 5000
MAX_TEXT_FILES = 5000
MAX_TEXT_BYTES = 10_000_000
MAX_ISSUES = 60
LIST_CAP = 10
OUTLIER_DB = 6.0
LEADING_SILENCE_S = 0.05
SEVERITY_ORDER = {"problem": 0, "warning": 1, "info": 2}

# Rule-of-thumb only, not a standard. Whole-game mix references exist (e.g. about -24 LUFS integrated, true peak
# <= -1 dBTP for console titles, per the IESD "Mix Ref Levels" document,
# https://www.audiogang.org/wp-content/uploads/2015/04/IESD-Mix-Ref-Levels-v03.02.pdf and
# https://designingsound.org/2013/02/loudness-in-game-audio/ , accessed 2026-10-13), but there's no published standard
# for per-asset levels by category; these offsets are a common starting point that the final mix (buses,
# volume_db) refines.
GUIDELINE = (
    "Guideline, not a standard: keep each category consistent within itself and normalize per category, never "
    "music and SFX to one target. Starting points by loudest moment (LUFS momentary max) relative to SFX: voice "
    "about 0 to +3 LU (dialogue is the anchor), UI -6 to 0, music -12 to -4, ambience -18 to -8. The real balance "
    "is set by bus and player volume_db, which this check doesn't see."
)
GUIDE_RANGES = {"voice": (0, 3), "ui": (-6, 0), "music": (-12, -4), "ambience": (-18, -8)}

QUOTED_RES = re.compile(r"""["'](res://[^"'\r\n]+?\.(?:wav|ogg|mp3))["']""", re.IGNORECASE)
QUOTED_UID = re.compile(r"""["'](uid://[a-z0-9]+)["']""", re.IGNORECASE)
EXT_RESOURCE = re.compile(r"\[ext_resource\b([^\]]*)\]")
ATTR = re.compile(r'(\w+)="([^"]*)"')
TEMPLATE = re.compile(r"%[-+ 0-9.]*[sdifx]|\{[^}]*\}|\$\w+|\*")


# ---------------------------------------------------------------- project walk


def _find_project(project_dir: str) -> tuple[Path, int | None]:
    root = resolve_path(project_dir)
    if not root.is_dir():
        raise AudioError(f"Not a directory: {root}. Pass the Godot project folder (the one containing project.godot).")
    cfg = root / "project.godot"
    if not cfg.is_file():
        raise AudioError(
            f"No project.godot in {root}; this isn't a Godot project root. Pass the folder that contains project.godot."
        )
    m = re.search(r"^config_version\s*=\s*(\d+)", _read_text(cfg, 1_000_000), re.MULTILINE)
    version = int(m.group(1)) if m else None
    if version is not None and version < 5:
        raise AudioError(
            f"project.godot has config_version={version}, i.e. Godot 3.x. This check reads the Godot 4 import format; "
            "open the project in Godot 4 first, or use audio_batch_survey on its audio folder."
        )
    return root, version


def _read_text(p: Path, limit: int = MAX_TEXT_BYTES) -> str:
    with open(p, "rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


def _walk(root: Path):
    """(audio files, text files) under root, skipping hidden folders (.godot, .git, ...) and .gdignore'd ones."""
    audio, text, ignored_audio = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        skip = os.path.basename(dirpath).startswith(".") and Path(dirpath) != root
        if not skip and "project.godot" in filenames and Path(dirpath) != root:
            skip = True  # a nested Godot project (e.g. a sample inside addons) has its own res://
        if skip or ".gdignore" in filenames:
            ignored_audio += [Path(dirpath, f) for f in filenames if f.lower().endswith(AUDIO_EXT)]
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            low = f.lower()
            if low.endswith(AUDIO_EXT):
                audio.append(Path(dirpath, f))
            elif low.endswith(TEXT_EXT) or low == "project.godot":
                text.append(Path(dirpath, f))
    return audio, text, ignored_audio


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


# ---------------------------------------------------------------- categories


def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def _keyword_category(tokens: list[str]) -> str | None:
    for t in tokens:
        for cat, words in SYNONYMS.items():
            if t in words:
                return cat
    return None


def _normalize_category(value: str) -> str:
    v = str(value).strip().lower()
    if v in CATEGORIES:
        return v
    cat = _keyword_category([v]) or {"ambient": "ambience", "dialogue": "voice"}.get(v)
    if cat:
        return cat
    raise AudioError(f"Unknown category '{value}'. Use one of: {', '.join(CATEGORIES)}.")


def _caller_rules(categories) -> list[tuple[str, str]]:
    if categories is None:
        return []
    if not isinstance(categories, dict):
        raise AudioError(
            'categories must be an object mapping a glob or folder to a category, e.g. {"audio/sfx": "sfx"}.'
        )
    rules = []
    for pattern, cat in categories.items():
        pat = str(pattern).strip().replace("\\", "/")
        pat = pat[len("res://") :] if pat.lower().startswith("res://") else pat
        rules.append((pat.strip("/").lower(), _normalize_category(cat)))
    return sorted(rules, key=lambda rc: -len(rc[0]))  # most specific first


def categorize(rel: str, rules: list[tuple[str, str]]) -> tuple[str, str]:
    """(category, source) for a res-relative path. source: 'caller', 'folder', 'name' or 'none'."""
    low = rel.lower()
    for pat, cat in rules:
        if fnmatch.fnmatchcase(low, pat) or low.startswith(pat + "/") or (pat and fnmatch.fnmatchcase(low, pat + "/*")):
            return cat, "caller"
    parts = rel.split("/")
    for folder in reversed(parts[:-1]):  # deepest folder wins
        cat = _keyword_category(_tokens(folder))
        if cat:
            return cat, "folder"
    cat = _keyword_category(_tokens(Path(parts[-1]).stem))
    return (cat, "name") if cat else ("uncategorized", "none")


# ---------------------------------------------------------------- import settings


def _value(raw: str):
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_import(text: str) -> dict[str, dict]:
    """INI-like .import file -> {section: {key: value}}. Continuation lines of multi-line values are ignored."""
    sections: dict[str, dict] = {"": {}}
    cur = sections[""]
    for line in text.splitlines():
        if line.startswith("["):
            m = re.match(r"^\[([^\]]+)\]\s*$", line)
            if m:
                cur = sections.setdefault(m.group(1).strip(), {})
            continue
        m = re.match(r"^([A-Za-z0-9_./-]+)\s*=(.*)$", line)
        if m:
            cur[m.group(1)] = _value(m.group(2))
    return sections


def wav_smpl_loop(p: Path) -> tuple[str, int, int] | None:
    """Loop (kind, start, end) from a WAV's RIFF 'smpl' chunk, read the way Godot's importer does; None if none."""
    kinds = {0: "forward", 1: "ping-pong", 2: "backward"}
    try:
        with open(p, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            for _ in range(10000):
                ch = f.read(8)
                if len(ch) < 8:
                    return None
                size = struct.unpack("<I", ch[4:])[0]
                if ch[:4] == b"smpl":
                    data = f.read(min(size, 60))
                    if len(data) < 52 or struct.unpack("<I", data[28:32])[0] == 0:
                        return None
                    kind, start, end = struct.unpack("<III", data[40:52])
                    return (kinds[kind], start, end) if kind in kinds else None
                f.seek(size + (size & 1), 1)
    except OSError:
        return None
    return None


def import_settings(p: Path) -> dict | None:
    """The Godot import settings that matter for audio, or None when there's no .import sidecar."""
    side = p.with_name(p.name + ".import")
    if not side.is_file():
        return None
    try:
        sec = parse_import(_read_text(resolve_path(side), 1_000_000))
    except (OSError, AudioError):
        return {"importer": None, "unreadable": True}
    remap, params = sec.get("remap", {}), sec.get("params", {})
    out = {
        "importer": remap.get("importer"),
        "valid": remap.get("valid", True) is not False,
        "uid": remap.get("uid") if isinstance(remap.get("uid"), str) else None,
    }
    imp = out["importer"]
    if imp == "wav":
        mode = params.get("edit/loop_mode", 0 if "edit/loop" not in params else None)
        if mode is None:  # Godot 3-style bool
            out["loop"] = params.get("edit/loop") is True
            out["loop_kind"] = "forward" if out["loop"] else None
        elif isinstance(mode, int) and not isinstance(mode, bool) and 0 <= mode <= 4:
            if mode == 0:
                smpl = wav_smpl_loop(p)
                out["loop_source"] = "Detect From WAV"
                out["loop"] = smpl is not None
                out["loop_kind"] = smpl[0] if smpl else None
                if smpl:
                    out["loop_begin"], out["loop_end"] = smpl[1], smpl[2] + 1  # smpl end is inclusive
            else:
                out["loop"] = mode >= 2
                out["loop_kind"] = {1: None, 2: "forward", 3: "ping-pong", 4: "backward"}[mode]
                b, e = params.get("edit/loop_begin", 0), params.get("edit/loop_end", -1)
                out["loop_begin"] = b if isinstance(b, int) and b > 0 else 0
                out["loop_end"] = e if isinstance(e, int) and e > 0 else None
        else:
            out["loop"] = None  # unknown value: say so rather than guess
        for key, name in (("edit/trim", "trim"), ("edit/normalize", "normalize"), ("force/mono", "force_mono")):
            out[name] = params.get(key) is True
        rate = params.get("force/max_rate_hz")
        if params.get("force/max_rate") is True and isinstance(rate, (int, float)) and rate > 0:
            out["max_rate_hz"] = int(rate)
    elif imp in ("oggvorbisstr", "mp3"):
        loop = params.get("loop")
        out["loop"] = loop if isinstance(loop, bool) else None
        out["loop_kind"] = "forward" if loop is True else None
        off = params.get("loop_offset", 0)
        out["loop_offset_s"] = float(off) if isinstance(off, (int, float)) and not isinstance(off, bool) else 0.0
    else:
        out["loop"] = None  # "keep", "skip" or a custom importer
    return out


def _loop_fix(ext: str, on: bool) -> str:
    if ext == ".wav":
        return f"Import dock -> Edit > Loop Mode: {'Forward' if on else 'Disabled'} -> Reimport"
    return f"Import dock -> Loop: {'On' if on else 'Off'} -> Reimport"


# ---------------------------------------------------------------- per-file analysis


def _seam(p: Path, sr: int, frames: int, imp: dict) -> dict:
    """edges.loop_seam across the loop point: the end of the loop region followed by its start."""
    if imp.get("loop_offset_s"):
        begin, end = int(round(imp["loop_offset_s"] * sr)), frames
    else:
        begin = int(imp.get("loop_begin") or 0)
        end = imp.get("loop_end") or frames
    end = min(int(end), frames)
    if not 0 <= begin < end - 16:  # loop points we can't interpret (e.g. seconds vs samples): use the whole file
        begin, end = 0, frames
    w = int(0.25 * sr)
    head = load(p, begin / sr, min(end, begin + w) / sr).samples
    tail = load(p, max(begin, end - w) / sr, None if end >= frames else end / sr).samples
    seam = edge_stats(Audio(np.concatenate([head, tail]), sr))["loop_seam"]
    seam["loop_s"] = [r(begin / sr, 3), r(end / sr, 3)]
    return seam


def _analyze_file(p: Path, max_seconds: float) -> dict:
    audio = load(p, end=max_seconds)  # end beyond the file is clamped
    rep = analyze(audio, pitch=False, rhythm=False)
    info = _info(p)
    lv, si, ed = rep["levels"], rep["silence"], rep["edges"]
    return {
        "sr": info.samplerate,
        "frames": info.frames,
        "duration_s": info.duration,
        "channels": info.channels,
        "truncated": info.duration > max_seconds + 1e-6,
        "lufs": lv["lufs_integrated"],
        "lufs_mmax": lv["lufs_momentary_max"],
        "peak_dbfs": lv["peak_dbfs"],
        "clipped": lv["clipped_samples"],
        "clip_events": lv["clip_events"],
        "dc": lv["dc_offset"],
        "silent": bool(si.get("all_silent")),
        "leading_s": si.get("leading_s") or 0.0,
        "start_amp": ed["start_amplitude"] or 0.0,
        "end_amp": ed["end_amplitude"] or 0.0,
        "mono_identical": rep.get("stereo", {}).get("width", "").startswith("mono"),
    }


# ---------------------------------------------------------------- references


def scan_references(root: Path, text_files: list[Path], uid_map: dict[str, str]):
    """{rel audio path: [where]} for literal res:// paths, plus uid hits, stale ext_resource paths and patterns."""
    refs: dict[str, list[str]] = defaultdict(list)
    by_uid: dict[str, list[str]] = defaultdict(list)
    stale: dict[str, tuple[str, str]] = {}  # referenced path -> (uid target, where)
    patterns: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for tf in text_files[:MAX_TEXT_FILES]:
        try:
            tf = resolve_path(tf)
            if tf.stat().st_size > MAX_TEXT_BYTES:
                skipped += 1
                continue
            text = _read_text(tf)
        except (OSError, AudioError):
            skipped += 1
            continue
        relf = _rel(root, tf)
        comment = "#" if tf.suffix == ".gd" else "//" if tf.suffix == ".cs" else None
        for n, line in enumerate(text.splitlines(), 1):
            if "res://" not in line and "uid://" not in line:
                continue
            if comment and line.lstrip().startswith(comment):
                continue  # commented-out code isn't a live reference
            where = f"{relf}:{n}"
            stale_paths = set()
            for m in EXT_RESOURCE.finditer(line):
                attrs = dict(ATTR.findall(m.group(1)))
                path, uid = attrs.get("path", ""), attrs.get("uid", "")
                if path.lower().endswith(AUDIO_EXT) and uid in uid_map and path[6:] != uid_map[uid]:
                    stale[path[6:]] = (uid_map[uid], where)
                    stale_paths.add(path)
            for m in QUOTED_RES.finditer(line):
                path = m.group(1)
                if path in stale_paths:
                    continue
                rel = path[len("res://") :]
                if TEMPLATE.search(rel):
                    patterns[TEMPLATE.sub("*", rel)].append(where)
                else:
                    refs[rel].append(where)
            for m in QUOTED_UID.finditer(line):
                if m.group(1) in uid_map:
                    by_uid[uid_map[m.group(1)]].append(where)
    return refs, by_uid, stale, patterns, skipped + max(0, len(text_files) - MAX_TEXT_FILES)


# ---------------------------------------------------------------- the check


def check_project(project_dir: str, categories: dict | None = None, max_files: int = 150, max_seconds: float = 90.0):
    root, version = _find_project(project_dir)
    rules = _caller_rules(categories)
    max_files = int(max(1, min(max_files, 1000)))
    max_seconds = float(min(max(max_seconds, 5.0), 600.0))
    audio_paths, text_paths, ignored_audio = _walk(root)
    if not audio_paths:
        raise AudioError(
            f"No audio files (.wav, .ogg, .mp3) found in the Godot project {root} (hidden and .gdignore'd folders "
            "are skipped)."
        )
    notes = []
    if len(audio_paths) > MAX_INVENTORY:
        notes.append(f"Inventory capped at {MAX_INVENTORY} of {len(audio_paths)} audio files.")
        audio_paths = audio_paths[:MAX_INVENTORY]

    issues: list[dict] = []

    def add(severity, res, category, problem, fix=None, **extra):
        item = {"severity": severity, "file": res, "category": category, "problem": problem}
        if fix:
            item["fix"] = fix
        item.update(extra)
        issues.append(item)

    files, outside = [], 0
    for p in audio_paths:
        rel = _rel(root, p)
        try:
            resolve_path(p)  # a symlink can point outside the sandbox: skip it rather than read it
        except AudioError:
            outside += 1
            continue
        cat, source = categorize(rel, rules)
        files.append({"path": p, "rel": rel, "res": "res://" + rel, "ext": p.suffix.lower(), "cat": cat, "src": source})
    if outside:
        notes.append(
            f"Skipped {outside} audio file(s) linking outside the allowed directories (AUDIO_MCP_ALLOWED_DIRS)."
        )
    if not files:
        raise AudioError(
            "None of the project's audio files are inside the allowed directories (AUDIO_MCP_ALLOWED_DIRS)."
        )
    uid_map = {}
    for f in files:
        f["imp"] = import_settings(f["path"])
        if f["imp"] and f["imp"].get("uid"):
            uid_map[f["imp"]["uid"]] = f["rel"]

    # ------------------------------------------------ analysis
    analyzed = 0
    unreadable = []
    for f in files:
        if analyzed >= max_files:
            break
        try:
            f["a"] = _analyze_file(resolve_path(f["path"]), max_seconds)
            analyzed += 1
        except AudioError as e:
            unreadable.append(f"{f['res']}: {str(e)[:120]}")
    if len(files) > analyzed + len(unreadable):
        notes.append(
            f"Analyzed the first {analyzed} of {len(files)} files (sorted by path; max_files={max_files}). Import "
            "settings and references were checked for all of them."
        )
    if any(f.get("a", {}).get("truncated") for f in files):
        notes.append(
            f"Files longer than {max_seconds:g}s were measured on their first {max_seconds:g}s (loudness, clipping, "
            "DC); loop seams always use the real loop points."
        )

    for f in files:
        imp, a, cat, res = f["imp"], f.get("a"), f["cat"], f["res"]
        if imp and not imp.get("valid", True):
            add(
                "problem",
                res,
                cat,
                "Godot failed to import this file (valid=false in its .import).",
                {
                    "editor": "Select it in the FileSystem dock and check the Output panel; re-export the file and Reimport"
                },
            )
        if imp and imp.get("importer") in ("keep", "skip"):
            add("info", res, cat, f"Import mode is '{imp['importer']}', so Godot doesn't load it as an AudioStream.")
        loops = imp.get("loop") if imp else None
        f["loops"] = loops
        if a is None:
            continue
        if a["silent"]:
            add("problem", res, cat, "The file is silent.", {"editor": "Replace it with the intended sound"})
            continue
        # Godot's Normalize import option scales the peak to 0 dB, so the game hears a different level.
        gain = -a["peak_dbfs"] if imp and imp.get("normalize") and a["peak_dbfs"] is not None else 0.0
        f["lufs"] = None if a["lufs"] is None else a["lufs"] + gain
        f["lufs_mmax"] = None if a["lufs_mmax"] is None else a["lufs_mmax"] + gain
        f["sr_eff"] = min(a["sr"], imp["max_rate_hz"]) if imp and imp.get("max_rate_hz") else a["sr"]
        one_shot = cat not in LONG_FORM
        if a["clipped"]:
            add(
                "problem",
                res,
                cat,
                f"Digital clipping: {a['clipped']} clipped samples in {a['clip_events']} places.",
                {"editor": "Re-export from the source at a lower level; turning it down won't remove the distortion"},
            )
        if a["dc"] > 0.005:
            add(
                "warning",
                res,
                cat,
                f"DC offset of {a['dc']} (can click when the sound starts or stops).",
                {"audio_process_operations": [{"op": "remove_dc"}]},
            )
        trims = bool(imp and imp.get("trim"))
        if one_shot and not loops and a["leading_s"] > LEADING_SILENCE_S and not trims:
            ops = [{"op": "trim_silence", "pad_ms": 5}]
            fix = {"audio_process_operations": ops}
            if f["ext"] == ".wav":
                fix["editor"] = "or Import dock -> Edit > Trim: On -> Reimport"
            add(
                "warning",
                res,
                cat,
                f"{a['leading_s'] * 1000:.0f} ms of silence before the sound starts; it plays late.",
                fix,
            )
        if one_shot and not loops and not a["truncated"]:
            if a["end_amp"] > 0.05:
                add(
                    "warning",
                    res,
                    cat,
                    f"Ends abruptly at amplitude {a['end_amp']}; clicks when it stops.",
                    {"audio_process_operations": [{"op": "fade_out", "seconds": 0.01}]},
                )
            if a["start_amp"] > 0.05:
                add(
                    "info",
                    res,
                    cat,
                    f"Starts at amplitude {a['start_amp']}; may click (fine if it's a hard attack).",
                    {"audio_process_operations": [{"op": "fade_in", "seconds": 0.003}]},
                )

        # ------------------------------------------------ loops
        if cat in LONG_FORM and loops is False:
            how = " (Loop Mode is 'Detect From WAV' and the file has no loop points)" if imp.get("loop_source") else ""
            sev = "warning" if a["duration_s"] >= 10 else "info"
            add(
                sev,
                res,
                cat,
                f"{cat.capitalize()} doesn't loop in its import settings{how}; it will stop after "
                f"{a['duration_s']:.1f}s unless code sets loop or it's a one-off stinger.",
                {"editor": _loop_fix(f["ext"], True)},
            )
        if cat not in LONG_FORM and cat != "uncategorized" and loops:
            add(
                "warning",
                res,
                cat,
                "Loops in its import settings: it plays forever and never emits `finished` (fine only for "
                "deliberate loops like engines or fire).",
                {"editor": _loop_fix(f["ext"], False)},
            )
        if loops and imp.get("loop_kind") != "ping-pong":
            try:
                seam = _seam(resolve_path(f["path"]), a["sr"], a["frames"], imp)
            except AudioError:
                seam = None
            if seam and seam["likely_click"]:
                whole = seam["loop_s"][0] == 0 and abs(seam["loop_s"][1] - a["duration_s"]) < 0.01
                fix = (
                    {"audio_process_operations": [{"op": "loop_crossfade", "seconds": 0.05}]}
                    if whole
                    else {"editor": "Move the loop points to zero crossings (Advanced Import Settings)"}
                )
                add(
                    "warning",
                    res,
                    cat,
                    f"Loop seam clicks: the waveform jumps {seam['sample_jump']} at the loop point (typical step "
                    f"{seam['typical_step']}).",
                    fix,
                )
            elif seam and abs(seam["level_jump_db"] or 0) > 6:
                add(
                    "warning",
                    res,
                    cat,
                    f"Level jumps {seam['level_jump_db']:+.1f} dB at the loop point (fade-out at the end, or a "
                    "loud intro?).",
                    {"editor": "Loop a section without the intro/fade (loop offset or loop points)"},
                )

    # ------------------------------------------------ loudness and consistency per category
    stats = {}
    for cat in CATEGORIES:
        group = [f for f in files if f["cat"] == cat]
        if not group:
            continue
        measured = [f for f in group if f.get("lufs_mmax") is not None]
        st = {"files": len(group), "analyzed": len(measured)}
        loops_n = sum(1 for f in group if f["loops"])
        st["looping"] = loops_n
        stats[cat] = st
        if not measured:
            continue
        durations = [f["a"]["duration_s"] for f in measured]
        long_measure = cat in LONG_FORM or float(np.median(durations)) >= 5.0
        if long_measure and sum(f["lufs"] is not None for f in measured) >= 0.8 * len(measured):
            key, unit = "lufs", "LUFS integrated"
        else:
            key, unit = "lufs_mmax", "LUFS momentary max"
        vals = [(f, f[key]) for f in measured if f[key] is not None]
        if not vals:
            continue
        arr = np.array([v for _, v in vals])
        med = float(np.median(arr))
        spread = float(np.percentile(arr, 90) - np.percentile(arr, 10)) if len(arr) > 1 else 0.0
        st.update(
            {
                "measure": unit,
                "median": r(med, 1),
                "spread_p10_p90_db": r(spread, 1),
                "range": [r(arr.min(), 1), r(arr.max(), 1)],
            }
        )
        st["median_lufs_momentary_max"] = r(float(np.median([f["lufs_mmax"] for f in measured])), 1)
        if key == "lufs":
            op = {"op": "normalize_lufs", "lufs": round(med, 1), "limit": True}
        else:
            op = {"op": "normalize_lufs", "lufs": round(med, 1), "measure": "momentary_max", "ceiling_db": -1}
        outliers = [(f, v) for f, v in vals if abs(v - med) > OUTLIER_DB] if len(vals) >= 3 else []
        for f, v in outliers:
            fix = {"audio_process_operations": [op]}
            if f["imp"] and f["imp"].get("normalize"):
                fix["editor"] = "first turn off Import dock -> Edit > Normalize (it would undo the change), Reimport"
            add(
                "warning",
                f["res"],
                cat,
                f"{abs(v - med):.1f} dB {'louder' if v > med else 'quieter'} than the other {cat} files "
                f"({v:.1f} vs median {med:.1f} {unit}).",
                fix,
            )
        if len(vals) == 2 and abs(vals[0][1] - vals[1][1]) > OUTLIER_DB:
            add(
                "info",
                [vals[0][0]["res"], vals[1][0]["res"]],
                cat,
                f"The two {cat} files differ by {abs(vals[0][1] - vals[1][1]):.1f} dB ({unit}).",
            )
        if outliers or spread > OUTLIER_DB:
            st["suggested_batch"] = _batch_suggestion(root, files, cat, op, spread)
        rates = Counter(f["sr_eff"] for f in measured)
        st["sample_rates"] = dict(rates)
        if len(rates) > 1:
            common = rates.most_common(1)[0][0]
            odd = [f["res"] for f in measured if f["sr_eff"] != common]
            add(
                "info",
                odd[:LIST_CAP],
                cat,
                f"Mixed sample rates in {cat}: most are {common} Hz, {len(odd)} differ (Godot resamples at runtime).",
                {"audio_process_operations": [{"op": "resample", "rate": common}]},
            )
        if cat == "sfx":
            stereo = [f["res"] for f in measured if f["a"]["channels"] > 1 and not (f["imp"] or {}).get("force_mono")]
            if stereo:
                add(
                    "info",
                    stereo[:LIST_CAP],
                    cat,
                    f"{len(stereo)} stereo SFX: sounds played by AudioStreamPlayer2D/3D are positioned in mono, so "
                    "stereo only costs memory there.",
                    {
                        "editor": "Import dock -> Force > Mono (WAV) -> Reimport",
                        "audio_process_operations": [{"op": "to_mono"}],
                    },
                )

    # ------------------------------------------------ cross-category guideline (info only)
    anchor = stats.get("sfx", {}).get("median_lufs_momentary_max")
    if anchor is not None:
        for cat, (lo, hi) in GUIDE_RANGES.items():
            v = stats.get(cat, {}).get("median_lufs_momentary_max")
            if v is None:
                continue
            rel = v - anchor
            stats[cat]["vs_sfx_lu"] = r(rel, 1)
            if rel > hi + 3 or rel < lo - 6:
                add(
                    "info",
                    None,
                    cat,
                    f"{cat} files peak {rel:+.1f} LU relative to SFX (guideline {lo:+d} to {hi:+d} LU); fine if the "
                    "bus or player volume_db compensates.",
                )

    # ------------------------------------------------ import sidecars and references
    never = [f["res"] for f in files if f["imp"] is None]
    if never:
        add(
            "info",
            never[:LIST_CAP],
            None,
            f"{len(never)} file(s) have no .import file: never imported, so the game can't load them yet.",
            {"editor": "Open the project in the Godot editor (it imports new files automatically)"},
        )

    refs, by_uid, stale, patterns, skipped = scan_references(root, text_paths, uid_map)
    inventory = {f["rel"] for f in files}
    lower = {f["rel"].lower(): f["rel"] for f in files}
    ignored = {_rel(root, p).lower() for p in ignored_audio if root in p.parents}
    missing = []
    for rel, where in sorted(refs.items()):
        if rel in inventory:
            continue
        cat, _ = categorize(rel, rules)
        at = ", ".join(where[:3]) + (" ..." if len(where) > 3 else "")
        if rel.lower() in lower:
            add(
                "problem",
                "res://" + rel,
                cat,
                f"Referenced with the wrong letter case (file is res://{lower[rel.lower()]}), at {at}. Works in the "
                "editor on Windows/macOS but breaks in exported builds.",
                {"editor": "Fix the path's case where it's referenced"},
            )
        elif rel.lower() in ignored:
            add(
                "problem",
                "res://" + rel,
                cat,
                f"Referenced at {at}, but the file is in a hidden or .gdignore'd folder, so Godot never imports it.",
                {"editor": "Move the file into an imported folder and update the reference"},
            )
        else:
            missing.append("res://" + rel)
            add(
                "problem",
                "res://" + rel,
                cat,
                f"Broken reference: used at {at}, but the file doesn't exist.",
                {
                    "editor": "Restore the file, or fix the path (opening the scene shows Godot's Fix Dependencies dialog)"
                },
            )
    for rel, (target, where) in sorted(stale.items()):
        add(
            "info",
            "res://" + rel,
            None,
            f"Scene path at {where} is stale; Godot finds res://{target} through its uid. Re-save the scene to "
            "update the path.",
        )
    referenced = set(refs) | set(by_uid) | {t for t, _ in stale.values()}
    for pat in patterns:
        referenced |= {rel for rel in inventory if fnmatch.fnmatchcase(rel, pat)}
    unreferenced = [f["res"] for f in files if f["rel"] not in referenced]
    if unreferenced and (refs or by_uid or patterns):
        add(
            "info",
            unreferenced[:LIST_CAP],
            None,
            f"{len(unreferenced)} audio file(s) aren't referenced by any scene, resource or script (possibly unused; "
            "files loaded through paths built at runtime can't be seen).",
        )

    # ------------------------------------------------ output
    issues.sort(key=lambda i: (SEVERITY_ORDER[i["severity"]], str(i["file"])))
    counts = Counter(i["severity"] for i in issues)
    cat_counts = Counter(f["cat"] for f in files)
    summary = [
        f"{counts['problem']} problems, {counts['warning']} warnings, {counts['info']} info",
        f"{len(files)} audio files ({analyzed} analyzed): "
        + ", ".join(f"{c} {cat_counts[c]}" for c in CATEGORIES if cat_counts[c]),
        f"References: {sum(map(len, refs.values()))} res:// paths in {len(text_paths)} scene/resource/script files; "
        f"{len(missing)} broken, {len(unreferenced)} audio files unreferenced",
    ]
    out = {
        "project": str(root),
        "godot_config_version": version,
        "summary": summary,
        "categories": stats,
        "issues": issues[:MAX_ISSUES],
    }
    if len(issues) > MAX_ISSUES:
        out["issues_truncated"] = len(issues) - MAX_ISSUES
    out["inventory"] = [_inventory_line(f) for f in files[:150]]
    if len(files) > 150:
        out["inventory_truncated"] = len(files) - 150
    out["references"] = {
        "broken": missing[:LIST_CAP],
        "unreferenced": unreferenced[:LIST_CAP],
        "dynamic_patterns": sorted(patterns)[:LIST_CAP],
        **({"unreadable_text_files": skipped} if skipped else {}),
    }
    out["guideline"] = GUIDELINE
    notes.append(f"res:// paths are relative to {root}; audio_process needs the absolute path.")
    if unreadable:
        out["unreadable"] = unreadable[:LIST_CAP]
    out["notes"] = notes
    return out


def _inventory_line(f) -> str:
    a, imp = f.get("a"), f["imp"]
    parts = [f["res"], f["cat"] + (" (from name)" if f["src"] == "name" else "")]
    if a:
        layout = "mono" if a["channels"] == 1 else f"{a['channels']}ch"
        parts.append(f"{a['duration_s']:.2f}s {a['sr']}Hz {layout}")
        lv = f.get("lufs") if f["cat"] in LONG_FORM else f.get("lufs_mmax")
        if lv is not None:
            parts.append(f"{lv:.1f} LUFS{'' if f['cat'] in LONG_FORM else ' Mmax'}")
    if imp is None:
        parts.append("not imported")
    else:
        loop = imp.get("loop")
        parts.append("loop:" + ("?" if loop is None else (imp.get("loop_kind") or "yes") if loop else "no"))
    return " | ".join(parts)


def _batch_suggestion(root: Path, files: list[dict], cat: str, op: dict, spread: float) -> dict:
    members = [f for f in files if f["cat"] == cat]
    paths = [f["path"] for f in members]
    common = Path(os.path.commonpath([str(p.parent) for p in paths]))
    under = [f for f in files if common in f["path"].parents]
    if all(f["cat"] == cat for f in under):
        target: str | list[str] = str(common)
    else:
        target = [str(p) for p in paths[:50]]
    return {
        "why": f"{cat} loudness spread {spread:.1f} dB (10th-90th percentile)",
        "audio_batch_process": {
            "target": target,
            "operations": [op],
            "output_dir": str(root.parent / f"{root.name}_audio_fixed" / cat),
        },
        "then": "Listen, then copy the results over the originals; Godot reimports them when the editor regains focus.",
    }
