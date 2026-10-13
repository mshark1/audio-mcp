"""MCP server exposing the audio tools."""

from __future__ import annotations

import functools

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import analysis, audio_io, batch, events, pitch, synth
from . import compare as compare_mod
from . import process as process_mod
from . import render as render_mod
from . import transcribe as transcribe_mod
from .audio_io import AudioError

INSTRUCTIONS = """Tools for perceiving and editing audio. You cannot hear audio directly; these tools turn sound into
numbers, images and note lists you can reason about.

Typical loop: audio_analyze (measurements + an `issues` list) -> audio_events / audio_view (what happens when) ->
audio_process (fix, writes a new file) -> audio_compare (verify what changed).
For many files: audio_batch_survey (consistency + suggested fixes) -> audio_batch_process.
Use absolute paths. Times are seconds. Levels are dBFS (0 = digital maximum). Loudness references:
-14 LUFS music streaming, -16 LUFS podcasts/voice, -23 LUFS broadcast (EBU R128).
Pitch/melody tools assume one note at a time; chords and full mixes give approximate results."""

mcp = MCPServer("audio", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES_FILE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)


def tool(**kwargs):
    """Register an MCP tool whose unexpected exceptions still reach the caller as a readable error.

    Expected problems raise AudioError (a ToolError) with advice. Anything else is a bug: by default the MCP SDK
    would hide it behind "Error executing tool", leaving the agent nothing to act on or report."""

    def register(fn):
        @functools.wraps(fn)
        def safe(*args, **kw):
            try:
                return fn(*args, **kw)
            except ToolError:
                raise
            except MemoryError:
                raise AudioError("Ran out of memory. Pass start/end to work on a shorter section.") from None
            except Exception as e:
                raise AudioError(
                    f"Internal error in {fn.__name__}: {type(e).__name__}: {e}. This is a bug in audio-mcp; "
                    "please report it at the project's issue tracker with the call and a description of the input."
                ) from e

        mcp.tool(**kwargs)(safe)
        return safe

    return register


@tool(annotations=READ_ONLY)
def audio_info(path: str) -> dict:
    """Fast metadata without decoding: format, sample rate, channels, duration.
    If `path` is a directory, lists the audio files inside it (recursively)."""
    p = audio_io.resolve_path(path)
    return audio_io.list_audio(p) if p.is_dir() else audio_io.file_info(p)


@tool(annotations=READ_ONLY)
def audio_analyze(
    path: str, start: float | None = None, end: float | None = None, pitch: bool = True, rhythm: bool = True
) -> dict:
    """Measure an audio file (or the start..end section): peak/true-peak/RMS/LUFS loudness, clipping, DC offset,
    a loudness sparkline over time, silence and noise floor, stereo correlation/width/balance, spectral balance
    (band energy, brightness, tonal vs noisy, strongest frequencies as notes), pitch, onsets/tempo,
    start/end clicks and loop-seam quality. Ends with `issues`: plain-language problems ranked by severity.
    Set pitch/rhythm to false to run faster."""
    return analysis.analyze(audio_io.load(path, start, end), pitch=pitch, rhythm=rhythm)


@tool(annotations=READ_ONLY)
def audio_timeline(path: str, segments: int = 24, start: float | None = None, end: float | None = None) -> dict:
    """How the sound changes over time: per-segment RMS, peak and brightness, plus a loudness sparkline.
    Streams the file, so it works on recordings of any length; use it to find where to zoom in."""
    return analysis.timeline(path, segments, start, end)


@tool(annotations=READ_ONLY)
def audio_view(
    path: str,
    kind: str = "both",
    start: float | None = None,
    end: float | None = None,
    freq_scale: str = "log",
    max_freq: float | None = None,
    pitch_overlay: bool = False,
    save_to: str | None = None,
) -> Image:
    """Render a picture of the sound. kind: 'both' (waveform above spectrogram), 'waveform' or 'spectrogram'.
    Spectrogram: time left to right, frequency bottom to top, brighter = louder. Horizontal lines are pitched
    tones/harmonics, vertical stripes are clicks/transients, diffuse haze is noise. Red dashed lines on the
    waveform mark 0 dBFS (clipping). Zoom with start/end; pitch_overlay draws the detected f0 in cyan.
    save_to optionally also writes the PNG to disk."""
    png = render_mod.render(audio_io.load(path, start, end), kind, freq_scale, max_freq, pitch_overlay)
    if save_to:
        out = audio_io.resolve_path(save_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png)
    return Image(data=png, format="png")


@tool(annotations=READ_ONLY)
def audio_melody(
    path: str, start: float | None = None, end: float | None = None, fmin: float = 50.0, fmax: float = 2000.0
) -> dict:
    """Transcribe a monophonic line (voice, whistle, solo instrument, synth lead, SFX pitch) into notes with
    start times, durations, note names and cents deviation, plus a compact text form.
    Narrow fmin/fmax to the expected range for better accuracy (e.g. bass 40-400, voice 80-1000)."""
    audio = audio_io.load(path, start, end)
    return pitch.melody(audio.mono, audio.sr, audio.offset, fmin, fmax)


@tool(annotations=READ_ONLY)
def audio_events(
    path: str,
    start: float | None = None,
    end: float | None = None,
    threshold_db: float | None = None,
    split_on_onsets: bool = True,
) -> dict:
    """Split the audio into distinct sound events (hits, notes, words/syllables, SFX layers) and describe each:
    start/end, level, attack, tonal vs noisy, brightness, pitch and pitch glide. `as_text` gives one line per
    event, e.g. "0.52-0.81s | -14dB | tonal | bright | A5 glide -12st | sharp attack". Also reports spacing
    regularity and the sequence of pitches. The threshold adapts to the background level; set threshold_db
    (dBFS) to override. split_on_onsets=false keeps continuous sounds (e.g. a sentence) as one event."""
    return events.detect_events(audio_io.load(path, start, end), threshold_db, split_on_onsets)


@tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
def audio_transcribe(
    path: str,
    start: float | None = None,
    end: float | None = None,
    model: str = transcribe_mod.DEFAULT_MODEL,
    language: str | None = None,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
) -> dict:
    """Speech to text (Whisper, runs locally). Returns the text, timed segments, per-word timings with
    confidence, detected language, speaking rate, long pauses and `uncertain_words` (likely mis-hearings).
    model: tiny/base/small/medium/large-v3/turbo (bigger = more accurate, slower; first use downloads it).
    language: ISO code like "en" to skip detection. initial_prompt: names or jargon to help spelling.
    Use start/end to transcribe part of a long recording."""
    return transcribe_mod.transcribe(audio_io.load(path, start, end), model, language, word_timestamps, initial_prompt)


@tool(annotations=READ_ONLY)
def audio_batch_survey(target: str | list[str], pattern: str = "*") -> dict:
    """Analyze a folder (recursive; filter with a glob `pattern` like "*.wav" or "footstep*") or a list of
    files, and report how consistent they are: loudness outliers, mixed sample rates/channels/formats,
    inconsistent leading silence, files with problems. Includes `suggested_fixes` with ready-to-use
    operations for audio_batch_process, and a per-file table."""
    return batch.survey(target, pattern)


@tool(annotations=WRITES_FILE)
def audio_batch_process(
    target: str | list[str], operations: list[dict], output_dir: str, pattern: str = "*", overwrite: bool = False
) -> dict:
    """Apply one operations chain (same format as audio_process) to every file in a folder or list, writing
    results with the same relative names into output_dir. Refuses to write into the input folder unless
    overwrite=true. Example: normalize a set of sound effects to the same loudness:
    [{"op":"trim_silence"}, {"op":"normalize_lufs","lufs":-18,"limit":true}]."""
    return batch.process_many(target, operations, output_dir, pattern, overwrite)


@tool(annotations=READ_ONLY)
def audio_compare(path_a: str, path_b: str) -> dict:
    """A/B two files: plain-language summary, metric deltas (loudness, peak, brightness, tempo, pitch),
    per-band level changes, spectral-shape similarity, issues fixed or introduced, and a null test
    (aligns and subtracts) that tells whether B is identical, gain-changed, lightly or heavily changed.
    Use after audio_process to verify an edit did what you intended and nothing else."""
    return compare_mod.compare(path_a, path_b)


@tool(annotations=WRITES_FILE)
def audio_process(
    path: str, operations: list[dict], output: str | None = None, overwrite: bool = False, subtype: str | None = None
) -> dict:
    """Apply operations in order and write a NEW file (default: <name>_processed.<ext> next to the input;
    the input is never modified unless output==path and overwrite=true). Each operation is an object, e.g.
    [{"op":"highpass","hz":80}, {"op":"normalize_lufs","lufs":-16}, {"op":"fade_out","seconds":0.5}].
    Ops: gain(db) normalize_peak(db=-1) normalize_lufs(lufs=-16, ceiling_db=-1, limit=false) trim_silence(threshold_db=-60, pad_ms=10)
    denoise(reduction_db=12, sensitivity=1, noise_start, noise_end) [verify speech afterwards with audio_transcribe]
    limit(ceiling_db=-1, release_ms=60, lookahead_ms=5) compress(threshold_db=-18, ratio=3, attack_ms=10, release_ms=120, knee_db=6, makeup_db=0)
    fade_in/fade_out(seconds, curve=cosine|linear|equal_power) crop(start, end) pad(start, end)
    highpass/lowpass(hz, order=4) eq(hz, db, q=1) low_shelf/high_shelf(hz, db) remove_dc to_mono to_stereo
    resample(rate) reverse loop_crossfade(seconds=0.05) mix(path, at=0, gain_db=0) append(path, gap=0).
    subtype: e.g. PCM_16, PCM_24, FLOAT (WAV). Returns before/after levels and any warnings."""
    audio = audio_io.load(path)
    x, sr, log = process_mod.apply_chain(audio, operations)
    target = output or audio_io.default_output(audio.path, "processed")
    written, warnings = audio_io.save(x, sr, target, overwrite=overwrite, subtype=subtype)
    return {
        "output": str(written),
        "operations": log,
        "before": analysis.quick_levels(audio.samples, audio.sr),
        "after": analysis.quick_levels(x, sr),
        "warnings": warnings,
    }


SYNTH_DESCRIPTION = (
    """Generate audio from a declarative spec: tones, sweeps, chords, noise bursts, simple melodies and
layered sound effects (band-limited oscillators, ADSR envelopes, vibrato, tremolo, filters, panning).
Example laser zap: {"layers":[{"wave":"square","freq":1800,"freq_end":200,"duration":0.25,
"envelope":{"attack":0.002,"release":0.1},"lowpass":6000}]}.
Example melody: {"layers":[{"wave":"triangle","sequence":[["C4",0.25],["E4",0.25],["G4",0.5]]}]}.
Full schema:
"""
    + synth.SPEC_HELP
)


@tool(description=SYNTH_DESCRIPTION, annotations=WRITES_FILE)
def audio_synthesize(spec: dict, output: str, overwrite: bool = False) -> dict:
    x, sr = synth.synthesize(spec)
    written, warnings = audio_io.save(x, sr, output, overwrite=overwrite)
    return {"output": str(written), **analysis.quick_levels(x, sr), "warnings": warnings}


def main():
    mcp.run()
