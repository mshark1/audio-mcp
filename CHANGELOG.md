# Changelog

All notable changes to this project are documented here. Tool names, parameters and output keys are a public API
for the agents that call them, so any change to them is listed explicitly.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-14

Three capabilities chosen from research into real audio needs (podcast delivery, indie game audio, AI audio
perception), built in parallel and checked against real recordings before release.

### Added
- **`audio_check_delivery(path, spec)`**: checks a file against `apple_podcasts`, `spotify`, `youtube`,
  `ebu_r128`, `atsc_a85`, `acx_audiobook` or `game_sfx` (`spec="list"` describes them). Returns a pass / pass with
  warnings / fail verdict, a per-criterion table (required, measured, status, note), the cited sources with access
  dates, and an ordered `audio_process` fix (noise handling before gain, true-peak limiting last) that has been
  **simulated on the file** (`fix.verified`). Problems no processing can solve are listed in `fix.cannot_fix`.
  Where a platform doesn't publish how it measures (ACX), the notes say what we approximate.
- **`audio_check_game_project(project_dir)`**: audits a Godot 4 project's audio. It inventories files by
  category, reads `.import` settings (loop mode, trim, normalize, force mono), finds music that doesn't loop, SFX
  that do, and clicky loop seams, checks loudness consistency per category with ready-to-run fixes, flags
  clipping, leading silence, clicks, DC offset, mixed sample rates and stereo SFX, and checks `res://`
  references in scenes and scripts (broken paths, wrong letter case, unreferenced files). Read-only.
- **M4A/AAC, WMA and video files** (MP4, MOV, WebM, MKV, AVI, ...) are read automatically by every tool when
  FFmpeg is installed (or `AUDIO_MCP_FFMPEG` points to it): decoded once to a temporary WAV and cached for the
  session. FFmpeg runs without a shell, with file-only protocol and demuxer whitelists, after the sandbox check.
  `audio_info` reports the codec (via ffprobe when available) and `decoded_via: "ffmpeg"`. New environment
  variables: `AUDIO_MCP_FFMPEG`, `AUDIO_MCP_FFPROBE`, `AUDIO_MCP_FFMPEG_TIMEOUT` (default 300 s),
  `AUDIO_MCP_FFMPEG_CACHE_MB` (default 2048).

### Fixed
- **True peak read about 1.1 dB high on audio longer than about 5.5 s** (block-boundary filter ringing was
  included in the maximum). Affected `true_peak_dbtp`, the true-peak warning, `normalize_peak(true_peak=true)` and
  `normalize_lufs` ceilings. Found by the perception benchmark in audio-mcp-assets.

### Changed
- Without FFmpeg, opening these formats fails with an error naming the format and per-OS install commands. Folder
  listings report them under `needs_ffmpeg`. Batch tools include them only when FFmpeg is available.

## [0.2.0] - 2026-10-13

Improvements found by using audio-mcp for real work: an AI agent built a public-domain reference-signal pack with
these tools only, and logged every failure. See https://github.com/mshark1/audio-mcp-assets (FRICTION.md).

### Added
- Synth waves `impulse` (one full-scale sample, or a train at `freq` per second), `dc` (constant offset) and
  `silence`. These test-signal waves skip the default fade unless an `envelope` is given.
- Oscillator `phase` (degrees). Oscillators now start exactly at that phase on the first sample (previously one
  sample in), which matters for test signals.
- `audio_process` ops `polarity` (invert all channels or one) and `normalize_rms`.
- `levels.lufs_momentary_max` in `audio_analyze` (and in process/synthesize level summaries): EBU momentary
  loudness, the loudest 400 ms window, defined for clips of any length. Integrated LUFS stays `null` under 0.4 s.
- `normalize_lufs(measure="momentary_max")` to match the loudness of short sound effects.
- `normalize_peak(true_peak=true)` to normalize to a true-peak (dBTP) target.

### Fixed
- Pitch was biased sharp at higher frequencies (+3.3 cents at 997 Hz, +10 cents at 1760 Hz). It's now within
  1 cent from 82 Hz to 1760 Hz (`audio_analyze` pitch, `audio_melody`, event pitch).
- Onsets in the last frame of a file were missed, and `audio_events` couldn't see a sound in the final few
  milliseconds.
- `audio_events` ignored sounds shorter than 30 ms (clicks, ticks). Events can now be as short as 10 ms.
- Synthesized pink and brown noise had energy down to 0 Hz (DC drift, "rumble" warnings); it now has nothing
  below 20 Hz and still loops seamlessly.
- The mains-hum warning fired for a single note on a mains harmonic (e.g. 250 Hz); it now needs the 50/60 Hz
  fundamental or its 2nd harmonic.

### Changed (output an agent may rely on)
- `spectrum.flatness` / `spectrum.character` and event `character` now use per-octave flatness. Pink and brown
  noise (and other sloped noise) report `noise-like` / `noisy` instead of `mixed` / `tonal`, and speech now
  reports `mixed tonal/noise` instead of `tonal`. Numeric flatness values differ from 0.1.x.
- `normalize_lufs`'s `ceiling_db` is now always a true-peak ceiling. Without `limit` it used to cap the sample
  peak, so true peaks could exceed it.
- `audio_batch_survey` compares sets of short one-shots by `lufs_momentary_max` (was RMS), and suggests
  `normalize_lufs` with `measure="momentary_max"` and no limiter (was a generic `normalize_peak`). Per-file rows
  include `lufs_momentary_max`.
- `stereo.width` is `phase problem (negative correlation)` only below -0.3 correlation (was: any negative
  value). Near-zero correlation now reports `wide`, and the out-of-phase issue no longer fires for wide stereo
  material. The issue message now includes the mono-sum level change.

## [0.1.1] - 2026-10-12

Packaging and discoverability only; no changes to tools or their output.

### Added
- Listed in the official MCP Registry as `io.github.mshark1/audio-mcp` (`server.json`); releases now publish there
  automatically.
- `Dockerfile` for running the server in a container (used by directories that inspect MCP servers), built and
  smoke-tested in CI.

### Changed
- `scripts/smoke_stdio.py` now fails when a call that should succeed errors, and can test any server command
  (e.g. the Docker image).

## [0.1.0] - 2026-10-12

First public release.

### Tools
- `audio_info`, `audio_analyze`, `audio_timeline`, `audio_view`, `audio_melody`, `audio_events`,
  `audio_transcribe` (optional `speech` extra), `audio_compare`, `audio_process`, `audio_batch_survey`,
  `audio_batch_process`, `audio_synthesize`.

### Highlights
- Streaming ITU-R BS.1770 loudness meter, validated against pyloudnorm (within 0.01 LU, 8-96 kHz).
- True-peak look-ahead limiter, compressor, speech-safe denoiser.
- Optional sandbox: `AUDIO_MCP_ALLOWED_DIRS`.
- Bounded memory: about 1.2 GB peak to analyze a 20-minute stereo 48 kHz file.
