# Changelog

All notable changes to this project are documented here. Tool names, parameters and output keys are a public API
for the agents that call them, so any change to them is listed explicitly.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - unreleased

Improvements found by using audio-mcp for real work: an AI agent built a public-domain reference-signal pack with
these tools only, and logged every failure. See https://github.com/mshark1/audio-mcp-assets (FRICTION.md).

### Added
- Synth waves `impulse` (one full-scale sample, or a train at `freq` per second), `dc` (constant offset) and
  `silence`. These test-signal waves skip the default fade unless an `envelope` is given.
- Oscillator `phase` (degrees). Oscillators now start exactly at that phase on the first sample (previously one
  sample in), which matters for test signals.
- `audio_process` ops `polarity` (invert all channels or one) and `normalize_rms`.

### Fixed
- Pitch was biased sharp at higher frequencies (+3.3 cents at 997 Hz, +10 cents at 1760 Hz). It's now within
  1 cent from 82 Hz to 1760 Hz (`audio_analyze` pitch, `audio_melody`, event pitch).
- Onsets in the last frame of a file were missed, and `audio_events` couldn't see a sound in the final few
  milliseconds.
- `audio_events` ignored sounds shorter than 30 ms (clicks, ticks). Events can now be as short as 10 ms.

### Changed (output an agent may rely on)
- `spectrum.flatness` / `spectrum.character` and event `character` now use per-octave flatness. Pink and brown
  noise (and other sloped noise) report `noise-like` / `noisy` instead of `mixed` / `tonal`, and speech now
  reports `mixed tonal/noise` instead of `tonal`. Numeric flatness values differ from 0.1.x.
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
