# Changelog

All notable changes to this project are documented here. Tool names, parameters and output keys are a public API
for the agents that call them, so any change to them is listed explicitly.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - unreleased

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
