# Guide for AI coding agents

This file is for AI agents working *on* this codebase. (Agents *using* the tools get their guidance from the MCP
server's `instructions` and the tool descriptions in `server.py`.) Read [CONTRIBUTING.md](CONTRIBUTING.md) for the
quality bar; this file covers the mechanics.

## Map

```
src/audio_mcp/
  server.py      MCP tools. Docstrings ARE the user interface for calling agents; edit them with care.
                 The @tool wrapper turns unexpected exceptions into readable "Internal error" messages.
  audio_io.py    load/save, path resolution + AUDIO_MCP_ALLOWED_DIRS sandbox, AudioError (a ToolError).
                 Formats libsndfile can't read go through FFmpeg (if installed) into a cached temp WAV;
                 read files via open_source(), never sf.* on the user's path directly.
  dsp.py         shared helpers: dB conversions, chunked STFT/frame RMS/resample, true peak, note names.
  loudness.py    streaming ITU-R BS.1770 integrated loudness (validated against pyloudnorm in tests).
  analysis.py    audio_analyze: levels, spectrum, silence, stereo, rhythm, edges, and find_issues(); timeline.
  pitch.py       YIN pitch tracking and note segmentation (audio_melody).
  events.py      event segmentation (audio_events).
  compare.py     A/B metrics and the null test.
  process.py     edit operations: one function per op, registered with @op; signature = parameters.
  dynamics.py    limiter and compressor (control-rate gain, applied in chunks).
  restore.py     denoise (spectral gate with speech-safe two-level gain).
  synth.py       declarative synthesizer.
  batch.py       folder survey and batch processing.
  transcribe.py  optional faster-whisper wrapper.
  render.py      matplotlib waveform/spectrogram PNGs.
tests/
  test_audio_mcp.py  behaviour against ground truth
  test_contract.py   every tool x odd inputs: succeed or AudioError, never an internal error
  test_loudness.py   loudness meter vs. reference; chunked helpers vs. direct computation
  test_ffmpeg_input.py  FFmpeg decoding via a fake ffmpeg script (runs anywhere) + real round trips (CI, Linux)
scripts/
  smoke_stdio.py     end-to-end over real MCP stdio
  demo.py            guided demo writing listenable files to demo_output/
```

## Invariants: don't break these

1. **Expected failures raise `AudioError`** with an actionable message. Any other exception is a bug (the contract
   test fails on "Internal error").
2. **Every path goes through `audio_io.resolve_path`**, which enforces the sandbox. Never call `open`,
   `Path.write_*` or `sf.read` on a user-supplied path without resolving it first.
3. **Nothing overwrites user files by default.** Writing tools take `overwrite=False`.
4. **Memory stays bounded.** Full-length float64 copies of a signal are not OK: a 30-minute stereo file is about
   0.7 GB as float32. Use `dsp.CHUNK`-sized loops, `stft_chunks`, `frame_rms`, `resample` (all chunked). If you
   add a whole-signal computation, profile it on a 20-minute file (see "Verifying" below).
5. **Output is JSON-friendly and compact.** Round with `dsp.r()` (non-finite becomes `None`), cap lists, keep keys
   stable. Tool names, parameters and output keys are a public API.
6. **No confident wrong answers.** Add confidence or flags where a result can be unreliable.
7. **MCP SDK 2.x** (`mcp.server.mcpserver.MCPServer`). Only `ToolError` messages reach clients, which is why
   `AudioError` subclasses it.

## Verifying your change

- **Write the test first, from ground truth.** Synthesize the input with `audio_synthesize` or numpy so the
  correct answer is known, then assert the tool reports it. Add odd inputs to `test_contract.py` if you add a
  tool or op.
- `pytest` must pass and `ruff check . && ruff format --check .` must be clean.
- Run `python scripts/smoke_stdio.py <tmpdir>` when you touch `server.py`. Unit tests call the functions directly
  and don't go through MCP serialization or error handling.
- **Look at and listen to your output.** Render with `audio_view` and inspect the image; for speech edits,
  compare `audio_transcribe` before and after. That's how a denoiser turning "fox" into "box" was caught.
- For performance-sensitive changes, time and measure peak memory on a long file. Generate one with numpy
  (20 minutes, stereo, 48 kHz) rather than committing large audio.

## Gotchas found the hard way

- `scipy.signal.find_peaks` never reports index 0; `ndimage.median_filter(mode="nearest")` copies edge values
  and can hide an onset at t=0.
- Raw cross-correlation favours small lags on periodic material; normalise by overlap energy (see `compare.py`).
- Wiener gain `1 - N/P` versus `sqrt(1 - N/P)` is a real trade-off between noise removal and keeping consonants.
- On Windows, Smart App Control can block DLLs from some wheels (PyAV) and from downloaded Python builds; it's
  an OS policy, not a code bug, and shouldn't be worked around.
- FFmpeg runs as an argument list (never a shell) with the input as `file:<path>`, `-protocol_whitelist file`
  and a demuxer whitelist, so a crafted file name or playlist can't reach the network or other files. Keep it so.
- Windows PowerShell 5.1 writes UTF-8 *with BOM* when redirecting output, which breaks TOML and JSON parsers.
