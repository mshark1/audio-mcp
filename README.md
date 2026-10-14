# audio-mcp

<!-- mcp-name: io.github.mshark1/audio-mcp -->

**Audio perception for AI agents.** Language models can't hear. This [MCP](https://modelcontextprotocol.io) server
turns sound into things they *can* reason about: measurements, plain-language problem reports, spectrogram images,
note lists, event timelines, transcripts and before/after diffs. It can also edit and synthesize audio, so an agent
can work in a loop: **make or fix a sound → measure it → look at it → verify the change.**

Everything runs locally. No GPU, no API keys. Core dependencies: numpy, scipy, soundfile, matplotlib.

Real output (JSON abbreviated) from the bundled demo, `scripts/demo.py`:

```
> audio_analyze voice_raw.wav          # speech recorded with mains hum and hiss
  levels: lufs_integrated -32.16, true_peak_dbtp -11.76, clipped_samples 0
  issues: [warning] Mains hum at 60 Hz (peaks at 60, 180 Hz). Fix: notch each with
                    {"op":"eq","hz":60,"db":-30,"q":30}, {"op":"eq","hz":180,"db":-30,"q":30};
                    for voice, highpass at ~80-100 Hz also removes the fundamental.
          [info]    Noise floor around -33.4 dBFS in the quietest parts; hiss or background noise may be audible.

> audio_events sfx.wav                 # a zap, a noise hit, another hit, a rising chirp
  0.09-0.36s | -4dB  | tonal | balanced          | glide G#6->B3 (-33st) | sharp attack
  0.59-0.69s | -16dB | noisy | very bright/harsh | sharp attack
  ...

> audio_melody melody.wav              # "Mary Had a Little Lamb", transcribed blind
  E4(0.23) D4(0.24) C4(0.23) D4(0.24) E4(0.25) E4(0.25) E4(0.5) D4(0.24) D4(0.26) D4(0.49) E4(0.25) G4(0.26) G4(0.47)
```

## Tools

| Tool | What it gives the agent |
|---|---|
| `audio_analyze` | Peak, true peak, RMS, LUFS loudness, clipping, DC offset, a loudness sparkline, silence and noise floor, stereo correlation/width/balance, band energy, brightness, tonal vs. noisy, strongest frequencies as note names, pitch, onsets, tempo, edge clicks, loop-seam quality, and an **`issues`** list in plain language with suggested fixes |
| `audio_events` | Splits audio into distinct sounds and describes each one: timing, level, attack, tonal/noisy, brightness, pitch or glide |
| `audio_view` | PNG waveform and/or spectrogram, with an optional pitch overlay. Every column covers its full time span, so short clicks aren't skipped on long files |
| `audio_timeline` | Level and brightness over time. Streams the file, so it works on recordings of any length |
| `audio_melody` | Single-line transcription: notes, timings, cents, compact text |
| `audio_transcribe` | Speech to text with local Whisper: timed words, language, speaking rate, long pauses and **uncertain words** *(optional extra)* |
| `audio_compare` | A/B summary, metric deltas, per-band dB changes, issues fixed or introduced, and a **null test** (identical? only a gain change? lightly changed?) |
| `audio_process` | An ordered chain of edits written to a **new** file: gain, peak/LUFS normalize, true-peak limiter, compressor, noise reduction, RMS normalize, trim, fades, crop, pad, high/low-pass, peaking EQ, shelves, DC removal, polarity, mono/stereo, resample, reverse, seamless-loop crossfade, mix, append |
| `audio_batch_survey` | Audits a folder for consistency (loudness outliers, mixed sample rates/channels/formats, uneven leading silence) and returns ready-to-run fixes |
| `audio_batch_process` | Applies one edit chain to many files and writes them to an output folder |
| `audio_synthesize` | Declarative synth: band-limited oscillators with phase control, noise colours (pink/brown loop seamlessly), sweeps, ADSR, vibrato, tremolo, filters, pan, note sequences, plus exact test signals (impulse, DC, silence) |
| `audio_info` | Metadata for a file, or a listing of the audio files in a folder |

Reads WAV, FLAC, OGG/Vorbis, Opus, MP3, AIFF and CAF (via libsndfile). Convert M4A/AAC with ffmpeg first.

## Install

```bash
pip install audio-mcp              # or: uv tool install audio-mcp
pip install "audio-mcp[speech]"    # adds audio_transcribe (faster-whisper; downloads a ~145 MB model on first use)
```

## Connect it to your MCP client

**Claude Code**

```bash
claude mcp add audio -- uvx audio-mcp
# or, if installed with pip:
claude mcp add audio -- audio-mcp
```

**Claude Desktop, Cursor, and other clients** (JSON config):

```json
{
  "mcpServers": {
    "audio": { "command": "uvx", "args": ["audio-mcp"] }
  }
}
```

Then ask things like *"check the sound effects in ./sfx for problems"*, *"clean up interview.wav for a podcast"*
or *"design a short UI confirm sound and check it isn't harsh"*.

### Restricting file access

By default the server can read and write any file your user account can. To confine it, set
`AUDIO_MCP_ALLOWED_DIRS` to a list of directories (separated by `;` on Windows, `:` elsewhere):

```json
{ "command": "uvx", "args": ["audio-mcp"], "env": { "AUDIO_MCP_ALLOWED_DIRS": "/home/me/audio" } }
```

Paths are checked after resolving `..` and symlinks. Other settings: `AUDIO_MCP_MAX_SECONDS` (largest section loaded
into memory per call, default 1800), `AUDIO_MCP_WHISPER_MODEL` (default `base`), `AUDIO_MCP_ROOT` (base for
relative paths).

## How accuracy is checked

The tests mostly use **synthetic signals whose correct answer is known by construction**, so they check
correctness rather than just "no crash":

- **Loudness:** our streaming BS.1770 meter matches pyloudnorm's exact-filter mode within 0.01 LU from 8 to 96 kHz,
  and reads the EBU reference tone (997 Hz, 0 dBFS) at −3.01 LUFS.
- **Limiter:** a +15 dBTP input comes out at exactly −1.0 dBTP with no samples touched by the safety clip.
- **Pitch and melody:** sines, chirps and synthesized melodies are recovered note for note, including repeated notes;
  pitch is within 1 cent from 82 Hz to 1760 Hz.
- **Tempo, onsets, events, silence, stereo phase, hum, clipping, alignment offsets:** each is checked against a
  constructed signal.
- **A contract test** runs every tool on every content type (silence, DC, noise, full-scale square), length
  (1 sample to 3 s), format (16/24/float WAV, FLAC, Vorbis), channel count (1, 2, 6) and sample rate (8–96 kHz).
  Each call must either succeed or fail with an `AudioError` that explains itself.
- **Memory:** analysis is chunked. A 20-minute stereo 48 kHz file analyzes in about 19 s with about 1.2 GB peak
  memory, 0.7 GB of which is the audio itself.

## Design principles

- **Never destructive by default.** Edits write `<name>_processed.wav`; overwriting needs `overwrite=true`, and
  batch output can't target the input folder.
- **Errors say what to do next.** An unknown op lists every op with its parameters; an undecodable file names the
  supported formats and the ffmpeg command to convert it. Unexpected errors are reported as bugs instead of being
  hidden.
- **Honest about uncertainty.** Chords set `polyphony_likely` instead of reporting a confident wrong pitch, weak
  tempos carry a low-confidence note, Whisper's uncertain words are listed, and LUFS is `null` for clips under
  0.4 s.
- **Compact output.** Rounded numbers, sparklines and one-line event summaries keep responses small enough for
  an agent's context.

## Limitations

- Pitch, melody and per-event pitch assume one note at a time. Chords and full mixes are flagged, not transcribed.
- Denoise is a spectral gate for *steady* noise (hiss, hum, fans, room tone). It doesn't remove clicks, music or
  other voices. Too much `sensitivity` dulls quiet consonants, so verify speech edits with `audio_transcribe`.
- Tempo is a classic autocorrelation estimate; half- and double-tempo errors happen on complex music.
- The tools describe sound, but they don't judge taste ("is this satisfying?").

## Reference signals

[audio-mcp-assets](https://github.com/mshark1/audio-mcp-assets) is a public-domain pack of test and calibration
audio (tones, sweeps, calibrated noise, loudness, stereo and phase tests, impulses, problem files) that an AI agent
built using only these tools. Every file is verified against known ground truth, twice, including independently of
this project.

## Contributing

Improvements from people and AI agents are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md); AI coding agents
should also read [AGENTS.md](AGENTS.md). The bar for a feature is that it does one thing well, with a test against
known ground truth.

## License

[MIT](LICENSE)
