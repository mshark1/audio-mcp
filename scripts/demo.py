"""Guided demo: runs the tools on realistic tasks and writes listenable results to demo_output/.

Usage: python scripts/demo.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_mcp import server

OUT = Path(__file__).resolve().parent.parent / "demo_output"
MEDIA = Path("C:/Windows/Media")


def header(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_design_sfx():
    header("1. Design a sound effect from a description, then check it")
    spec = {
        "layers": [
            {
                "wave": "saw",
                "freq": "C5",
                "freq_end": "C3",
                "duration": 0.45,
                "lowpass": 5000,
                "envelope": {"attack": 0.003, "decay": 0.1, "sustain": 0.5, "release": 0.2},
                "vibrato": {"rate_hz": 30, "depth_cents": 60},
            },
            {
                "wave": "noise",
                "duration": 0.12,
                "highpass": 3000,
                "gain_db": -10,
                "envelope": {"attack": 0.001, "release": 0.1},
            },
            {
                "wave": "sine",
                "freq": "C2",
                "start": 0.05,
                "duration": 0.4,
                "gain_db": -4,
                "envelope": {"attack": 0.005, "release": 0.3},
            },
        ]
    }
    res = server.audio_synthesize(spec, str(OUT / "1_power_down.wav"), overwrite=True)
    print(f"wrote {Path(res['output']).name}: {res['duration_s']}s, peak {res['peak_dbfs']} dBFS")
    for line in server.audio_events(res["output"], split_on_onsets=False)["as_text"]:
        print("  event:", line)
    server.audio_view(res["output"], save_to=str(OUT / "1_power_down.png"), pitch_overlay=True)


def demo_melody():
    header("2. 'Hear' a melody: synthesize a tune, then transcribe it back blind")
    tune = [
        ["E4", 0.25],
        ["D4", 0.25],
        ["C4", 0.25],
        ["D4", 0.25],
        ["E4", 0.25],
        ["E4", 0.25],
        ["E4", 0.5],
        ["D4", 0.25],
        ["D4", 0.25],
        ["D4", 0.5],
        ["E4", 0.25],
        ["G4", 0.25],
        ["G4", 0.5],
    ]
    path = OUT / "2_melody.wav"
    server.audio_synthesize({"layers": [{"wave": "triangle", "sequence": tune}]}, str(path), overwrite=True)
    mel = server.audio_melody(str(path))
    print("played:      ", " ".join(n for n, _ in tune))
    print("transcribed: ", " ".join(n["note"] for n in mel["notes"]))


def demo_real_sound():
    header("3. Inspect a real recording: C:/Windows/Media/Alarm01.wav")
    rep = server.audio_analyze(str(MEDIA / "Alarm01.wav"))
    lv, sp = rep["levels"], rep["spectrum"]
    print(f"loudness {lv['lufs_integrated']} LUFS, peak {lv['peak_dbfs']} dBFS, stereo {rep['stereo']['width']}")
    print(f"loudness over time: {rep['loudness_shape']['sparkline']}")
    print(
        f"character: {sp['character']}, {sp['brightness']}; strongest notes:",
        [p["note"] for p in sp["spectral_peaks"][:4]],
    )
    print(f"chord/polyphony flagged: {rep['pitch'].get('polyphony_likely', False)}")
    print(f"rhythm: {rep['rhythm']['onset_count']} onsets, {rep['rhythm']['timing']}")
    for i in rep["issues"]:
        print(f"  [{i['severity']}] {i['message']}")
    server.audio_view(str(MEDIA / "Alarm01.wav"), save_to=str(OUT / "3_alarm.png"))


def demo_batch():
    header("4. Audit a whole folder: every sound in C:/Windows/Media")
    survey = server.audio_batch_survey(str(MEDIA), pattern="*.wav")
    print(f"{survey['file_count']} files, stats: {survey['stats']}")
    for f in survey["findings"]:
        print("  -", f[:220] + ("..." if len(f) > 220 else ""))
    for fix in survey["suggested_fixes"]:
        print("  suggested fix:", fix)


def make_speech(path: Path, text: str) -> bool:
    if sys.platform != "win32" or not shutil.which("powershell"):
        return False
    script = (
        "Add-Type -AssemblyName System.Speech; $v = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$v.SetOutputToWaveFile('{path}'); $v.Speak('{text}'); $v.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)
    return True


def demo_cleanup():
    header("5. Fix a bad voice recording end to end (analyze -> process -> verify)")
    clean = OUT / "_tts.wav"
    if not make_speech(clean, "Welcome to the show. Today we are talking about tools that let computers listen."):
        print("(skipped: needs Windows text-to-speech)")
        return
    x, sr = sf.read(str(clean))
    x = np.concatenate([np.zeros(sr), x, np.zeros(sr)])
    t = np.arange(len(x)) / sr
    rng = np.random.default_rng(0)
    raw = (
        0.25 * x
        + 0.03 * np.sin(2 * np.pi * 60 * t)
        + 0.01 * np.sin(2 * np.pi * 180 * t)
        + 0.004 * rng.standard_normal(len(x))
    )
    raw_path = OUT / "5_voice_raw.wav"
    sf.write(str(raw_path), raw.astype(np.float32), sr)
    clean.unlink()

    rep = server.audio_analyze(str(raw_path), rhythm=False)
    print(f"BEFORE: {rep['levels']['lufs_integrated']} LUFS, noise floor {rep['silence']['noise_floor_dbfs']} dBFS")
    for i in rep["issues"]:
        print(f"  [{i['severity']}] {i['message'][:150]}")

    ops = [
        {"op": "highpass", "hz": 90},
        {"op": "eq", "hz": 180, "db": -30, "q": 30},
        {"op": "denoise", "reduction_db": 15},
        {"op": "trim_silence", "threshold_db": -50, "pad_ms": 200},
        {"op": "normalize_lufs", "lufs": -16, "limit": True},
    ]
    res = server.audio_process(str(raw_path), ops, output=str(OUT / "5_voice_fixed.wav"), overwrite=True)
    print("APPLIED:")
    for line in res["operations"]:
        print("  ", line)
    after = server.audio_analyze(res["output"], rhythm=False)
    print(
        f"AFTER: {after['levels']['lufs_integrated']} LUFS, noise floor {after['silence']['noise_floor_dbfs']} dBFS, "
        f"issues: {[i['message'][:60] for i in after['issues']] or 'none'}"
    )
    cmp = server.audio_compare(str(raw_path), res["output"])
    print("COMPARE:", cmp["summary"])
    try:
        before_txt = server.audio_transcribe(str(raw_path), language="en")["text"]
        after_txt = server.audio_transcribe(res["output"], language="en")["text"]
        print("TRANSCRIPT before:", before_txt)
        print(
            "TRANSCRIPT after: ",
            after_txt,
            "(unchanged words = edit didn't damage the speech)" if before_txt == after_txt else "",
        )
    except Exception as e:
        print("(transcription unavailable:", e, ")")
    server.audio_view(str(raw_path), kind="spectrogram", save_to=str(OUT / "5_voice_raw.png"))
    server.audio_view(res["output"], kind="spectrogram", save_to=str(OUT / "5_voice_fixed.png"))


if __name__ == "__main__":
    import logging

    logging.disable(logging.INFO)
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    demo_design_sfx()
    demo_melody()
    demo_real_sound()
    demo_batch()
    demo_cleanup()
    print(f"\nListen to / view the results in {OUT}")
