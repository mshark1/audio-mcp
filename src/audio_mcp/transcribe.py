"""Speech-to-text with faster-whisper (optional: `pip install -e ".[speech]"`). Models download on first use."""

from __future__ import annotations

import logging
import os
import sys
import threading
import types

import numpy as np

from .audio_io import Audio, AudioError
from .dsp import r, resample

DEFAULT_MODEL = os.environ.get("AUDIO_MCP_WHISPER_MODEL", "base")
MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-large-v3",
)
_models: dict[str, object] = {}
_lock = threading.Lock()


def _ensure_av_importable():
    """faster-whisper imports PyAV at import time but only uses it to decode file paths. We always pass sample
    arrays, so if PyAV can't load (not installed, or its DLLs are blocked by an OS policy) register a placeholder
    that explains itself if anything ever touches it."""
    try:
        import av  # noqa: F401
    except ImportError as e:
        reason = str(e)
        stub = types.ModuleType("av")

        def __getattr__(name):
            raise ImportError(
                f"PyAV is unavailable ({reason}); audio-mcp decodes audio itself, so this should not be needed."
            )

        stub.__getattr__ = __getattr__
        sys.modules["av"] = stub


def _model(name: str):
    if name not in MODELS:
        raise AudioError(
            f"Unknown Whisper model '{name}'. Options: {', '.join(MODELS)} (bigger = more accurate, slower)."
        )
    with _lock:
        if name not in _models:
            for noisy in ("faster_whisper", "httpx", "huggingface_hub"):
                logging.getLogger(noisy).setLevel(logging.WARNING)
            _ensure_av_importable()
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise AudioError(
                    f"Speech transcription needs the optional dependency faster-whisper ({e}). "
                    'Install it with: pip install -e ".[speech]" in the audio-mcp folder.'
                ) from None
            try:
                _models[name] = WhisperModel(name, device="cpu", compute_type="int8")
            except Exception as e:
                raise AudioError(
                    f"Could not load Whisper model '{name}' (first use downloads it from Hugging Face): {e}"
                ) from None
        return _models[name]


def transcribe(
    audio: Audio,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    word_timestamps: bool = True,
    initial_prompt: str | None = None,
) -> dict:
    x = resample(audio.mono, audio.sr, 16000).astype(np.float32)
    segments, info = _model(model).transcribe(
        x,
        language=language,
        word_timestamps=word_timestamps,
        vad_filter=True,
        beam_size=5,
        initial_prompt=initial_prompt,
    )
    off = audio.offset
    segs, words = [], []
    for s in segments:
        seg = {"start_s": r(s.start + off, 2), "end_s": r(s.end + off, 2), "text": s.text.strip()}
        if s.avg_logprob < -1.0 or s.no_speech_prob > 0.6:
            seg["low_confidence"] = True
        segs.append(seg)
        for w in s.words or []:
            words.append(
                {
                    "word": w.word.strip(),
                    "start_s": r(w.start + off, 2),
                    "end_s": r(w.end + off, 2),
                    "p": r(w.probability, 2),
                }
            )

    out = {
        "model": model,
        "language": info.language,
        "language_probability": r(info.language_probability, 2),
        "text": " ".join(s["text"] for s in segs),
        "segments": segs,
    }
    if not segs:
        out["comment"] = "No speech detected (voice-activity detection found nothing to transcribe)."
        return out
    if words:
        speech_time = sum(w["end_s"] - w["start_s"] for w in words)
        span = words[-1]["end_s"] - words[0]["start_s"]
        out["words_per_minute"] = r(60 * len(words) / span, 0) if span > 0 else None
        out["speech_fraction_of_span"] = r(speech_time / span, 2) if span > 0 else None
        pauses = [
            {"after_word": a["word"], "at_s": a["end_s"], "length_s": r(b["start_s"] - a["end_s"], 2)}
            for a, b in zip(words, words[1:])
            if b["start_s"] - a["end_s"] >= 0.7
        ]
        out["long_pauses"] = pauses[:30]
        unsure = [w for w in words if (w["p"] or 0) < 0.5]
        out["uncertain_words"] = unsure[:50]
        out["words"] = words[:3000]
        if len(words) > 3000:
            out["truncated"] = "word list capped at 3000; pass start/end for later sections"
    return out
