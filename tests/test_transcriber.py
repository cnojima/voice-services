"""Unit tests for the Transcriber hallucination filter (Technical Design §4.5).

Stubbed Whisper segment dicts -- no real model needed.
"""
from __future__ import annotations

from voice_services.stt.transcriber import Transcriber


def _seg(no_speech_prob, avg_logprob):
    return {"no_speech_prob": no_speech_prob, "avg_logprob": avg_logprob}


def test_empty_text_is_hallucination():
    assert Transcriber._is_hallucination("", [], 0.0, 0.0) is True
    assert Transcriber._is_hallucination("   ", [], 0.0, 0.0) is True
    assert Transcriber._is_hallucination("...", [], 0.0, 0.0) is True


def test_high_no_speech_and_low_logprob_is_hallucination():
    segments = [_seg(0.9, -1.5), _seg(0.8, -2.0)]
    assert Transcriber._is_hallucination("some text", segments, 0.9, -1.5) is True


def test_known_phrase_filtered_unconditionally():
    # Technical Design originally gated this on max_no_speech_prob > 0.3.
    # M0/Phase 4 testing showed Whisper hallucinates these phrases on real
    # silence with max_no_speech_prob near zero, so the gate never fired
    # for the case it existed to catch. Filtered regardless now -- see the
    # comment in voice/stt.py.
    assert Transcriber._is_hallucination("Thank you.", [_seg(0.5, -0.2)], 0.5, -0.2) is True
    assert Transcriber._is_hallucination("Thank you.", [_seg(0.1, -0.2)], 0.1, -0.2) is True
    assert Transcriber._is_hallucination("you", [], 0.0, 0.0) is True


def test_normal_transcript_passes_through():
    segments = [_seg(0.05, -0.3)]
    assert Transcriber._is_hallucination(
        "What's the weather like today?", segments, 0.05, -0.3
    ) is False


def test_no_segments_only_checks_text_content():
    assert Transcriber._is_hallucination("Hello there", [], 0.0, 0.0) is False


def test_detect_language_can_be_narrowed_per_call_without_touching_the_loaded_set():
    import numpy as np

    from voice_services.config import WhisperConfig
    from voice_services.stt.transcriber import Transcriber

    class Param:
        dtype = "f4"
        device = "cpu"

    class Model:
        dims = type("D", (), {"n_mels": 80})()
        device = "cpu"

        def parameters(self):
            return iter([Param()])

        def detect_language(self, mel):
            return None, {"en": 0.3, "ja": 0.2, "it": 0.5}  # Italian wins overall

    import voice_services.stt.transcriber as stt_mod

    t = object.__new__(Transcriber)
    t.cfg, t.model = WhisperConfig(languages=("en", "ja", "it")), Model()
    orig_mel, orig_pad = stt_mod.whisper.log_mel_spectrogram, stt_mod.whisper.pad_or_trim
    stt_mod.whisper.log_mel_spectrogram = lambda audio, n_mels: type("M", (), {"to": lambda s, *a: s})()
    stt_mod.whisper.pad_or_trim = lambda a: a
    try:
        audio = np.zeros(10, dtype="f4")
        assert t.detect_language(audio) == "it"  # default: every loaded language
        assert t.detect_language(audio, ("ja", "en")) == "en"  # a tutor session hears only these
        assert t.detect_language(audio, ("ja",)) == "ja"  # one candidate: no model call needed
    finally:
        stt_mod.whisper.log_mel_spectrogram, stt_mod.whisper.pad_or_trim = orig_mel, orig_pad
