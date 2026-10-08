"""Transcriber (Whisper wrapper).

Moved from the `voice` project's voice/stt.py (commit 650dd4a).
"""
from __future__ import annotations

import re

import numpy as np
import whisper

from voice_services.config import WhisperConfig
from voice_services.devices import resolve_whisper_device
from voice_services.lang import normalize_text as _normalize


# Stock lines Whisper emits on silence/noise (subtitle credits, YouTube outros).
# Matched against the whole normalized transcript, so entries are written the
# natural way and normalized below. Bare "thank you" in it/fr/es (grazie, merci,
# gracias) is deliberately left out: unlike English, those are plausible whole
# turns for a language learner.
_KNOWN_HALLUCINATIONS = {
    _normalize(p) for p in (
        "you", "thank you", "thanks for watching", "bye",
        # Japanese (YouTube outro captions).
        "ご視聴ありがとうございました", "チャンネル登録お願いします",
        # Spanish
        "Gracias por ver el video", "Gracias por ver",
        "Subtítulos realizados por la comunidad de Amara.org",
        "Subtítulos por la comunidad de Amara.org",
        "Suscríbete", "Suscríbete al canal",
        # French
        "Merci d'avoir regardé cette vidéo", "Merci d'avoir regardé la vidéo",
        "Sous-titres réalisés par la communauté d'Amara.org",
        "Sous-titrage ST' 501", "Sous-titrage ST'501", "Sous-titrage Société Radio-Canada",
        "Abonnez-vous", "N'oubliez pas de vous abonner",
        # Italian
        "Grazie per la visione", "Grazie per aver guardato il video",
        "Sottotitoli creati dalla comunità Amara.org",
        "Sottotitoli e revisione a cura di QTSS", "Sottotitoli a cura di QTSS",
        "Iscriviti al canale",
    )
}


class Transcriber:
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self.device = resolve_whisper_device(cfg.device)
        self.model = whisper.load_model(cfg.model, device=self.device)

    def detect_language(self, audio: np.ndarray, candidates: tuple[str, ...] | None = None) -> str:
        """Pick the likeliest of `candidates` (default cfg.languages) for this
        utterance. A caller may narrow the set -- e.g. a tutor session hears only
        the target and native language.

        Whisper's own detector scores all ~100 languages, and on short clips of
        accented speech it wanders off (Welsh, Nynorsk...). Restricting the argmax
        to the languages we can actually answer in avoids that. A single
        candidate skips the model call entirely.
        """
        candidates = candidates or self.cfg.languages
        if len(candidates) == 1:
            return candidates[0]
        dtype = next(self.model.parameters()).dtype
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio), n_mels=self.model.dims.n_mels
        ).to(self.model.device, dtype)
        _, probs = self.model.detect_language(mel)
        return max(candidates, key=lambda lang: probs.get(lang, 0.0))

    def transcribe(self, audio: np.ndarray, language: str = "en") -> tuple[str, dict]:
        result = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            fp16=(self.device != "cpu"),
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
        )
        segments = result.get("segments", [])
        text = result.get("text", "").strip()

        max_no_speech = max((s.get("no_speech_prob", 0.0) for s in segments), default=0.0)
        avg_logprob = (
            sum(s.get("avg_logprob", 0.0) for s in segments) / len(segments)
            if segments
            else 0.0
        )
        duration = len(audio) / 16000.0

        info = {
            "max_no_speech_prob": max_no_speech,
            "avg_logprob": avg_logprob,
            "duration_s": duration,
        }

        if self._is_hallucination(text, segments, max_no_speech, avg_logprob):
            return "", info
        return text, info

    @staticmethod
    def _is_hallucination(
        text: str, segments: list[dict], max_no_speech: float, avg_logprob: float
    ) -> bool:
        # 1. Empty or no letter/digit content (\w also covers kana and kanji).
        if not text or not re.search(r"\w", text):
            return True

        # 2. Every segment looks like silence, per Whisper's own heuristic.
        if segments and all(
            s.get("no_speech_prob", 0.0) > 0.6 and s.get("avg_logprob", 0.0) < -1.0
            for s in segments
        ):
            return True

        # 3. A known filler phrase. Technical Design originally gated this on
        #    max_no_speech_prob > 0.3, on the assumption that a hallucinated
        #    "Thank you." would come with high no_speech_prob. Empirically
        #    (M0/Phase 4 smoke test, both pink noise and true digital
        #    silence) Whisper hallucinates these exact phrases with
        #    max_no_speech_prob near zero -- the confidence signal doesn't
        #    correlate with the hallucination at all here. Filtering
        #    unconditionally trades a rare false negative (someone's entire
        #    turn really is just "thank you") for reliably catching the
        #    much more common silence/noise hallucination.
        if _normalize(text) in _KNOWN_HALLUCINATIONS:
            return True

        return False
