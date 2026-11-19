"""Kokoro-82M voices, run in-process beside the GPT-SoVITS pool (contract: docs/TTS_API.md).

One small model and no subprocess, so a Kokoro voice never takes a pool slot or evicts a pinned
voice. The voices are a fixed catalogue (`KOKORO_VOICES`), not directories under the weights root.
It honours `speed_factor` only: the other sampling fields are GPT-SoVITS's and are ignored.

`kokoro` is optional. Without it, or when its weights or espeak-ng cannot load, `available` is false
and `error` says why; the server then lists no Kokoro voices and has no fallback.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Iterator

import numpy as np

from voice_services.tts.voices import VoiceNotFoundError

SAMPLE_RATE = 24000
_REPO_ID = "hexgrad/Kokoro-82M"

# Voice name -> language of the text it speaks. English and Japanese are the fallback for a failed
# GPT-SoVITS voice (placeholders until the listening check in docs/TTS_API.md picks them).
KOKORO_VOICES: dict[str, str] = {
    "af_heart": "en",
    "jf_alpha": "ja",
    "ff_siwis": "fr",
    "if_sara": "it",
}
_LANG_CODE = {"en": "a", "ja": "j", "fr": "f", "it": "i"}


def use_system_espeak() -> None:
    """Point misaki at an installed espeak-ng. The one pip bundles (espeakng-loader) has its build
    machine's data path baked in and crashes on macOS. VOICE_ESPEAK_LIBRARY and VOICE_ESPEAK_DATA
    (e.g. Homebrew's lib/libespeak-ng.dylib and share/espeak-ng-data) win over the bundled paths;
    with neither set this does nothing."""
    lib, data = os.environ.get("VOICE_ESPEAK_LIBRARY"), os.environ.get("VOICE_ESPEAK_DATA")
    if not (lib or data):
        return
    import misaki.espeak  # noqa: F401 -- sets the bundled paths on import, so override after it
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    if lib:
        EspeakWrapper.set_library(lib)
    if data:
        EspeakWrapper.set_data_path(data)


def _real_factory() -> Callable[[str], object]:
    """A factory of KPipelines that share one KModel (an 82M model per language would be four copies)."""
    use_system_espeak()
    from kokoro import KModel, KPipeline

    model = KModel(repo_id=_REPO_ID)
    return lambda code: KPipeline(lang_code=code, repo_id=_REPO_ID, model=model)


def _to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):  # a torch tensor, possibly on the GPU
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class _PcmStream:
    """Audio chunks as 16-bit PCM bytes, with a `close()` for the server to call."""

    def __init__(self, first: np.ndarray | None, parts: Iterator[np.ndarray]):
        self._first, self._parts = first, parts

    def __iter__(self) -> "_PcmStream":
        return self

    def __next__(self) -> bytes:
        if self._first is not None:
            audio, self._first = self._first, None
        else:
            audio = next(self._parts)  # StopIteration ends the stream
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()

    def close(self) -> None:
        self._parts.close()


class KokoroEngine:
    """`pipeline_factory(lang_code) -> pipeline` is injectable for tests; by default the real Kokoro, whose
    model is loaded now so a missing package or weight shows as `available == False` at startup."""

    def __init__(self, pipeline_factory: Callable[[str], object] | None = None):
        self.errors: dict[str, str] = {}  # language -> why it cannot be spoken; empty when all can
        self._lock = threading.Lock()
        self._pipelines: dict[str, object] = {}
        try:
            self._factory = pipeline_factory or _real_factory()
        except Exception as e:  # noqa: BLE001 -- ImportError, a missing download, a bad espeak path
            self.errors = {lang: f"{type(e).__name__}: {e}" for lang in _LANG_CODE}
            return
        for lang in _LANG_CODE:  # fail here, not on a user's turn; one language failing (Japanese needs unidic) spares the rest
            try:
                self._pipeline(lang)
            except Exception as e:  # noqa: BLE001
                self.errors[lang] = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        """At least one language can be spoken."""
        return len(self.errors) < len(_LANG_CODE)

    @property
    def error(self) -> str | None:
        """Why nothing can be spoken, or None when something can."""
        return None if self.available else "; ".join(sorted(set(self.errors.values())))

    def voices(self) -> dict[str, str]:
        """The catalogue voices that can be spoken now, name -> language."""
        return {name: lang for name, lang in KOKORO_VOICES.items() if lang not in self.errors}

    def voice_for(self, language: str) -> str | None:
        """The catalogue's voice for `language`, or None."""
        return next((name for name, lang in self.voices().items() if lang == language), None)

    def _pipeline(self, language: str):
        code = _LANG_CODE[language]
        if code not in self._pipelines:
            self._pipelines[code] = self._factory(code)
        return self._pipelines[code]

    def _chunks(self, text: str, voice: str, sampling: dict) -> Iterator[np.ndarray]:
        """Audio as the pipeline produces it, one chunk per inference step."""
        if voice not in KOKORO_VOICES:
            raise VoiceNotFoundError(f"no Kokoro voice {voice!r}")
        if KOKORO_VOICES[voice] in self.errors:
            raise RuntimeError(f"Kokoro cannot speak {KOKORO_VOICES[voice]}: {self.errors[KOKORO_VOICES[voice]]}")
        # The lock covers one inference step at a time, never the time a client spends reading: a client that
        # disconnects leaves this generator unfinished until a garbage collection, and must not hold the engine.
        with self._lock:
            steps = iter(self._pipeline(KOKORO_VOICES[voice])(text, voice=voice, speed=sampling.get("speed_factor", 1)))
        while True:
            with self._lock:
                step = next(steps, None)
            if step is None:
                return
            if step[2] is not None:
                yield _to_numpy(step[2])

    def synthesize(self, text: str, voice: str, sampling: dict | None = None) -> tuple[np.ndarray, int]:
        parts = list(self._chunks(text, voice, sampling or {}))
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), SAMPLE_RATE

    def stream_tts(self, text: str, voice: str, sampling: dict | None = None) -> tuple[int, Iterator[bytes]]:
        """(sample_rate, chunks) of raw little-endian 16-bit mono PCM, returned once the first audio exists so a
        failure surfaces before the caller commits to a response. Nothing is held between chunks."""
        parts = self._chunks(text, voice, sampling or {})
        first = next(parts, None)  # raises here on any failure to start

        return SAMPLE_RATE, _PcmStream(first, parts)
