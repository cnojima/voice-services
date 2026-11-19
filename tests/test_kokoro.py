"""voice_services/tts/kokoro.py against a fake pipeline: no model, no espeak."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from voice_services.tts.kokoro import KOKORO_VOICES, KokoroEngine
from voice_services.tts.voices import VoiceNotFoundError


class FakePipeline:
    def __init__(self, lang_code, log):
        self.lang_code, self.log = lang_code, log

    def __call__(self, text, voice=None, speed=1, **_):
        self.log.append({"lang_code": self.lang_code, "text": text, "voice": voice, "speed": speed})
        for n in (1, 2):  # two chunks, like a text split in two
            yield f"g{n}", f"p{n}", np.full(240 * n, 0.5, dtype=np.float32)


def engine(log=None, factory=None):
    log = [] if log is None else log
    return KokoroEngine(pipeline_factory=factory or (lambda code: FakePipeline(code, log))), log


def test_synthesize_returns_mono_float32_at_24khz_and_passes_voice_lang_and_speed():
    eng, log = engine()
    wav, sr = eng.synthesize("Bonjour.", "ff_siwis", {"speed_factor": 1.25})
    assert sr == 24000
    assert wav.dtype == np.float32 and wav.ndim == 1 and len(wav) == 240 + 480
    assert log == [{"lang_code": "f", "text": "Bonjour.", "voice": "ff_siwis", "speed": 1.25}]


@pytest.mark.parametrize("voice, code", [("ff_siwis", "f"), ("if_sara", "i")])
def test_each_voice_uses_its_languages_code(voice, code):
    eng, log = engine()
    eng.synthesize("x", voice, {})
    assert log[-1]["lang_code"] == code


def test_only_speed_factor_is_read_from_sampling():
    eng, log = engine()
    eng.synthesize("x", "if_sara", {"temperature": 0.5, "top_p": 0.9, "top_k": 5, "fragment_interval": 1.0})
    assert log[0]["speed"] == 1


def test_stream_yields_whole_16_bit_samples():
    eng, _ = engine()
    sr, chunks = eng.stream_tts("Ciao.", "if_sara", {})
    data = list(chunks)
    assert sr == 24000
    assert data and all(len(c) % 2 == 0 for c in data)
    assert sum(len(c) for c in data) == (240 + 480) * 2
    assert np.frombuffer(data[0], dtype="<i2")[0] == 16383  # 0.5 of full scale


def test_stream_fails_before_returning_when_the_pipeline_fails():
    def boom(text, voice=None, speed=1, **_):
        raise RuntimeError("no model")
        yield  # pragma: no cover

    eng, _ = engine(factory=lambda code: type("P", (), {"__call__": staticmethod(boom)})())
    with pytest.raises(RuntimeError, match="no model"):
        eng.stream_tts("x", "ff_siwis", {})


def test_voice_for_and_catalogue():
    eng, _ = engine()
    assert eng.voice_for("fr") == "ff_siwis"
    assert eng.voice_for("it") == "if_sara"
    assert eng.voice_for("de") is None
    assert {KOKORO_VOICES["ff_siwis"], KOKORO_VOICES["if_sara"]} == {"fr", "it"}
    assert set(KOKORO_VOICES.values()) == {"en", "ja", "fr", "it"}


def test_an_unknown_voice_is_voice_not_found():
    eng, _ = engine()
    with pytest.raises(VoiceNotFoundError):
        eng.synthesize("x", "nobody", {})


def test_a_factory_that_raises_makes_the_engine_unavailable_with_the_reason():
    def factory(code):
        raise ImportError("No module named 'kokoro'")

    eng = KokoroEngine(pipeline_factory=factory)
    assert not eng.available
    assert "kokoro" in eng.error
    with pytest.raises(RuntimeError, match="kokoro"):
        eng.synthesize("x", "ff_siwis", {})


def test_a_working_engine_is_available():
    eng, _ = engine()
    assert eng.available and eng.error is None


def test_two_threads_never_run_the_model_at_once():
    running, peak, lock = [0], [0], threading.Lock()

    class Slow:
        def __call__(self, text, voice=None, speed=1, **_):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1
            yield "g", "p", np.zeros(10, dtype=np.float32)

    eng = KokoroEngine(pipeline_factory=lambda code: Slow())
    threads = [threading.Thread(target=eng.synthesize, args=("x", "ff_siwis", {})) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_closing_a_stream_that_was_never_read_releases_the_engine():
    eng, _ = engine()
    _, chunks = eng.stream_tts("x", "ff_siwis", {})
    chunks.close()
    done = threading.Event()
    threading.Thread(target=lambda: (eng.synthesize("y", "ff_siwis", {}), done.set()), daemon=True).start()
    assert done.wait(2), "the engine is still held by the closed stream"


def test_a_language_that_cannot_load_is_not_offered_but_the_others_still_work():
    def factory(code):
        if code == "j":
            raise RuntimeError("Failed initializing MeCab")
        return FakePipeline(code, [])

    eng = KokoroEngine(pipeline_factory=factory)

    assert eng.available and eng.error is None  # most of it works
    assert "jf_alpha" not in eng.voices() and "ff_siwis" in eng.voices()
    assert eng.voice_for("ja") is None and eng.voice_for("fr") == "ff_siwis"
    assert "MeCab" in eng.errors["ja"]
    with pytest.raises(RuntimeError, match="MeCab"):
        eng.synthesize("こんにちは", "jf_alpha", {})
    assert eng.synthesize("Bonjour.", "ff_siwis", {})[1] == 24000


def test_a_stream_the_client_stops_reading_does_not_hold_the_engine():
    """A disconnect may leave the response generator unfinished until a garbage collection: nothing may
    be held while the client reads, or every later Kokoro request waits on it."""
    eng, _ = engine()
    _, chunks = eng.stream_tts("x", "ff_siwis", {})
    next(chunks)  # one chunk read, then the client goes away without closing anything
    done = threading.Event()
    threading.Thread(target=lambda: (eng.synthesize("y", "ff_siwis", {}), done.set()), daemon=True).start()
    assert done.wait(2), "an unfinished stream still holds the engine"
