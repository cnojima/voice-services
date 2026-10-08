"""UpstreamInstance against a faked api_v2.py (httpx.MockTransport): no subprocess, no GPU."""
from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from voice_services.config import TTSConfig
from voice_services.tts import backend as backend_mod
from voice_services.tts.backend import UpstreamInstance
from voice_services.tts.voices import VoiceNotFoundError

SR = 32000
MANIFEST = {"gpt_weights": "g.ckpt", "sovits_weights": "s.pth", "ref_audio_path": "r.wav", "ref_text": "hi"}


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(SR * seconds), dtype=np.float32), SR, format="WAV")
    return buf.getvalue()


def _voice(tmp_path, name="ayaka", **extra):
    d = tmp_path / name
    d.mkdir()
    (d / "voice.json").write_text(json.dumps({**MANIFEST, **extra}), encoding="utf-8")


class FakeUpstream:
    """Stands in for api_v2.py: records requests, answers /tts from a queue of durations."""

    def __init__(self, durations=(1.0,), running=True):
        self.durations = list(durations)
        self.running = running
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.running:
            raise httpx.ConnectError("refused")
        self.requests.append(request)
        if request.url.path == "/tts":
            seconds = self.durations.pop(0) if len(self.durations) > 1 else self.durations[0]
            return httpx.Response(200, content=_wav(seconds))
        return httpx.Response(200, text="ok")

    def paths(self):
        return [r.url.path for r in self.requests]


def _instance(tmp_path, fake, **cfg):
    inst = UpstreamInstance(TTSConfig(weights_root=str(tmp_path), **cfg), port=9999)
    inst._client = httpx.Client(transport=httpx.MockTransport(fake))
    return inst


def test_request_sends_the_reference_clips_own_language_as_prompt_lang(tmp_path):
    _voice(tmp_path, "juliet", ref_lang="en")
    fake = FakeUpstream()
    inst = _instance(tmp_path, fake)
    inst.start("juliet")

    inst.synthesize("hello", "ja")

    body = json.loads([r for r in fake.requests if r.url.path == "/tts"][-1].content)
    assert body["prompt_lang"] == "en"
    assert body["text_lang"] == "ja"
    assert body["prompt_text"] == "hi"


def test_start_attaches_to_a_running_upstream_and_loads_the_voice(tmp_path):
    _voice(tmp_path)
    fake = FakeUpstream()
    inst = _instance(tmp_path, fake)

    inst.start("ayaka")

    assert inst.ready and inst.voice_name == "ayaka"
    assert fake.paths().count("/set_gpt_weights") == 1 and fake.paths().count("/set_sovits_weights") == 1
    assert "/tts" in fake.paths()  # the warm-up synthesis


def test_start_without_a_gpt_sovits_root_fails_clearly_and_cleans_up(tmp_path):
    _voice(tmp_path)
    inst = _instance(tmp_path, FakeUpstream(running=False))  # nothing running, no root configured

    with pytest.raises(RuntimeError, match="VOICE_GPT_SOVITS_ROOT"):
        inst.start("ayaka")

    assert not inst.ready and inst.closed


def test_start_with_an_unknown_voice_raises_voice_not_found(tmp_path):
    inst = _instance(tmp_path, FakeUpstream())

    with pytest.raises(VoiceNotFoundError):
        inst.start("nobody")


def test_switch_voice_sets_both_weights_and_updates_the_voice(tmp_path):
    _voice(tmp_path, "ayaka")
    _voice(tmp_path, "juliet", ref_lang="en")
    fake = FakeUpstream()
    inst = _instance(tmp_path, fake)
    inst.start("ayaka")
    fake.requests.clear()

    inst.switch_voice("juliet")

    assert fake.paths() == ["/set_gpt_weights", "/set_sovits_weights"]
    assert inst.voice_name == "juliet"
    assert inst._voice["ref_lang"] == "en"


def test_switch_voice_failure_keeps_the_old_voice(tmp_path):
    _voice(tmp_path, "ayaka")
    _voice(tmp_path, "juliet")
    fake = FakeUpstream()
    inst = _instance(tmp_path, fake)
    inst.start("ayaka")
    inst._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))

    with pytest.raises(RuntimeError, match="500"):
        inst.switch_voice("juliet")

    assert inst.voice_name == "ayaka"


def test_synthesize_retries_an_implausibly_short_result(tmp_path):
    _voice(tmp_path)
    fake = FakeUpstream(durations=[2.0])
    inst = _instance(tmp_path, fake)
    inst.start("ayaka")
    fake.requests.clear()
    fake.durations = [0.05, 0.05, 2.0]  # two degenerate decodes, then a good one

    wav, sr = inst.synthesize("これはテストです。", "ja")

    assert len(wav) / sr == pytest.approx(2.0)
    assert fake.paths().count("/tts") == 3


def test_synthesize_gives_up_after_the_retry_cap_and_returns_the_last_result(tmp_path, capsys):
    _voice(tmp_path)
    fake = FakeUpstream(durations=[2.0])
    inst = _instance(tmp_path, fake)
    inst.start("ayaka")
    fake.requests.clear()
    fake.durations = [0.05]

    wav, sr = inst.synthesize("これはテストです。", "ja")

    assert len(wav) / sr == pytest.approx(0.05, abs=0.001)
    assert fake.paths().count("/tts") == backend_mod.SYNTHESIZE_RETRIES
    assert "implausibly short" in capsys.readouterr().err


def test_synthesize_raises_on_an_upstream_error(tmp_path):
    _voice(tmp_path)
    inst = _instance(tmp_path, FakeUpstream())
    inst.start("ayaka")
    inst._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")))

    with pytest.raises(RuntimeError, match="400"):
        inst.synthesize("x", "ja")


def test_close_is_repeatable(tmp_path):
    inst = _instance(tmp_path, FakeUpstream())
    assert inst.closed is False
    inst.close()
    inst.close()
    assert inst.closed is True
