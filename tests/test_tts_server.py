"""voice_services/tts/server.py HTTP surface against a pool of FakeInstances."""
from __future__ import annotations

import io
import json

import pytest
import soundfile as sf
from starlette.testclient import TestClient

import voice_services.tts.server as tts_server
from voice_services.tts import pronunciations
from tests.fakes import SR, FakeInstance, FakeKokoro, make_voices
from voice_services.config import TTSConfig
from voice_services.tts.pool import TTSPool


@pytest.fixture
def client(tmp_path):
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "juliet", juliet={"ref_lang": "en", "version": "v2Pro"})
    tts_server.app.state.pool = TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9100, factory=FakeInstance)
    tts_server.app.state.preload = []
    tts_server.app.state.kokoro = None
    with TestClient(tts_server.app) as c:
        yield c
    tts_server.app.state.kokoro = None


@pytest.fixture
def kokoro(client):
    """The same server with a (fake) Kokoro engine."""
    tts_server.app.state.kokoro = FakeKokoro()
    return tts_server.app.state.kokoro


def test_health_lists_instances(client):
    body = client.get("/").json()

    assert body["status"] == "ok" and body["ready"] is True
    assert [i["voice"] for i in body["instances"]] == [None, None]


def test_voices_lists_available_and_loaded(client):
    client.post("/voice", json={"name": "ayaka"})

    body = client.get("/voices").json()

    assert body["loaded"] == ["ayaka"]
    assert body["voices"] == [
        {"name": "ayaka", "ref_lang": "ja", "version": "v2", "engine": "gpt-sovits"},
        {"name": "juliet", "ref_lang": "en", "version": "v2Pro", "engine": "gpt-sovits"},
    ]
    assert [v["name"] for v in client.get("/voices", params={"language": "en"}).json()["voices"]] == ["juliet"]


def test_voices_lists_the_pinned_ones(tmp_path):
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "juliet", juliet={"ref_lang": "en"})
    tts_server.app.state.pool = TTSPool(
        TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9200, factory=FakeInstance, pinned=["juliet"]
    )
    tts_server.app.state.preload = []
    with TestClient(tts_server.app) as c:
        assert c.get("/voices").json()["pinned"] == ["juliet"]
        # asking for other voices never displaces it
        for name in ("ayaka", "juliet", "ayaka"):
            assert c.post("/voice", json={"name": name}).status_code == 200
        assert "juliet" in c.get("/voices").json()["loaded"]
        assert c.get("/").json()["instances"][0]["pinned"] is True


def test_voices_pinned_is_empty_without_a_pin(client):
    assert client.get("/voices").json()["pinned"] == []


def test_the_pin_flag_is_parsed():
    assert tts_server.build_parser().parse_args(["--pin", "rosamund,kafka"]).pin == "rosamund,kafka"
    assert tts_server.build_parser().parse_args([]).pin == ""


def test_set_voice_reports_whether_it_swapped(client):
    assert client.post("/voice", json={"name": "ayaka"}).json() == {"voice": "ayaka", "swapped": True}
    assert client.post("/voice", json={"name": "ayaka"}).json() == {"voice": "ayaka", "swapped": False}


def test_set_unknown_voice_is_404(client):
    assert client.post("/voice", json={"name": "nobody"}).status_code == 404


def test_tts_returns_a_16_bit_wav_with_the_sample_rate(client):
    resp = client.post("/tts", json={"text": "こんにちは", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.headers["x-sample-rate"] == str(SR)
    data, sr = sf.read(io.BytesIO(resp.content), dtype="int16")
    assert sr == SR and len(data) == SR
    assert sf.info(io.BytesIO(resp.content)).subtype == "PCM_16"


def test_tts_streaming_returns_raw_pcm_chunks(client):
    with client.stream("POST", "/tts", params={"stream": "true"}, json={"text": "hi", "text_lang": "en", "voice": "juliet"}) as resp:
        body = b"".join(resp.iter_bytes())

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/L16"
    assert resp.headers["x-sample-rate"] == str(SR)
    assert body == b"\x01\x00" * 100 + b"\x02\x00" * 100


def test_the_instance_is_released_after_a_streamed_response(client):
    client.post("/tts", params={"stream": "true"}, json={"text": "hi", "text_lang": "en", "voice": "juliet"})

    assert client.get("/").json()["instances"][0]["busy"] is False


@pytest.mark.parametrize(
    "body,status",
    [
        ({"text": "  ", "text_lang": "ja", "voice": "ayaka"}, 400),
        ({"text": "hi", "text_lang": "de", "voice": "ayaka"}, 400),
        ({"text": "hi", "text_lang": "ja", "voice": "nobody"}, 404),
        ({"text": "hi", "text_lang": "ja"}, 422),  # voice is required
    ],
)
def test_tts_rejects_bad_requests(client, body, status):
    resp = client.post("/tts", json=body)

    assert resp.status_code == status
    assert "detail" in resp.json()


def test_tts_is_503_when_no_instance_can_start(client):
    FakeInstance.fail_start = {"ayaka"}

    resp = client.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 503
    assert "cannot start ayaka" in resp.json()["detail"]
    assert client.get("/").json()["status"] == "degraded"


def test_tts_respells_words_from_the_lexicon(client):
    from voice_services.tts.pronunciations import Lexicon

    client.app.state.pronunciations = {"en": Lexicon({"API": "ay pee eye"})}
    seen = []
    orig = FakeInstance.synthesize
    FakeInstance.synthesize = lambda self, text, language, sampling=None: (seen.append(text), orig(self, text, language, sampling))[1]
    try:
        client.post("/tts", json={"text": "The api is down.", "text_lang": "en", "voice": "juliet"})
        client.post("/tts", json={"text": "The api is down.", "text_lang": "ja", "voice": "ayaka"})
    finally:
        FakeInstance.synthesize = orig

    assert seen == ["The ay pee eye is down.", "The api is down."]


def _pron_client(client, tmp_path, token="s3cret"):
    d = tmp_path / "pron"
    d.mkdir()
    (d / "en.json").write_text('{"API": "ay pee eye"}', encoding="utf-8")
    client.app.state.pronunciations_dir = d
    client.app.state.admin_token = token
    client.app.state.pronunciations = pronunciations.load(d)
    return d


AUTH = {"Authorization": "Bearer s3cret"}


def test_patch_pronunciations_updates_file_and_live_lexicon(client, tmp_path):
    d = _pron_client(client, tmp_path)

    r = client.patch("/pronunciations/en", json={"set": {"GPU": "jee pee you"}, "remove": ["API"]}, headers=AUTH)

    assert r.status_code == 200 and r.json() == {"lang": "en", "entries": {"GPU": "jee pee you"}}
    assert json.loads((d / "en.json").read_text(encoding="utf-8")) == {"GPU": "jee pee you"}
    assert client.get("/pronunciations/en").json()["entries"] == {"GPU": "jee pee you"}
    assert client.app.state.pronunciations["en"].apply("a GPU") == "a jee pee you"


def test_patch_pronunciations_needs_the_token(client, tmp_path):
    d = _pron_client(client, tmp_path)
    body = {"set": {"GPU": "jee pee you"}}

    assert client.patch("/pronunciations/en", json=body).status_code == 401
    assert client.patch("/pronunciations/en", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    client.app.state.admin_token = ""
    assert client.patch("/pronunciations/en", json=body, headers=AUTH).status_code == 403
    assert json.loads((d / "en.json").read_text(encoding="utf-8")) == {"API": "ay pee eye"}


def test_patch_pronunciations_validates_language_and_input(client, tmp_path):
    _pron_client(client, tmp_path)

    assert client.patch("/pronunciations/de", json={"set": {"a": "b"}}, headers=AUTH).status_code == 404
    assert client.patch("/pronunciations/en", json={"set": {"a": ""}}, headers=AUTH).status_code == 400
    assert client.get("/pronunciations/de").status_code == 404


def test_tts_passes_sampling_overrides_to_the_instance(client):
    body = {"text": "hi", "text_lang": "en", "voice": "juliet"}
    client.post("/tts", json={**body, "temperature": 0.6, "top_p": 0.8, "top_k": 10, "speed_factor": 1.2})
    inst = next(i for i in FakeInstance.created if i.voice_name == "juliet")
    assert inst.sampling == {"temperature": 0.6, "top_p": 0.8, "top_k": 10, "speed_factor": 1.2}

    client.post("/tts", params={"stream": "true"}, json={**body, "temperature": 0.5})
    assert inst.sampling == {"temperature": 0.5}

    client.post("/tts", json=body)
    assert inst.sampling == {}


@pytest.mark.parametrize("extra", [{"temperature": 0}, {"top_p": 1.5}, {"top_k": 0}, {"speed_factor": 0.2}, {"speed_factor": 3}])
def test_tts_rejects_out_of_range_sampling(client, extra):
    resp = client.post("/tts", json={"text": "hi", "text_lang": "en", "voice": "juliet", **extra})

    assert resp.status_code == 422


def test_tts_turns_dashes_colons_and_semicolons_into_pauses(client):
    seen = []
    orig = FakeInstance.synthesize
    FakeInstance.synthesize = lambda self, text, language, sampling=None: (seen.append(text), orig(self, text, language, sampling))[1]
    try:
        client.post("/tts", json={"text": "Wait — what: now; go.", "text_lang": "en", "voice": "juliet"})
    finally:
        FakeInstance.synthesize = orig
    assert seen == ["Wait, what, now, go."]


def test_tts_fragment_interval_request_value_beats_the_server_default(client):
    body = {"text": "hi", "text_lang": "en", "voice": "juliet"}
    client.app.state.fragment_interval = 0.6
    try:
        client.post("/tts", json=body)
        inst = next(i for i in FakeInstance.created if i.voice_name == "juliet")
        assert inst.sampling == {"fragment_interval": 0.6}
        client.post("/tts", json={**body, "fragment_interval": 0.1})
        assert inst.sampling == {"fragment_interval": 0.1}
        assert client.post("/tts", json={**body, "fragment_interval": -1}).status_code == 422
    finally:
        client.app.state.fragment_interval = None


def test_voices_lists_kokoro_voices_with_their_engine(client, kokoro):
    body = client.get("/voices").json()

    assert {"name": "ff_siwis", "ref_lang": "fr", "version": None, "engine": "kokoro"} in body["voices"]
    assert {"name": "if_sara", "ref_lang": "it", "version": None, "engine": "kokoro"} in body["voices"]
    assert body["voices"][0]["engine"] == "gpt-sovits"  # the voice directories come first
    assert body["loaded"] == []  # `loaded` is the pool's: Kokoro voices never need loading
    assert [v["name"] for v in client.get("/voices", params={"language": "fr"}).json()["voices"]] == ["ff_siwis"]


def test_voices_has_no_kokoro_voices_when_the_engine_is_missing_or_unavailable(client):
    assert all(v["engine"] == "gpt-sovits" for v in client.get("/voices").json()["voices"])
    tts_server.app.state.kokoro = FakeKokoro(available=False, error="ImportError: no kokoro")
    assert all(v["engine"] == "gpt-sovits" for v in client.get("/voices").json()["voices"])


@pytest.mark.parametrize("lang, voice", [("fr", "ff_siwis"), ("it", "if_sara")])
def test_tts_routes_a_kokoro_voice_to_kokoro_not_the_pool(client, kokoro, lang, voice):
    resp = client.post("/tts", json={"text": "Bonjour.", "text_lang": lang, "voice": voice})

    assert resp.status_code == 200 and resp.headers["X-Sample-Rate"] == "24000"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 24000 and len(data) == 24000
    assert kokoro.calls[0][:3] == ("synth", voice, "Bonjour.")
    assert FakeInstance.created == []  # the pool was never touched
    assert client.get("/").json()["instances"][0]["voice"] is None


def test_tts_streams_a_kokoro_voice(client, kokoro):
    resp = client.post("/tts", params={"stream": "true"}, json={"text": "Ciao.", "text_lang": "it", "voice": "if_sara"})

    assert resp.status_code == 200 and resp.headers["X-Sample-Rate"] == "24000"
    assert resp.content == b"\x03\x00" * 50
    assert FakeInstance.created == []


def test_tts_accepts_fr_and_it_and_still_400s_for_de(client, kokoro):
    for lang in ("fr", "it"):
        assert client.post("/tts", json={"text": "x", "text_lang": lang, "voice": "ayaka"}).status_code == 200
    resp = client.post("/tts", json={"text": "x", "text_lang": "de", "voice": "ayaka"})
    assert resp.status_code == 400 and "ja" in resp.json()["detail"] and "fr" in resp.json()["detail"]


def test_tts_kokoro_ignores_sampling_except_speed(client, kokoro):
    resp = client.post("/tts", json={
        "text": "Bonjour.", "text_lang": "fr", "voice": "ff_siwis",
        "temperature": 0.5, "top_p": 0.9, "top_k": 5, "fragment_interval": 1.0, "speed_factor": 1.5,
    })

    assert resp.status_code == 200
    assert kokoro.calls[0][3]["speed_factor"] == 1.5  # the engine reads only this; the rest it is free to drop


def test_set_voice_for_a_kokoro_voice_is_not_swapped(client, kokoro):
    resp = client.post("/voice", json={"name": "ff_siwis"})

    assert resp.status_code == 200 and resp.json() == {"voice": "ff_siwis", "swapped": False}
    assert FakeInstance.created == []


def test_a_kokoro_voice_is_404_when_kokoro_is_unavailable(client):
    tts_server.app.state.kokoro = FakeKokoro(available=False, error="ImportError: no kokoro")

    for call in (
        lambda: client.post("/tts", json={"text": "x", "text_lang": "fr", "voice": "ff_siwis"}),
        lambda: client.post("/voice", json={"name": "ff_siwis"}),
    ):
        resp = call()
        assert resp.status_code == 404 and "no kokoro" in resp.json()["detail"]


def test_a_kokoro_voice_is_404_when_there_is_no_engine_at_all(client):
    resp = client.post("/tts", json={"text": "x", "text_lang": "fr", "voice": "ff_siwis"})

    assert resp.status_code == 404 and "Kokoro" in resp.json()["detail"]


def test_a_kokoro_failure_is_503(client, kokoro):
    kokoro.fail = True

    resp = client.post("/tts", json={"text": "x", "text_lang": "fr", "voice": "ff_siwis"})

    assert resp.status_code == 503 and "kokoro failed" in resp.json()["detail"]


def test_pronunciations_cover_fr_and_it(client, tmp_path):
    d = _pron_client(client, tmp_path)

    for lang in ("fr", "it"):
        assert client.get(f"/pronunciations/{lang}").json() == {"lang": lang, "entries": {}}
    r = client.patch("/pronunciations/fr", json={"set": {"GPU": "gé pé u"}}, headers=AUTH)
    assert r.status_code == 200 and json.loads((d / "fr.json").read_text(encoding="utf-8")) == {"GPU": "gé pé u"}


def test_a_language_kokoro_could_not_load_is_not_listed_and_its_voice_is_404(client):
    tts_server.app.state.kokoro = FakeKokoro(languages=("fr", "it"), errors={"ja": "Failed initializing MeCab"})

    names = [v["name"] for v in client.get("/voices").json()["voices"] if v["engine"] == "kokoro"]
    resp = client.post("/tts", json={"text": "x", "text_lang": "ja", "voice": "jf_alpha"})

    assert names == ["ff_siwis", "if_sara"]
    assert resp.status_code == 404 and "MeCab" in resp.json()["detail"]


# -- fallback to Kokoro -------------------------------------------------------------------------------


def test_pool_503_falls_back_to_kokoro_with_headers(client, kokoro):
    FakeInstance.fail_start = {"ayaka"}

    resp = client.post("/tts", json={"text": "こんにちは", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 200 and resp.headers["X-Sample-Rate"] == "24000"
    assert resp.headers["X-Fallback-Voice"] == "jf_alpha" and resp.headers["X-Fallback-From"] == "ayaka"
    assert kokoro.calls[0][:3] == ("synth", "jf_alpha", "こんにちは")


def test_a_streaming_start_failure_falls_back_too(client, kokoro):
    FakeInstance.fail_start = {"juliet"}

    resp = client.post("/tts", params={"stream": "true"}, json={"text": "Hello.", "text_lang": "en", "voice": "juliet"})

    assert resp.status_code == 200 and resp.content == b"\x03\x00" * 50
    assert resp.headers["X-Fallback-Voice"] == "af_heart" and resp.headers["X-Fallback-From"] == "juliet"
    assert kokoro.calls[0][0] == "stream"


def test_the_fallback_voice_follows_text_lang_not_the_requested_voice(client, kokoro):
    FakeInstance.fail_start = {"ayaka"}

    resp = client.post("/tts", json={"text": "Bonjour.", "text_lang": "fr", "voice": "ayaka"})

    assert resp.headers["X-Fallback-Voice"] == "ff_siwis"


def test_no_fallback_for_a_language_kokoro_lacks(client):
    tts_server.app.state.kokoro = FakeKokoro(languages=("fr", "it"))
    FakeInstance.fail_start = {"ayaka"}

    resp = client.post("/tts", json={"text": "こんにちは", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 503 and "cannot start ayaka" in resp.json()["detail"]
    assert "X-Fallback-Voice" not in resp.headers


def test_no_fallback_without_a_kokoro_engine(client):
    FakeInstance.fail_start = {"ayaka"}

    assert client.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "ayaka"}).status_code == 503


def test_no_fallback_for_an_unknown_voice(client, kokoro):
    resp = client.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "nobody"})

    assert resp.status_code == 404 and kokoro.calls == []


def test_a_kokoro_voice_that_fails_is_not_retried(client, kokoro):
    kokoro.fail = True

    resp = client.post("/tts", json={"text": "x", "text_lang": "fr", "voice": "ff_siwis"})

    assert resp.status_code == 503 and len(kokoro.calls) == 1 and "X-Fallback-Voice" not in resp.headers


def test_when_the_fallback_fails_too_the_error_names_both(client, kokoro):
    kokoro.fail = True
    FakeInstance.fail_start = {"ayaka"}

    resp = client.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 503
    assert "cannot start ayaka" in resp.json()["detail"] and "kokoro failed" in resp.json()["detail"]


def test_normal_responses_carry_no_fallback_headers(client, kokoro):
    resp = client.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "ayaka"})

    assert resp.status_code == 200 and "X-Fallback-Voice" not in resp.headers and kokoro.calls == []


def test_a_fallback_leaves_a_pinned_voice_loaded(tmp_path):
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "juliet", juliet={"ref_lang": "en"})
    tts_server.app.state.pool = TTSPool(
        TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9300, factory=FakeInstance, pinned=["juliet"]
    )
    tts_server.app.state.preload = []
    tts_server.app.state.kokoro = FakeKokoro()
    try:
        with TestClient(tts_server.app) as c:
            c.post("/voice", json={"name": "juliet"})
            FakeInstance.fail_start = {"ayaka"}
            resp = c.post("/tts", json={"text": "hi", "text_lang": "ja", "voice": "ayaka"})
            assert resp.headers["X-Fallback-Voice"] == "jf_alpha"
            assert "juliet" in c.get("/voices").json()["loaded"]
    finally:
        tts_server.app.state.kokoro = None
