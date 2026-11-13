"""voice_services/tts/server.py HTTP surface against a pool of FakeInstances."""
from __future__ import annotations

import io
import json

import pytest
import soundfile as sf
from starlette.testclient import TestClient

import voice_services.tts.server as tts_server
from voice_services.tts import pronunciations
from tests.fakes import SR, FakeInstance, make_voices
from voice_services.config import TTSConfig
from voice_services.tts.pool import TTSPool


@pytest.fixture
def client(tmp_path):
    FakeInstance.reset()
    make_voices(tmp_path, "ayaka", "juliet", juliet={"ref_lang": "en", "version": "v2Pro"})
    tts_server.app.state.pool = TTSPool(TTSConfig(weights_root=str(tmp_path)), size=2, base_port=9100, factory=FakeInstance)
    tts_server.app.state.preload = []
    with TestClient(tts_server.app) as c:
        yield c


def test_health_lists_instances(client):
    body = client.get("/").json()

    assert body["status"] == "ok" and body["ready"] is True
    assert [i["voice"] for i in body["instances"]] == [None, None]


def test_voices_lists_available_and_loaded(client):
    client.post("/voice", json={"name": "ayaka"})

    body = client.get("/voices").json()

    assert body["loaded"] == ["ayaka"]
    assert body["voices"] == [
        {"name": "ayaka", "ref_lang": "ja", "version": "v2"},
        {"name": "juliet", "ref_lang": "en", "version": "v2Pro"},
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
        ({"text": "hi", "text_lang": "fr", "voice": "ayaka"}, 400),
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

    assert client.patch("/pronunciations/fr", json={"set": {"a": "b"}}, headers=AUTH).status_code == 404
    assert client.patch("/pronunciations/en", json={"set": {"a": ""}}, headers=AUTH).status_code == 400
    assert client.get("/pronunciations/fr").status_code == 404


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
