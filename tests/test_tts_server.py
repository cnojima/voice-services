"""voice_services/tts/server.py HTTP surface against a pool of FakeInstances."""
from __future__ import annotations

import io

import pytest
import soundfile as sf
from starlette.testclient import TestClient

import voice_services.tts.server as tts_server
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
