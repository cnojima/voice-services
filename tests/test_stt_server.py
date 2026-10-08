"""voice/stt_server.py: FastAPI routing and wiring. Transcriber is stubbed --
this only exercises the HTTP surface (body decoding, language/candidates
params, error responses), not real Whisper.
"""
from __future__ import annotations

import numpy as np
import pytest
from starlette.testclient import TestClient

import voice_services.stt.server as stt_server
from voice_services.config import WhisperConfig


class _FakeTranscriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cpu"

    def detect_language(self, audio, candidates):
        return (candidates or ("en",))[0]

    def transcribe(self, audio, language):
        return f"[{language}] {len(audio)} samples", {
            "max_no_speech_prob": 0.0,
            "avg_logprob": -0.1,
            "duration_s": len(audio) / 16000.0,
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(stt_server, "Transcriber", _FakeTranscriber)
    stt_server.app.state.cfg = WhisperConfig()
    with TestClient(stt_server.app) as c:
        yield c


def _pcm(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype="<f4").tobytes()


def test_health_reports_model_and_device(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "turbo", "device": "cpu"}


def test_transcribe_with_explicit_language_skips_detection(client):
    resp = client.post("/transcribe", params={"language": "ja"}, content=_pcm(1600))

    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "ja"
    assert body["text"] == "[ja] 1600 samples"
    assert body["duration_s"] == pytest.approx(0.1)


def test_transcribe_without_language_detects_among_candidates(client):
    resp = client.post("/transcribe", params={"candidates": "ja,en"}, content=_pcm(800))

    assert resp.status_code == 200
    assert resp.json()["language"] == "ja"  # first of the candidates, per _FakeTranscriber


def test_transcribe_without_language_or_candidates_falls_back_to_en(client):
    resp = client.post("/transcribe", content=_pcm(800))

    assert resp.status_code == 200
    assert resp.json()["language"] == "en"


def test_malformed_body_length_is_rejected(client):
    resp = client.post("/transcribe", content=b"\x00\x00\x00")

    assert resp.status_code == 400
    assert "multiple of 4 bytes" in resp.json()["detail"]
