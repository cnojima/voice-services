"""Whisper speech-to-text as a local HTTP API (contract: docs/STT_API.md).

    python -m voice_services.stt                      # :8771
    python -m voice_services.stt --model turbo --port 8771

Audio in is raw little-endian float32 PCM, mono, 16 kHz. The caller
resamples/mixes down before sending; this server does not.

Moved from the `voice` project's voice/stt_server.py (commit 650dd4a).
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from voice_services import env
from voice_services.config import WhisperConfig
from voice_services.stt.transcriber import Transcriber


@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg: WhisperConfig = app.state.cfg
    print(f"Loading Whisper ({cfg.model}, device={cfg.device})...", flush=True)
    app.state.transcriber = Transcriber(cfg)
    print(f"Ready on device {app.state.transcriber.device}.", flush=True)
    yield


app = FastAPI(lifespan=_lifespan)


@app.get("/")
def health() -> JSONResponse:
    t: Transcriber = app.state.transcriber
    return JSONResponse({"status": "ok", "model": t.cfg.model, "device": t.device})


def _pcm(body: bytes) -> np.ndarray:
    if len(body) % 4 != 0:
        raise HTTPException(400, "body length is not a multiple of 4 bytes (expected float32 PCM)")
    return np.frombuffer(body, dtype="<f4").copy()  # writable: torch warns on read-only arrays


def _candidates(candidates: str | None) -> tuple[str, ...] | None:
    return tuple(c.strip() for c in candidates.split(",") if c.strip()) if candidates else None


@app.post("/detect")
async def detect(request: Request, candidates: str | None = None) -> JSONResponse:
    """Detection alone, for callers that detect and transcribe as two steps. `candidates` as
    for /transcribe."""
    t: Transcriber = app.state.transcriber
    return JSONResponse({"language": t.detect_language(_pcm(await request.body()), _candidates(candidates))})


@app.post("/transcribe")
async def transcribe(request: Request, language: str | None = None, candidates: str | None = None) -> JSONResponse:
    """`language` pins the transcript language, skipping detection (fastest,
    matches Transcriber.transcribe's default). Omit it and pass `candidates`
    (comma-separated) to auto-detect among those first, same as
    Transcriber.detect_language. Neither given: detects among the server's
    own cfg.languages."""
    audio = _pcm(await request.body())

    t: Transcriber = app.state.transcriber
    if language is None:
        language = t.detect_language(audio, _candidates(candidates))

    text, info = t.transcribe(audio, language)
    return JSONResponse({"text": text, "language": language, **info})


def build_parser():
    import argparse

    cfg = WhisperConfig()
    p = argparse.ArgumentParser(prog="voice_services.stt", description=__doc__)
    p.add_argument("--host", default=os.environ.get("VOICE_STT_HOST", "127.0.0.1"),
                   help="interface to bind (default: localhost only; no auth, so tailnet/localhost only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("VOICE_STT_PORT", "8771")))
    p.add_argument("--model", default=os.environ.get("VOICE_STT_MODEL", cfg.model), help="Whisper model name")
    p.add_argument("--device", default=os.environ.get("VOICE_STT_DEVICE", cfg.device), help="auto, cpu, cuda, ...")
    p.add_argument("--languages", default=os.environ.get("VOICE_STT_LANGUAGES", ",".join(cfg.languages)),
                   help="comma-separated languages to detect among when a request names none")
    return p


def main(argv: list[str] | None = None) -> None:
    env.load_env()
    args = build_parser().parse_args(argv)
    languages = tuple(l.strip() for l in args.languages.split(",") if l.strip())
    app.state.cfg = WhisperConfig(model=args.model, device=args.device, languages=languages)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
