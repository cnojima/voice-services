"""GPT-SoVITS voices as a local HTTP API (contract: docs/TTS_API.md).

    python -m voice_services.tts                                  # :8772, pool of 2
    python -m voice_services.tts --pool-size 3 --preload ayaka,rosamund,kafka
    python -m voice_services.tts --pin rosamund --preload ayaka    # rosamund is never swapped out
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO

import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from voice_services import env
from voice_services.config import TTSConfig
from voice_services.tts.pool import TTSPool, TTSUnavailable
from voice_services.tts.voices import VoiceNotFoundError, default_weights_root, list_voices, load_voice

TEXT_LANGUAGES = ("ja", "en")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    pool: TTSPool = app.state.pool
    pool.preload(getattr(app.state, "preload", []))  # background: the server accepts connections while it loads
    yield
    pool.close()


app = FastAPI(lifespan=_lifespan)


class TTSRequest(BaseModel):
    text: str
    text_lang: str
    voice: str


class VoiceRequest(BaseModel):
    name: str


def _pool() -> TTSPool:
    return app.state.pool


def _acquire(voice: str):
    try:
        return _pool().acquire(voice)
    except VoiceNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except TTSUnavailable as e:
        raise HTTPException(503, str(e)) from e


@app.get("/")
def health() -> JSONResponse:
    return JSONResponse(_pool().status())


@app.get("/voices")
def voices(language: str | None = None) -> JSONResponse:
    pool = _pool()
    out = []
    for name in list_voices(pool.cfg.weights_root, language):
        try:
            manifest = load_voice(pool.cfg.weights_root, name)
        except VoiceNotFoundError:
            continue
        out.append({"name": name, "ref_lang": manifest["ref_lang"], "version": manifest["version"]})
    return JSONResponse({"loaded": pool.loaded(), "pinned": pool.pinned(), "voices": out})


@app.post("/voice")
def set_voice(req: VoiceRequest) -> JSONResponse:
    with _acquire(req.name) as lease:
        return JSONResponse({"voice": req.name, "swapped": lease.swapped})


@app.post("/tts")
def tts(req: TTSRequest, stream: bool = Query(False)):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if req.text_lang not in TEXT_LANGUAGES:
        raise HTTPException(400, f"unsupported text_lang {req.text_lang!r}, expected one of {TEXT_LANGUAGES}")

    lease = _acquire(req.voice)
    if not stream:
        try:
            with lease:
                wav, sr = lease.inst.synthesize(req.text, req.text_lang)
        except Exception as e:  # noqa: BLE001 -- the lease already discarded a dead instance
            raise HTTPException(503, f"{type(e).__name__}: {e}") from e
        buf = BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav", headers={"X-Sample-Rate": str(sr)})

    try:
        sr, chunks = lease.inst.stream_tts(req.text, req.text_lang)
    except Exception as e:  # noqa: BLE001
        lease.release(e)
        raise HTTPException(503, f"{type(e).__name__}: {e}") from e

    def body():
        failed = None
        try:
            yield from chunks
        except Exception as e:  # noqa: BLE001 -- bytes are already out; the connection just ends
            failed = e
            print(f"TTS stream failed mid-response: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        finally:
            chunks.close()
            lease.release(failed)

    return StreamingResponse(body(), media_type="audio/L16", headers={"X-Sample-Rate": str(sr)})


def build_parser():
    import argparse

    p = argparse.ArgumentParser(prog="voice_services.tts", description=__doc__)
    p.add_argument("--host", default=os.environ.get("VOICE_TTS_HOST", "127.0.0.1"),
                   help="interface to bind (default: localhost only; no auth, so tailnet/localhost only)")
    p.add_argument("--port", type=int, default=int(os.environ.get("VOICE_TTS_PORT", "8772")))
    p.add_argument("--pool-size", type=int, default=int(os.environ.get("VOICE_TTS_POOL_SIZE", "2")),
                   help="upstream api_v2.py processes, one voice each")
    p.add_argument("--preload", default=os.environ.get("VOICE_TTS_PRELOAD", ""),
                   help="comma-separated voices to load at startup, one per instance")
    p.add_argument("--pin", default=os.environ.get("VOICE_TTS_PIN", ""),
                   help="comma-separated voices that keep their instance: loaded at startup and never swapped out "
                        "(fewer than the pool size, so one instance can still swap)")
    p.add_argument("--upstream-base-port", type=int, default=int(os.environ.get("VOICE_TTS_UPSTREAM_PORT", "9890")),
                   help="first localhost port for the upstream processes (one per instance)")
    return p


def main(argv: list[str] | None = None) -> None:
    env.load_env()
    args = build_parser().parse_args(argv)
    cfg = TTSConfig(
        weights_root=default_weights_root(),
        gpt_sovits_root=os.environ.get("VOICE_GPT_SOVITS_ROOT", ""),
        ffmpeg_bin=os.environ.get("VOICE_FFMPEG_BIN", ""),
    )
    pins = [v.strip() for v in args.pin.split(",") if v.strip()]
    try:
        for name in pins:
            load_voice(cfg.weights_root, name)  # a typo should stop startup, not surface at the first request
        app.state.pool = TTSPool(cfg, size=args.pool_size, base_port=args.upstream_base_port, pinned=pins)
    except (VoiceNotFoundError, ValueError) as e:
        sys.exit(f"--pin: {e}")
    app.state.preload = [v.strip() for v in args.preload.split(",") if v.strip()]

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
