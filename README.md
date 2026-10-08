# voice-services

Whisper speech-to-text and GPT-SoVITS text-to-speech as HTTP services. CUDA-first.
Extracted from the `voice` project (commit 650dd4a); voice-js is the main consumer.

| service | run | default port | contract |
|---|---|---|---|
| STT | `python -m voice_services.stt` | 8771 | [docs/STT_API.md](docs/STT_API.md) |
| TTS | `python -m voice_services.tts --preload ayaka,rosamund` | 8772 | [docs/TTS_API.md](docs/TTS_API.md) |

No authentication: bind to localhost or a Tailscale interface only (`--host` / `VOICE_*_HOST`).

## Setup

1. Install the CUDA build of torch for your machine first (a plain `pip install` pulls a CPU wheel on Windows).
2. `pip install -e ".[dev,weights]"`
3. Copy `.env.example` to `.env` and set `VOICE_GPT_SOVITS_ROOT` (and `VOICE_WEIGHTS_ROOT`, `VOICE_FFMPEG_BIN` if needed).
4. `python scripts/fetch_weights.py` for the voice weights (needs `hf auth login`).
5. `python scripts/check_setup.py` says what is still missing.
6. `pytest`

## TTS needs more than this repo

TTS drives upstream GPT-SoVITS's own `api_v2.py`, in its own CUDA venv (`<VOICE_GPT_SOVITS_ROOT>/runtime`) with the
pretrained base models. This repo does not install that checkout. Each pool instance is one such process on
`VOICE_TTS_UPSTREAM_PORT` + its index, holding one voice; see the contract for how requests are routed.

## Status

Verified on Windows (RTX 5090): both services, pool of 2, streaming, voice swap, STT round trip of TTS output.
Not yet tried on the DGX Spark (aarch64 + CUDA): the upstream GPT-SoVITS stack there is the main unknown.
