# voice-services

Whisper speech-to-text and GPT-SoVITS text-to-speech as HTTP services. CUDA-first.
Extracted from the `voice` project (commit 650dd4a); voice-js is the main consumer.

| service | run | default port | contract |
|---|---|---|---|
| STT | `python -m voice_services.stt` | 8771 | [docs/STT_API.md](docs/STT_API.md) |
| TTS | `python -m voice_services.tts` (in progress) | 8772 | [docs/TTS_API.md](docs/TTS_API.md) |

No authentication: bind to localhost or a Tailscale interface only.

## Setup

1. Install the CUDA build of torch for your machine first (a plain `pip install` pulls a CPU wheel on Windows).
2. `pip install -e ".[dev]"`
3. Copy `.env.example` to `.env` and adjust.
4. `pytest`
