# voice-services

Whisper speech-to-text and GPT-SoVITS text-to-speech as HTTP services. CUDA-first.cle
Extracted from the `voice` project (commit 650dd4a); voice-js is the main consumer.

| service | run | default port | contract |
|---|---|---|---|
| STT | `python -m voice_services.stt` | 8771 | [docs/STT_API.md](docs/STT_API.md) |
| TTS | `python -m voice_services.tts --preload ayaka,rosamund` | 8772 | [docs/TTS_API.md](docs/TTS_API.md) |

No authentication: bind to localhost or a Tailscale interface only (`--host` / `VOICE_*_HOST`).

## Setup

1. Install the CUDA build of torch for your machine first (a plain `pip install` pulls a CPU wheel on Windows).
    `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; nvidia-smi | grep -o "CUDA Version: [0-9.]*"; grep -rn "download.pytorch.org\|index-url\|cu130\|cu128" --include=*.md --include=*.toml --include=*.sh --include=*.txt . 2>/dev/null | grep -v ".venv\|node_modules" | head; .venv/Scripts/python.exe -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"; cat .venv/Lib/site-packages/torch-*.dist-info/direct_url.json 2>/dev/null | head -3; grep -c . .venv/pyvenv.cfg && grep -i "version\|uv" .venv/pyvenv.cfg`
2. `pip install -e ".[dev,weights]"`
3. Copy `.env.example` to `.env` and set `VOICE_GPT_SOVITS_ROOT` (and `VOICE_WEIGHTS_ROOT`, `VOICE_FFMPEG_BIN` if needed).
4. `python scripts/fetch_weights.py` for the voice weights (needs `hf auth login`).
5. `python scripts/check_setup.py` says what is still missing.
6. `pytest`

```
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 1. CUDA torch first, from the PyTorch index. The default PyPI wheel on Windows is CPU-only.
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2. Then the project. Whisper will see torch is already satisfied and keep it.
.\.venv\Scripts\python.exe -m pip install -e ".[dev,weights]"

# 3. Verify
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running

| | Windows | Ubuntu |
|---|---|---|
| both services | `start-services.bat` (one window each) | `./start-services.sh` (one terminal; Ctrl+C stops both) |
| STT only | `start-stt.bat` | `./start-stt.sh` |
| TTS only | `start-tts.bat` | `./start-tts.sh` |

The scripts use `.venv` in this repo (or the python in `VOICE_SERVICES_PYTHON`), take settings from `.env`
(pool size, preloaded and pinned voices: `VOICE_TTS_POOL_SIZE`, `VOICE_TTS_PRELOAD`, `VOICE_TTS_PIN`), and pass extra arguments through.
Stop the Windows TTS window with Ctrl+C: closing it can leave the `api_v2.py` processes running.

## TTS needs more than this repo

TTS drives upstream GPT-SoVITS's own `api_v2.py`, in its own CUDA venv (`<VOICE_GPT_SOVITS_ROOT>/runtime`) with the
pretrained base models. This repo does not install that checkout. Each pool instance is one such process on
`VOICE_TTS_UPSTREAM_PORT` + its index, holding one voice; see the contract for how requests are routed.

## Status

Verified on Windows (RTX 5090): both services, pool of 2, streaming, voice swap, STT round trip of TTS output.
Not yet tried on the DGX Spark (aarch64 + CUDA): the upstream GPT-SoVITS stack there is the main unknown.
