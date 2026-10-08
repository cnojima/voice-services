#!/usr/bin/env bash
# TTS service (GPT-SoVITS voices) on :8772. Needs VOICE_GPT_SOVITS_ROOT in .env (run scripts/check_setup.py).
# Pool size and preloaded voices come from VOICE_TTS_POOL_SIZE / VOICE_TTS_PRELOAD in .env.
# Extra arguments are passed through, e.g.: ./start-tts.sh --pool-size 3 --preload ayaka,rosamund,kafka
# exec keeps SIGINT/SIGTERM going straight to the server, which then stops its api_v2.py processes.
set -euo pipefail
cd "$(dirname "$0")"
PY="${VOICE_SERVICES_PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python not found: $PY" >&2
  echo "Create the venv per README.md, or set VOICE_SERVICES_PYTHON to a python that has this project installed." >&2
  exit 1
fi
export PYTHONUTF8=1
exec "$PY" -m voice_services.tts "$@"
