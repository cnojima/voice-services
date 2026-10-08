#!/usr/bin/env bash
# STT service (Whisper) on :8771. Settings (model, port, host) come from .env / VOICE_STT_* -- see .env.example.
# Extra arguments are passed through, e.g.: ./start-stt.sh --model small
set -euo pipefail
cd "$(dirname "$0")"
PY="${VOICE_SERVICES_PYTHON:-./.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python not found: $PY" >&2
  echo "Create the venv per README.md, or set VOICE_SERVICES_PYTHON to a python that has this project installed." >&2
  exit 1
fi
export PYTHONUTF8=1
exec "$PY" -m voice_services.stt "$@"
