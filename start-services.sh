#!/usr/bin/env bash
# Both services in one terminal (STT :8771, TTS :8772). Ctrl+C stops both, and the TTS server
# stops its api_v2.py processes on the way out. If either service dies, the other is stopped too.
set -uo pipefail
cd "$(dirname "$0")"

./start-stt.sh &
stt=$!
./start-tts.sh &
tts=$!

stop() {
  trap - INT TERM EXIT
  kill "$stt" "$tts" 2>/dev/null
  wait 2>/dev/null
}
trap stop INT TERM EXIT

wait -n
