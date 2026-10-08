@echo off
rem STT service (Whisper) on :8771. Settings (model, port, host) come from .env / VOICE_STT_* -- see .env.example.
rem Extra arguments are passed through, e.g.: start-stt.bat --model small
cd /d "%~dp0"
if not defined VOICE_SERVICES_PYTHON set "VOICE_SERVICES_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VOICE_SERVICES_PYTHON%" (
    echo Python not found: %VOICE_SERVICES_PYTHON% 1>&2
    echo Create the venv per README.md, or set VOICE_SERVICES_PYTHON to a python that has this project installed. 1>&2
    exit /b 1
)
set PYTHONUTF8=1
"%VOICE_SERVICES_PYTHON%" -m voice_services.stt %*
