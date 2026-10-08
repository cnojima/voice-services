@echo off
rem TTS service (GPT-SoVITS voices) on :8772. Needs VOICE_GPT_SOVITS_ROOT in .env (run scripts\check_setup.py).
rem Pool size and preloaded voices come from VOICE_TTS_POOL_SIZE / VOICE_TTS_PRELOAD in .env.
rem Stop with Ctrl+C in this window: closing the window can leave the api_v2.py processes running.
rem Extra arguments are passed through, e.g.: start-tts.bat --pool-size 3 --preload ayaka,rosamund,kafka
cd /d "%~dp0"
if not defined VOICE_SERVICES_PYTHON set "VOICE_SERVICES_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VOICE_SERVICES_PYTHON%" (
    echo Python not found: %VOICE_SERVICES_PYTHON% 1>&2
    echo Create the venv per README.md, or set VOICE_SERVICES_PYTHON to a python that has this project installed. 1>&2
    exit /b 1
)
set PYTHONUTF8=1
"%VOICE_SERVICES_PYTHON%" -m voice_services.tts %*
