@echo off
rem Both services, each in its own window (STT :8771, TTS :8772). Stop each with Ctrl+C.
cd /d "%~dp0"
start "voice-services STT" cmd /k call "%~dp0start-stt.bat"
start "voice-services TTS" cmd /k call "%~dp0start-tts.bat"
