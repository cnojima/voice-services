@echo off
rem Installs STT and TTS as Windows services via NSSM (voice-stt, voice-tts): start at boot, restart on crash, logs in logs\.
rem Run from an elevated prompt. Re-running reinstalls. Install NSSM first: winget install NSSM.NSSM
rem Services run as LocalSystem by default. To run as your own account (user-profile caches, mapped paths):
rem   install-services.bat .\youruser yourpassword
setlocal
cd /d "%~dp0"
net session >nul 2>&1
if errorlevel 1 (
    echo Run this from an elevated prompt ^(Administrator^). 1>&2
    exit /b 1
)
where nssm >nul 2>&1
if errorlevel 1 (
    echo nssm not found on PATH. Install it with: winget install NSSM.NSSM ^(then open a new prompt^) 1>&2
    exit /b 1
)
if not defined VOICE_SERVICES_PYTHON set "VOICE_SERVICES_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VOICE_SERVICES_PYTHON%" (
    echo Python not found: %VOICE_SERVICES_PYTHON% 1>&2
    echo Create the venv per README.md, or set VOICE_SERVICES_PYTHON to a python that has this project installed. 1>&2
    exit /b 1
)
if not exist logs mkdir logs
set "svcuser=%~1"
set "svcpass=%~2"
call :install voice-stt stt "Whisper speech-to-text" || exit /b 1
call :install voice-tts tts "GPT-SoVITS text-to-speech" || exit /b 1
echo.
echo Installed. Start with: services.bat start
exit /b 0

:install
set "svc=%~1"
sc query %svc% >nul 2>&1
if not errorlevel 1 (
    echo Reinstalling %svc%
    nssm stop %svc% >nul 2>&1
    nssm remove %svc% confirm >nul
)
rem Run python directly, not via the .bat: a cmd wrapper would answer the stop signal with "Terminate batch job (Y/N)?".
nssm install %svc% "%VOICE_SERVICES_PYTHON%" -m voice_services.%2 >nul
if errorlevel 1 (
    echo Failed to install %svc% 1>&2
    exit /b 1
)
nssm set %svc% DisplayName "voice-services %~3" >nul
nssm set %svc% Description "voice-services %~3. Settings come from .env in the repo." >nul
nssm set %svc% AppDirectory "%CD%" >nul
nssm set %svc% AppEnvironmentExtra PYTHONUTF8=1 >nul
nssm set %svc% Start SERVICE_AUTO_START >nul
nssm set %svc% AppStdout "%CD%\logs\%svc%.log" >nul
nssm set %svc% AppStderr "%CD%\logs\%svc%.log" >nul
nssm set %svc% AppRotateFiles 1 >nul
nssm set %svc% AppRotateOnline 1 >nul
nssm set %svc% AppRotateBytes 10485760 >nul
rem Ctrl+C first and give it time, so TTS can shut down its api_v2.py children; NSSM kills the process tree after that.
nssm set %svc% AppStopMethodConsole 15000 >nul
if defined svcuser nssm set %svc% ObjectName "%svcuser%" "%svcpass%" >nul
echo Installed %svc% ^(log: logs\%svc%.log^)
exit /b 0
