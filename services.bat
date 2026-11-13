@echo off
rem Control the voice-stt / voice-tts Windows services from any shell, including an SSH session.
rem   services.bat <start|stop|restart|status> [stt|tts|all]      (default: all)
rem start/stop/restart need Administrator rights (an SSH login as an admin account has them); status does not.
rem Install the services first with install-services.bat.
setlocal
cd /d "%~dp0"
set "action=%~1"
set "which=%~2"
if not defined which set "which=all"
set "ok="
for %%a in (start stop restart status) do if /i "%action%"=="%%a" set "ok=1"
if not defined ok goto usage
set "list="
if /i "%which%"=="all" set "list=STT TTS"
if /i "%which%"=="stt" set "list=STT"
if /i "%which%"=="tts" set "list=TTS"
if not defined list goto usage
set "rc=0"
for %%s in (%list%) do call :%action% %%s
exit /b %rc%

:usage
echo usage: services.bat ^<start^|stop^|restart^|status^> [stt^|tts^|all] 1>&2
exit /b 2

:start
call :state %1
if "%state%"=="RUNNING" (
    echo voice-%1 already running
    exit /b 0
)
net start voice-%1 || set "rc=1"
exit /b 0

:stop
call :state %1
if "%state%"=="STOPPED" (
    echo voice-%1 already stopped
    exit /b 0
)
net stop voice-%1 || set "rc=1"
exit /b 0

:restart
call :stop %1
call :start %1
exit /b 0

:status
call :state %1
if "%state%"=="not installed" (
    echo voice-%1: not installed
    exit /b 0
)
set "port=8771"
if /i "%1"=="TTS" set "port=8772"
if exist .env for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"VOICE_%1_PORT=" .env') do set "port=%%b"
set "code=000"
for /f %%c in ('curl.exe -s -o nul -w "%%{http_code}" --max-time 3 http://localhost:%port%/') do set "code=%%c"
if "%code%"=="000" (
    echo voice-%1: %state%, no http response on :%port% ^(still loading models, or down - see logs\voice-%1.log^)
) else (
    echo voice-%1: %state%, http %code% on :%port%
)
exit /b 0

:state
set "state=not installed"
for /f "tokens=4" %%a in ('sc query voice-%1 2^>nul ^| findstr STATE') do set "state=%%a"
exit /b 0
