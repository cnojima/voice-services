@echo off
rem Stops and removes the voice-stt and voice-tts Windows services (see install-services.bat). Run from an elevated prompt.
setlocal
net session >nul 2>&1
if errorlevel 1 (
    echo Run this from an elevated prompt ^(Administrator^). 1>&2
    exit /b 1
)
where nssm >nul 2>&1
if errorlevel 1 (
    echo nssm not found on PATH. 1>&2
    exit /b 1
)
for %%s in (voice-stt voice-tts) do (
    sc query %%s >nul 2>&1
    if not errorlevel 1 (
        nssm stop %%s >nul 2>&1
        nssm remove %%s confirm
    ) else (
        echo %%s is not installed
    )
)
