@echo off
title MIAL - Voice-First AI Machine Learning Assistant
cd /d "%~dp0"

echo =====================================================================
echo   MIAL - Voice-First AI Machine Learning Assistant
echo   Built on SmartML - Anna University HMTK Hackathon
echo =====================================================================
echo.
echo Launching Mial desktop floating overlay...
echo Hotkey: Press [Ctrl + Alt + M] anywhere to summon Mial!
echo.

python -m smartml.mial

if %errorlevel% neq 0 (
    echo.
    echo [Notice] If python command was not found or exited with error, check:
    echo python -m smartml.mial
    pause
)

