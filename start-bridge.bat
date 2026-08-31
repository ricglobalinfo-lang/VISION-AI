@echo off
title Camera bridge - camera stream
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0bridge.ps1"
pause
