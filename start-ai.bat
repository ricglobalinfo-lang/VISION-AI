@echo off
title AI kamera - lokal
cd /d "%~dp0"

set "PY=%~dp0ai\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo XATO: venv topilmadi. Avval Python o'rnating va:
  echo   py -3 -m venv ai\.venv
  echo   ai\.venv\Scripts\pip install -r ai\requirements.txt
  pause
  exit /b 1
)

echo === Lokal AI kamera ===
echo.

REM Stop conflicting local server / previous AI worker on 8080 best-effort
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -match 'worker\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Get-Process ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue"

timeout /t 2 /nobreak >nul

echo Yuz bazasini yangilash (faces\ papkasi)...
"%PY%" "%~dp0ai\enroll_faces.py"
if errorlevel 1 (
  echo enroll xatosi — bo'sh DB bilan davom etiladi.
)

echo.
echo AI worker ishga tushmoqda...
echo Brauzer: http://127.0.0.1:8080/ai.html
echo.
start "" "http://127.0.0.1:8080/ai.html"
"%PY%" "%~dp0ai\worker.py"
pause
