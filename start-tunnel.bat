@echo off
title Cloudflare Tunnel - AI kamera
cd /d "%~dp0"

set "CF=%~dp0tools\cloudflared.exe"
if not exist "%CF%" (
  echo XATO: tools\cloudflared.exe topilmadi.
  echo Avval cloudflared yuklab qo'ying.
  pause
  exit /b 1
)

echo === Cloudflare Tunnel ===
echo Lokal: http://127.0.0.1:8080/ai.html
echo.
echo AI worker ishlayotganiga ishonch hosil qiling (start-ai.bat).
echo Quyida https://....trycloudflare.com havolasi chiqadi.
echo Telefon (mobil internet) shu havola orqali kiradi.
echo.
echo To'xtatish: Ctrl+C
echo.

"%CF%" tunnel --url http://127.0.0.1:8080 --no-autoupdate
pause
