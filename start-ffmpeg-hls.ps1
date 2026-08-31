<#
.SYNOPSIS
  Local-only live camera: RTSP -> HLS in www/live (no hosting).
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Ffmpeg = Join-Path $Root "tools\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
if (-not (Test-Path $Ffmpeg)) { $Ffmpeg = "ffmpeg" }

$HlsDir = Join-Path $Root "www\live"
New-Item -ItemType Directory -Force -Path $HlsDir | Out-Null
Get-ChildItem $HlsDir -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\.(ts|m3u8)$' } | Remove-Item -Force -ErrorAction SilentlyContinue

# Local LAN can handle main stream; change to 102 if PC is slow.
$Rtsp = "rtsp://admin:YOUR_CAMERA_PASSWORD@192.168.1.100:554/Streaming/Channels/101"
$Out  = Join-Path $HlsDir "index.m3u8"
$Seg  = Join-Path $HlsDir "seg_%05d.ts"
$Log  = Join-Path $env:TEMP "camera-kurbonov-ffmpeg.log"

Write-Host "ffmpeg HLS (local) -> $HlsDir"
Write-Host "RTSP: Channels/101"

$ffArgs = @(
    "-hide_banner", "-loglevel", "warning",
    "-rtsp_transport", "tcp",
    "-i", $Rtsp,
    "-an",
    "-c:v", "copy",
    "-f", "hls",
    "-hls_time", "1",
    "-hls_list_size", "8",
    "-hls_flags", "delete_segments+append_list+omit_endlist+independent_segments",
    "-hls_segment_filename", $Seg,
    $Out
)

while ($true) {
    $p = Start-Process -FilePath $Ffmpeg -ArgumentList $ffArgs -WorkingDirectory $HlsDir `
        -RedirectStandardError $Log -PassThru -NoNewWindow
    Write-Host "ffmpeg pid=$($p.Id)"
    Wait-Process -Id $p.Id
    Write-Host "ffmpeg exited; restarting in 3s..."
    Start-Sleep -Seconds 3
}
