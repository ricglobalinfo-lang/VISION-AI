<#
.SYNOPSIS
  Serves www/ on http://0.0.0.0:8080 (this PC + LAN). No hosting.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Www = Join-Path $Root "www"
$Port = 8080

if (-not (Test-Path $Www)) {
    throw "www folder missing: $Www"
}

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:$Port/")
$listener.Prefixes.Add("http://localhost:$Port/")

# Bind LAN IPs so phones/PCs on same network can open the stream (no admin urlacl).
$lanIps = @()
try {
    $lanIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.PrefixOrigin -ne "WellKnown" -and
            $_.IPAddress -notlike "169.254.*"
        } |
        Select-Object -ExpandProperty IPAddress -Unique)
} catch {
    $lanIps = @()
}
foreach ($ip in $lanIps) {
    try { $listener.Prefixes.Add("http://${ip}:$Port/") } catch {}
}

try {
    $listener.Start()
} catch {
    throw "Cannot bind port $Port : $($_.Exception.Message)"
}

Write-Host "Local camera server: http://127.0.0.1:$Port/"
foreach ($ip in $lanIps) {
    Write-Host "LAN: http://${ip}:$Port/"
}
Write-Host "Serving: $Www"
Write-Host "Press Ctrl+C to stop."

$mime = @{
    ".html" = "text/html; charset=utf-8"
    ".js"   = "application/javascript; charset=utf-8"
    ".css"  = "text/css; charset=utf-8"
    ".m3u8" = "application/vnd.apple.mpegurl"
    ".ts"   = "video/mp2t"
    ".jpg"  = "image/jpeg"
    ".png"  = "image/png"
    ".ico"  = "image/x-icon"
    ".svg"  = "image/svg+xml"
    ".json" = "application/json"
}

function Get-SafePath([string]$urlPath) {
    $rel = [Uri]::UnescapeDataString($urlPath.Split("?")[0])
    if ($rel -eq "/" -or $rel -eq "") { $rel = "/index.html" }
    $rel = $rel.TrimStart("/").Replace("/", [IO.Path]::DirectorySeparatorChar)
    $full = [IO.Path]::GetFullPath((Join-Path $Www $rel))
    $rootFull = [IO.Path]::GetFullPath($Www)
    if (-not $full.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        return $null
    }
    return $full
}

while ($listener.IsListening) {
    $ctx = $null
    try {
        $ctx = $listener.GetContext()
        $req = $ctx.Request
        $res = $ctx.Response
        $path = Get-SafePath $req.Url.AbsolutePath

        if ($null -eq $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $res.StatusCode = 404
            $buf = [Text.Encoding]::UTF8.GetBytes("Not found")
            $res.ContentType = "text/plain; charset=utf-8"
            $res.OutputStream.Write($buf, 0, $buf.Length)
            $res.Close()
            continue
        }

        $ext = [IO.Path]::GetExtension($path).ToLowerInvariant()
        $res.StatusCode = 200
        if ($mime.ContainsKey($ext)) {
            $res.ContentType = $mime[$ext]
        } else {
            $res.ContentType = "application/octet-stream"
        }

        if ($ext -eq ".m3u8" -or $ext -eq ".ts") {
            $res.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate")
            $res.Headers.Add("Access-Control-Allow-Origin", "*")
        }

        $bytes = [IO.File]::ReadAllBytes($path)
        $res.ContentLength64 = $bytes.Length
        $res.OutputStream.Write($bytes, 0, $bytes.Length)
        $res.Close()
    } catch {
        if ($ctx -ne $null) {
            try { $ctx.Response.Abort() } catch {}
        }
        Start-Sleep -Milliseconds 50
    }
}
