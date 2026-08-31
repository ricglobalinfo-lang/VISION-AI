<#
.SYNOPSIS
  LAN bridge: pulls Hikvision JPEG and atomically uploads feed.jpg to hosting.

.DESCRIPTION
  Writes feed.jpg.tmp on the server, then renames to feed.jpg so browsers
  never read a half-written file (that caused flicker / broken frames).
#>

$ErrorActionPreference = "Continue"

$CameraUser = "admin"
$CameraPass = "YOUR_CAMERA_PASSWORD"
$CameraUrl  = "http://192.168.1.100/ISAPI/Streaming/channels/101/picture"

$SshHost    = "your-hosting-domain.uz"
$SshUser    = "your_ssh_user"
$SshPass    = "YOUR_SSH_PASSWORD"
$RemoteDir  = "/var/www/user/data/www/camera.example.uz"
$RemotePath = "$RemoteDir/feed.jpg"
$RemoteTmp  = "$RemoteDir/feed.jpg.tmp"

$IntervalSec = 0.8
$MinJpegBytes = 20000
$LocalTmp    = Join-Path $env:TEMP "camera-kurbonov-feed.jpg"
$LogPath     = Join-Path $env:TEMP "camera-kurbonov-bridge.log"

function Write-Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Ensure-PoshSsh {
    if (Get-Module -ListAvailable -Name Posh-SSH) { return $true }
    try {
        Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force -Scope CurrentUser | Out-Null
        Set-PSRepository -Name PSGallery -InstallationPolicy Trusted -ErrorAction SilentlyContinue
        Install-Module -Name Posh-SSH -Force -Scope CurrentUser -AllowClobber -ErrorAction Stop
        return $true
    } catch {
        Write-Log "Posh-SSH install failed: $($_.Exception.Message)"
        return $false
    }
}

function Test-JpegFile([string]$path, [int]$minBytes) {
    if (-not (Test-Path $path)) { return $false }
    $item = Get-Item $path
    if ($item.Length -lt $minBytes) { return $false }
    $fs = [System.IO.File]::OpenRead($path)
    try {
        $b0 = $fs.ReadByte()
        $b1 = $fs.ReadByte()
        return ($b0 -eq 0xFF -and $b1 -eq 0xD8)
    } finally {
        $fs.Dispose()
    }
}

function Open-Sessions {
    param($Credential)
    $sftp = New-SFTPSession -ComputerName $SshHost -Credential $Credential -AcceptKey -ErrorAction Stop
    $ssh  = New-SSHSession -ComputerName $SshHost -Credential $Credential -AcceptKey -ErrorAction Stop
    return @{ Sftp = $sftp; Ssh = $ssh }
}

Write-Log "Bridge starting (atomic upload). Camera=$CameraUrl -> ${SshUser}@${SshHost}:$RemotePath"
Write-Log "Log file: $LogPath"

if (-not (Ensure-PoshSsh)) {
    Write-Log "FATAL: Posh-SSH missing"
    exit 1
}

Import-Module Posh-SSH -ErrorAction Stop
$cred = New-Object System.Management.Automation.PSCredential(
    $SshUser,
    (ConvertTo-SecureString $SshPass -AsPlainText -Force)
)

$sessions = $null
try {
    $sessions = Open-Sessions -Credential $cred
    Write-Log "SFTP+SSH sessions OK"
} catch {
    Write-Log "FATAL: session failed: $($_.Exception.Message)"
    exit 1
}

$ok = 0
$fail = 0
while ($true) {
    $loopStart = Get-Date
    try {
        $curl = & curl.exe -sS -o $LocalTmp -w "%{http_code}" --digest -u "${CameraUser}:${CameraPass}" --connect-timeout 5 --max-time 10 $CameraUrl 2>$null
        if ($curl -ne "200" -or -not (Test-JpegFile -path $LocalTmp -minBytes $MinJpegBytes)) {
            $fail++
            Write-Log "Camera fetch failed (http=$curl size=$(if (Test-Path $LocalTmp) { (Get-Item $LocalTmp).Length } else { 0 }))"
        } else {
            $bytes = [System.IO.File]::ReadAllBytes($LocalTmp)
            $stream = New-SFTPFileStream -SessionId $sessions.Sftp.SessionId -Path $RemoteTmp -FileMode Create -FileAccess Write
            try {
                $stream.Write($bytes, 0, $bytes.Length)
                $stream.Flush()
            } finally {
                $stream.Dispose()
            }

            # Atomic replace on Linux: rename is atomic within same filesystem.
            $mv = Invoke-SSHCommand -SessionId $sessions.Ssh.SessionId -Command "mv -f '$RemoteTmp' '$RemotePath'"
            if ($mv.ExitStatus -ne 0) {
                throw "remote mv failed: $($mv.Error)$($mv.Output)"
            }

            $ok++
            if (($ok % 40) -eq 1) {
                Write-Log "OK uploads=$ok fails=$fail size=$($bytes.Length)"
            }
        }
    } catch {
        $fail++
        Write-Log "Upload error: $($_.Exception.Message)"
        try {
            if ($sessions.Sftp) { Remove-SFTPSession -SessionId $sessions.Sftp.SessionId | Out-Null }
            if ($sessions.Ssh)  { Remove-SSHSession  -SessionId $sessions.Ssh.SessionId  | Out-Null }
        } catch {}
        try {
            $sessions = Open-Sessions -Credential $cred
            Write-Log "Sessions reconnected"
        } catch {
            Write-Log "Reconnect failed"
            Start-Sleep -Seconds 5
        }
    }

    $elapsed = ((Get-Date) - $loopStart).TotalSeconds
    $sleep = [Math]::Max(0.2, $IntervalSec - $elapsed)
    Start-Sleep -Seconds $sleep
}
