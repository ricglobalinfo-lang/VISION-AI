<#
.SYNOPSIS
  Sync local HLS (index.m3u8 + .ts) to hosting for public live playback.
  Uses SSH + base64 (SFTP file streams are aborted by this host).
#>
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$HlsDir = Join-Path $Root "hls"

$SshHost = "your-hosting-domain.uz"
$SshUser = "your_ssh_user"
$SshPass = "YOUR_SSH_PASSWORD"
$RemoteDir = "/var/www/user/data/www/camera.example.uz/live"

$LogPath = Join-Path $env:TEMP "camera-kurbonov-hls-sync.log"

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

function Open-Ssh($Credential) {
    New-SSHSession -ComputerName $SshHost -Credential $Credential -AcceptKey -Force -ErrorAction Stop
}

function Upload-ViaSsh($SessionId, [string]$RemotePath, [byte[]]$Bytes) {
    $b64 = [Convert]::ToBase64String($Bytes)
    # Escape single quotes for remote single-quoted printf
    $cmd = "printf '%s' '$b64' | base64 -d > '$RemotePath.tmp' && mv -f '$RemotePath.tmp' '$RemotePath'"
    $r = Invoke-SSHCommand -SessionId $SessionId -Command $cmd -TimeOut 120
    if ($r.ExitStatus -ne 0) {
        throw "upload failed $($RemotePath): $($r.Error)"
    }
}

if (-not (Ensure-PoshSsh)) { exit 1 }
Import-Module Posh-SSH -ErrorAction Stop
$cred = New-Object System.Management.Automation.PSCredential(
    $SshUser,
    (ConvertTo-SecureString $SshPass -AsPlainText -Force)
)

$ssh = $null
try {
    $ssh = Open-Ssh $cred
    Invoke-SSHCommand -SessionId $ssh.SessionId -Command "mkdir -p '$RemoteDir'" | Out-Null
    Write-Log "Sync started (ssh/base64) -> $RemoteDir session=$($ssh.SessionId)"
} catch {
    Write-Log "FATAL session: $($_.Exception.Message)"
    exit 1
}

$uploadedTs = @{}
$ok = 0
$fail = 0

while ($true) {
    try {
        if ($null -eq $ssh) { throw "no ssh" }

        $tsFiles = @(Get-ChildItem -Path $HlsDir -Filter "seg_*.ts" -ErrorAction SilentlyContinue | Sort-Object Name)
        foreach ($f in $tsFiles) {
            $key = $f.Name + ":" + $f.Length + ":" + $f.LastWriteTimeUtc.Ticks
            if ($uploadedTs.ContainsKey($f.Name) -and $uploadedTs[$f.Name] -eq $key) { continue }
            $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
            if ($bytes.Length -lt 100) { continue }
            Upload-ViaSsh $ssh.SessionId "$RemoteDir/$($f.Name)" $bytes
            $uploadedTs[$f.Name] = $key
        }

        $m3u8 = Join-Path $HlsDir "index.m3u8"
        if (Test-Path $m3u8) {
            $bytes = [System.IO.File]::ReadAllBytes($m3u8)
            Upload-ViaSsh $ssh.SessionId "$RemoteDir/index.m3u8" $bytes
        }

        $localNames = New-Object 'System.Collections.Generic.HashSet[string]'
        foreach ($f in $tsFiles) { [void]$localNames.Add($f.Name) }
        $stale = @($uploadedTs.Keys | Where-Object { -not $localNames.Contains($_) })
        foreach ($name in $stale) {
            try {
                Invoke-SSHCommand -SessionId $ssh.SessionId -Command "rm -f '$RemoteDir/$name'" | Out-Null
            } catch {}
            $uploadedTs.Remove($name)
        }

        $ok++
        if (($ok % 30) -eq 1) {
            Write-Log "OK cycles=$ok fails=$fail local_ts=$($tsFiles.Count) tracked=$($uploadedTs.Count)"
        }
    } catch {
        $fail++
        Write-Log "Sync error: $($_.Exception.Message)"
        try {
            if ($null -ne $ssh) { Remove-SSHSession -SessionId $ssh.SessionId | Out-Null }
        } catch {}
        $ssh = $null
        try {
            $ssh = Open-Ssh $cred
            Write-Log "Reconnected session=$($ssh.SessionId)"
        } catch {
            Write-Log "Reconnect failed"
            Start-Sleep -Seconds 4
        }
    }
    Start-Sleep -Milliseconds 700
}
