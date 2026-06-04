param(
    [string]$Server = $env:MULTISAT_REMOTE_HOST,
    [string]$KeyPath = $env:MULTISAT_SSH_KEY,
    [string]$RemoteRoot = $(if ($env:MULTISAT_SERVER_WORKSPACE) { "$env:MULTISAT_SERVER_WORKSPACE/data/raw/era5_gate1_forecast" } else { "data/raw/era5_gate1_forecast" }),
    [string]$LocalRoot = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/raw/era5_gate1_forecast" } else { "data/raw/era5_gate1_forecast" }),
    [string]$Subset = "exact6h",
    [string[]]$Parts = @("pl", "sfc"),
    [int]$MaxFiles = 0,
    [string]$LogPath = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/logs/sync_era5_forecast_from_server.log" } else { "logs/sync_era5_forecast_from_server.log" })
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "$stamp $Message"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    Add-Content -Encoding UTF8 -Path $LogPath -Value $line
    Write-Output $line
}

function Invoke-SshText {
    param([string]$Command)
    $output = & ssh -i $KeyPath -o StrictHostKeyChecking=no $Server $Command
    if ($LASTEXITCODE -ne 0) {
        throw "ssh command failed with exit code $LASTEXITCODE"
    }
    return @($output) | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ }
}

New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
Write-Log "sync_start subset=$Subset remote=$RemoteRoot local=$LocalRoot max_files=$MaxFiles"

$copied = 0
$skipped = 0

foreach ($part in $Parts) {
    $remotePart = "$RemoteRoot/$Subset/$part"
    $localPart = Join-Path (Join-Path $LocalRoot $Subset) $part
    New-Item -ItemType Directory -Force -Path $localPart | Out-Null

    $okFiles = @(Invoke-SshText "find '$remotePart' -maxdepth 1 -type f -name '*.grib.ok' -print | sort") |
        Where-Object { $_ -match '\.grib\.ok$' -and $_ -like "$remotePart/*" }
    foreach ($remoteOk in $okFiles) {
        if ([string]::IsNullOrWhiteSpace($remoteOk)) {
            continue
        }
        if ($MaxFiles -gt 0 -and $copied -ge $MaxFiles) {
            Write-Log "max_files_reached copied=$copied skipped=$skipped"
            break
        }
        $remoteGrib = $remoteOk -replace '\.ok$', ''
        if ($remoteGrib -notmatch '\.grib$') {
            Write-Log "skip_invalid_remote_grib part=$part path=$remoteGrib"
            $skipped += 1
            continue
        }
        $name = Split-Path -Leaf $remoteGrib
        if ($name -notmatch '^era5_forecast_.*\.grib$') {
            Write-Log "skip_unexpected_name part=$part file=$name path=$remoteGrib"
            $skipped += 1
            continue
        }
        $localGrib = Join-Path $localPart $name
        $localOk = "$localGrib.ok"
        if ((Test-Path -LiteralPath $localGrib) -and (Test-Path -LiteralPath $localOk)) {
            $skipped += 1
            continue
        }
        Write-Log "copy part=$part file=$name"
        & scp -i $KeyPath -o StrictHostKeyChecking=no "${Server}:$remoteGrib" $localGrib
        if ($LASTEXITCODE -ne 0) {
            throw "scp grib failed for $remoteGrib"
        }
        & scp -i $KeyPath -o StrictHostKeyChecking=no "${Server}:$remoteOk" $localOk
        if ($LASTEXITCODE -ne 0) {
            throw "scp ok failed for $remoteOk"
        }
        $copied += 1
    }
}

Write-Log "sync_done copied=$copied skipped=$skipped"
