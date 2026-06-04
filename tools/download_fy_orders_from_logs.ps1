param(
    [string]$LogDir = $(if ($env:FY_DATA_SERVICE_LOG_DIR) { $env:FY_DATA_SERVICE_LOG_DIR } else { "fy_logs" }),

    [string]$OutputRoot = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/raw/fy_orders" } else { "data/raw/fy_orders" }),

    [string[]]$OrderCode = @(),

    [int]$Last = 0,

    [switch]$All,

    [int]$MaxConcurrentDownloads = 6,

    [switch]$DryRun,

    [string]$DownloaderScript = (Join-Path $PSScriptRoot "download_fy_ftp_order.ps1")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LogDir)) {
    throw "Log directory not found: $LogDir"
}

if (-not (Test-Path -LiteralPath $DownloaderScript)) {
    throw "Downloader script not found: $DownloaderScript"
}

if ($MaxConcurrentDownloads -lt 1) {
    throw "MaxConcurrentDownloads must be >= 1"
}

if ($Last -lt 0) {
    throw "Last must be >= 0"
}

$orders = New-Object System.Collections.Generic.List[object]
$logFiles = Get-ChildItem -LiteralPath $LogDir -File -Filter "*.log" | Sort-Object LastWriteTime, FullName

foreach ($logFile in $logFiles) {
    $text = Get-Content -LiteralPath $logFile.FullName -Raw -ErrorAction Stop
    $matches = [regex]::Matches($text, '\{[^{}]*"targetserver"[^{}]*"ftpaccount"[^{}]*"ordercode"[^{}]*\}')
    $matchIndex = 0

    foreach ($match in $matches) {
        $matchIndex++
        try {
            $order = $match.Value | ConvertFrom-Json
        } catch {
            continue
        }

        if (-not $order.ftppassword) {
            continue
        }

        $orders.Add([pscustomobject]@{
            ordercode = [string]$order.ordercode
            targetserver = [string]$order.targetserver
            serverport = [int]$order.serverport
            ftpaccount = [string]$order.ftpaccount
            ftppassword = [string]$order.ftppassword
            datatotalnumber = [int]$order.datatotalnumber
            dataquantity_gb = [math]::Round([double]$order.dataquantity / 1GB, 3)
            sourceLog = $logFile.FullName
            sourceLastWriteTime = $logFile.LastWriteTime
            matchIndex = $matchIndex
        })
    }
}

$deduped = $orders |
    Group-Object ordercode |
    ForEach-Object {
        $_.Group | Sort-Object sourceLastWriteTime, matchIndex -Descending | Select-Object -First 1
    }

if (-not $deduped -or $deduped.Count -eq 0) {
    throw "No FTP orders with passwords were found in $LogDir"
}

if ($OrderCode.Count -gt 0) {
    $selected = $deduped | Where-Object { $OrderCode -contains $_.ordercode }
    $missing = $OrderCode | Where-Object { $_ -notin @($selected | ForEach-Object { $_.ordercode }) }
    if ($missing.Count -gt 0) {
        throw "Order code(s) not found in logs: $($missing -join ', ')"
    }
} elseif ($All) {
    $selected = $deduped | Sort-Object sourceLastWriteTime, matchIndex, ordercode
} elseif ($Last -gt 0) {
    $selected = $deduped | Sort-Object sourceLastWriteTime, matchIndex, ordercode | Select-Object -Last $Last
} else {
    throw "Specify -OrderCode, -Last, or -All. Use -Last 1 for the newest order."
}

Write-Host "Selected orders:"
$selected |
    Select-Object ordercode, targetserver, serverport, ftpaccount, datatotalnumber, dataquantity_gb |
    Format-Table -AutoSize

if ($DryRun) {
    Write-Host "Dry run only. Remove -DryRun to start downloading."
    exit 0
}

if (-not (Test-Path -LiteralPath $OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

foreach ($order in $selected) {
    $outputDir = Join-Path $OutputRoot $order.ordercode
    Write-Host "Downloading order $($order.ordercode) to $outputDir"

    & powershell -ExecutionPolicy Bypass -File $DownloaderScript `
        -FtpAccount $order.ftpaccount `
        -FtpPassword $order.ftppassword `
        -TargetServer $order.targetserver `
        -ServerPort $order.serverport `
        -OutputDir $outputDir `
        -MaxConcurrentDownloads $MaxConcurrentDownloads

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for order $($order.ordercode)"
    }
}
