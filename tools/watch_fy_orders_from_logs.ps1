param(
    [Parameter(Mandatory = $true)]
    [string[]]$OrderCode,

    [string]$LogDir = $(if ($env:FY_DATA_SERVICE_LOG_DIR) { $env:FY_DATA_SERVICE_LOG_DIR } else { "fy_logs" }),

    [string]$OutputRoot = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/raw/fy_orders" } else { "data/raw/fy_orders" }),

    [int]$IntervalSeconds = 300,

    [int]$MaxHours = 72,

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

if ($IntervalSeconds -lt 10) {
    throw "IntervalSeconds must be >= 10"
}

if ($MaxHours -lt 1) {
    throw "MaxHours must be >= 1"
}

if ($MaxConcurrentDownloads -lt 1) {
    throw "MaxConcurrentDownloads must be >= 1"
}

function Get-FyOrdersFromLogs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceLogDir
    )

    $orders = New-Object System.Collections.Generic.List[object]
    $logFiles = Get-ChildItem -LiteralPath $SourceLogDir -File -Filter "*.log" | Sort-Object LastWriteTime, FullName

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

    return $orders |
        Group-Object ordercode |
        ForEach-Object {
            $_.Group | Sort-Object sourceLastWriteTime, matchIndex -Descending | Select-Object -First 1
        }
}

function Test-OrderDownloadComplete {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OrderOutputDir
    )

    return Test-Path -LiteralPath (Join-Path $OrderOutputDir ".fy_download_complete")
}

function Write-OrderDownloadComplete {
    param(
        [Parameter(Mandatory = $true)]
        [string]$OrderOutputDir,

        [Parameter(Mandatory = $true)]
        [object]$Order
    )

    $marker = Join-Path $OrderOutputDir ".fy_download_complete"
    [pscustomobject]@{
        ordercode = $Order.ordercode
        datatotalnumber = $Order.datatotalnumber
        dataquantity_gb = $Order.dataquantity_gb
        completed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    } | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
}

$targetCodes = @($OrderCode | Sort-Object -Unique)
$deadline = (Get-Date).AddHours($MaxHours)

if (-not (Test-Path -LiteralPath $OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}

Write-Host "Watching FYDataService logs for $($targetCodes.Count) order(s)."
Write-Host "Output root: $OutputRoot"
Write-Host "Deadline: $($deadline.ToString("yyyy-MM-dd HH:mm:ss"))"

if ($DryRun) {
    $foundOrders = @(Get-FyOrdersFromLogs -SourceLogDir $LogDir)
    $targetCodes | ForEach-Object {
        $code = $_
        $orderOutputDir = Join-Path $OutputRoot $code
        $order = $foundOrders | Where-Object { $_.ordercode -eq $code } | Select-Object -First 1
        [pscustomobject]@{
            ordercode = $code
            status = if (Test-OrderDownloadComplete -OrderOutputDir $orderOutputDir) {
                "downloaded"
            } elseif ($order) {
                "ftp_ready"
            } else {
                "waiting_for_ftp_credentials"
            }
            datatotalnumber = if ($order) { $order.datatotalnumber } else { $null }
            dataquantity_gb = if ($order) { $order.dataquantity_gb } else { $null }
            outputDir = $orderOutputDir
        }
    } | Format-Table -AutoSize

    Write-Host "Dry run only. Remove -DryRun to keep watching and download when ready."
    exit 0
}

while ((Get-Date) -lt $deadline) {
    $foundOrders = @(Get-FyOrdersFromLogs -SourceLogDir $LogDir)
    $pending = New-Object System.Collections.Generic.List[string]

    foreach ($code in $targetCodes) {
        $orderOutputDir = Join-Path $OutputRoot $code
        if (Test-OrderDownloadComplete -OrderOutputDir $orderOutputDir) {
            continue
        }

        $order = $foundOrders | Where-Object { $_.ordercode -eq $code } | Select-Object -First 1
        if (-not $order) {
            $pending.Add($code)
            continue
        }

        Write-Host "Found FTP credentials for $code. Files: $($order.datatotalnumber), size: $($order.dataquantity_gb) GB"
        & powershell -ExecutionPolicy Bypass -File $DownloaderScript `
            -FtpAccount $order.ftpaccount `
            -FtpPassword $order.ftppassword `
            -TargetServer $order.targetserver `
            -ServerPort $order.serverport `
            -OutputDir $orderOutputDir `
            -MaxConcurrentDownloads $MaxConcurrentDownloads

        if ($LASTEXITCODE -ne 0) {
            throw "Download failed for order $code"
        }

        Write-OrderDownloadComplete -OrderOutputDir $orderOutputDir -Order $order
    }

    $remaining = @($targetCodes | Where-Object {
        -not (Test-OrderDownloadComplete -OrderOutputDir (Join-Path $OutputRoot $_))
    })

    if ($remaining.Count -eq 0) {
        Write-Host "All target orders are downloaded."
        exit 0
    }

    Write-Host ("{0} pending: {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), ($remaining -join ", "))
    Start-Sleep -Seconds $IntervalSeconds
}

throw "Timed out before all target orders became downloadable: $($targetCodes -join ', ')"
