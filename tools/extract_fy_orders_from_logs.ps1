param(
    [string]$LogDir = $(if ($env:FY_DATA_SERVICE_LOG_DIR) { $env:FY_DATA_SERVICE_LOG_DIR } else { "fy_logs" }),

    [string]$OutputPath = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/interim/fy_orders.csv" } else { "tmp/fy_orders.csv" }),

    [switch]$IncludePassword
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LogDir)) {
    throw "Log directory not found: $LogDir"
}

$orders = New-Object System.Collections.Generic.List[object]

Get-ChildItem -LiteralPath $LogDir -File -Filter "*.log" | ForEach-Object {
    $sourceLog = $_.FullName
    $text = Get-Content -LiteralPath $sourceLog -Raw -ErrorAction Stop
    $matches = [regex]::Matches($text, '\{[^{}]*"targetserver"[^{}]*"ftpaccount"[^{}]*"ordercode"[^{}]*\}')

    foreach ($match in $matches) {
        try {
            $order = $match.Value | ConvertFrom-Json
        } catch {
            continue
        }

        $row = [ordered]@{
            ordercode = $order.ordercode
            targetserver = $order.targetserver
            serverport = $order.serverport
            ftpaccount = $order.ftpaccount
            datatotalnumber = $order.datatotalnumber
            dataquantity_gb = [math]::Round([double]$order.dataquantity / 1GB, 3)
            orderinfoid = $order.orderinfoid
            sourceLog = $sourceLog
        }

        if ($IncludePassword) {
            $row.ftppassword = $order.ftppassword
        }

        $orders.Add([pscustomobject]$row)
    }
}

$deduped = $orders |
    Sort-Object ordercode, sourceLog -Unique |
    Sort-Object ordercode

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$deduped | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding UTF8

Write-Host "Wrote orders: $OutputPath"
Write-Host "Orders: $($deduped.Count)"
if (-not $IncludePassword) {
    Write-Host "Passwords were omitted. Re-run with -IncludePassword only if you need a local download manifest."
}
