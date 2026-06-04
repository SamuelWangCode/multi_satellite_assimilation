param(
    [Parameter(Mandatory = $true)]
    [string]$UrlList,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [int]$MaxConcurrentDownloads = 6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $UrlList)) {
    throw "URL list not found: $UrlList"
}

if (-not (Get-Command aria2c -ErrorAction SilentlyContinue)) {
    throw "aria2c not found in PATH. Install aria2 first, or use the official wget method from NSMC."
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$resolvedList = (Resolve-Path -LiteralPath $UrlList).Path
$resolvedOutput = (Resolve-Path -LiteralPath $OutputDir).Path

$args = @(
    "--input-file=$resolvedList"
    "--dir=$resolvedOutput"
    "--continue=true"
    "--max-concurrent-downloads=$MaxConcurrentDownloads"
    "--split=1"
    "--summary-interval=30"
    "--auto-file-renaming=false"
)

Write-Host "Running aria2c with $MaxConcurrentDownloads concurrent downloads..."
& aria2c @args

