param(
    [Parameter(Mandatory = $true)]
    [string]$FtpAccount,

    [Parameter(Mandatory = $true)]
    [string]$FtpPassword,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$TargetServer = "ftp.nsmc.org.cn",

    [int]$ServerPort = 21,

    [string]$RemoteDir = "/",

    [int]$MaxConcurrentDownloads = 6,

    [switch]$KeepUrlList
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-RemotePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $base = if ($BaseDir.StartsWith("/")) { $BaseDir } else { "/" + $BaseDir }
    if (-not $base.EndsWith("/")) {
        $base += "/"
    }

    if ($Name.StartsWith("/")) {
        return $Name
    }

    return $base + $Name
}

function Escape-FtpPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $parts = $Path -split "/"
    $escaped = foreach ($part in $parts) {
        if ($part -eq "") {
            ""
        } else {
            [System.Uri]::EscapeDataString($part)
        }
    }

    return ($escaped -join "/")
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe not found in PATH."
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$remoteDirNormalized = if ($RemoteDir.StartsWith("/")) { $RemoteDir } else { "/" + $RemoteDir }
if (-not $remoteDirNormalized.EndsWith("/")) {
    $remoteDirNormalized += "/"
}

$listUrl = "ftp://${TargetServer}:$ServerPort$(Escape-FtpPath -Path $remoteDirNormalized)"
$credential = "${FtpAccount}:$FtpPassword"

Write-Host "Listing FTP order directory..."
$listing = & curl.exe --silent --show-error --ftp-method nocwd --user $credential --list-only $listUrl
if ($LASTEXITCODE -ne 0) {
    throw "FTP listing failed."
}

$files = $listing |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -and -not $_.EndsWith("/") -and $_ -notmatch '^\s*total\s+' }

if (-not $files -or $files.Count -eq 0) {
    throw "No files were found in $RemoteDir"
}

$encodedUser = [System.Uri]::EscapeDataString($FtpAccount)
$encodedPassword = [System.Uri]::EscapeDataString($FtpPassword)
$urlListPath = Join-Path ([System.IO.Path]::GetTempPath()) ("fy_ftp_order_urls_{0}.txt" -f ([guid]::NewGuid().ToString("N")))

$urls = foreach ($file in $files) {
    $remotePath = Join-RemotePath -BaseDir $remoteDirNormalized -Name $file
    "ftp://${encodedUser}:${encodedPassword}@${TargetServer}:$ServerPort$(Escape-FtpPath -Path $remotePath)"
}

Set-Content -LiteralPath $urlListPath -Encoding UTF8 -Value $urls
Write-Host "Files: $($files.Count)"

if (Get-Command aria2c -ErrorAction SilentlyContinue) {
    $resolvedOutput = (Resolve-Path -LiteralPath $OutputDir).Path
    & aria2c `
        "--input-file=$urlListPath" `
        "--dir=$resolvedOutput" `
        "--continue=true" `
        "--max-concurrent-downloads=$MaxConcurrentDownloads" `
        "--split=1" `
        "--summary-interval=30" `
        "--auto-file-renaming=false"
} else {
    Write-Host "aria2c not found. Falling back to sequential curl downloads."
    foreach ($url in $urls) {
        $name = [System.IO.Path]::GetFileName(([System.Uri]$url).AbsolutePath)
        $outPath = Join-Path $OutputDir $name
        & curl.exe --fail --location --continue-at - --output $outPath $url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed: $name"
        }
    }
}

if (-not $KeepUrlList) {
    Remove-Item -LiteralPath $urlListPath -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "Kept URL list with embedded FTP credentials: $urlListPath"
}

