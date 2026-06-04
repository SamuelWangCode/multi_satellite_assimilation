param(
    [string]$ExternalRoot = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { $env:MULTISAT_LOCAL_DATA_ROOT } else { "external_data" }),
    [switch]$IncludeLogs,
    [switch]$DeleteAfterCopy
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path $ExternalRoot)) {
    New-Item -ItemType Directory -Force -Path $ExternalRoot | Out-Null
}

$sources = @("data")
if ($IncludeLogs) {
    $sources += "logs"
}

$manifestDir = Join-Path $ExternalRoot "migration_manifests"
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifest = Join-Path $manifestDir "project_data_migration_$stamp.txt"

Add-Content -Path $manifest -Value "repo=$RepoRoot" -Encoding utf8
Add-Content -Path $manifest -Value "external_root=$ExternalRoot" -Encoding utf8
Add-Content -Path $manifest -Value "delete_after_copy=$DeleteAfterCopy" -Encoding utf8

foreach ($source in $sources) {
    if (-not (Test-Path $source)) {
        Add-Content -Path $manifest -Value "skip missing $source" -Encoding utf8
        continue
    }

    $sourceFull = (Resolve-Path $source).Path
    $dest = Join-Path $ExternalRoot $source
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    Add-Content -Path $manifest -Value "copy $sourceFull -> $dest" -Encoding utf8
    robocopy $sourceFull $dest /E /R:2 /W:5 /NFL /NDL /NP | Out-File -FilePath $manifest -Append -Encoding utf8
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed for $source with exit code $code"
    }

    if ($DeleteAfterCopy) {
        $sourceResolved = (Resolve-Path $source).Path
        $repoResolved = (Resolve-Path $RepoRoot).Path
        if (-not $sourceResolved.StartsWith($repoResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "refusing to delete outside repo: $sourceResolved"
        }
        Remove-Item -LiteralPath $sourceResolved -Recurse -Force
        Add-Content -Path $manifest -Value "deleted $sourceResolved" -Encoding utf8
    }
}

Write-Output $manifest
