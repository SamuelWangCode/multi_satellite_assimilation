param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string[]]$Hours = @("0", "6", "12", "18"),

    [string[]]$Minutes = @("0"),

    [datetime]$StartDate,

    [datetime]$EndDate,

    [string]$IncludePattern,

    [string]$ExcludePattern
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FyObservationStart {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $match = [regex]::Match($Text, "_(?<start>\d{14})_(?<end>\d{14})_")
    if (-not $match.Success) {
        return $null
    }

    return [datetime]::ParseExact(
        $match.Groups["start"].Value,
        "yyyyMMddHHmmss",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Convert-ToIntSet {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Values,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $set = [System.Collections.Generic.HashSet[int]]::new()

    foreach ($rawValue in $Values) {
        foreach ($piece in ($rawValue -split ",")) {
            $trimmed = $piece.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) {
                continue
            }

            $parsed = 0
            if (-not [int]::TryParse($trimmed, [ref]$parsed)) {
                throw "Invalid $Name value: $trimmed"
            }

            [void]$set.Add($parsed)
        }
    }

    return ,$set
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input file not found: $InputPath"
}

$hourSet = Convert-ToIntSet -Values $Hours -Name "hour"
$minuteSet = Convert-ToIntSet -Values $Minutes -Name "minute"

$selected = New-Object System.Collections.Generic.List[string]
$skippedWithoutTime = 0
$skippedByTime = 0
$skippedByDate = 0

Get-Content -LiteralPath $InputPath | ForEach-Object {
    $line = $_.Trim()
    if ([string]::IsNullOrWhiteSpace($line)) {
        return
    }

    if ($IncludePattern -and ($line -notmatch $IncludePattern)) {
        return
    }

    if ($ExcludePattern -and ($line -match $ExcludePattern)) {
        return
    }

    $obsStart = Get-FyObservationStart -Text $line
    if ($null -eq $obsStart) {
        $script:skippedWithoutTime++
        return
    }

    if ($PSBoundParameters.ContainsKey("StartDate") -and $obsStart.Date -lt $StartDate.Date) {
        $script:skippedByDate++
        return
    }

    if ($PSBoundParameters.ContainsKey("EndDate") -and $obsStart.Date -gt $EndDate.Date) {
        $script:skippedByDate++
        return
    }

    if ((-not $hourSet.Contains($obsStart.Hour)) -or (-not $minuteSet.Contains($obsStart.Minute))) {
        $script:skippedByTime++
        return
    }

    $selected.Add($line)
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

Set-Content -LiteralPath $OutputPath -Encoding utf8 -Value $selected

Write-Host "Input:    $InputPath"
Write-Host "Output:   $OutputPath"
Write-Host "Selected: $($selected.Count)"
Write-Host "Skipped (time mismatch): $skippedByTime"
Write-Host "Skipped (date mismatch): $skippedByDate"
Write-Host "Skipped (no FY timestamp found): $skippedWithoutTime"
Write-Host ""
Write-Host "Hours filter:   $($Hours -join ',')"
Write-Host "Minutes filter: $($Minutes -join ',')"
