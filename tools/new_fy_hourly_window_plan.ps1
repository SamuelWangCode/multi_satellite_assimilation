param(
    [datetime]$StartDate = "2024-03-05",

    [datetime]$EndDate = "2025-12-31",

    [string]$OutputPath = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/interim/FY_hourly_window_plan.csv" } else { "tmp/FY_hourly_window_plan.csv" }),

    [int]$BatchMonths = 1,

    [string[]]$Hours = @("0"),

    [string]$ProductName = "FY satellite product",

    [string]$SuggestedPrefix = "FY_PRODUCT"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToHourList {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Values
    )

    $seen = @{}
    $list = New-Object System.Collections.Generic.List[int]
    foreach ($rawValue in $Values) {
        foreach ($piece in ($rawValue -split ",")) {
            $trimmed = $piece.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) {
                continue
            }

            $parsed = 0
            if (-not [int]::TryParse($trimmed, [ref]$parsed)) {
                throw "Invalid hour value: $trimmed"
            }

            if ($parsed -lt 0 -or $parsed -gt 23) {
                throw "Hour must be between 0 and 23: $parsed"
            }

            if (-not $seen.ContainsKey($parsed)) {
                $seen[$parsed] = $true
                $list.Add($parsed)
            }
        }
    }

    if ($list.Count -eq 0) {
        throw "At least one hour is required."
    }

    return ,$list
}

if ($BatchMonths -lt 1) {
    throw "BatchMonths must be >= 1"
}

$hourList = Convert-ToHourList -Values $Hours
$rows = New-Object System.Collections.Generic.List[object]

$batchStart = $StartDate.Date
$batchIndex = 1
while ($batchStart -le $EndDate.Date) {
    $candidateEnd = $batchStart.AddMonths($BatchMonths).AddDays(-1)
    $batchEnd = if ($candidateEnd -lt $EndDate.Date) { $candidateEnd } else { $EndDate.Date }

    foreach ($hour in $hourList) {
        $windowStart = $batchStart.Date.AddHours($hour)
        $windowEnd = $batchEnd.Date.AddHours($hour).AddMinutes(59).AddSeconds(59)

        $rows.Add([pscustomobject]@{
            Product = $ProductName
            Batch = "{0:D2}" -f $batchIndex
            HourUTC = "{0:D2}" -f $hour
            ClientStartUTC = $windowStart.ToString("yyyy-MM-dd HH:mm:ss")
            ClientEndUTC = $windowEnd.ToString("yyyy-MM-dd HH:mm:ss")
            Everyday = "Yes"
            TimeSelection = "All"
            SuggestedOrderName = ("{0}_{1:yyyyMMdd}_{2:yyyyMMdd}_{3:D2}UTC" -f $SuggestedPrefix, $batchStart, $batchEnd, $hour)
        })
    }

    $batchStart = $batchEnd.AddDays(1)
    $batchIndex++
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir -and -not (Test-Path -LiteralPath $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$rows | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Encoding UTF8

Write-Host "Wrote plan: $OutputPath"
Write-Host "Rows: $($rows.Count)"
Write-Host "Use UTC time, tick Everyday, and keep Time Selection as All for each row."
