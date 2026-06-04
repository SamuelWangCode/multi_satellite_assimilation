param(
    [datetime]$StartDate = "2024-03-05",

    [datetime]$EndDate = "2025-12-31",

    [string]$OutputPath = $(if ($env:MULTISAT_LOCAL_DATA_ROOT) { "$env:MULTISAT_LOCAL_DATA_ROOT/interim/GIIRS_AVP_batch_plan.csv" } else { "tmp/GIIRS_AVP_batch_plan.csv" }),

    [int]$BatchMonths = 3,

    [string[]]$Hours = @("0", "6", "12", "18"),

    [Alias("WindowMinutes")]
    [int]$WindowRadiusMinutes = 60,

    [int]$ChunkMinutes = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Convert-ToIntList {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Values,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $list = New-Object System.Collections.Generic.List[int]
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

            if ($parsed -lt 0 -or $parsed -gt 23) {
                throw "$Name must be between 0 and 23: $parsed"
            }

            $list.Add($parsed)
        }
    }

    return ,($list | Sort-Object -Unique)
}

if ($BatchMonths -lt 1) {
    throw "BatchMonths must be >= 1"
}

if ($WindowRadiusMinutes -lt 1 -or $WindowRadiusMinutes -gt 23 * 60 + 59) {
    throw "WindowRadiusMinutes must be between 1 and 1439"
}

if ($ChunkMinutes -lt 1 -or $ChunkMinutes -gt 23 * 60 + 59) {
    throw "ChunkMinutes must be between 1 and 1439"
}

$hourList = Convert-ToIntList -Values $Hours -Name "hour"
$rows = New-Object System.Collections.Generic.List[object]

function Add-PlanRow {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]]$Rows,

        [Parameter(Mandatory = $true)]
        [int]$BatchIndex,

        [Parameter(Mandatory = $true)]
        [datetime]$BatchStart,

        [Parameter(Mandatory = $true)]
        [datetime]$BatchEnd,

        [Parameter(Mandatory = $true)]
        [int]$CenterHour,

        [Parameter(Mandatory = $true)]
        [datetime]$WindowStart,

        [Parameter(Mandatory = $true)]
        [datetime]$WindowEnd,

        [Parameter(Mandatory = $true)]
        [int]$OffsetStartMinutes,

        [Parameter(Mandatory = $true)]
        [int]$OffsetEndMinutes,

        [Parameter(Mandatory = $true)]
        [string]$WindowPart
    )

    $Rows.Add([pscustomobject]@{
        Product = "FY-4B GIIRS AVP"
        Batch = "{0:D2}" -f $BatchIndex
        CenterHourUTC = "{0:D2}" -f $CenterHour
        WindowPart = $WindowPart
        OffsetStartMinutes = $OffsetStartMinutes
        OffsetEndMinutes = $OffsetEndMinutes
        ClientStartUTC = $WindowStart.ToString("yyyy-MM-dd HH:mm:ss")
        ClientEndUTC = $WindowEnd.ToString("yyyy-MM-dd HH:mm:ss")
        Everyday = "Yes"
        TimeSelection = "All"
        SuggestedOrderName = ("GIIRS_AVP_{0:yyyyMMdd}_{1:yyyyMMdd}_{2:D2}UTC_{3}" -f $BatchStart, $BatchEnd, $CenterHour, $WindowPart)
    })
}

function Format-OffsetLabel {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Minutes
    )

    if ($Minutes -lt 0) {
        return "m{0:D3}" -f [math]::Abs($Minutes)
    }

    return "p{0:D3}" -f $Minutes
}

$batchStart = $StartDate.Date
$batchIndex = 1
while ($batchStart -le $EndDate.Date) {
    $candidateEnd = $batchStart.AddMonths($BatchMonths).AddDays(-1)
    $batchEnd = if ($candidateEnd -lt $EndDate.Date) { $candidateEnd } else { $EndDate.Date }

    foreach ($hour in $hourList) {
        $centerStart = $batchStart.Date.AddHours($hour)
        $centerEnd = $batchEnd.Date.AddHours($hour)

        for ($offsetStart = -1 * $WindowRadiusMinutes; $offsetStart -lt $WindowRadiusMinutes; $offsetStart += $ChunkMinutes) {
            $offsetEnd = [math]::Min($offsetStart + $ChunkMinutes, $WindowRadiusMinutes)
            $windowStart = $centerStart.AddMinutes($offsetStart)
            $windowEnd = $centerEnd.AddMinutes($offsetEnd).AddSeconds(-1)
            $windowPart = "$(Format-OffsetLabel -Minutes $offsetStart)_to_$(Format-OffsetLabel -Minutes $offsetEnd)"

            Add-PlanRow -Rows $rows -BatchIndex $batchIndex -BatchStart $batchStart -BatchEnd $batchEnd -CenterHour $hour -WindowStart $windowStart -WindowEnd $windowEnd -OffsetStartMinutes $offsetStart -OffsetEndMinutes $offsetEnd -WindowPart $windowPart
        }
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
Write-Host "The scientific window is centered: analysis time +/- $WindowRadiusMinutes minutes."
Write-Host "Each client operation is split into chunks of up to $ChunkMinutes minutes."
