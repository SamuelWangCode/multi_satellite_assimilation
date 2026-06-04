# FY satellite download helpers

These scripts are meant to work with the official NSMC order file list workflow:

1. Search and submit an order on the FY service website or desktop client.
2. Export or download the order URL list as a text file.
3. Filter the URL list locally by synoptic times.
4. Download the filtered list with `aria2c`.

## 1) Keep only 00/06/12/18 UTC AGRI files

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\filter_fy_order_list.ps1 `
  -InputPath .\orders\FY4B_AGRI_all_urls.txt `
  -OutputPath .\orders\FY4B_AGRI_00_06_12_18.txt `
  -Hours 0,6,12,18 `
  -Minutes 0 `
  -StartDate 2024-03-05 `
  -EndDate 2025-12-31
```

The example above keeps only the exact synoptic files at `00:00`, `06:00`, `12:00`, and `18:00` UTC.

## 2) Download the filtered list

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_fy_list.ps1 `
  -UrlList .\orders\FY4B_AGRI_00_06_12_18.txt `
  -OutputDir "$env:MULTISAT_LOCAL_DATA_ROOT\raw\FY4B_AGRI_synoptic" `
  -MaxConcurrentDownloads 6
```

`6` matches the ordinary-order parallel limit described by the official NSMC FAQ.

## GIIRS AVP batching

GIIRS AVP can create too many small files for the FY client cart UI. For the first experiment, keep only the GIIRS AVP windows nearest the analysis times instead of ordering all times.

Generate a client operation table:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\new_giirs_avp_batch_plan.ps1 `
  -StartDate 2024-03-05 `
  -EndDate 2025-12-31 `
  -BatchMonths 3 `
  -Hours 0,6,12,18 `
  -WindowRadiusMinutes 60 `
  -ChunkMinutes 60 `
  -OutputPath .\tmp\GIIRS_AVP_batch_plan.csv
```

The scientific window is still centered on the analysis hours, but the client operations are split into one-hour chunks. With the default settings, `00/06/12/18 UTC` become `23:00-23:59`, `00:00-00:59`, `05:00-05:59`, `06:00-06:59`, `11:00-11:59`, `12:00-12:59`, `17:00-17:59`, and `18:00-18:59` UTC.

For every row in the table, use UTC time, tick `Everyday`, keep `Time Selection` as `All`, add the search result to the cart, and submit that cart before starting the next row.

Extract completed FTP order metadata from FYDataService logs:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\extract_fy_orders_from_logs.ps1 `
  -OutputPath .\tmp\fy_orders.csv
```

Add `-IncludePassword` only when you need a local download manifest.

Download a submitted FTP order outside the FY client:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_fy_ftp_order.ps1 `
  -FtpAccount AYYYYMMDDHHMMSSNNNN `
  -FtpPassword your_temporary_order_password `
  -OutputDir "$env:MULTISAT_LOCAL_DATA_ROOT\raw\FY4B_GIIRS_AVP\GIIRS_AVP_20240305_20240531_00UTC" `
  -MaxConcurrentDownloads 6
```

Or download directly from FYDataService logs without writing passwords to CSV:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\download_fy_orders_from_logs.ps1 `
  -Last 1 `
  -OutputRoot "$env:MULTISAT_LOCAL_DATA_ROOT\raw\FY4B_GIIRS_AVP" `
  -MaxConcurrentDownloads 6
```

Use `-DryRun` first to verify which order will be selected. Replace `-Last 1` with `-OrderCode AYYYYMMDDHHMMSSNNNN` if you want one specific order.

Watch orders that are still preparing and download them once FTP credentials appear in the FYDataService logs:

```powershell
$orders = @("AYYYYMMDDHHMMSSNNNN", "AYYYYMMDDHHMMSSMMMM")

powershell -ExecutionPolicy Bypass -File .\tools\watch_fy_orders_from_logs.ps1 `
  -OrderCode $orders `
  -OutputRoot "$env:MULTISAT_LOCAL_DATA_ROOT\raw\FY4B_GIIRS_AVP" `
  -IntervalSeconds 300 `
  -MaxHours 72 `
  -MaxConcurrentDownloads 6
```

This watcher does not create FTP credentials. Keep FYDataService open and refresh the order or download list after orders become `Data preparation complete` so the official client can write the temporary FTP credentials to its logs.

## HIRAS L1C hourly windows

HIRAS L1C is too large for full-period bulk download. Generate exact one-hour UTC windows instead of selecting every month at once:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\new_fy_hourly_window_plan.ps1 `
  -StartDate 2024-03-05 `
  -EndDate 2025-12-31 `
  -BatchMonths 1 `
  -Hours 23,0 `
  -ProductName "FY-3E HIRAS L1C DESC" `
  -SuggestedPrefix "FY3E_HIRAS_L1C_DESC_MIN" `
  -OutputPath .\tmp\HIRAS_L1C_DESC_min_23_00_plan.csv
```

For fuller China-domain dawn descending-orbit coverage, use `-Hours 20,21,22,23,0`; this is much larger, so test one month first.
