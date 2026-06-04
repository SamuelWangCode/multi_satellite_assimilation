"""Build a small Gate 1 pilot batch of complete samples.

This local driver builds satellite features and forecast subsets locally, then
uses SSH to extract ERA5 reanalysis label subsets on the server and copies them
back before merging complete samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class SampleTime:
    analysis_time: datetime

    @property
    def stamp(self) -> str:
        return self.analysis_time.strftime("%Y%m%d%H")

    @property
    def iso(self) -> str:
        return self.analysis_time.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def iter_times(start_date: str, end_date: str, hours: list[int]) -> list[SampleTime]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    out: list[SampleTime] = []
    current = start
    while current <= end:
        for hour in hours:
            out.append(SampleTime(current.replace(hour=hour)))
        current += timedelta(days=1)
    return out


def forecast_init_day(analysis_time: datetime) -> tuple[datetime, bool]:
    if analysis_time.hour == 0:
        return (analysis_time - timedelta(days=1)).replace(hour=18), True
    if analysis_time.hour == 12:
        return analysis_time.replace(hour=6), True
    if analysis_time.hour == 6:
        return (analysis_time - timedelta(days=1)).replace(hour=18), False
    if analysis_time.hour == 18:
        return analysis_time.replace(hour=6), False
    raise ValueError(f"unsupported analysis hour: {analysis_time.hour}")


def run_command(command: list[str], log_path: Path, *, retries: int = 1, retry_sleep_seconds: float = 0.0) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    attempts_total = max(1, int(retries))
    for attempt in range(1, attempts_total + 1):
        result = subprocess.run(command, text=True, capture_output=True)
        attempts.append(
            {
                "attempt": attempt,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        log_path.write_text(
            json.dumps(
                {
                    "command": command,
                    "attempts": attempts,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if result.returncode == 0:
            return
        if attempt < attempts_total and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"command failed with code {attempts[-1]['returncode']}: {' '.join(command)}")


def ensure_forecast_available(forecast_root: Path, sample_time: SampleTime) -> bool:
    init_time, exact = forecast_init_day(sample_time.analysis_time)
    subset = "exact6h" if exact else "all_steps"
    pl = forecast_root / subset / "pl" / f"era5_forecast_pl_{subset}_init_{init_time:%Y%m%d}.grib"
    sfc = forecast_root / subset / "sfc" / f"era5_forecast_sfc_{subset}_init_{init_time:%Y%m%d}.grib"
    return pl.exists() and sfc.exists()


def ensure_remote_analysis(args: argparse.Namespace, sample_time: SampleTime, log_dir: Path) -> Path:
    local_dir = Path(args.analysis_output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_nc = local_dir / f"era5_analysis_subset_{sample_time.stamp}.nc"
    local_json = local_dir / f"era5_analysis_subset_{sample_time.stamp}.json"
    if local_nc.exists() and local_json.exists() and not args.overwrite:
        return local_nc

    remote_nc = f"{args.remote_analysis_output_dir}/era5_analysis_subset_{sample_time.stamp}.nc"
    remote_json = f"{args.remote_analysis_output_dir}/era5_analysis_subset_{sample_time.stamp}.json"
    remote_command = (
        f"cd {args.remote_workspace} && "
        f"{args.remote_python} tools/extract_era5_analysis_subset.py "
        f"--analysis-time {sample_time.iso} "
        f"--output-dir {args.remote_analysis_output_dir}"
    )
    run_command(
        [
            "ssh",
            "-i",
            args.ssh_key,
            "-o",
            "StrictHostKeyChecking=accept-new",
            args.remote_host,
            remote_command,
        ],
        log_dir / f"ssh_analysis_{sample_time.stamp}.json",
        retries=args.command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    run_command(
        [
            "scp",
            "-i",
            args.ssh_key,
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{args.remote_host}:{remote_nc}",
            str(local_nc),
        ],
        log_dir / f"scp_analysis_nc_{sample_time.stamp}.json",
        retries=args.command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    run_command(
        [
            "scp",
            "-i",
            args.ssh_key,
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{args.remote_host}:{remote_json}",
            str(local_json),
        ],
        log_dir / f"scp_analysis_json_{sample_time.stamp}.json",
        retries=args.command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    return local_nc


def build_one(args: argparse.Namespace, sample_time: SampleTime, log_dir: Path) -> dict[str, str | int]:
    stamp = sample_time.stamp
    py = sys.executable
    satellite_nc = Path(args.satellite_output_dir) / f"gate1_satellite_sample_{stamp}.nc"
    forecast_nc = Path(args.forecast_output_dir) / f"era5_forecast_subset_{stamp}.nc"
    complete_nc = Path(args.complete_output_dir) / f"gate1_complete_sample_{stamp}.nc"
    if complete_nc.exists() and not args.overwrite:
        return {
            "analysis_time_utc": sample_time.iso,
            "stamp": stamp,
            "status": "ok_existing",
            "complete_sample": str(complete_nc),
        }

    run_command(
        [
            py,
            "tools/build_gate1_satellite_sample.py",
            "--analysis-time",
            sample_time.iso,
            "--output-dir",
            args.satellite_output_dir,
            "--figure-dir",
            args.figure_output_dir,
        ],
        log_dir / f"satellite_{stamp}.json",
        retries=args.local_command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    run_command(
        [
            py,
            "tools/extract_era5_forecast_subset.py",
            "--analysis-time",
            sample_time.iso,
            "--forecast-root",
            args.forecast_root,
            "--output-dir",
            args.forecast_output_dir,
        ],
        log_dir / f"forecast_{stamp}.json",
        retries=args.local_command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    analysis_nc = ensure_remote_analysis(args, sample_time, log_dir)
    analysis_for_complete = analysis_nc
    humidity_label_policy = "derived_xa_q_smoke_fallback_pending_direct_q"
    if args.inject_direct_q:
        direct_grib = Path(args.direct_q_grib_dir) / f"era5_reanalysis_label_pl_{sample_time.analysis_time:%Y%m}.grib"
        if direct_grib.exists():
            direct_dir = Path(args.direct_analysis_output_dir)
            direct_dir.mkdir(parents=True, exist_ok=True)
            direct_analysis_nc = direct_dir / f"era5_analysis_subset_{stamp}.nc"
            run_command(
                [
                    py,
                    "tools/inject_direct_era5_q_to_analysis_subset.py",
                    "--analysis-time",
                    sample_time.iso,
                    "--input",
                    str(analysis_nc),
                    "--direct-q-grib",
                    str(direct_grib),
                    "--output",
                    str(direct_analysis_nc),
                ],
                log_dir / f"inject_direct_q_{stamp}.json",
                retries=args.local_command_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            analysis_for_complete = direct_analysis_nc
            humidity_label_policy = "direct_era5_reanalysis_xa_q"
        elif args.require_direct_q:
            raise FileNotFoundError(f"direct q GRIB not found for {sample_time.stamp}: {direct_grib}")

    run_command(
        [
            py,
            "tools/build_gate1_complete_sample.py",
            "--stamp",
            stamp,
            "--satellite-sample",
            str(satellite_nc),
            "--analysis-subset",
            str(analysis_for_complete),
            "--forecast-subset",
            str(forecast_nc),
            "--static-features",
            args.static_features,
            "--output-dir",
            args.complete_output_dir,
        ],
        log_dir / f"complete_{stamp}.json",
        retries=args.local_command_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    return {
        "analysis_time_utc": sample_time.iso,
        "stamp": stamp,
        "status": "ok",
        "satellite_sample": str(satellite_nc),
        "analysis_subset": str(analysis_for_complete),
        "forecast_subset": str(forecast_nc),
        "complete_sample": str(complete_nc),
        "humidity_label_policy": humidity_label_policy,
    }


def build(args: argparse.Namespace) -> None:
    hours = [int(item) for item in args.hours.split(",") if item.strip()]
    candidates = iter_times(args.start_date, args.end_date, hours)
    if args.max_samples > 0:
        candidates = candidates[: args.max_samples]

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    for directory in [
        args.satellite_output_dir,
        args.figure_output_dir,
        args.forecast_output_dir,
        args.analysis_output_dir,
        args.direct_analysis_output_dir,
        args.complete_output_dir,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for sample_time in candidates:
        row: dict[str, str | int] = {"analysis_time_utc": sample_time.iso, "stamp": sample_time.stamp}
        try:
            if not ensure_forecast_available(Path(args.forecast_root), sample_time):
                row.update({"status": "skipped_missing_forecast"})
            else:
                row = build_one(args, sample_time, log_dir)
        except Exception as exc:  # noqa: BLE001 - batch builder records failures and continues by default.
            row.update({"status": "failed", "error": str(exc)})
            if args.stop_on_error:
                rows.append(row)
                break
        rows.append(row)

    columns = sorted({key for row in rows for key in row})
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "manifest": str(manifest_path),
        "candidate_count": len(candidates),
        "ok_count": sum(1 for row in rows if str(row.get("status", "")).startswith("ok")),
        "failed_count": sum(1 for row in rows if row.get("status") == "failed"),
        "skipped_count": sum(1 for row in rows if str(row.get("status", "")).startswith("skipped")),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-05-01")
    parser.add_argument("--end-date", default="2024-05-02")
    parser.add_argument("--hours", default="0,12")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--forecast-root", default="data/raw/era5_gate1_forecast")
    parser.add_argument("--satellite-output-dir", default="data/processed/samples_025_pilot")
    parser.add_argument("--figure-output-dir", default="data/diagnostics/figures/gate1_samples_pilot")
    parser.add_argument("--forecast-output-dir", default="data/interim/era5_forecast_pilot")
    parser.add_argument("--analysis-output-dir", default="data/interim/era5_analysis_pilot")
    parser.add_argument("--direct-analysis-output-dir", default="data/interim/era5_analysis_pilot_directq")
    parser.add_argument("--complete-output-dir", default="data/processed/gate1_complete_samples_pilot")
    parser.add_argument("--static-features", default="data/interim/static/gate1_static_features_025.nc")
    parser.add_argument("--manifest", default="data/processed/gate1_complete_samples_pilot/manifest.csv")
    parser.add_argument("--log-dir", default="logs/gate1_pilot_batch")
    parser.add_argument("--remote-host", default=os.environ.get("MULTISAT_REMOTE_HOST", ""))
    parser.add_argument("--ssh-key", default=os.environ.get("MULTISAT_SSH_KEY", ""))
    parser.add_argument("--remote-workspace", default=os.environ.get("MULTISAT_SERVER_WORKSPACE", ""))
    parser.add_argument("--remote-python", default="python3")
    parser.add_argument(
        "--remote-analysis-output-dir",
        default=os.environ.get("MULTISAT_REMOTE_ANALYSIS_OUTPUT_DIR", "data/interim/era5_analysis_pilot"),
    )
    parser.add_argument("--direct-q-grib-dir", default="data/raw/era5_reanalysis_label_direct_q_2024/analysis/pl")
    parser.add_argument("--inject-direct-q", action="store_true")
    parser.add_argument("--require-direct-q", action="store_true")
    parser.add_argument("--command-retries", type=int, default=3)
    parser.add_argument("--local-command-retries", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
