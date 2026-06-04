"""Create Gate 1 ERA5 analysis/short-forecast time inventory tables.

This does not download ERA5. It records the valid-time to forecast-cycle
mapping approved for the first-stage experiment, so download and processing
scripts can use a single source of truth.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


VALID_TIME_MAPPING = {
    0: {"init_hour": 18, "init_day_offset": -1, "step_hours": 6, "lead_hours": 6},
    6: {"init_hour": 18, "init_day_offset": -1, "step_hours": 12, "lead_hours": 12},
    12: {"init_hour": 6, "init_day_offset": 0, "step_hours": 6, "lead_hours": 6},
    18: {"init_hour": 6, "init_day_offset": 0, "step_hours": 12, "lead_hours": 12},
}

PL_VARIABLES = ["z", "t", "u", "v", "q"]
PL_LEVELS = [1000, 925, 850, 700, 500, 300, 200]
SFC_VARIABLES = ["mslp", "sp", "t2m", "d2m", "q2m", "u10", "v10"]


def fmt(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_valid_times(start_date: str, end_date: str, hours: list[int]) -> list[datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out: list[datetime] = []
    day = start
    while day <= end:
        for hour in sorted(hours):
            out.append(day.replace(hour=hour))
        day += timedelta(days=1)
    return out


def forecast_init_for_valid(valid_time: datetime) -> tuple[datetime, int, int, bool]:
    mapping = VALID_TIME_MAPPING[valid_time.hour]
    init_day = valid_time + timedelta(days=mapping["init_day_offset"])
    init_time = init_day.replace(hour=mapping["init_hour"], minute=0, second=0, microsecond=0)
    step_hours = mapping["step_hours"]
    lead_hours = mapping["lead_hours"]
    exact_6h_subset = lead_hours == 6 and valid_time.hour in {0, 12}
    return init_time, step_hours, lead_hours, exact_6h_subset


def write_forecast_plan(path: Path, valid_times: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "kind",
        "valid_time_utc",
        "analysis_hour_utc",
        "forecast_init_time_utc",
        "forecast_cycle_hour_utc",
        "forecast_step_hours",
        "forecast_lead_hours",
        "exact_6h_subset",
        "domain",
        "pressure_level_variables",
        "pressure_levels_hpa",
        "surface_variables",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for valid_time in valid_times:
            init_time, step_hours, lead_hours, exact_6h_subset = forecast_init_for_valid(valid_time)
            writer.writerow(
                {
                    "source": "era5",
                    "kind": "short_range_forecast",
                    "valid_time_utc": fmt(valid_time),
                    "analysis_hour_utc": valid_time.hour,
                    "forecast_init_time_utc": fmt(init_time),
                    "forecast_cycle_hour_utc": init_time.hour,
                    "forecast_step_hours": step_hours,
                    "forecast_lead_hours": lead_hours,
                    "exact_6h_subset": int(exact_6h_subset),
                    "domain": "70-150E,0-60N",
                    "pressure_level_variables": " ".join(PL_VARIABLES),
                    "pressure_levels_hpa": " ".join(str(item) for item in PL_LEVELS),
                    "surface_variables": " ".join(SFC_VARIABLES),
                    "status": "required_not_downloaded",
                    "notes": "Gate 1 approved mapping: 00/12 lead=6h, 06/18 lead=12h.",
                }
            )


def write_analysis_plan(path: Path, valid_times: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "kind",
        "valid_time_utc",
        "analysis_hour_utc",
        "domain",
        "pressure_level_variables",
        "pressure_levels_hpa",
        "surface_variables",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for valid_time in valid_times:
            writer.writerow(
                {
                    "source": "era5",
                    "kind": "analysis",
                    "valid_time_utc": fmt(valid_time),
                    "analysis_hour_utc": valid_time.hour,
                    "domain": "70-150E,0-60N",
                    "pressure_level_variables": " ".join(PL_VARIABLES),
                    "pressure_levels_hpa": " ".join(str(item) for item in PL_LEVELS),
                    "surface_variables": " ".join(SFC_VARIABLES),
                    "status": "required_partial_local_analysis_2010_2024_only",
                    "notes": "2025 analysis still missing from checked /bigdata3/ERA_time_format.",
                }
            )


def write_summary(path: Path, valid_times: list[datetime]) -> None:
    counts = {
        "analysis_times": len(valid_times),
        "lead_6h": 0,
        "lead_12h": 0,
        "exact_6h_subset": 0,
    }
    for valid_time in valid_times:
        _, _, lead_hours, exact = forecast_init_for_valid(valid_time)
        counts[f"lead_{lead_hours}h"] += 1
        counts["exact_6h_subset"] += int(exact)

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["item", "count"])
        writer.writeheader()
        for key, value in counts.items():
            writer.writerow({"item": key, "count": value})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2024-03-05")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--analysis-hours", default="0,6,12,18")
    parser.add_argument("--output-dir", default="data/interim/indexes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hours = [int(item) for item in args.analysis_hours.split(",") if item.strip()]
    invalid = sorted(set(hours) - set(VALID_TIME_MAPPING))
    if invalid:
        raise ValueError(f"unsupported analysis hours for approved mapping: {invalid}")

    valid_times = iter_valid_times(args.start_date, args.end_date, hours)
    output_dir = Path(args.output_dir)
    write_forecast_plan(output_dir / "era5_short_forecast_plan.csv", valid_times)
    write_analysis_plan(output_dir / "era5_analysis_plan.csv", valid_times)
    write_summary(output_dir / "era5_short_forecast_plan_summary.csv", valid_times)

    print("ERA5 short forecast inventory plan completed.")
    print(f"Analysis times: {len(valid_times)}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
