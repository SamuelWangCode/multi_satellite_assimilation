"""QC complete Gate 1 NetCDF samples for required fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr


GROUP_PREFIXES = {
    "background": "xb_",
    "label": "xa_",
    "target": "dx_",
    "agri": "fy4b_agri",
    "clm": "fy4b_clm",
    "mwts": "fy3e_mwts",
    "mwhs": "fy3e_mwhs",
    "static": "static_",
}


def variable_count(names: list[str], prefix: str) -> int:
    return sum(1 for name in names if name.startswith(prefix))


def any_contains(names: list[str], pattern: str) -> bool:
    return any(pattern in name for name in names)


def mean_of_masks(ds: xr.Dataset, suffix: str) -> float:
    values = []
    for name in ds.data_vars:
        if name.endswith(suffix):
            arr = ds[name].values
            finite = np.isfinite(arr)
            if finite.any():
                values.append(float(np.nanmean(arr)))
    if not values:
        return float("nan")
    return float(np.mean(values))


def obs_count_max(ds: xr.Dataset, prefix: str) -> float:
    max_values = []
    for name in ds.data_vars:
        if name.startswith(prefix) and name.endswith("_obs_count"):
            arr = ds[name].values
            finite = np.isfinite(arr)
            if finite.any():
                max_values.append(float(np.nanmax(arr)))
    if not max_values:
        return float("nan")
    return float(np.nanmax(max_values))


def json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def qc_sample(path: Path) -> dict[str, object]:
    with xr.open_dataset(path) as ds:
        names = list(ds.data_vars)
        humidity = ds.attrs.get("humidity_source_json", "")
        row: dict[str, object] = {
            "path": str(path),
            "analysis_time_utc": json_safe(ds.attrs.get("analysis_time_utc", "")),
            "forecast_lead_hours": json_safe(ds.attrs.get("forecast_lead_hours", "")),
            "exact_6h_subset": json_safe(ds.attrs.get("exact_6h_subset", "")),
            "variable_count": len(names),
            "humidity_source_json": json_safe(humidity),
            "xa_q_direct": "direct_era5_reanalysis_q" in str(humidity),
            "mean_missing_mask": mean_of_masks(ds, "_missing_mask"),
            "mean_valid_mask": mean_of_masks(ds, "_valid_mask"),
            "mwts_obs_count_max": obs_count_max(ds, "fy3e_mwts"),
            "mwhs_obs_count_max": obs_count_max(ds, "fy3e_mwhs"),
        }
        for group, prefix in GROUP_PREFIXES.items():
            row[f"{group}_var_count"] = variable_count(names, prefix)

        required_checks = {
            "has_forecast_lead_hours": "forecast_lead_hours" in ds.attrs,
            "has_exact_6h_subset": "exact_6h_subset" in ds.attrs,
            "has_availability_class": any_contains(names, "availability_class"),
            "has_obs_count": any_contains(names, "_obs_count"),
            "has_missing_mask": any_contains(names, "_missing_mask"),
            "has_valid_mask": any_contains(names, "_valid_mask"),
            "has_time_offset_min": any_contains(names, "time_offset_min"),
            "has_sensor_id": any_contains(names, "_sensor_id"),
            "has_source_id": any_contains(names, "_source_id"),
            "has_channel_id": any_contains(names, "_channel_id"),
            "has_static_landmask": "static_landmask" in names,
            "has_static_elevation": "static_elevation_m" in names,
            "has_mwts_group": variable_count(names, "fy3e_mwts") > 0,
            "has_mwhs_group": variable_count(names, "fy3e_mwhs") > 0,
            "has_agri_group": variable_count(names, "fy4b_agri") > 0,
            "has_clm_group": variable_count(names, "fy4b_clm") > 0,
            "has_dx_target": variable_count(names, "dx_") > 0,
            "has_xb_background": variable_count(names, "xb_") > 0,
            "has_xa_label": variable_count(names, "xa_") > 0,
        }
        row.update(required_checks)
        row["required_check_pass"] = all(required_checks.values())
        return row


def run(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir)
    paths = sorted(sample_dir.glob("gate1_complete_sample_*.nc"))
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    rows = [qc_sample(path) for path in paths]

    csv_path = Path(args.output_csv)
    json_path = Path(args.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        columns = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    summary = {
        "sample_dir": str(sample_dir),
        "sample_count": len(rows),
        "required_check_pass_count": sum(1 for row in rows if row.get("required_check_pass")),
        "direct_q_ready_count": sum(1 for row in rows if row.get("xa_q_direct")),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default="data/processed/gate1_complete_samples_pilot")
    parser.add_argument("--output-csv", default="data/diagnostics/tables/gate1_complete_samples_pilot_qc.csv")
    parser.add_argument("--output-json", default="data/diagnostics/tables/gate1_complete_samples_pilot_qc.json")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
