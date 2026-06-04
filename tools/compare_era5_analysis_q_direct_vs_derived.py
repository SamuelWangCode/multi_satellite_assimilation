"""Compare direct ERA5 reanalysis q against derived q from t/r/p subsets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr


def open_direct_q(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def summarize(diff: np.ndarray) -> dict[str, float | int]:
    finite = diff[np.isfinite(diff)]
    row: dict[str, float | int] = {"finite": int(finite.size), "total": int(diff.size)}
    if finite.size:
        for q in [1, 5, 50, 95, 99]:
            row[f"p{q:02d}"] = float(np.nanpercentile(finite, q))
        row["mean"] = float(np.nanmean(finite))
        row["mae"] = float(np.nanmean(np.abs(finite)))
        row["rmse"] = float(np.sqrt(np.nanmean(finite * finite)))
        row["max_abs"] = float(np.nanmax(np.abs(finite)))
    return row


def main() -> None:
    args = parse_args()
    direct_path = Path(args.direct_q_grib)
    derived_path = Path(args.derived_subset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open_direct_q(direct_path) as direct, xr.open_dataset(derived_path) as derived:
        target = np.datetime64(args.analysis_time.replace("Z", ""))
        matches = np.where(direct["time"].values == target)[0]
        if matches.size != 1:
            raise ValueError(f"expected one direct q time match for {args.analysis_time}, found {matches.size}")
        q_direct = direct["q"].isel(time=int(matches[0])).rename({"isobaricInhPa": "level", "latitude": "lat", "longitude": "lon"})
        q_derived = derived["xa_q"]
        if not np.array_equal(q_direct["level"].values.astype(int), q_derived["level"].values.astype(int)):
            raise ValueError("level mismatch")
        if not np.allclose(q_direct["lat"].values, q_derived["lat"].values):
            raise ValueError("lat mismatch")
        if not np.allclose(q_direct["lon"].values, q_derived["lon"].values):
            raise ValueError("lon mismatch")

        rows = []
        for level in q_derived["level"].values:
            diff = (q_derived.sel(level=level) - q_direct.sel(level=level)).values
            row = {"analysis_time_utc": args.analysis_time, "level": int(level), **summarize(diff)}
            rows.append(row)

    fieldnames = ["analysis_time_utc", "level", "finite", "total", "mean", "mae", "rmse", "p01", "p05", "p50", "p95", "p99", "max_abs"]
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-time", default="2024-05-01T00:00:00Z")
    parser.add_argument("--direct-q-grib", default="data/raw/era5_analysis_q_check_20260601/era5_analysis_pl_202405.grib")
    parser.add_argument("--derived-subset", default="data/interim/era5_analysis_smoke/era5_analysis_subset_2024050100.nc")
    parser.add_argument("--output", default="data/diagnostics/tables/era5_analysis_q_direct_vs_derived_2024050100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
