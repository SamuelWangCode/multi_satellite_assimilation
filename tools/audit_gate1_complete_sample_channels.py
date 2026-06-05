"""Audit Gate 1 complete-sample channel residuals by simple regimes.

This is a lightweight diagnostic for complete NetCDF samples. It summarizes
ERA5-supervised dx labels against satellite channel availability/value fields
by sensor, channel, land/sea, cloud bin and forecast lead.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


CHANNEL_RE = re.compile(r"^(fy4b_agri|fy3e_mwts|fy3e_mwhs)_ch(\d{2})_value_mean$")
TARGETS = ("dx_t2m", "dx_u10", "dx_v10", "dx_mslp")


@dataclass
class Accumulator:
    n: int = 0
    sat_n: int = 0
    sum_dx: float = 0.0
    sum_dx2: float = 0.0
    sum_sat: float = 0.0
    sum_sat2: float = 0.0
    sum_obs_count: float = 0.0
    sum_missing: float = 0.0

    def add(self, dx: np.ndarray, sat: np.ndarray, obs_count: np.ndarray, missing: np.ndarray) -> None:
        valid = np.isfinite(dx)
        if not valid.any():
            return
        dxv = dx[valid].astype(np.float64, copy=False)
        satv = sat[valid].astype(np.float64, copy=False)
        obsv = obs_count[valid].astype(np.float64, copy=False)
        missv = missing[valid].astype(np.float64, copy=False)
        sat_valid = np.isfinite(satv)
        self.n += int(dxv.size)
        self.sat_n += int(sat_valid.sum())
        self.sum_dx += float(np.nansum(dxv))
        self.sum_dx2 += float(np.nansum(dxv * dxv))
        self.sum_sat += float(np.sum(satv[sat_valid]))
        self.sum_sat2 += float(np.sum(satv[sat_valid] * satv[sat_valid]))
        self.sum_obs_count += float(np.nansum(obsv))
        self.sum_missing += float(np.nansum(missv))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default="data/processed/gate1_complete_samples_202405_202407_directq")
    parser.add_argument("--output-csv", default="data/diagnostics/tables/channel_bias_audit.csv")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--grid-sample-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260605)
    return parser.parse_args()


def finite_std(sum_value: float, sum_value2: float, n: int) -> float:
    if n <= 1:
        return math.nan
    mean = sum_value / n
    var = max(0.0, (sum_value2 / n) - mean * mean)
    return math.sqrt(var)


def choose_points(rng: np.random.Generator, n_points: int, sample_size: int) -> np.ndarray:
    if sample_size <= 0 or sample_size >= n_points:
        return np.arange(n_points)
    return rng.choice(n_points, size=sample_size, replace=False)


def classify_land(landmask: np.ndarray) -> np.ndarray:
    out = np.full(landmask.shape, "unknown", dtype=object)
    finite = np.isfinite(landmask)
    out[finite & (landmask >= 0.5)] = "land"
    out[finite & (landmask < 0.5)] = "sea"
    return out


def classify_cloud(cloud_fraction: np.ndarray) -> np.ndarray:
    out = np.full(cloud_fraction.shape, "unknown", dtype=object)
    finite = np.isfinite(cloud_fraction)
    out[finite & (cloud_fraction < 0.2)] = "clear"
    out[finite & (cloud_fraction >= 0.2) & (cloud_fraction <= 0.8)] = "mixed"
    out[finite & (cloud_fraction > 0.8)] = "cloudy"
    return out


def flat_var(ds: xr.Dataset, name: str, flat_index: np.ndarray, fallback: float = math.nan) -> np.ndarray:
    if name not in ds:
        return np.full(flat_index.shape, fallback, dtype=np.float32)
    arr = np.asarray(ds[name].values)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D for this audit, got shape={arr.shape}")
    return arr.ravel()[flat_index]


def audit(args: argparse.Namespace) -> None:
    sample_dir = Path(args.sample_dir)
    paths = sorted(sample_dir.glob("gate1_complete_sample_*.nc"))
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise FileNotFoundError(f"no complete samples found in {sample_dir}")

    rng = np.random.default_rng(args.seed)
    acc: dict[tuple[str, str, str, str, int, str, str, str, str], Accumulator] = defaultdict(Accumulator)
    sample_count = 0

    for path in paths:
        with xr.open_dataset(path) as ds:
            lat_n = int(ds.sizes["lat"])
            lon_n = int(ds.sizes["lon"])
            flat_index = choose_points(rng, lat_n * lon_n, args.grid_sample_size)
            lead = int(ds.attrs.get("forecast_lead_hours", -1))
            land_class = classify_land(flat_var(ds, "static_landmask", flat_index))
            cloud_class = classify_cloud(flat_var(ds, "fy4b_clm_cloud_fraction", flat_index))
            channels = []
            for name in ds.data_vars:
                match = CHANNEL_RE.match(name)
                if match:
                    channels.append((match.group(1), match.group(2), name))

            target_values = {target: flat_var(ds, target, flat_index) for target in TARGETS if target in ds}

            for sensor, channel, value_name in channels:
                value = flat_var(ds, value_name, flat_index)
                obs_count = flat_var(ds, f"{sensor}_ch{channel}_obs_count", flat_index, fallback=0.0)
                missing = flat_var(ds, f"{sensor}_ch{channel}_missing_mask", flat_index, fallback=1.0)
                finite_value = np.where(np.isfinite(value), value, np.nan)
                for target, dx in target_values.items():
                    for land in ("land", "sea", "unknown"):
                        land_mask = land_class == land
                        if not land_mask.any():
                            continue
                        for cloud in ("clear", "mixed", "cloudy", "unknown"):
                            regime = land_mask & (cloud_class == cloud)
                            if not regime.any():
                                continue
                            key = (
                                sensor,
                                channel,
                                land,
                                cloud,
                                lead,
                                target,
                                "pending_precip_product",
                                "not_available_current_sample",
                                "range_time_coverage_sanity_only",
                            )
                            acc[key].add(dx[regime], finite_value[regime], obs_count[regime], missing[regime])
            sample_count += 1

    rows = []
    for key, item in acc.items():
        sensor, channel, land, cloud, lead, target, rain_proxy, scan_angle_bin, qc_flag_ours = key
        if item.n <= 0:
            continue
        rows.append(
            {
                "sensor_id": sensor,
                "channel_id": int(channel),
                "land_sea_mask": land,
                "cloud_flag": cloud,
                "rain_proxy": rain_proxy,
                "scan_angle_bin": scan_angle_bin,
                "forecast_lead_hours": lead,
                "target_variable": target,
                "qc_flag_ours": qc_flag_ours,
                "sample_count": sample_count,
                "grid_point_count": item.n,
                "satellite_value_count": item.sat_n,
                "mean_dx_label": item.sum_dx / item.n,
                "std_dx_label": finite_std(item.sum_dx, item.sum_dx2, item.n),
                "mean_satellite_value": item.sum_sat / item.sat_n if item.sat_n > 0 else math.nan,
                "std_satellite_value": finite_std(item.sum_sat, item.sum_sat2, item.sat_n),
                "mean_obs_count": item.sum_obs_count / item.n,
                "missing_rate": item.sum_missing / item.n,
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["sensor_id", "channel_id", "target_variable", "land_sea_mask", "cloud_flag", "forecast_lead_hours"]
    ).to_csv(output_csv, index=False)
    print(
        {
            "sample_dir": str(sample_dir),
            "sample_count": sample_count,
            "row_count": len(rows),
            "output_csv": str(output_csv),
        }
    )


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
