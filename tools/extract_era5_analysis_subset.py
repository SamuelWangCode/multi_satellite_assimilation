"""Extract one ERA5 reanalysis label (xa) subset on the Gate 1 grid.

This is a smoke-test utility for xa only. It reads existing monthly ERA5
reanalysis files, clips the approved China-buffer
domain, selects the configured pressure levels, and derives q/q2m from the
available relative humidity/dewpoint fields.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


DOMAIN = {
    "lon_min": 70.0,
    "lon_max": 150.0,
    "lat_min": 0.0,
    "lat_max": 60.0,
    "resolution": 0.25,
}

DEFAULT_LEVELS_HPA = [1000, 925, 850, 700, 500, 300, 200]
PL_VARS = ["z", "r", "t", "u", "v"]
SFC_VARS = ["msl", "sp", "t2m", "u10", "v10", "u100", "v100", "tcwv"]
EPSILON = 0.622


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def saturation_vapor_pressure_pa(temperature_k: np.ndarray) -> np.ndarray:
    temperature_c = temperature_k - 273.15
    return 611.2 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def specific_humidity_from_rh(
    temperature_k: np.ndarray,
    relative_humidity_percent: np.ndarray,
    pressure_hpa: np.ndarray,
) -> np.ndarray:
    pressure_pa = pressure_hpa[:, None, None] * 100.0
    vapor_pressure = np.clip(relative_humidity_percent, 0.0, 100.0) / 100.0
    vapor_pressure = vapor_pressure * saturation_vapor_pressure_pa(temperature_k)
    return EPSILON * vapor_pressure / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)


def specific_humidity_from_dewpoint(dewpoint_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    vapor_pressure = saturation_vapor_pressure_pa(dewpoint_k)
    return EPSILON * vapor_pressure / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)


def time_index(ds: Dataset, analysis_time: datetime) -> int:
    valid_time = np.asarray(ds.variables["valid_time"][:], dtype=np.int64)
    target = int(analysis_time.timestamp())
    matches = np.where(valid_time == target)[0]
    if matches.size != 1:
        raise ValueError(f"expected one valid_time match for {format_time(analysis_time)}, found {matches.size}")
    return int(matches[0])


def contiguous_domain_slices(ds: Dataset) -> tuple[slice, slice, np.ndarray, np.ndarray]:
    lat = np.asarray(ds.variables["latitude"][:], dtype=np.float32)
    lon = np.asarray(ds.variables["longitude"][:], dtype=np.float32)
    lat_mask = (lat >= DOMAIN["lat_min"]) & (lat <= DOMAIN["lat_max"])
    lon_mask = (lon >= DOMAIN["lon_min"]) & (lon <= DOMAIN["lon_max"])
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("domain selection produced an empty slice")
    lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    return lat_slice, lon_slice, lat[lat_slice], lon[lon_slice]


def pressure_level_indices(ds: Dataset, levels_hpa: list[int]) -> list[int]:
    available = np.asarray(ds.variables["pressure_level"][:], dtype=np.int32)
    out = []
    for level in levels_hpa:
        matches = np.where(available == level)[0]
        if matches.size != 1:
            raise ValueError(f"pressure level {level} hPa not found; available={available.tolist()}")
        out.append(int(matches[0]))
    return out


def write_coord(out: Dataset, name: str, values: np.ndarray, units: str) -> None:
    var = out.createVariable(name, values.dtype, (name,))
    var[:] = values
    var.units = units


def write_var(out: Dataset, name: str, dims: tuple[str, ...], values: np.ndarray, attrs: dict[str, str]) -> None:
    var = out.createVariable(name, "f4", dims, zlib=True, complevel=2, fill_value=np.nan)
    var[:] = values.astype(np.float32)
    for key, value in attrs.items():
        setattr(var, key, value)


def stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    if not finite.any():
        return {"finite": 0, "total": int(values.size)}
    return {
        "finite": int(finite.sum()),
        "total": int(values.size),
        "min": float(np.nanmin(values)),
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
    }


def extract(args: argparse.Namespace) -> None:
    analysis_time = parse_time(args.analysis_time)
    ym = analysis_time.strftime("%Y%m")
    root = Path(args.era5_root)
    pl_path = root / f"{ym}_pl.nc"
    sfc_path = root / f"{ym}_sfc.nc"
    d2m_path = root / f"{ym}_d2m.nc"
    for path in [pl_path, sfc_path, d2m_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    levels = [int(item) for item in args.pressure_levels.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"era5_analysis_subset_{analysis_time:%Y%m%d%H}"
    output_path = output_dir / f"{stem}.nc"
    summary_path = output_dir / f"{stem}.json"

    summary: dict[str, object] = {
        "analysis_time_utc": format_time(analysis_time),
        "source_kind": "era5_reanalysis_label",
        "source_product": "ERA5 reanalysis",
        "archive_type": "analysis/an",
        "note": "xa smoke subset only; ERA5 reanalysis label, not xb and not dx.",
        "input_paths": {
            "pl": str(pl_path),
            "sfc": str(sfc_path),
            "d2m": str(d2m_path),
        },
        "pressure_levels_hpa": levels,
        "variables": {},
    }

    with Dataset(pl_path, "r") as pl_ds, Dataset(sfc_path, "r") as sfc_ds, Dataset(d2m_path, "r") as d2m_ds:
        ti = time_index(pl_ds, analysis_time)
        if time_index(sfc_ds, analysis_time) != ti or time_index(d2m_ds, analysis_time) != ti:
            raise ValueError("time index mismatch across ERA5 files")
        lat_slice, lon_slice, lat, lon = contiguous_domain_slices(pl_ds)
        level_idx = pressure_level_indices(pl_ds, levels)
        levels_arr = np.asarray(levels, dtype=np.int32)

        with Dataset(output_path, "w") as out:
            out.createDimension("level", len(levels))
            out.createDimension("lat", len(lat))
            out.createDimension("lon", len(lon))
            write_coord(out, "level", levels_arr, "hPa")
            write_coord(out, "lat", lat.astype(np.float32), "degrees_north")
            write_coord(out, "lon", lon.astype(np.float32), "degrees_east")
            out.analysis_time_utc = format_time(analysis_time)
            out.source_kind = "era5_reanalysis_label"
            out.source_product = "ERA5 reanalysis"
            out.archive_type = "analysis/an"
            out.domain_json = json.dumps(DOMAIN, sort_keys=True)
            out.note = "xa smoke subset only; xb/dx require ERA5 short-range forecast."

            pl_arrays: dict[str, np.ndarray] = {}
            for name in PL_VARS:
                if name not in pl_ds.variables:
                    continue
                values = np.asarray(pl_ds.variables[name][ti, level_idx, lat_slice, lon_slice], dtype=np.float32)
                pl_arrays[name] = values
                write_var(out, f"xa_{name}", ("level", "lat", "lon"), values, {"source_variable": name})
                summary["variables"][f"xa_{name}"] = stats(values)  # type: ignore[index]

            if "t" in pl_arrays and "r" in pl_arrays:
                q = specific_humidity_from_rh(pl_arrays["t"], pl_arrays["r"], levels_arr.astype(np.float32))
                write_var(
                    out,
                    "xa_q",
                    ("level", "lat", "lon"),
                    q,
                    {"derived_from": "t,r,pressure_level", "q_source": "derived_from_relative_humidity"},
                )
                summary["variables"]["xa_q"] = stats(q)  # type: ignore[index]

            sfc_arrays: dict[str, np.ndarray] = {}
            for name in SFC_VARS:
                if name not in sfc_ds.variables:
                    continue
                out_name = "mslp" if name == "msl" else name
                values = np.asarray(sfc_ds.variables[name][ti, lat_slice, lon_slice], dtype=np.float32)
                sfc_arrays[out_name] = values
                write_var(out, f"xa_{out_name}", ("lat", "lon"), values, {"source_variable": name})
                summary["variables"][f"xa_{out_name}"] = stats(values)  # type: ignore[index]

            d2m = np.asarray(d2m_ds.variables["d2m"][ti, lat_slice, lon_slice], dtype=np.float32)
            write_var(out, "xa_d2m", ("lat", "lon"), d2m, {"source_variable": "d2m"})
            summary["variables"]["xa_d2m"] = stats(d2m)  # type: ignore[index]
            if "sp" in sfc_arrays:
                q2m = specific_humidity_from_dewpoint(d2m, sfc_arrays["sp"])
                write_var(out, "xa_q2m", ("lat", "lon"), q2m, {"derived_from": "d2m,sp", "q_source": "derived_from_dewpoint"})
                summary["variables"]["xa_q2m"] = stats(q2m)  # type: ignore[index]
            out.humidity_source_json = json.dumps(
                {
                    "xa_q": "derived_from_t_r_pressure_level",
                    "xa_q2m": "derived_from_d2m_sp",
                },
                sort_keys=True,
            )

    summary["output_path"] = str(output_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    server_workspace = Path(os.environ.get("MULTISAT_SERVER_WORKSPACE", "."))
    parser.add_argument("--analysis-time", required=True, help="UTC analysis time, e.g. 2024-05-01T00:00:00Z")
    parser.add_argument("--era5-root", default=os.environ.get("MULTISAT_ERA5_ANALYSIS_ROOT", "data/raw/era5_analysis"))
    parser.add_argument("--output-dir", default=str(server_workspace / "data/interim/era5_analysis_smoke"))
    parser.add_argument("--pressure-levels", default=",".join(str(item) for item in DEFAULT_LEVELS_HPA))
    return parser.parse_args()


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()
