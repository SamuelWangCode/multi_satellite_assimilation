"""Extract one ERA5 short-range forecast xb subset from Gate 1 GRIB files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr


DOMAIN = {
    "lon_min": 70.0,
    "lon_max": 150.0,
    "lat_min": 0.0,
    "lat_max": 60.0,
    "resolution": 0.25,
}

EPSILON = 0.622


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def forecast_mapping(analysis_time: datetime) -> tuple[datetime, int, bool]:
    hour = analysis_time.hour
    if hour == 0:
        return (analysis_time - timedelta(days=1)).replace(hour=18), 6, True
    if hour == 6:
        return (analysis_time - timedelta(days=1)).replace(hour=18), 12, False
    if hour == 12:
        return analysis_time.replace(hour=6), 6, True
    if hour == 18:
        return analysis_time.replace(hour=6), 12, False
    raise ValueError(f"unsupported analysis hour: {hour}")


def saturation_vapor_pressure_pa(temperature_k: xr.DataArray) -> xr.DataArray:
    temperature_c = temperature_k - 273.15
    return 611.2 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def specific_humidity_from_dewpoint(dewpoint_k: xr.DataArray, pressure_pa: xr.DataArray) -> xr.DataArray:
    vapor_pressure = saturation_vapor_pressure_pa(dewpoint_k)
    return EPSILON * vapor_pressure / (pressure_pa - (1.0 - EPSILON) * vapor_pressure)


def select_valid_time(ds: xr.Dataset, analysis_time: datetime) -> xr.Dataset:
    target = np.datetime64(analysis_time.replace(tzinfo=None))
    valid_time = ds["valid_time"]
    if "time" not in valid_time.dims:
        raise ValueError("valid_time coordinate does not include time dimension")
    matches = np.where(valid_time.values == target)
    if len(matches[0]) != 1:
        raise ValueError(f"expected one valid_time match for {format_time(analysis_time)}, found {len(matches[0])}")
    time_index = int(matches[0][0])
    selected = ds.isel(time=time_index)
    if "step" in selected.dims:
        step_index = int(matches[1][0])
        selected = selected.isel(step=step_index)
    return selected


def stats(values: xr.DataArray) -> dict[str, float | int]:
    arr = values.values
    finite = np.isfinite(arr)
    if not finite.any():
        return {"finite": 0, "total": int(arr.size)}
    return {
        "finite": int(finite.sum()),
        "total": int(arr.size),
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
    }


def rename_level(ds: xr.Dataset) -> xr.Dataset:
    if "isobaricInhPa" in ds.dims or "isobaricInhPa" in ds.coords:
        ds = ds.rename({"isobaricInhPa": "level"})
    return ds


def open_grib(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def assign_pl_var(out: xr.Dataset, name: str, values: xr.DataArray) -> None:
    out[name] = (("level", "lat", "lon"), values.values.astype(np.float32))


def assign_sfc_var(out: xr.Dataset, name: str, values: xr.DataArray) -> None:
    out[name] = (("lat", "lon"), values.values.astype(np.float32))


def extract(args: argparse.Namespace) -> None:
    analysis_time = parse_time(args.analysis_time)
    init_time, lead_hours, exact_6h = forecast_mapping(analysis_time)
    root = Path(args.forecast_root) / ("exact6h" if exact_6h else "all_steps")
    init_day = init_time.date()
    pl_path = root / "pl" / f"era5_forecast_pl_{'exact6h' if exact_6h else 'all_steps'}_init_{init_day:%Y%m%d}.grib"
    sfc_path = root / "sfc" / f"era5_forecast_sfc_{'exact6h' if exact_6h else 'all_steps'}_init_{init_day:%Y%m%d}.grib"
    for path in [pl_path, sfc_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"era5_forecast_subset_{analysis_time:%Y%m%d%H}"
    output_path = output_dir / f"{stem}.nc"
    summary_path = output_dir / f"{stem}.json"

    summary: dict[str, object] = {
        "analysis_time_utc": format_time(analysis_time),
        "forecast_init_time_utc": format_time(init_time),
        "forecast_lead_hours": lead_hours,
        "forecast_step_hours": lead_hours,
        "exact_6h_subset": exact_6h,
        "source_kind": "era5_short_range_forecast",
        "input_paths": {"pl": str(pl_path), "sfc": str(sfc_path)},
        "variables": {},
    }

    with open_grib(pl_path) as pl_ds, open_grib(sfc_path) as sfc_ds:
        pl = rename_level(select_valid_time(pl_ds, analysis_time))
        sfc = select_valid_time(sfc_ds, analysis_time)

        out = xr.Dataset(
            coords={
                "lat": ("lat", pl.latitude.values.astype(np.float32)),
                "lon": ("lon", pl.longitude.values.astype(np.float32)),
            }
        )
        if "level" in pl.coords:
            out = out.assign_coords(level=("level", pl.level.values.astype(np.int32)))

        for name in ["z", "t", "q", "u", "v"]:
            if name in pl:
                assign_pl_var(out, f"xb_{name}", pl[name])
                out[f"xb_{name}"].attrs["source_variable"] = name
                if name == "q":
                    out[f"xb_{name}"].attrs["q_source"] = "direct_era5_forecast_q"
                summary["variables"][f"xb_{name}"] = stats(out[f"xb_{name}"])  # type: ignore[index]

        sfc_names = {"msl": "mslp", "sp": "sp", "u10": "u10", "v10": "v10", "t2m": "t2m", "d2m": "d2m"}
        for src, dst in sfc_names.items():
            if src in sfc:
                assign_sfc_var(out, f"xb_{dst}", sfc[src])
                summary["variables"][f"xb_{dst}"] = stats(out[f"xb_{dst}"])  # type: ignore[index]

        if "xb_d2m" in out and "xb_sp" in out:
            out["xb_q2m"] = specific_humidity_from_dewpoint(out["xb_d2m"], out["xb_sp"]).astype("float32")
            out["xb_q2m"].attrs["derived_from"] = "d2m,sp"
            out["xb_q2m"].attrs["q_source"] = "derived_from_dewpoint"
            summary["variables"]["xb_q2m"] = stats(out["xb_q2m"])  # type: ignore[index]

        out.attrs.update(
            {
                "analysis_time_utc": format_time(analysis_time),
                "forecast_init_time_utc": format_time(init_time),
                "forecast_lead_hours": lead_hours,
                "forecast_step_hours": lead_hours,
                "exact_6h_subset": int(exact_6h),
                "domain": json.dumps(DOMAIN, sort_keys=True),
                "humidity_source_json": json.dumps(
                    {
                        "xb_q": "direct_era5_forecast_q",
                        "xb_q2m": "derived_from_d2m_sp",
                    },
                    sort_keys=True,
                ),
            }
        )
        out.to_netcdf(output_path)
        out.close()

    summary["output_path"] = str(output_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-time", required=True)
    parser.add_argument("--forecast-root", default="data/raw/era5_gate1_forecast")
    parser.add_argument("--output-dir", default="data/interim/era5_forecast_smoke")
    return parser.parse_args()


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()
