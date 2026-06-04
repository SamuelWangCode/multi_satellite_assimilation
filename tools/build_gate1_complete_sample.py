"""Merge satellite features, ERA5 reanalysis label xa, forecast xb, and dx labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


UPPER_AIR = ["z", "t", "q", "u", "v"]
SURFACE = ["mslp", "sp", "t2m", "q2m", "u10", "v10", "d2m"]
CANONICAL_COORDS = {"lat", "lon", "level"}


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


def assert_same_grid(a: xr.Dataset, b: xr.Dataset, name: str) -> None:
    if not np.array_equal(a["lat"].values, b["lat"].values):
        raise ValueError(f"lat mismatch with {name}")
    if not np.array_equal(a["lon"].values, b["lon"].values):
        raise ValueError(f"lon mismatch with {name}")


def drop_noncanonical_coords(ds: xr.Dataset) -> xr.Dataset:
    drop = [name for name in ds.coords if name not in CANONICAL_COORDS]
    if drop:
        ds = ds.drop_vars(drop, errors="ignore")
    extra_dims = set(ds.dims) - CANONICAL_COORDS
    if extra_dims:
        raise ValueError(f"non-canonical dimensions remain: {sorted(extra_dims)}")
    return ds


def parse_humidity_attr(value: object) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def build(args: argparse.Namespace) -> None:
    sat_path = Path(args.satellite_sample)
    xa_path = Path(args.analysis_subset)
    xb_path = Path(args.forecast_subset)
    static_path = Path(args.static_features) if args.static_features else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"gate1_complete_sample_{args.stamp}.nc"
    summary_path = output_dir / f"gate1_complete_sample_{args.stamp}.json"

    with xr.open_dataset(sat_path) as sat, xr.open_dataset(xa_path) as xa, xr.open_dataset(xb_path) as xb:
        assert_same_grid(sat, xa, "xa")
        assert_same_grid(sat, xb, "xb")
        if "level" in xa.coords and "level" in xb.coords and not np.array_equal(xa["level"].values, xb["level"].values):
            raise ValueError("pressure level mismatch between xa and xb")

        sat = drop_noncanonical_coords(sat)
        xa = drop_noncanonical_coords(xa)
        xb = drop_noncanonical_coords(xb)
        dx = xr.Dataset(coords={name: xb.coords[name] for name in xb.coords if name in CANONICAL_COORDS})
        summary: dict[str, object] = {
            "satellite_sample": str(sat_path),
            "analysis_subset": str(xa_path),
            "forecast_subset": str(xb_path),
            "output_path": str(output_path),
            "variables": {},
            "dx_variables": [],
            "static_features": str(static_path) if static_path else None,
        }

        for name in UPPER_AIR + SURFACE:
            xa_name = f"xa_{name}"
            xb_name = f"xb_{name}"
            if xa_name in xa and xb_name in xb:
                dx_name = f"dx_{name}"
                dx[dx_name] = (xa[xa_name] - xb[xb_name]).astype("float32")
                dx[dx_name].attrs.update(
                    {
                        "formula": f"{xa_name} - {xb_name}",
                        "role": "analysis_increment_label",
                    }
                )
                summary["dx_variables"].append(dx_name)  # type: ignore[union-attr]
                summary["variables"][dx_name] = stats(dx[dx_name])  # type: ignore[index]

        merge_parts = [sat, xa, xb, dx]
        static_count = 0
        if static_path and static_path.exists():
            with xr.open_dataset(static_path) as static:
                assert_same_grid(sat, static, "static_features")
                static = drop_noncanonical_coords(static).load()
                static_count = len(static.data_vars)
                merge_parts.append(static)
        elif static_path:
            raise FileNotFoundError(f"static feature file not found: {static_path}")

        merged = xr.merge(merge_parts, compat="override", combine_attrs="override")
        for key in [
            "analysis_time_utc",
            "forecast_init_time_utc",
            "forecast_lead_hours",
            "forecast_step_hours",
            "exact_6h_subset",
        ]:
            if key in xb.attrs:
                merged.attrs[key] = xb.attrs[key]
        humidity = {}
        humidity.update(parse_humidity_attr(xa.attrs.get("humidity_source_json", "")))
        humidity.update(parse_humidity_attr(xb.attrs.get("humidity_source_json", "")))
        humidity.setdefault("xa_q", "derived_from_t_r_pressure_level")
        humidity.setdefault("xb_q", "direct_era5_forecast_q")
        humidity.setdefault("xa_q2m", "derived_from_d2m_sp")
        humidity.setdefault("xb_q2m", "derived_from_d2m_sp")
        merged.attrs["humidity_source_json"] = json.dumps(humidity, sort_keys=True)
        merged.attrs["canonical_coordinates"] = "lat,lon,level"
        if static_path:
            merged.attrs["static_features"] = str(static_path)
        merged.attrs["xa_role"] = "ERA5 reanalysis training label at valid time."
        merged.attrs["xb_role"] = "ERA5 short-range forecast background valid at the same time."
        merged.attrs["sample_role"] = "Gate 1 complete smoke sample with satellite, xb, xa, and dx."
        merged.attrs["scientific_warning"] = "Single smoke sample only; not a trained result."

        encoding = {name: {"zlib": True, "complevel": 2} for name in merged.data_vars}
        merged.to_netcdf(output_path, encoding=encoding)
        summary["feature_count_satellite"] = len(sat.data_vars)
        summary["feature_count_static"] = static_count
        summary["xa_count"] = len(xa.data_vars)
        summary["xb_count"] = len(xb.data_vars)
        summary["merged_variable_count"] = len(merged.data_vars)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="2024050100")
    parser.add_argument("--satellite-sample", default="data/processed/samples_025/gate1_satellite_sample_2024050100.nc")
    parser.add_argument("--analysis-subset", default="data/interim/era5_analysis_smoke/era5_analysis_subset_2024050100.nc")
    parser.add_argument("--forecast-subset", default="data/interim/era5_forecast_smoke/era5_forecast_subset_2024050100.nc")
    parser.add_argument("--static-features", default="data/interim/static/gate1_static_features_025.nc")
    parser.add_argument("--output-dir", default="data/processed/gate1_complete_samples")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
