"""Build Gate 1 static land/sea and topography features on the 0.25 degree grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def target_axis(start: float, stop: float, resolution: float, *, descending: bool = False) -> np.ndarray:
    count = int(round((stop - start) / resolution)) + 1
    values = np.linspace(start, stop, count, dtype=np.float32)
    if descending:
        values = values[::-1]
    return values


def stats(values: xr.DataArray) -> dict[str, float | int]:
    arr = values.values
    finite = np.isfinite(arr)
    out: dict[str, float | int] = {"finite": int(finite.sum()), "total": int(arr.size)}
    if finite.any():
        out.update(
            {
                "min": float(np.nanmin(arr)),
                "mean": float(np.nanmean(arr)),
                "max": float(np.nanmax(arr)),
            }
        )
    return out


def load_surface_height(source: Path) -> xr.DataArray:
    with xr.open_dataset(source) as ds:
        if "z" in ds:
            height = ds["z"].rename("static_surface_height_m")
            if "latitude" in height.dims:
                height = height.rename({"latitude": "lat"})
            if "longitude" in height.dims:
                height = height.rename({"longitude": "lon"})
            return height.load()

        for name in ["surface_height_m", "elevation_m", "topography_m"]:
            if name in ds:
                height = ds[name].rename("static_surface_height_m")
                rename: dict[str, str] = {}
                if "latitude" in height.dims:
                    rename["latitude"] = "lat"
                if "longitude" in height.dims:
                    rename["longitude"] = "lon"
                if rename:
                    height = height.rename(rename)
                return height.load()

    raise ValueError(f"Cannot find a surface-height variable in {source}")


def interp_regular_latlon(field: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> xr.DataArray:
    src_lat = field["lat"].values.astype(np.float64)
    src_lon = field["lon"].values.astype(np.float64)
    values = field.values.astype(np.float64)

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        values = values[::-1, :]
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        values = values[:, ::-1]

    lon_interp = np.empty((values.shape[0], target_lon.size), dtype=np.float64)
    for row_idx in range(values.shape[0]):
        lon_interp[row_idx, :] = np.interp(target_lon, src_lon, values[row_idx, :])

    out = np.empty((target_lat.size, target_lon.size), dtype=np.float32)
    target_lat64 = target_lat.astype(np.float64)
    for col_idx in range(target_lon.size):
        out[:, col_idx] = np.interp(target_lat64, src_lat, lon_interp[:, col_idx]).astype(np.float32)

    return xr.DataArray(out, coords={"lat": target_lat, "lon": target_lon}, dims=("lat", "lon"), name=field.name)


def build(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_suffix(".json")

    target_lat = target_axis(args.lat_min, args.lat_max, args.resolution, descending=True)
    target_lon = target_axis(args.lon_min, args.lon_max, args.resolution)

    height = load_surface_height(source)
    if "lat" not in height.coords or "lon" not in height.coords:
        raise ValueError("source height field must have lat/lon coordinates")

    height = height.sortby("lat").sortby("lon")
    height_025 = interp_regular_latlon(height, target_lat, target_lon).astype("float32")
    height_025.attrs.update(
        {
            "long_name": "surface height relative to mean sea level",
            "units": "m",
            "source_file": str(source),
            "positive": "up",
        }
    )

    elevation = xr.where(height_025 > 0, height_025, 0.0).astype("float32").rename("static_elevation_m")
    bathymetry = xr.where(height_025 < 0, height_025, 0.0).astype("float32").rename("static_bathymetry_m")
    landmask = xr.where(height_025 > args.land_threshold_m, 1.0, 0.0).astype("float32").rename("static_landmask")
    oceanmask = (1.0 - landmask).astype("float32").rename("static_oceanmask")
    land_sea_class = landmask.astype("int8").rename("static_land_sea_class")

    max_elev = float(elevation.max(skipna=True).values)
    if max_elev > 0:
        topo_norm = (np.log1p(elevation) / np.log1p(max_elev)).astype("float32")
    else:
        topo_norm = (elevation * 0.0).astype("float32")
    topo_norm = topo_norm.rename("static_topography_norm")

    elevation.attrs.update({"long_name": "non-negative surface elevation", "units": "m"})
    bathymetry.attrs.update({"long_name": "non-positive bathymetry over ocean", "units": "m"})
    landmask.attrs.update(
        {
            "long_name": "land mask from surface height",
            "units": "1",
            "threshold_m": float(args.land_threshold_m),
            "class_value": "1=land,0=ocean_or_below_threshold",
        }
    )
    oceanmask.attrs.update({"long_name": "ocean mask complement of static_landmask", "units": "1"})
    land_sea_class.attrs.update({"long_name": "integer land sea class", "class_value": "1=land,0=ocean"})
    topo_norm.attrs.update({"long_name": "log-normalized non-negative topography", "units": "1"})

    ds = xr.Dataset(
        {
            "static_surface_height_m": height_025,
            "static_elevation_m": elevation,
            "static_bathymetry_m": bathymetry,
            "static_landmask": landmask,
            "static_oceanmask": oceanmask,
            "static_land_sea_class": land_sea_class,
            "static_topography_norm": topo_norm,
        },
        coords={"lat": target_lat, "lon": target_lon},
        attrs={
            "role": "Gate 1 static auxiliary predictors for land/sea bias bins and topographic conditioning.",
            "source_file": str(source),
            "grid_resolution_deg": float(args.resolution),
            "domain": json.dumps(
                {
                    "lat_min": float(args.lat_min),
                    "lat_max": float(args.lat_max),
                    "lon_min": float(args.lon_min),
                    "lon_max": float(args.lon_max),
                },
                sort_keys=True,
            ),
            "leakage_policy": "static fields only; no analysis-side atmospheric label fields.",
        },
    )

    encoding = {name: {"zlib": True, "complevel": 2} for name in ds.data_vars}
    ds.to_netcdf(output, encoding=encoding)

    summary = {
        "source": str(source),
        "output": str(output),
        "shape": {"lat": int(ds.sizes["lat"]), "lon": int(ds.sizes["lon"])},
        "variables": {name: stats(ds[name]) for name in ds.data_vars},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ETOPO_2022_v1_r3600x1800_surface.nc")
    parser.add_argument("--output", default="data/interim/static/gate1_static_features_025.nc")
    parser.add_argument("--lat-min", type=float, default=0.0)
    parser.add_argument("--lat-max", type=float, default=60.0)
    parser.add_argument("--lon-min", type=float, default=70.0)
    parser.add_argument("--lon-max", type=float, default=150.0)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--land-threshold-m", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
