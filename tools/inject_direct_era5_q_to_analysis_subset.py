"""Replace derived xa_q in an ERA5 analysis subset with direct ERA5 q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def open_direct_q(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def parse_humidity_attr(value: object) -> dict[str, str]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def inject(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    direct_path = Path(args.direct_q_grib)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target = np.datetime64(args.analysis_time.replace("Z", ""))
    with xr.open_dataset(input_path) as subset, open_direct_q(direct_path) as direct:
        matches = np.where(direct["time"].values == target)[0]
        if matches.size != 1:
            raise ValueError(f"expected one direct q time match for {args.analysis_time}, found {matches.size}")
        q_direct = direct["q"].isel(time=int(matches[0])).rename(
            {"isobaricInhPa": "level", "latitude": "lat", "longitude": "lon"}
        )
        q_direct = q_direct.assign_coords(level=q_direct["level"].values.astype(np.int32))
        q_direct = q_direct.sel(level=subset["level"].values.astype(np.int32))

        if not np.array_equal(q_direct["level"].values.astype(np.int32), subset["level"].values.astype(np.int32)):
            raise ValueError("level mismatch between direct q and subset")
        if not np.allclose(q_direct["lat"].values.astype(np.float32), subset["lat"].values.astype(np.float32)):
            raise ValueError("lat mismatch between direct q and subset")
        if not np.allclose(q_direct["lon"].values.astype(np.float32), subset["lon"].values.astype(np.float32)):
            raise ValueError("lon mismatch between direct q and subset")

        out = subset.load()
        out["xa_q"] = (("level", "lat", "lon"), q_direct.values.astype(np.float32))
        out["xa_q"].attrs.update(
            {
                "source_variable": "q",
                "q_source": "direct_era5_reanalysis_q",
                "direct_q_grib": str(direct_path),
                "replaced": "derived_from_t_r_pressure_level",
            }
        )
        humidity = parse_humidity_attr(out.attrs.get("humidity_source_json", ""))
        humidity["xa_q"] = "direct_era5_reanalysis_q"
        humidity.setdefault("xa_q2m", "derived_from_d2m_sp")
        out.attrs["humidity_source_json"] = json.dumps(humidity, sort_keys=True)
        out.attrs["direct_q_grib"] = str(direct_path)
        out.attrs["q_label_policy"] = "xa_q is direct ERA5 reanalysis q; xa_q2m remains derived from d2m/sp."
        out.to_netcdf(output_path)

    print(json.dumps({"input": str(input_path), "direct_q_grib": str(direct_path), "output": str(output_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-time", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--direct-q-grib", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    inject(parse_args())


if __name__ == "__main__":
    main()
