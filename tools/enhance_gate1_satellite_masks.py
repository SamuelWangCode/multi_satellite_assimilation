"""Add explicit masks/time/geometry aliases to an existing satellite sample.

This is useful when the raw satellite archive is temporarily unavailable but
the gridded smoke sample already exists. The main builder also emits these
fields directly when raw data are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def add_masks(ds: xr.Dataset, count_name: str) -> None:
    if not count_name.endswith("_obs_count"):
        return
    prefix = count_name[: -len("_obs_count")]
    valid_name = f"{prefix}_valid_mask"
    missing_name = f"{prefix}_missing_mask"
    count = ds[count_name]
    if valid_name not in ds:
        ds[valid_name] = (count > 0).astype("float32")
        ds[valid_name].attrs["definition"] = f"{count_name} > 0"
    if missing_name not in ds:
        ds[missing_name] = (count <= 0).astype("float32")
        ds[missing_name].attrs["definition"] = f"{count_name} == 0"


def constant_like_grid(ds: xr.Dataset, value: float) -> xr.DataArray:
    data = np.full((ds.sizes["lat"], ds.sizes["lon"]), value, dtype=np.float32)
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": ds["lat"], "lon": ds["lon"]})


def enhance(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(input_path) as src:
        ds = src.load()

    for name in list(ds.data_vars):
        add_masks(ds, name)

    metadata = {}
    if "metadata_json" in ds.attrs:
        try:
            metadata = json.loads(ds.attrs["metadata_json"])
        except json.JSONDecodeError:
            metadata = {}
    agri_offset = float(metadata.get("agri_time_offset_min", 0.0))
    clm_offset = float(metadata.get("clm_time_offset_min", 0.0))
    ds["fy4b_agri_time_offset_min"] = constant_like_grid(ds, agri_offset)
    ds["fy4b_agri_time_offset_min"].attrs["source"] = "analysis_time_minus_AGRI_start_time"
    ds["fy4b_clm_time_offset_min"] = constant_like_grid(ds, clm_offset)
    ds["fy4b_clm_time_offset_min"].attrs["source"] = "analysis_time_minus_CLM_start_time"

    for sensor in ["mwts", "mwhs"]:
        old = f"fy3e_{sensor}_geometry0_mean"
        if old in ds:
            raw = f"fy3e_{sensor}_geometry0_raw_mean"
            scan = f"fy3e_{sensor}_scan_angle_raw_mean"
            if raw not in ds:
                ds[raw] = ds[old]
                ds[raw].attrs["source_alias"] = old
            if scan not in ds:
                ds[scan] = ds[old]
                ds[scan].attrs["source_alias"] = old
                ds[scan].attrs["caution"] = "Raw geometry field 0; exact official meaning still under parser validation."

    ds.attrs["mask_fields_added"] = "valid_mask/missing_mask derived from obs_count"
    ds.attrs["time_offset_fields_added"] = "fy4b_agri_time_offset_min, fy4b_clm_time_offset_min"
    ds.attrs["geometry_alias_note"] = "MW geometry aliases are raw-field aliases pending official format validation."
    encoding = {name: {"zlib": True, "complevel": 1} for name in ds.data_vars}
    ds.to_netcdf(output_path, encoding=encoding)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "variable_count": len(ds.data_vars),
        "valid_mask_count": sum(1 for name in ds.data_vars if name.endswith("_valid_mask")),
        "missing_mask_count": sum(1 for name in ds.data_vars if name.endswith("_missing_mask")),
        "time_offset_variables": [name for name in ds.data_vars if "time_offset" in name],
        "geometry_alias_variables": [name for name in ds.data_vars if "geometry" in name or "scan_angle" in name],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/samples_025/gate1_satellite_sample_2024050100.nc")
    parser.add_argument("--output", default="data/processed/samples_025/gate1_satellite_sample_2024050100_enhanced.nc")
    return parser.parse_args()


def main() -> None:
    enhance(parse_args())


if __name__ == "__main__":
    main()
