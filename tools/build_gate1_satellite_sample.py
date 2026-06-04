"""Build a Gate 1 gridded satellite-feature sample for one analysis time.

This script creates the satellite side of sample(t) on the approved 0.25 deg
China-buffer grid. It intentionally does not fabricate xb from ERA5 analysis;
ERA5 forecast/analysis fields are handled separately once forecast files are
available.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pyproj import Proj, Transformer


DOMAIN = {
    "lon_min": 70.0,
    "lon_max": 150.0,
    "lat_min": 0.0,
    "lat_max": 60.0,
    "resolution": 0.25,
}

FY4B_NOM_N = 2748
FY4B_SUB_LON = 105.0
FY4B_CFAC_4KM = 10233137.0
FY4B_LFAC_4KM = 10233137.0
GEOS_HEIGHT_M = 35785863.0

MW_RECORDS = {
    "mwts": {"record_len": 202, "value_offset": 108, "channels": 17, "encoding": "int24_in_i32"},
    "mwhs": {"record_len": 184, "value_offset": 107, "channels": 15, "encoding": "u16_stride4"},
}

SENSOR_IDS = {
    "fy4b_agri": 401.0,
    "fy4b_clm": 402.0,
    "fy3e_mwts": 301.0,
    "fy3e_mwhs": 302.0,
}

SOURCE_IDS = {
    "fy4b_agri": 1401.0,
    "fy4b_clm": 1402.0,
    "fy3e_mwts": 1301.0,
    "fy3e_mwhs": 1302.0,
}


def parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def grid_coords() -> tuple[np.ndarray, np.ndarray]:
    res = DOMAIN["resolution"]
    lons = np.arange(DOMAIN["lon_min"], DOMAIN["lon_max"] + 0.5 * res, res, dtype=np.float32)
    lats = np.arange(DOMAIN["lat_max"], DOMAIN["lat_min"] - 0.5 * res, -res, dtype=np.float32)
    return lats, lons


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def row_for_analysis_time(path: Path, analysis_time: datetime) -> dict[str, str]:
    target = format_time(analysis_time)
    for row in load_csv_rows(path):
        if row["analysis_time_utc"] == target:
            return row
    raise ValueError(f"analysis time {target} not found in {path}")


def rows_in_window(path: Path, analysis_time: datetime, window_minutes: float) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in load_csv_rows(path):
        dt = parse_time(row["start_time_utc"])
        offset_min = (dt - analysis_time).total_seconds() / 60.0
        if abs(offset_min) <= window_minutes:
            row = dict(row)
            row["time_offset_min"] = f"{offset_min:.3f}"
            out.append(row)
    return out


def fy4b_grid_index(cache_dir: Path) -> tuple[np.ndarray, tuple[int, int]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "fy4b_4km_grid_index_70_150E_0_60N_025deg.npz"
    lats, lons = grid_coords()
    shape = (len(lats), len(lons))
    if cache_path.exists():
        data = np.load(cache_path)
        return data["grid_index"].astype(np.int32), shape

    n = FY4B_NOM_N
    coff = (n - 1) / 2.0
    loff = (n - 1) / 2.0
    cols = np.arange(n, dtype=np.float64)
    rows = np.arange(n, dtype=np.float64)
    x_rad = (cols - coff) * (2.0**16) / FY4B_CFAC_4KM * np.pi / 180.0
    y_rad = -(rows - loff) * (2.0**16) / FY4B_LFAC_4KM * np.pi / 180.0
    xx, yy = np.meshgrid(GEOS_HEIGHT_M * x_rad, GEOS_HEIGHT_M * y_rad)

    geos = Proj(
        proj="geos",
        h=GEOS_HEIGHT_M,
        lon_0=FY4B_SUB_LON,
        a=6378137.0,
        b=6356752.31414,
        sweep="x",
    )
    transformer = Transformer.from_proj(geos, "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    del xx, yy

    res = DOMAIN["resolution"]
    jj_float = np.rint((lon - DOMAIN["lon_min"]) / res)
    ii_float = np.rint((DOMAIN["lat_max"] - lat) / res)
    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= DOMAIN["lat_min"])
        & (lat <= DOMAIN["lat_max"])
        & (lon >= DOMAIN["lon_min"])
        & (lon <= DOMAIN["lon_max"])
        & (ii_float >= 0)
        & (ii_float < shape[0])
        & (jj_float >= 0)
        & (jj_float < shape[1])
    )
    ii = np.zeros_like(ii_float, dtype=np.int32)
    jj = np.zeros_like(jj_float, dtype=np.int32)
    ii[valid] = ii_float[valid].astype(np.int32)
    jj[valid] = jj_float[valid].astype(np.int32)
    grid_index = np.full((n, n), -1, dtype=np.int32)
    grid_index[valid] = ii[valid] * shape[1] + jj[valid]
    np.savez_compressed(cache_path, grid_index=grid_index)
    return grid_index, shape


def stats_from_values(values: np.ndarray, grid_index: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    flat_index = grid_index.ravel()
    flat_values = values.ravel()
    valid = (flat_index >= 0) & np.isfinite(flat_values)
    ngrid = shape[0] * shape[1]
    idx = flat_index[valid]
    val = flat_values[valid].astype(np.float64)

    count = np.bincount(idx, minlength=ngrid).astype(np.float32)
    total = np.bincount(idx, weights=val, minlength=ngrid)
    total2 = np.bincount(idx, weights=val * val, minlength=ngrid)
    mean = np.full(ngrid, np.nan, dtype=np.float32)
    std = np.full(ngrid, np.nan, dtype=np.float32)
    good = count > 0
    mean[good] = (total[good] / count[good]).astype(np.float32)
    variance = np.maximum(total2[good] / count[good] - mean[good].astype(np.float64) ** 2, 0.0)
    std[good] = np.sqrt(variance).astype(np.float32)

    vmin = np.full(ngrid, np.inf, dtype=np.float32)
    vmax = np.full(ngrid, -np.inf, dtype=np.float32)
    np.minimum.at(vmin, idx, val.astype(np.float32))
    np.maximum.at(vmax, idx, val.astype(np.float32))
    vmin[~good] = np.nan
    vmax[~good] = np.nan

    return {
        "mean": mean.reshape(shape),
        "std": std.reshape(shape),
        "min": vmin.reshape(shape),
        "max": vmax.reshape(shape),
        "count": count.reshape(shape),
    }


def stats_from_points(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    lats, lons = grid_coords()
    res = DOMAIN["resolution"]
    ii_float = np.rint((DOMAIN["lat_max"] - lat) / res)
    jj_float = np.rint((lon - DOMAIN["lon_min"]) / res)
    valid = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= DOMAIN["lat_min"])
        & (lat <= DOMAIN["lat_max"])
        & (lon >= DOMAIN["lon_min"])
        & (lon <= DOMAIN["lon_max"])
        & (ii_float >= 0)
        & (ii_float < len(lats))
        & (jj_float >= 0)
        & (jj_float < len(lons))
        & np.isfinite(values)
    )
    ii = np.zeros_like(ii_float, dtype=np.int32)
    jj = np.zeros_like(jj_float, dtype=np.int32)
    ii[valid] = ii_float[valid].astype(np.int32)
    jj[valid] = jj_float[valid].astype(np.int32)
    grid_index = ii * shape[1] + jj
    return stats_from_flat_points(grid_index[valid], values[valid], shape)


def stats_from_flat_points(idx: np.ndarray, values: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    ngrid = shape[0] * shape[1]
    idx = idx.astype(np.int64)
    val = values.astype(np.float64)
    count = np.bincount(idx, minlength=ngrid).astype(np.float32)
    total = np.bincount(idx, weights=val, minlength=ngrid)
    total2 = np.bincount(idx, weights=val * val, minlength=ngrid)
    mean = np.full(ngrid, np.nan, dtype=np.float32)
    std = np.full(ngrid, np.nan, dtype=np.float32)
    good = count > 0
    mean[good] = (total[good] / count[good]).astype(np.float32)
    variance = np.maximum(total2[good] / count[good] - mean[good].astype(np.float64) ** 2, 0.0)
    std[good] = np.sqrt(variance).astype(np.float32)
    vmin = np.full(ngrid, np.inf, dtype=np.float32)
    vmax = np.full(ngrid, -np.inf, dtype=np.float32)
    if idx.size:
        np.minimum.at(vmin, idx, val.astype(np.float32))
        np.maximum.at(vmax, idx, val.astype(np.float32))
    vmin[~good] = np.nan
    vmax[~good] = np.nan
    return {
        "mean": mean.reshape(shape),
        "std": std.reshape(shape),
        "min": vmin.reshape(shape),
        "max": vmax.reshape(shape),
        "count": count.reshape(shape),
    }


def mask_from_count(count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = (count > 0).astype(np.float32)
    missing = (count <= 0).astype(np.float32)
    return valid, missing


def availability_class_from_count(count: np.ndarray) -> np.ndarray:
    availability = np.full(count.shape, 1, dtype=np.float32)
    availability[count > 0] = 0.0
    return availability


def constant_grid(value: float, shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def add_identity_features(
    features: dict[str, np.ndarray],
    prefix: str,
    shape: tuple[int, int],
    sensor_key: str,
    channel_id: int | None = None,
) -> None:
    features[f"{prefix}_sensor_id"] = constant_grid(SENSOR_IDS[sensor_key], shape)
    features[f"{prefix}_source_id"] = constant_grid(SOURCE_IDS[sensor_key], shape)
    if channel_id is not None:
        features[f"{prefix}_channel_id"] = constant_grid(float(channel_id), shape)


def calibrated_agri_channel(path: Path, channel: int) -> np.ndarray:
    name = f"NOMChannel{channel:02d}"
    cal_name = f"CALChannel{channel:02d}"
    with Dataset(path, "r") as ds:
        var = ds.groups["Data"].variables[name]
        dn = np.array(var[:])
        cal = np.array(ds.groups["Calibration"].variables[cal_name][:], dtype=np.float32)
        fill = getattr(var, "FillValue", 65535)
    out = np.full(dn.shape, np.nan, dtype=np.float32)
    valid = (dn >= 0) & (dn < len(cal))
    out[valid] = cal[dn[valid]]
    out[dn == fill] = np.nan
    return out


def read_clm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path, "r") as ds:
        clm = np.array(ds.variables["CLM"][:]).astype(np.int16)
        dqf = np.array(ds.variables["DQF"][:]).astype(np.int16)
    clm[clm < 0] += 256
    dqf[dqf < 0] += 256
    return clm, dqf


def decode_uint24_i32(values: np.ndarray) -> np.ndarray:
    raw = (values >> 8).astype(np.int32)
    out = raw.astype(np.float32) / 100.0
    out[(raw >= 999000) | (raw <= -999000)] = np.nan
    out[(out < 50.0) | (out > 400.0)] = np.nan
    return out


def decode_u16_stride4(records: np.ndarray, offset: int, channel: int) -> np.ndarray:
    start = offset + channel * 4
    chunk = np.ascontiguousarray(records[:, start : start + 2])
    raw = chunk.view("<u2").reshape(-1)
    out = raw.astype(np.float32) / 100.0
    out[(out < 50.0) | (out > 400.0)] = np.nan
    return out


def int32_at(records: np.ndarray, offset: int) -> np.ndarray:
    chunk = np.ascontiguousarray(records[:, offset : offset + 4])
    return chunk.view("<i4").reshape(-1)


def read_mw_file(path: Path, sensor: str, analysis_time: datetime) -> dict[str, np.ndarray]:
    cfg = MW_RECORDS[sensor]
    data = path.read_bytes()
    rec_len = cfg["record_len"]
    if len(data) % rec_len != 0:
        raise ValueError(f"{path} size {len(data)} is not divisible by record length {rec_len}")
    records = np.frombuffer(data, dtype=np.uint8).reshape(len(data) // rec_len, rec_len)
    magic = records[:, 0:5].tobytes()[0:5]
    if magic != b"FY-3E":
        raise ValueError(f"{path} does not start with FY-3E record magic")

    year = int32_at(records, 28)
    month = int32_at(records, 32)
    day = int32_at(records, 36)
    hour = int32_at(records, 40)
    minute = int32_at(records, 44)
    second = int32_at(records, 48)
    lat = int32_at(records, 52).astype(np.float32) / 100.0
    lon = int32_at(records, 56).astype(np.float32) / 100.0
    lon = np.where(lon < 0, lon + 360.0, lon)
    geom = np.vstack(
        [
            int32_at(records, 68).astype(np.float32) / 100.0,
            int32_at(records, 72).astype(np.float32) / 100.0,
            int32_at(records, 76).astype(np.float32) / 100.0,
            int32_at(records, 80).astype(np.float32) / 100.0,
        ]
    ).T
    values = []
    for channel in range(cfg["channels"]):
        if cfg["encoding"] == "int24_in_i32":
            values.append(decode_uint24_i32(int32_at(records, cfg["value_offset"] + channel * 4)))
        elif cfg["encoding"] == "u16_stride4":
            values.append(decode_u16_stride4(records, cfg["value_offset"], channel))
        else:
            raise ValueError(f"unsupported microwave encoding: {cfg['encoding']}")
    value_arr = np.vstack(values).T

    offsets = np.empty(records.shape[0], dtype=np.float32)
    for i in range(records.shape[0]):
        dt = datetime(
            int(year[i]),
            int(month[i]),
            int(day[i]),
            int(hour[i]),
            int(minute[i]),
            int(second[i]),
            tzinfo=timezone.utc,
        )
        offsets[i] = (dt - analysis_time).total_seconds() / 60.0
    return {"lat": lat, "lon": lon, "geom": geom, "values": value_arr, "time_offset_min": offsets}


def add_grid_var(ds: Dataset, name: str, values: np.ndarray, units: str = "", long_name: str = "") -> None:
    var = ds.createVariable(name, "f4", ("lat", "lon"), zlib=True, complevel=1, fill_value=np.nan)
    var[:, :] = values.astype(np.float32)
    if units:
        var.units = units
    if long_name:
        var.long_name = long_name


def write_sample(
    output_path: Path,
    analysis_time: datetime,
    features: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lats, lons = grid_coords()
    with Dataset(output_path, "w") as ds:
        ds.createDimension("lat", len(lats))
        ds.createDimension("lon", len(lons))
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lat_var[:] = lats
        lon_var[:] = lons
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        ds.analysis_time_utc = format_time(analysis_time)
        ds.domain = json.dumps(DOMAIN, sort_keys=True)
        ds.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        for name, values in sorted(features.items()):
            add_grid_var(ds, name, values)


def plot_diagnostics(features: dict[str, np.ndarray], fig_path: Path, analysis_time: datetime) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("fy4b_agri_ch09_value_mean", "AGRI ch09"),
        ("fy4b_agri_ch13_value_mean", "AGRI ch13"),
        ("fy4b_clm_cloud_fraction", "CLM cloud fraction"),
        ("fy3e_mwts_ch01_obs_count", "MWTS ch01 count"),
        ("fy3e_mwhs_ch01_obs_count", "MWHS ch01 count"),
        ("fy3e_mwts_ch03_value_mean", "MWTS ch03 mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, (name, title) in zip(axes.ravel(), panels):
        data = features.get(name)
        if data is None:
            ax.set_title(title + " missing")
            ax.axis("off")
            continue
        im = ax.imshow(
            data,
            origin="upper",
            extent=[DOMAIN["lon_min"], DOMAIN["lon_max"], DOMAIN["lat_min"], DOMAIN["lat_max"]],
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"Gate 1 satellite sample {format_time(analysis_time)}")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def add_count_masks(features: dict[str, np.ndarray], prefix: str, count: np.ndarray) -> None:
    valid, missing = mask_from_count(count)
    features[f"{prefix}_valid_mask"] = valid
    features[f"{prefix}_missing_mask"] = missing
    features[f"{prefix}_availability_class"] = availability_class_from_count(count)


def build_sample(args: argparse.Namespace) -> None:
    analysis_time = parse_time(args.analysis_time)
    indexes = Path(args.index_dir)
    grid_index, shape = fy4b_grid_index(Path(args.cache_dir))

    match = row_for_analysis_time(indexes / "fy4b_agri_clm_matches.csv", analysis_time)
    agri_path = Path(match["agri_path"])
    clm_path = Path(match["clm_path"])
    if not agri_path.exists() or not clm_path.exists():
        raise FileNotFoundError("AGRI/CLM match paths are missing for the requested analysis time")

    features: dict[str, np.ndarray] = {}
    agri_offset_min = float(match.get("agri_offset_min") or 0.0)
    clm_offset_min = float(match.get("clm_offset_min") or 0.0)
    metadata: dict[str, object] = {
        "analysis_time_utc": format_time(analysis_time),
        "agri_path": str(agri_path),
        "agri_time_utc": match.get("agri_time_utc", ""),
        "agri_time_offset_min": agri_offset_min,
        "clm_path": str(clm_path),
        "clm_time_utc": match.get("clm_time_utc", ""),
        "clm_time_offset_min": clm_offset_min,
        "mwts_paths": [],
        "mwhs_paths": [],
        "availability_class_encoding": {
            "0": "valid",
            "1": "no_observation",
            "2": "rejected_by_sensor_qc",
            "3": "rejected_by_project_qc",
            "4": "unknown_or_not_available",
        },
        "availability_class_current_scope": "0/1 only. Sensor/project QC rejection classes are reserved until reliable QC fields are decoded.",
        "sensor_id_encoding": SENSOR_IDS,
        "source_id_encoding": SOURCE_IDS,
        "note": "Satellite-only Gate 1 sample. xb/xa/dx are added by ERA5 processing once forecast is available.",
    }
    add_identity_features(features, "fy4b_agri", shape, "fy4b_agri")
    add_identity_features(features, "fy4b_clm", shape, "fy4b_clm")
    add_identity_features(features, "fy3e_mwts", shape, "fy3e_mwts")
    add_identity_features(features, "fy3e_mwhs", shape, "fy3e_mwhs")
    features["fy4b_agri_time_offset_min"] = constant_grid(agri_offset_min, shape)
    features["fy4b_clm_time_offset_min"] = constant_grid(clm_offset_min, shape)

    for channel in args.agri_channels:
        values = calibrated_agri_channel(agri_path, channel)
        stats = stats_from_values(values, grid_index, shape)
        prefix = f"fy4b_agri_ch{channel:02d}"
        features[f"{prefix}_value_mean"] = stats["mean"]
        features[f"{prefix}_value_std"] = stats["std"]
        features[f"{prefix}_value_min"] = stats["min"]
        features[f"{prefix}_value_max"] = stats["max"]
        features[f"{prefix}_obs_count"] = stats["count"]
        add_identity_features(features, prefix, shape, "fy4b_agri", channel)
        add_count_masks(features, prefix, stats["count"])

    clm, dqf = read_clm(clm_path)
    valid_clm = np.isin(clm, [0, 1, 2, 3]).astype(np.float32)
    cloud = np.isin(clm, [0, 1]).astype(np.float32)
    clear = np.isin(clm, [2, 3]).astype(np.float32)
    valid_stats = stats_from_values(valid_clm, grid_index, shape)
    cloud_stats = stats_from_values(np.where(valid_clm > 0, cloud, np.nan), grid_index, shape)
    clear_stats = stats_from_values(np.where(valid_clm > 0, clear, np.nan), grid_index, shape)
    dqf_stats = stats_from_values(dqf.astype(np.float32), grid_index, shape)
    features["fy4b_clm_obs_count"] = valid_stats["count"]
    features["fy4b_clm_channel_id"] = constant_grid(0.0, shape)
    add_count_masks(features, "fy4b_clm", valid_stats["count"])
    features["fy4b_clm_cloud_fraction"] = cloud_stats["mean"]
    features["fy4b_clm_clear_fraction"] = clear_stats["mean"]
    features["fy4b_clm_dqf_mean"] = dqf_stats["mean"]

    for sensor in ["mwts", "mwhs"]:
        rows = rows_in_window(indexes / f"fy3e_{sensor}_index.csv", analysis_time, args.mw_window_minutes)
        metadata[f"{sensor}_paths"] = [row["path"] for row in rows]
        if not rows:
            continue
        chunks = []
        for row in rows:
            chunks.append(read_mw_file(Path(row["path"]), sensor, analysis_time))
        lat = np.concatenate([item["lat"] for item in chunks])
        lon = np.concatenate([item["lon"] for item in chunks])
        values = np.concatenate([item["values"] for item in chunks], axis=0)
        time_offset = np.concatenate([item["time_offset_min"] for item in chunks])
        geom = np.concatenate([item["geom"] for item in chunks], axis=0)
        shape2 = shape
        for channel in range(values.shape[1]):
            stats = stats_from_points(lat, lon, values[:, channel], shape2)
            prefix = f"fy3e_{sensor}_ch{channel + 1:02d}"
            features[f"{prefix}_value_mean"] = stats["mean"]
            features[f"{prefix}_value_std"] = stats["std"]
            features[f"{prefix}_value_min"] = stats["min"]
            features[f"{prefix}_value_max"] = stats["max"]
            features[f"{prefix}_obs_count"] = stats["count"]
            add_identity_features(features, prefix, shape2, f"fy3e_{sensor}", channel + 1)
            add_count_masks(features, prefix, stats["count"])
        features[f"fy3e_{sensor}_time_offset_min_mean"] = stats_from_points(lat, lon, time_offset, shape2)["mean"]
        for geom_index in range(geom.shape[1]):
            geom_stats = stats_from_points(lat, lon, geom[:, geom_index], shape2)
            features[f"fy3e_{sensor}_geometry{geom_index}_raw_mean"] = geom_stats["mean"]
            if geom_index == 0:
                features[f"fy3e_{sensor}_geometry0_mean"] = geom_stats["mean"]
                features[f"fy3e_{sensor}_scan_angle_raw_mean"] = geom_stats["mean"]

    stamp = analysis_time.strftime("%Y%m%d%H")
    output_path = Path(args.output_dir) / f"gate1_satellite_sample_{stamp}.nc"
    fig_path = Path(args.figure_dir) / f"gate1_satellite_sample_{stamp}.png"
    write_sample(output_path, analysis_time, features, metadata)
    plot_diagnostics(features, fig_path, analysis_time)

    summary = {
        "analysis_time_utc": format_time(analysis_time),
        "output_path": str(output_path),
        "figure_path": str(fig_path),
        "feature_count": len(features),
        "features": sorted(features),
        "metadata": metadata,
    }
    summary_path = Path(args.output_dir) / f"gate1_satellite_sample_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-time", default="2024-05-01T00:00:00Z")
    parser.add_argument("--index-dir", default="data/interim/indexes")
    parser.add_argument("--output-dir", default="data/processed/samples_025")
    parser.add_argument("--figure-dir", default="data/diagnostics/figures/gate1_samples")
    parser.add_argument("--cache-dir", default="data/interim/cache")
    parser.add_argument("--agri-channels", type=int, nargs="+", default=[9, 10, 11, 12, 13, 14, 15])
    parser.add_argument("--mw-window-minutes", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    build_sample(parse_args())


if __name__ == "__main__":
    main()
