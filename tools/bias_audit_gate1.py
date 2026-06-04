"""Gate 1 bias audit for satellite-feature samples.

This script is intentionally lightweight and has two modes:

1) Index-level audit (always available once Gate 1 indexes exist):
   - Coverage / missingness proxy from the match tables
   - Time offset distributions by hour and forecast lead

2) Sample-level audit (optional, requires produced samples_025 zarr stores):
   - Missing rates and feature distributions grouped by
     channel × scan_angle × solar_elev × cloud × hour × lead

The goal is to prevent the model from learning instrument/diurnal/air-mass
biases (AGRI diurnal bias, scan-angle bias, air-mass bias) as real analysis
increments.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_mvp.yaml")
    parser.add_argument("--index-dir", default="data/interim/indexes")
    parser.add_argument("--samples", default="", help="Zarr store path or directory containing *.zarr samples.")
    parser.add_argument("--output-dir", default="data/diagnostics/bias_audit")
    parser.add_argument("--max-zarr", type=int, default=12, help="Max number of zarr stores to audit (sample-level).")
    parser.add_argument("--max-times", type=int, default=128, help="Max time steps per zarr store (sample-level).")
    parser.add_argument("--grid-sample-size", type=int, default=4096, help="Grid point samples per time.")
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    # Expect ISO like: 2024-03-05T00:00:00Z
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class LeadMapping:
    by_hour: dict[int, int]

    def lead_hours(self, hour_utc: int) -> int:
        if hour_utc not in self.by_hour:
            raise KeyError(f"missing lead mapping for hour={hour_utc}")
        return int(self.by_hour[hour_utc])


def load_lead_mapping(config_path: Path) -> LeadMapping:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mapping: dict[int, int] = {}
    vt = data.get("background", {}).get("valid_time_mapping", {})
    for hour_str, item in vt.items():
        hour = int(hour_str)
        mapping[hour] = int(item["lead_hours"])
    if not mapping:
        # Fallback to the locked Gate 1 mapping.
        mapping = {0: 6, 6: 12, 12: 6, 18: 12}
    return LeadMapping(by_hour=mapping)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_md(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def q(series: pd.Series, quantiles: list[float]) -> dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {f"p{int(p*100):02d}": math.nan for p in quantiles}
    values = series.quantile(quantiles, interpolation="linear")
    return {f"p{int(p*100):02d}": float(values.loc[p]) for p in quantiles}


def audit_indexes(index_dir: Path, lead_mapping: LeadMapping, output_dir: Path) -> None:
    agri_match = index_dir / "fy4b_agri_clm_matches.csv"
    mw_match = index_dir / "fy3e_mwts_mwhs_matches.csv"

    outputs: list[str] = []
    outputs.append("# Gate 1 bias audit（index-level）")
    outputs.append("")
    outputs.append(f"- index_dir: `{index_dir}`")
    outputs.append("")

    if agri_match.exists():
        df = pd.read_csv(agri_match)
        df["analysis_time_utc"] = df["analysis_time_utc"].apply(parse_utc)
        df["hour_utc"] = df["analysis_time_utc"].dt.hour
        df["lead_hours"] = df["hour_utc"].map(lambda h: lead_mapping.lead_hours(int(h)))

        group_cols = ["hour_utc", "lead_hours"]
        grouped = df.groupby(group_cols, dropna=False)

        rows: list[dict[str, Any]] = []
        for (hour, lead), g in grouped:
            rows.append(
                {
                    "hour_utc": int(hour),
                    "lead_hours": int(lead),
                    "n_times": int(len(g)),
                    "agri_available_rate": float(g["agri_available"].mean()),
                    "clm_available_rate": float(g["clm_available"].mean()),
                    "both_available_rate": float(g["both_available"].mean()),
                    "agri_offset_p05": q(g["agri_offset_min"], [0.05]).get("p05"),
                    "agri_offset_p50": q(g["agri_offset_min"], [0.50]).get("p50"),
                    "agri_offset_p95": q(g["agri_offset_min"], [0.95]).get("p95"),
                    "clm_offset_p05": q(g["clm_offset_min"], [0.05]).get("p05"),
                    "clm_offset_p50": q(g["clm_offset_min"], [0.50]).get("p50"),
                    "clm_offset_p95": q(g["clm_offset_min"], [0.95]).get("p95"),
                }
            )

        out_csv = output_dir / "index_level_fy4b_agri_clm_summary.csv"
        pd.DataFrame(rows).sort_values(group_cols).to_csv(out_csv, index=False)

        outputs.append("## FY-4B AGRI/CLM match summary")
        outputs.append(f"- output: `{out_csv}`")
        outputs.append("- focus: availability + time_offset_min distributions by hour×lead")
        outputs.append("")
    else:
        outputs.append("## FY-4B AGRI/CLM match summary")
        outputs.append(f"- missing: `{agri_match}`")
        outputs.append("")

    if mw_match.exists():
        df = pd.read_csv(mw_match)
        df["analysis_time_utc"] = df["analysis_time_utc"].apply(parse_utc)
        df["hour_utc"] = df["analysis_time_utc"].dt.hour
        df["lead_hours"] = df["hour_utc"].map(lambda h: lead_mapping.lead_hours(int(h)))

        group_cols = ["hour_utc", "lead_hours"]
        grouped = df.groupby(group_cols, dropna=False)
        rows = []
        for (hour, lead), g in grouped:
            rows.append(
                {
                    "hour_utc": int(hour),
                    "lead_hours": int(lead),
                    "n_times": int(len(g)),
                    "mwts_primary_count_p50": q(g["mwts_primary_count_pm90"], [0.50]).get("p50"),
                    "mwts_primary_count_p95": q(g["mwts_primary_count_pm90"], [0.95]).get("p95"),
                    "mwhs_primary_count_p50": q(g["mwhs_primary_count_pm90"], [0.50]).get("p50"),
                    "mwhs_primary_count_p95": q(g["mwhs_primary_count_pm90"], [0.95]).get("p95"),
                    "mwts_nearest_offset_p50": q(g["mwts_nearest_offset_min_pm90"], [0.50]).get("p50"),
                    "mwhs_nearest_offset_p50": q(g["mwhs_nearest_offset_min_pm90"], [0.50]).get("p50"),
                }
            )

        out_csv = output_dir / "index_level_fy3e_mwts_mwhs_summary.csv"
        pd.DataFrame(rows).sort_values(group_cols).to_csv(out_csv, index=False)

        outputs.append("## FY-3E MWTS/MWHS match summary")
        outputs.append(f"- output: `{out_csv}`")
        outputs.append("- focus: coverage proxy (counts) + nearest time offsets by hour×lead")
        outputs.append("")
    else:
        outputs.append("## FY-3E MWTS/MWHS match summary")
        outputs.append(f"- missing: `{mw_match}`")
        outputs.append("")

    md_path = output_dir / "index_level_bias_audit.md"
    write_md(md_path, outputs)


def discover_zarr_paths(samples: str, max_zarr: int) -> list[Path]:
    if not samples:
        return []
    root = Path(samples)
    if root.is_file() and root.name.endswith(".zarr"):
        return [root]
    if root.is_dir():
        paths = sorted(root.rglob("*.zarr"))
        return paths[: max_zarr if max_zarr > 0 else None]
    return []


def open_zarr_dataset(path: Path):
    import xarray as xr

    try:
        return xr.open_zarr(path, consolidated=False)
    except Exception as exc:
        raise RuntimeError(f"failed to open zarr: {path}") from exc


def sample_grid_indices(rng: np.random.Generator, n_lat: int, n_lon: int, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = n_lat * n_lon
    if total <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    k = min(int(sample_size), int(total))
    flat = rng.choice(total, size=k, replace=False)
    lat_idx = flat // n_lon
    lon_idx = flat % n_lon
    return lat_idx, lon_idx


def bin_values(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    # returns bin index [0..len(bins)-2], -1 for NaN/outside
    out = np.full(values.shape, -1, dtype=int)
    mask = np.isfinite(values)
    out[mask] = np.digitize(values[mask], bins) - 1
    out[(out < 0) | (out >= len(bins) - 1)] = -1
    return out


def cloud_class_from_fraction(frac: np.ndarray) -> np.ndarray:
    # 0: clear, 1: mixed, 2: cloudy, -1: unknown
    out = np.full(frac.shape, -1, dtype=int)
    mask = np.isfinite(frac)
    out[mask & (frac < 0.2)] = 0
    out[mask & (frac >= 0.2) & (frac <= 0.8)] = 1
    out[mask & (frac > 0.8)] = 2
    return out


VAR_RE = re.compile(r"^sat\.fy4b_agri\.ch(?P<ch>\d{2})\.(?P<field>value_nearest|value_mean|missing_mask|scan_angle|satellite_zenith|time_offset_min)$")


def audit_samples(samples: str, lead_mapping: LeadMapping, output_dir: Path, seed: int, max_zarr: int, max_times: int, grid_sample_size: int) -> None:
    zarr_paths = discover_zarr_paths(samples, max_zarr=max_zarr)
    if not zarr_paths:
        return

    import xarray as xr

    rng = np.random.default_rng(int(seed))
    scan_bins = np.array([0.0, 20.0, 40.0, 60.0, 90.0])
    solar_bins = np.array([-10.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for zarr_path in zarr_paths:
        ds = open_zarr_dataset(zarr_path)
        if not {"lat", "lon"} <= set(ds.dims):
            continue

        # Determine time axis.
        if "time" in ds.dims:
            times = ds["time"].values
            if len(times) > max_times:
                time_idx = rng.choice(len(times), size=max_times, replace=False)
                time_idx.sort()
                ds_sel = ds.isel(time=time_idx)
            else:
                ds_sel = ds
        else:
            ds_sel = ds

        # Prepare group keys per time.
        if "time" in ds_sel.dims:
            time_vals = pd.to_datetime(ds_sel["time"].values, utc=True)
            hour_utc = time_vals.hour.astype(int)
            lead = np.array([lead_mapping.lead_hours(int(h)) for h in hour_utc], dtype=int)
        else:
            hour_utc = np.array([math.nan])
            lead = np.array([math.nan])

        # Discover aux / cloud / scan / solar.
        cloud_var = None
        for cand in ("sat.fy4b_clm.cloud_fraction", "sat.fy4b_clm.cloudy_fraction"):
            if cand in ds_sel:
                cloud_var = cand
                break
        solar_var = "aux.solar_elev" if "aux.solar_elev" in ds_sel else None

        # Choose a representative scan-angle-like variable if possible.
        scan_var = None
        for name in ds_sel.data_vars:
            if name.endswith(".scan_angle") or name.endswith(".satellite_zenith"):
                if name.startswith("sat.fy4b_agri."):
                    scan_var = name
                    break

        # Build variable groups for AGRI channels.
        channels: dict[str, dict[str, str]] = {}
        for name in ds_sel.data_vars:
            match = VAR_RE.match(name)
            if not match:
                continue
            ch = match.group("ch")
            field = match.group("field")
            channels.setdefault(ch, {})[field] = name

        if not channels:
            continue

        n_lat = int(ds_sel.dims["lat"])
        n_lon = int(ds_sel.dims["lon"])

        # Grid sampling indices shared for this store.
        lat_idx, lon_idx = sample_grid_indices(rng, n_lat=n_lat, n_lon=n_lon, sample_size=grid_sample_size)
        if lat_idx.size == 0:
            continue

        ds_point = ds_sel.isel(lat=xr.DataArray(lat_idx, dims="point"), lon=xr.DataArray(lon_idx, dims="point"))

        if cloud_var:
            cloud_frac = ds_point[cloud_var].values
            if "time" in ds_point[cloud_var].dims:
                cloud_cls = np.stack([cloud_class_from_fraction(cloud_frac[i]) for i in range(cloud_frac.shape[0])])
            else:
                cloud_cls = cloud_class_from_fraction(cloud_frac)
        else:
            cloud_cls = None

        if solar_var:
            solar_elev = ds_point[solar_var].values
            if "time" in ds_point[solar_var].dims:
                solar_bin = np.stack([bin_values(solar_elev[i], solar_bins) for i in range(solar_elev.shape[0])])
            else:
                solar_bin = bin_values(solar_elev, solar_bins)
        else:
            solar_bin = None

        if scan_var:
            scan_val = ds_point[scan_var].values
            if "time" in ds_point[scan_var].dims:
                scan_bin = np.stack([bin_values(scan_val[i], scan_bins) for i in range(scan_val.shape[0])])
            else:
                scan_bin = bin_values(scan_val, scan_bins)
        else:
            scan_bin = None

        # Meta audit: missingness proxy of aux fields if present.
        for aux_name in (
            "aux.solar_elev",
            "aux.solar_azim",
            "aux.local_solar_time_hours",
            "aux.hour_angle_deg",
            "aux.surface_skin_T",
            "aux.tcwv",
            "aux.thickness_1000_300",
            "aux.thickness_200_50",
        ):
            if aux_name not in ds_point:
                continue
            arr = ds_point[aux_name].values
            miss = np.mean(~np.isfinite(arr))
            meta_rows.append({"zarr": str(zarr_path), "aux": aux_name, "missing_rate_sampled": float(miss)})

        # Main grouped audit for AGRI per channel.
        for ch, fields in sorted(channels.items()):
            if "missing_mask" not in fields or ("value_nearest" not in fields and "value_mean" not in fields):
                continue

            miss = ds_point[fields["missing_mask"]].values
            miss = np.asarray(miss)
            if miss.dtype != bool and not np.issubdtype(miss.dtype, np.floating):
                miss = miss.astype(float)

            value_field = fields.get("value_nearest") or fields.get("value_mean")
            vals = np.asarray(ds_point[value_field].values)

            if "time" in ds_point[fields["missing_mask"]].dims:
                for t in range(miss.shape[0]):
                    key_hour = int(hour_utc[t])
                    key_lead = int(lead[t])

                    mask_valid = np.isfinite(vals[t]) & (miss[t] < 0.5)
                    missing_rate = float(np.mean(~mask_valid))

                    scan_b = scan_bin[t] if scan_bin is not None else None
                    solar_b = solar_bin[t] if solar_bin is not None else None
                    cloud_c = cloud_cls[t] if cloud_cls is not None else None

                    # Aggregate stats by bins (limited to keep output size manageable).
                    # If any grouping variable missing, fall back to -1 category.
                    for idx in range(vals.shape[-1]):
                        row = {
                            "zarr": str(zarr_path),
                            "channel": f"ch{ch}",
                            "hour_utc": key_hour,
                            "lead_hours": key_lead,
                            "scan_bin": int(scan_b[idx]) if scan_b is not None else -1,
                            "solar_bin": int(solar_b[idx]) if solar_b is not None else -1,
                            "cloud_class": int(cloud_c[idx]) if cloud_c is not None else -1,
                            "missing_rate_point": float(not mask_valid[idx]),
                            "value": float(vals[t, idx]) if mask_valid[idx] else math.nan,
                        }
                        rows.append(row)
            else:
                key_hour = math.nan
                key_lead = math.nan
                mask_valid = np.isfinite(vals) & (miss < 0.5)
                for idx in range(vals.shape[-1]):
                    row = {
                        "zarr": str(zarr_path),
                        "channel": f"ch{ch}",
                        "hour_utc": key_hour,
                        "lead_hours": key_lead,
                        "scan_bin": int(scan_bin[idx]) if scan_bin is not None else -1,
                        "solar_bin": int(solar_bin[idx]) if solar_bin is not None else -1,
                        "cloud_class": int(cloud_cls[idx]) if cloud_cls is not None else -1,
                        "missing_rate_point": float(not mask_valid[idx]),
                        "value": float(vals[idx]) if mask_valid[idx] else math.nan,
                    }
                    rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Convert point-wise rows into grouped summaries.
        group_cols = ["channel", "scan_bin", "solar_bin", "cloud_class", "hour_utc", "lead_hours"]
        agg = df.groupby(group_cols, dropna=False).agg(
            n=("missing_rate_point", "size"),
            missing_rate=("missing_rate_point", "mean"),
            value_p05=("value", lambda s: np.nanquantile(s.to_numpy(), 0.05) if np.isfinite(s.to_numpy()).any() else math.nan),
            value_p50=("value", lambda s: np.nanmedian(s.to_numpy()) if np.isfinite(s.to_numpy()).any() else math.nan),
            value_p95=("value", lambda s: np.nanquantile(s.to_numpy(), 0.95) if np.isfinite(s.to_numpy()).any() else math.nan),
        )
        out_csv = output_dir / "sample_level_agri_grouped_summary.csv"
        agg.reset_index().sort_values(group_cols).to_csv(out_csv, index=False)

        md = [
            "# Gate 1 bias audit（sample-level）",
            "",
            f"- samples: `{samples}`",
            f"- audited_zarr_count: {len(discover_zarr_paths(samples, max_zarr=max_zarr))}",
            f"- output: `{out_csv}`",
            "",
            "备注：该表按 `channel×scan_bin×solar_bin×cloud_class×hour×lead` 聚合，",
            "其中 bin 值为离散编号（-1 表示缺失/不可用），用于快速发现系统性偏差混杂。",
            "",
        ]
        write_md(output_dir / "sample_level_bias_audit.md", md)

    if meta_rows:
        out_csv = output_dir / "sample_level_aux_missingness.csv"
        pd.DataFrame(meta_rows).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    lead_mapping = load_lead_mapping(config_path) if config_path.exists() else LeadMapping({0: 6, 6: 12, 12: 6, 18: 12})

    audit_indexes(index_dir=index_dir, lead_mapping=lead_mapping, output_dir=output_dir)
    audit_samples(
        samples=args.samples,
        lead_mapping=lead_mapping,
        output_dir=output_dir,
        seed=args.seed,
        max_zarr=args.max_zarr,
        max_times=args.max_times,
        grid_sample_size=args.grid_sample_size,
    )


if __name__ == "__main__":
    main()

