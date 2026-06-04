"""Summarize dx quantiles for Gate 1 complete samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr


QUANTILES = [1, 5, 50, 95, 99]


def summarize_var(name: str, da: xr.DataArray) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    if "level" in da.dims:
        for level in da["level"].values:
            values = da.sel(level=level).values
            rows.extend(summarize_values(name, values, level=int(level)))
    else:
        rows.extend(summarize_values(name, da.values, level="surface"))
    return rows


def summarize_values(name: str, values: np.ndarray, level: int | str) -> list[dict[str, str | float | int]]:
    finite = values[np.isfinite(values)]
    row: dict[str, str | float | int] = {
        "variable": name,
        "level": level,
        "finite": int(finite.size),
        "total": int(values.size),
    }
    if finite.size:
        qs = np.nanpercentile(finite, QUANTILES)
        for q, value in zip(QUANTILES, qs):
            row[f"p{q:02d}"] = float(value)
        row["mean"] = float(np.nanmean(finite))
        row["min"] = float(np.nanmin(finite))
        row["max"] = float(np.nanmax(finite))
    return [row]


def main() -> None:
    args = parse_args()
    sample = Path(args.sample)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(sample) as ds:
        rows: list[dict[str, str | float | int]] = []
        for name in sorted(ds.data_vars):
            if name.startswith("dx_"):
                rows.extend(summarize_var(name, ds[name]))
    fieldnames = ["variable", "level", "finite", "total", "min", "p01", "p05", "p50", "p95", "p99", "max", "mean"]
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="data/processed/gate1_complete_samples/gate1_complete_sample_2024050100.nc")
    parser.add_argument("--output", default="data/diagnostics/tables/gate1_complete_sample_dx_quantiles_2024050100.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
