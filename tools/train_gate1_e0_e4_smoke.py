"""Run a minimal E0/E1/E4 dataloader and model smoke test on complete samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import xarray as xr


TARGETS = ["dx_t2m", "dx_u10", "dx_v10", "dx_mslp"]
BACKGROUND = ["xb_t2m", "xb_u10", "xb_v10", "xb_mslp", "xb_q2m", "xb_sp"]
STATIC = ["static_landmask", "static_elevation_m", "static_topography_norm"]
GEO = [
    "fy4b_agri_ch09_value_mean",
    "fy4b_agri_ch10_value_mean",
    "fy4b_agri_ch11_value_mean",
    "fy4b_agri_ch12_value_mean",
    "fy4b_agri_ch13_value_mean",
    "fy4b_agri_ch14_value_mean",
    "fy4b_agri_ch15_value_mean",
    "fy4b_clm_cloud_fraction",
    "fy4b_clm_clear_fraction",
    "fy4b_agri_time_offset_min",
    "fy4b_clm_time_offset_min",
]


def matching_vars(names: list[str], prefix: str) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith(prefix)
        and (
            name.endswith("_value_mean")
            or name.endswith("_obs_count")
            or name.endswith("_missing_mask")
            or name.endswith("_time_offset_min_mean")
            or name.endswith("_scan_angle_raw_mean")
        )
    )


def finite_stats(paths: list[Path], variables: list[str]) -> dict[str, tuple[float, float]]:
    sums = {name: 0.0 for name in variables}
    sums2 = {name: 0.0 for name in variables}
    counts = {name: 0 for name in variables}
    for path in paths:
        with xr.open_dataset(path) as ds:
            for name in variables:
                if name not in ds:
                    continue
                arr = ds[name].values.astype(np.float64)
                finite = np.isfinite(arr)
                if not finite.any():
                    continue
                vals = arr[finite]
                sums[name] += float(vals.sum())
                sums2[name] += float(np.square(vals).sum())
                counts[name] += int(vals.size)
    out: dict[str, tuple[float, float]] = {}
    for name in variables:
        if counts[name] == 0:
            out[name] = (0.0, 1.0)
            continue
        mean = sums[name] / counts[name]
        var = max(sums2[name] / counts[name] - mean * mean, 1.0e-12)
        out[name] = (float(mean), float(np.sqrt(var)))
    return out


def load_stack(ds: xr.Dataset, variables: list[str], stats: dict[str, tuple[float, float]]) -> torch.Tensor:
    arrays = []
    for name in variables:
        if name in ds:
            arr = ds[name].values.astype(np.float32)
        else:
            shape = (ds.sizes["lat"], ds.sizes["lon"])
            arr = np.full(shape, np.nan, dtype=np.float32)
        mean, std = stats[name]
        arr = np.where(np.isfinite(arr), arr, mean)
        arrays.append(((arr - mean) / std).astype(np.float32))
    return torch.from_numpy(np.stack(arrays, axis=0))


class TinyConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = tensor.shape
    if height <= size or width <= size:
        return tensor
    top = random.randint(0, height - size)
    left = random.randint(0, width - size)
    return tensor[:, top : top + size, left : left + size]


def run_zero_increment_baseline(
    paths: list[Path],
    target_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict[str, object]:
    losses: list[float] = []
    zero_standardized = torch.tensor(
        [(-target_stats[name][0]) / target_stats[name][1] for name in TARGETS],
        dtype=torch.float32,
    ).view(1, len(TARGETS), 1, 1)

    for path in paths:
        with xr.open_dataset(path) as ds:
            y = crop(load_stack(ds, TARGETS, target_stats), args.crop_size).unsqueeze(0)
        pred = zero_standardized.expand_as(y)
        loss = torch.mean((pred - y) ** 2)
        losses.append(float(loss.detach().cpu().item()))

    return {
        "experiment": "E0-RAW-NO-TRAIN-SMOKE",
        "sample_count": len(paths),
        "input_channel_count": 0,
        "target_channel_count": len(TARGETS),
        "epochs": 0,
        "steps": len(losses),
        "final_eval_loss_standardized": losses[-1] if losses else float("nan"),
        "mean_eval_loss_standardized": float(np.mean(losses)) if losses else float("nan"),
        "formal_definition": "xa_model = xb; dx_pred = 0; no trained model and no static input.",
    }


def run_experiment(
    paths: list[Path],
    name: str,
    input_vars: list[str],
    target_stats: dict[str, tuple[float, float]],
    input_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict[str, object]:
    model = TinyConv(len(input_vars), len(TARGETS))
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []

    for _epoch in range(args.epochs):
        for path in paths:
            with xr.open_dataset(path) as ds:
                x = crop(load_stack(ds, input_vars, input_stats), args.crop_size).unsqueeze(0)
                y = crop(load_stack(ds, TARGETS, target_stats), args.crop_size).unsqueeze(0)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

    return {
        "experiment": name,
        "sample_count": len(paths),
        "input_channel_count": len(input_vars),
        "target_channel_count": len(TARGETS),
        "epochs": args.epochs,
        "steps": len(losses),
        "final_train_loss_standardized": losses[-1] if losses else float("nan"),
        "mean_train_loss_standardized": float(np.mean(losses)) if losses else float("nan"),
    }


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(Path(args.sample_dir).glob("gate1_complete_sample_*.nc"))
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if not paths:
        raise FileNotFoundError(f"no complete samples found in {args.sample_dir}")

    with xr.open_dataset(paths[0]) as first:
        names = list(first.data_vars)
    mwts = matching_vars(names, "fy3e_mwts")
    mwhs = matching_vars(names, "fy3e_mwhs")
    e1_inputs = BACKGROUND + STATIC
    e4_inputs = BACKGROUND + STATIC + GEO + mwts + mwhs
    all_inputs = sorted(set(e1_inputs + e4_inputs))

    input_stats = finite_stats(paths, all_inputs)
    target_stats = finite_stats(paths, TARGETS)
    rows = [
        run_zero_increment_baseline(paths, target_stats, args),
        run_experiment(paths, "E1-B0-REG-STATIC-SMOKE", e1_inputs, target_stats, input_stats, args),
        run_experiment(paths, "E4-B3-MULTI-SMOKE", e4_inputs, target_stats, input_stats, args),
    ]

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sample_dir": args.sample_dir,
        "samples": [str(path) for path in paths],
        "targets": TARGETS,
        "background_inputs": BACKGROUND,
        "static_inputs": STATIC,
        "geo_inputs": GEO,
        "mwts_input_count": len(mwts),
        "mwhs_input_count": len(mwhs),
        "scientific_warning": "Smoke only. Loss values are not scientific results. Formal E0 is no-train xb baseline; learned static/background model is E1-style.",
        "rows": rows,
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default="data/processed/gate1_complete_samples_pilot")
    parser.add_argument("--output-json", default="data/diagnostics/tables/gate1_e0_e4_smoke.json")
    parser.add_argument("--output-csv", default="data/diagnostics/tables/gate1_e0_e4_smoke.csv")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
