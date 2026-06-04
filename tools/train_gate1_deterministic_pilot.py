"""Train a lightweight deterministic E0-E4 pilot on Gate 1 complete samples."""

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


def read_stamp(path: Path) -> str:
    return path.stem.replace("gate1_complete_sample_", "")


def split_paths(paths: list[Path], train_fraction: float, val_fraction: float) -> dict[str, list[Path]]:
    paths = sorted(paths, key=read_stamp)
    n = len(paths)
    if n < 3:
        return {"train": paths, "val": paths, "test": paths}
    n_train = max(1, int(round(n * train_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return {
        "train": paths[:n_train],
        "val": paths[n_train : n_train + n_val],
        "test": paths[n_train + n_val :],
    }


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
            arr = np.full((ds.sizes["lat"], ds.sizes["lon"]), np.nan, dtype=np.float32)
        mean, std = stats[name]
        arr = np.where(np.isfinite(arr), arr, mean)
        arrays.append(((arr - mean) / std).astype(np.float32))
    return torch.from_numpy(np.stack(arrays, axis=0))


def crop_pair(
    x: torch.Tensor, y: torch.Tensor, size: int, random_crop: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = y.shape
    if height <= size or width <= size:
        return x, y
    if random_crop:
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    return x[:, top : top + size, left : left + size], y[:, top : top + size, left : left + size]


def load_branch_stacks(
    ds: xr.Dataset, branch_vars: dict[str, list[str]], stats: dict[str, tuple[float, float]]
) -> dict[str, torch.Tensor]:
    return {name: load_stack(ds, variables, stats) for name, variables in branch_vars.items()}


def crop_branch_pair(
    x_by_branch: dict[str, torch.Tensor],
    y: torch.Tensor,
    size: int,
    random_crop: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    _, height, width = y.shape
    if height <= size or width <= size:
        return x_by_branch, y
    if random_crop:
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    return (
        {name: x[:, top : top + size, left : left + size] for name, x in x_by_branch.items()},
        y[:, top : top + size, left : left + size],
    )


class BranchTinyConv(nn.Module):
    def __init__(
        self,
        branch_channels: dict[str, int],
        out_channels: int,
        branch_hidden: int,
        head_hidden: int,
    ) -> None:
        super().__init__()
        self.branch_names = list(branch_channels)
        self.stems = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(in_channels, branch_hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(branch_hidden, branch_hidden, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                for name, in_channels in branch_channels.items()
            }
        )
        self.head = nn.Sequential(
            nn.Conv2d(branch_hidden * len(branch_channels), head_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(head_hidden, out_channels, kernel_size=1),
        )

    def forward(self, x_by_branch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.stems[name](x_by_branch[name]) for name in self.branch_names]
        return self.head(torch.cat(features, dim=1))


def eval_zero_increment(paths: list[Path], target_stats: dict[str, tuple[float, float]], args: argparse.Namespace) -> dict[str, float]:
    zero_standardized = torch.tensor([(-target_stats[name][0]) / target_stats[name][1] for name in TARGETS]).view(
        1, len(TARGETS), 1, 1
    )
    out: dict[str, float] = {}
    losses = []
    for path in paths:
        with xr.open_dataset(path) as ds:
            y = load_stack(ds, TARGETS, target_stats)
        if args.eval_crop_size > 0:
            _, y = crop_pair(y, y, args.eval_crop_size, random_crop=False)
        pred = zero_standardized.expand(1, -1, y.shape[-2], y.shape[-1])
        losses.append(float(torch.mean((pred - y.unsqueeze(0)) ** 2).item()))
    out["loss_standardized"] = float(np.mean(losses)) if losses else float("nan")
    return out


def evaluate_model(
    model: nn.Module,
    paths: list[Path],
    branch_vars: dict[str, list[str]],
    input_stats: dict[str, tuple[float, float]],
    target_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        for path in paths:
            with xr.open_dataset(path) as ds:
                x_by_branch = load_branch_stacks(ds, branch_vars, input_stats)
                y = load_stack(ds, TARGETS, target_stats)
            if args.eval_crop_size > 0:
                x_by_branch, y = crop_branch_pair(x_by_branch, y, args.eval_crop_size, random_crop=False)
            pred = model({name: x.unsqueeze(0) for name, x in x_by_branch.items()})
            losses.append(float(torch.mean((pred - y.unsqueeze(0)) ** 2).item()))
    return {"loss_standardized": float(np.mean(losses)) if losses else float("nan")}


def train_model(
    paths: list[Path],
    name: str,
    branch_vars: dict[str, list[str]],
    input_stats: dict[str, tuple[float, float]],
    target_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[nn.Module, list[float]]:
    del name
    model = BranchTinyConv(
        {branch_name: len(variables) for branch_name, variables in branch_vars.items()},
        len(TARGETS),
        branch_hidden=args.branch_hidden,
        head_hidden=args.head_hidden,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    model.train()
    for _epoch in range(args.epochs):
        shuffled = list(paths)
        random.shuffle(shuffled)
        for path in shuffled:
            with xr.open_dataset(path) as ds:
                x_by_branch = load_branch_stacks(ds, branch_vars, input_stats)
                y = load_stack(ds, TARGETS, target_stats)
            x_by_branch, y = crop_branch_pair(x_by_branch, y, args.crop_size)
            pred = model({branch_name: x.unsqueeze(0) for branch_name, x in x_by_branch.items()})
            loss = torch.mean((pred - y.unsqueeze(0)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    return model, losses


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(Path(args.sample_dir).glob("gate1_complete_sample_*.nc"), key=read_stamp)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    if len(paths) < 3:
        raise FileNotFoundError(f"need at least 3 complete samples in {args.sample_dir}; found {len(paths)}")

    splits = split_paths(paths, args.train_fraction, args.val_fraction)
    with xr.open_dataset(paths[0]) as first:
        names = list(first.data_vars)
    mwts = matching_vars(names, "fy3e_mwts")
    mwhs = matching_vars(names, "fy3e_mwhs")

    background_static = BACKGROUND + STATIC
    experiments: dict[str, dict[str, list[str]]] = {
        "E1-B0-REG": {"background_static": background_static},
        "E2-B1-GEO": {"background_static": background_static, "geo": GEO},
        "E3-B2-MW": {"background_static": background_static, "mwts": mwts, "mwhs": mwhs},
        "E4-B3-MULTI": {"background_static": background_static, "geo": GEO, "mwts": mwts, "mwhs": mwhs},
    }
    all_inputs = sorted(
        {
            variable
            for branch_vars in experiments.values()
            for variables in branch_vars.values()
            for variable in variables
        }
    )
    input_stats = finite_stats(splits["train"], all_inputs)
    target_stats = finite_stats(splits["train"], TARGETS)

    rows: list[dict[str, object]] = []
    for split_name, split_paths_list in splits.items():
        metrics = eval_zero_increment(split_paths_list, target_stats, args)
        rows.append(
            {
                "experiment": "E0-RAW",
                "split": split_name,
                "sample_count": len(split_paths_list),
                "input_channel_count": 0,
                "target_channel_count": len(TARGETS),
                "epochs": 0,
                "train_steps": 0,
                "loss_standardized": metrics["loss_standardized"],
                "definition": "xa_model=xb; dx_pred=0; no training.",
            }
        )

    for name, branch_vars in experiments.items():
        model, train_losses = train_model(splits["train"], name, branch_vars, input_stats, target_stats, args)
        branch_channel_counts = {branch_name: len(variables) for branch_name, variables in branch_vars.items()}
        for split_name, split_paths_list in splits.items():
            metrics = evaluate_model(model, split_paths_list, branch_vars, input_stats, target_stats, args)
            rows.append(
                {
                    "experiment": name,
                    "split": split_name,
                    "sample_count": len(split_paths_list),
                    "input_channel_count": sum(branch_channel_counts.values()),
                    "branch_names": ",".join(branch_vars),
                    "branch_channel_counts_json": json.dumps(branch_channel_counts, sort_keys=True),
                    "target_channel_count": len(TARGETS),
                    "epochs": args.epochs,
                    "train_steps": len(train_losses),
                    "train_loss_last": train_losses[-1] if train_losses else float("nan"),
                    "loss_standardized": metrics["loss_standardized"],
                }
            )

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sample_dir": args.sample_dir,
        "sample_count": len(paths),
        "splits": {name: [str(path) for path in values] for name, values in splits.items()},
        "targets": TARGETS,
        "mwts_input_count": len(mwts),
        "mwhs_input_count": len(mwhs),
        "model_family": "branch_tiny_conv",
        "branch_policy": "MWTS and MWHS use separate encoder stems before fusion in E3/E4.",
        "warning": "Pilot only. ERA5-supervised development metric, not final paper claim.",
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
    parser.add_argument("--sample-dir", default="data/processed/gate1_complete_samples_may_directq")
    parser.add_argument("--output-json", default="data/diagnostics/tables/gate1_e0_e4_deterministic_pilot.json")
    parser.add_argument("--output-csv", default="data/diagnostics/tables/gate1_e0_e4_deterministic_pilot.csv")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--eval-crop-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--branch-hidden", type=int, default=16)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
