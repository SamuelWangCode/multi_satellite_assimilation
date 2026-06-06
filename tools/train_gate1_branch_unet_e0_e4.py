"""Train formal Gate 1 branch residual U-Net E0-E4 experiments.

This script is the publishable-model training path. The older
train_gate1_deterministic_pilot.py remains a small pipeline smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import xarray as xr


LEVELS = [1000, 925, 850, 700, 500, 300, 200]
SURFACE_TARGETS = ["dx_t2m", "dx_q2m", "dx_u10", "dx_v10", "dx_mslp"]
HUMIDITY_850_TARGETS = [
    "dx_t2m",
    "dx_q2m",
    "dx_u10",
    "dx_v10",
    "dx_mslp",
    "dx_q@1000",
    "dx_q@925",
    "dx_q@850",
    "dx_q@700",
    "dx_t@850",
    "dx_u@850",
    "dx_v@850",
]
FULL_3D_BASES = ["dx_t", "dx_q", "dx_u", "dx_v", "dx_z"]

BACKGROUND_SURFACE = ["xb_t2m", "xb_q2m", "xb_u10", "xb_v10", "xb_mslp", "xb_sp"]
BACKGROUND_3D_BASES = ["xb_t", "xb_q", "xb_u", "xb_v", "xb_z"]
STATIC = [
    "static_landmask",
    "static_oceanmask",
    "static_elevation_m",
    "static_topography_norm",
]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    level: int | None = None

    @property
    def label(self) -> str:
        return self.name if self.level is None else f"{self.name}_{self.level}"


def parse_field_spec(text: str) -> FieldSpec:
    if "@" in text:
        name, level = text.split("@", 1)
        return FieldSpec(name=name, level=int(level))
    return FieldSpec(name=text)


def read_stamp(path: Path) -> str:
    return path.stem.replace("gate1_complete_sample_", "")


def available_paths(sample_dir: Path, max_samples: int) -> list[Path]:
    paths = sorted(sample_dir.glob("gate1_complete_sample_*.nc"), key=read_stamp)
    if max_samples > 0:
        paths = paths[:max_samples]
    if len(paths) < 3:
        raise FileNotFoundError(f"need at least 3 samples in {sample_dir}, found {len(paths)}")
    return paths


def chronological_split(paths: list[Path], train_fraction: float, val_fraction: float) -> dict[str, list[Path]]:
    n = len(paths)
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


def load_or_make_splits(args: argparse.Namespace, paths: list[Path]) -> dict[str, list[Path]]:
    if args.split_json:
        data = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        return {name: [Path(item) for item in values] for name, values in data["splits"].items()}
    return chronological_split(paths, args.train_fraction, args.val_fraction)


def target_specs(package: str) -> list[FieldSpec]:
    if package == "surface":
        return [parse_field_spec(item) for item in SURFACE_TARGETS]
    if package == "humidity_850":
        return [parse_field_spec(item) for item in HUMIDITY_850_TARGETS]
    if package == "full_3d":
        return [FieldSpec(name, level) for name in FULL_3D_BASES for level in LEVELS] + [
            parse_field_spec(item) for item in SURFACE_TARGETS
        ]
    raise ValueError(f"unknown target package: {package}")


def find_level_index(ds: xr.Dataset, level: int) -> int:
    values = np.asarray(ds["level"].values).astype(int)
    matches = np.where(values == int(level))[0]
    if matches.size == 0:
        raise KeyError(f"level {level} not found; available={values.tolist()}")
    return int(matches[0])


def field_values(ds: xr.Dataset, spec: FieldSpec) -> np.ndarray:
    if spec.name not in ds:
        raise KeyError(f"missing variable: {spec.name}")
    arr = ds[spec.name].values.astype(np.float32)
    if spec.level is None:
        if arr.ndim != 2:
            raise ValueError(f"{spec.name} expected 2-D, got {arr.shape}")
        return arr
    if arr.ndim != 3:
        raise ValueError(f"{spec.name}@{spec.level} expected 3-D, got {arr.shape}")
    return arr[find_level_index(ds, spec.level)]


def optional_values(ds: xr.Dataset, spec: FieldSpec) -> np.ndarray:
    try:
        return field_values(ds, spec)
    except KeyError:
        shape = (int(ds.sizes["lat"]), int(ds.sizes["lon"]))
        return np.full(shape, np.nan, dtype=np.float32)


def finite_stats(paths: list[Path], specs: list[FieldSpec]) -> dict[str, tuple[float, float]]:
    sums = {spec.label: 0.0 for spec in specs}
    sums2 = {spec.label: 0.0 for spec in specs}
    counts = {spec.label: 0 for spec in specs}
    for path in paths:
        with xr.open_dataset(path) as ds:
            for spec in specs:
                arr = optional_values(ds, spec).astype(np.float64)
                finite = np.isfinite(arr)
                if not finite.any():
                    continue
                vals = arr[finite]
                sums[spec.label] += float(vals.sum())
                sums2[spec.label] += float(np.square(vals).sum())
                counts[spec.label] += int(vals.size)
    stats: dict[str, tuple[float, float]] = {}
    for spec in specs:
        label = spec.label
        if counts[label] == 0:
            stats[label] = (0.0, 1.0)
            continue
        mean = sums[label] / counts[label]
        var = max(sums2[label] / counts[label] - mean * mean, 1.0e-12)
        stats[label] = (float(mean), float(math.sqrt(var)))
    return stats


def load_stack(ds: xr.Dataset, specs: list[FieldSpec], stats: dict[str, tuple[float, float]]) -> torch.Tensor:
    arrays = []
    for spec in specs:
        arr = optional_values(ds, spec)
        mean, std = stats[spec.label]
        arr = np.where(np.isfinite(arr), arr, mean)
        arrays.append(((arr - mean) / std).astype(np.float32))
    return torch.from_numpy(np.stack(arrays, axis=0))


def expand_3d_specs(names: Iterable[str], levels: Iterable[int]) -> list[FieldSpec]:
    return [FieldSpec(name, level) for name in names for level in levels]


def channel_specs(prefix: str, channel_count: int) -> list[FieldSpec]:
    fields = []
    for channel in range(1, channel_count + 1):
        stem = f"{prefix}_ch{channel:02d}"
        for suffix in ("value_mean", "obs_count", "missing_mask"):
            fields.append(FieldSpec(f"{stem}_{suffix}"))
    for shared in ("time_offset_min_mean", "scan_angle_raw_mean", "geometry0_raw_mean", "geometry1_raw_mean", "geometry2_raw_mean", "geometry3_raw_mean"):
        fields.append(FieldSpec(f"{prefix}_{shared}"))
    return fields


def geo_specs() -> list[FieldSpec]:
    specs = []
    for channel in range(9, 16):
        stem = f"fy4b_agri_ch{channel:02d}"
        for suffix in ("value_mean", "obs_count", "missing_mask"):
            specs.append(FieldSpec(f"{stem}_{suffix}"))
    specs.extend(
        FieldSpec(name)
        for name in (
            "fy4b_agri_time_offset_min",
            "fy4b_clm_cloud_fraction",
            "fy4b_clm_clear_fraction",
            "fy4b_clm_obs_count",
            "fy4b_clm_missing_mask",
            "fy4b_clm_time_offset_min",
        )
    )
    return specs


def branch_specs(target_package: str) -> dict[str, list[FieldSpec]]:
    background = [FieldSpec(name) for name in BACKGROUND_SURFACE + STATIC]
    if target_package in {"humidity_850", "full_3d"}:
        background += expand_3d_specs(BACKGROUND_3D_BASES, LEVELS)
    return {
        "background_static": background,
        "geo": geo_specs(),
        "mwts": channel_specs("fy3e_mwts", 17),
        "mwhs": channel_specs("fy3e_mwhs", 15),
    }


def experiment_branches(all_branches: dict[str, list[FieldSpec]]) -> dict[str, dict[str, list[FieldSpec]]]:
    return {
        "E1-B0-REG": {"background_static": all_branches["background_static"]},
        "E2-B1-GEO": {"background_static": all_branches["background_static"], "geo": all_branches["geo"]},
        "E3-B2-MW": {
            "background_static": all_branches["background_static"],
            "mwts": all_branches["mwts"],
            "mwhs": all_branches["mwhs"],
        },
        "E4-B3-MULTI": all_branches,
    }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(group for group in (8, 4, 2, 1) if out_channels % group == 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class BranchResidualUNet(nn.Module):
    def __init__(self, branch_channels: dict[str, int], out_channels: int, stem_width: int, base_width: int) -> None:
        super().__init__()
        self.branch_names = list(branch_channels)
        self.stems = nn.ModuleDict({name: ConvBlock(channels, stem_width) for name, channels in branch_channels.items()})
        self.fuse = nn.Conv2d(stem_width * len(self.branch_names), base_width, kernel_size=1)
        self.enc1 = ConvBlock(base_width, base_width)
        self.down1 = nn.Conv2d(base_width, base_width * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(base_width * 2, base_width * 2)
        self.down2 = nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, stride=2, padding=1)
        self.mid = ConvBlock(base_width * 4, base_width * 4)
        self.up2 = nn.Conv2d(base_width * 4 + base_width * 2, base_width * 2, kernel_size=3, padding=1)
        self.dec2 = ConvBlock(base_width * 2, base_width * 2)
        self.up1 = nn.Conv2d(base_width * 2 + base_width, base_width, kernel_size=3, padding=1)
        self.dec1 = ConvBlock(base_width, base_width)
        self.head = nn.Conv2d(base_width, out_channels, kernel_size=1)

    def forward(self, x_by_branch: dict[str, torch.Tensor], drop_branches: set[str] | None = None) -> torch.Tensor:
        drop_branches = drop_branches or set()
        feats = []
        for name in self.branch_names:
            x = x_by_branch[name]
            if name in drop_branches:
                x = torch.zeros_like(x)
            feats.append(self.stems[name](x))
        x0 = self.fuse(torch.cat(feats, dim=1))
        e1 = self.enc1(x0)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        u2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(self.up2(torch.cat([u2, e2], dim=1)))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(self.up1(torch.cat([u1, e1], dim=1)))
        return self.head(d1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def crop_tensors(
    x_by_branch: dict[str, torch.Tensor],
    y: torch.Tensor,
    size: int,
    random_crop: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    _, height, width = y.shape
    if size <= 0 or height <= size or width <= size:
        return x_by_branch, y
    if random_crop:
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    return {name: x[:, top : top + size, left : left + size] for name, x in x_by_branch.items()}, y[:, top : top + size, left : left + size]


def load_branches(ds: xr.Dataset, branches: dict[str, list[FieldSpec]], stats: dict[str, tuple[float, float]]) -> dict[str, torch.Tensor]:
    return {name: load_stack(ds, specs, stats) for name, specs in branches.items()}


def regime_masks(ds: xr.Dataset, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {"all": np.ones(shape, dtype=bool)}
    if "static_landmask" in ds:
        land = ds["static_landmask"].values
        masks["land"] = np.isfinite(land) & (land >= 0.5)
        masks["sea"] = np.isfinite(land) & (land < 0.5)
    if "fy4b_clm_cloud_fraction" in ds:
        cloud = ds["fy4b_clm_cloud_fraction"].values
        finite = np.isfinite(cloud)
        masks["clear"] = finite & (cloud < 0.2)
        masks["mixed"] = finite & (cloud >= 0.2) & (cloud <= 0.8)
        masks["cloudy"] = finite & (cloud > 0.8)
    mw = np.zeros(shape, dtype=np.float32)
    for prefix, channels in (("fy3e_mwts", 17), ("fy3e_mwhs", 15)):
        for channel in range(1, channels + 1):
            name = f"{prefix}_ch{channel:02d}_obs_count"
            if name in ds:
                mw += np.nan_to_num(ds[name].values.astype(np.float32), nan=0.0)
    masks["mw_covered"] = mw > 0
    masks["no_mw"] = mw <= 0
    return masks


def update_metric(acc: dict[str, dict[str, float]], key: str, sqerr: np.ndarray, mask: np.ndarray) -> None:
    if mask.shape != sqerr.shape[-2:]:
        return
    vals = sqerr[..., mask]
    if vals.size == 0:
        return
    item = acc.setdefault(key, {"sum": 0.0, "count": 0.0})
    item["sum"] += float(np.nansum(vals))
    item["count"] += float(np.isfinite(vals).sum())


def evaluate(
    model: nn.Module | None,
    paths: list[Path],
    branches: dict[str, list[FieldSpec]],
    input_stats: dict[str, tuple[float, float]],
    targets: list[FieldSpec],
    target_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
    device: torch.device,
    drop_branches: set[str] | None = None,
) -> dict[str, float]:
    if model is not None:
        model.eval()
    acc: dict[str, dict[str, float]] = {}
    zero = torch.tensor([(-target_stats[spec.label][0]) / target_stats[spec.label][1] for spec in targets], device=device).view(1, -1, 1, 1)
    with torch.no_grad():
        for path in paths:
            with xr.open_dataset(path) as ds:
                y = load_stack(ds, targets, target_stats)
                if model is not None:
                    x = load_branches(ds, branches, input_stats)
                masks = regime_masks(ds, (int(ds.sizes["lat"]), int(ds.sizes["lon"])))
            if args.eval_crop_size > 0:
                dummy = {name: tensor for name, tensor in (x if model is not None else {}).items()}
                dummy, y = crop_tensors(dummy, y, args.eval_crop_size, random_crop=False)
                # Center-crop masks consistently with eval crop.
                _, h, w = y.shape
                full_h, full_w = next(iter(masks.values())).shape
                top = (full_h - h) // 2
                left = (full_w - w) // 2
                masks = {name: mask[top : top + h, left : left + w] for name, mask in masks.items()}
                if model is not None:
                    x = dummy
            yb = y.unsqueeze(0).to(device)
            if model is None:
                pred = zero.expand(1, -1, y.shape[-2], y.shape[-1])
            else:
                xb = {name: tensor.unsqueeze(0).to(device) for name, tensor in x.items()}
                pred = model(xb, drop_branches=drop_branches)
            sqerr = torch.square(pred - yb).squeeze(0).detach().cpu().numpy()
            update_metric(acc, "mse_all", sqerr, masks["all"])
            for idx, spec in enumerate(targets):
                update_metric(acc, f"mse_{spec.label}", sqerr[idx : idx + 1], masks["all"])
            for regime, mask in masks.items():
                update_metric(acc, f"mse_regime_{regime}", sqerr, mask)
    out: dict[str, float] = {}
    for key, item in acc.items():
        out[key] = item["sum"] / item["count"] if item["count"] > 0 else float("nan")
        out[f"{key}_count"] = item["count"]
    return out


def train_one(
    train_paths: list[Path],
    branches: dict[str, list[FieldSpec]],
    input_stats: dict[str, tuple[float, float]],
    targets: list[FieldSpec],
    target_stats: dict[str, tuple[float, float]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    model = BranchResidualUNet(
        {name: len(specs) for name, specs in branches.items()},
        len(targets),
        stem_width=args.stem_width,
        base_width=args.base_width,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses: list[float] = []
    model.train()
    for _epoch in range(args.epochs):
        shuffled = list(train_paths)
        random.shuffle(shuffled)
        for path in shuffled:
            with xr.open_dataset(path) as ds:
                x = load_branches(ds, branches, input_stats)
                y = load_stack(ds, targets, target_stats)
            x, y = crop_tensors(x, y, args.crop_size, random_crop=True)
            xb = {name: tensor.unsqueeze(0).to(device) for name, tensor in x.items()}
            yb = y.unsqueeze(0).to(device)
            pred = model(xb)
            loss = torch.mean(torch.square(pred - yb))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
    return model, losses


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_splits(path: Path, splits: dict[str, list[Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": {name: [str(item) for item in values] for name, values in splits.items()}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = available_paths(Path(args.sample_dir), args.max_samples)
    splits = load_or_make_splits(args, paths)
    targets = target_specs(args.target_package)
    branches_all = branch_specs(args.target_package)
    experiments = experiment_branches(branches_all)
    all_input_specs = sorted({spec for exp in experiments.values() for specs in exp.values() for spec in specs}, key=lambda s: s.label)
    input_stats = finite_stats(splits["train"], all_input_specs)
    target_stats = finite_stats(splits["train"], targets)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_splits(output_dir / "split_metadata.json", splits)
    (output_dir / "normalization_stats.json").write_text(
        json.dumps(
            {
                "input": {key: {"mean": value[0], "std": value[1]} for key, value in input_stats.items()},
                "target": {key: {"mean": value[0], "std": value[1]} for key, value in target_stats.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for split_name, split_paths_list in splits.items():
        metrics = evaluate(None, split_paths_list, {}, input_stats, targets, target_stats, args, device)
        rows.append(
            {
                "seed": -1,
                "experiment": "E0-RAW",
                "split": split_name,
                "sample_count": len(split_paths_list),
                "target_package": args.target_package,
                "target_channel_count": len(targets),
                "input_channel_count": 0,
                "parameter_count": 0,
                "model_family": "no_training_xb_baseline",
                **metrics,
            }
        )

    for seed in seeds:
        set_seed(seed)
        for name, branches in experiments.items():
            model, train_losses = train_one(splits["train"], branches, input_stats, targets, target_stats, args, device)
            param_count = parameter_count(model)
            branch_counts = {branch: len(specs) for branch, specs in branches.items()}
            for split_name, split_paths_list in splits.items():
                metrics = evaluate(model, split_paths_list, branches, input_stats, targets, target_stats, args, device)
                rows.append(
                    {
                        "seed": seed,
                        "experiment": name,
                        "split": split_name,
                        "sample_count": len(split_paths_list),
                        "target_package": args.target_package,
                        "target_channel_count": len(targets),
                        "input_channel_count": sum(branch_counts.values()),
                        "branch_channel_counts_json": json.dumps(branch_counts, sort_keys=True),
                        "parameter_count": param_count,
                        "epochs": args.epochs,
                        "train_steps": len(train_losses),
                        "train_loss_last": train_losses[-1] if train_losses else float("nan"),
                        "model_family": "branch_residual_unet_encoder_decoder",
                        **metrics,
                    }
                )
            if name == "E4-B3-MULTI" and args.source_dropout_eval:
                drops = {
                    "drop_geo": {"geo"},
                    "drop_mwts": {"mwts"},
                    "drop_mwhs": {"mwhs"},
                    "drop_mwts_mwhs": {"mwts", "mwhs"},
                }
                for drop_name, drop_set in drops.items():
                    metrics = evaluate(model, splits["test"], branches, input_stats, targets, target_stats, args, device, drop_branches=drop_set)
                    rows.append(
                        {
                            "seed": seed,
                            "experiment": f"E4-B3-MULTI::{drop_name}",
                            "split": "test",
                            "sample_count": len(splits["test"]),
                            "target_package": args.target_package,
                            "target_channel_count": len(targets),
                            "input_channel_count": sum(branch_counts.values()),
                            "branch_channel_counts_json": json.dumps(branch_counts, sort_keys=True),
                            "parameter_count": param_count,
                            "epochs": args.epochs,
                            "train_steps": len(train_losses),
                            "model_family": "branch_residual_unet_encoder_decoder_source_dropout_eval",
                            **metrics,
                        }
                    )

    summary = {
        "sample_dir": args.sample_dir,
        "sample_count": len(paths),
        "target_package": args.target_package,
        "targets": [spec.label for spec in targets],
        "device": str(device),
        "seeds": seeds,
        "splits": {name: [str(item) for item in values] for name, values in splits.items()},
        "model_family": "branch_residual_unet_encoder_decoder",
        "formal_warnings": [
            "ERA5-supervised metrics are development metrics, not independent validation.",
            "D4-B3-RESDIFF remains blocked until deterministic E4 is stable or interpretable by regime.",
        ],
        "rows": rows,
    }
    output_json = output_dir / "metrics.json"
    output_csv = output_dir / "metrics.csv"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as fp:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_json": str(output_json), "output_csv": str(output_csv), "rows": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default="data/processed/gate1_complete_samples_202405_202407_directq")
    parser.add_argument("--output-dir", default="data/diagnostics/formal_e0_e4_branch_unet")
    parser.add_argument("--target-package", choices=["surface", "humidity_850", "full_3d"], default="surface")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--split-json", default="")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seeds", default="20260606")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--eval-crop-size", type=int, default=160)
    parser.add_argument("--stem-width", type=int, default=24)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="")
    parser.add_argument("--source-dropout-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
