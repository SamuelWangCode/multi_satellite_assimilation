# Multi-Satellite Assimilation

Code utilities for building multi-satellite analysis-increment datasets and running deterministic source-ablation pilots.

## Repository Scope

This repository is intended for public code release. It stores source code and reusable configuration templates only.

Bulk datasets, generated samples, logs, credentials, model checkpoints, and research-management notes are intentionally excluded from version control.

## Configuration

Copy the template and set machine-specific paths outside Git:

```powershell
Copy-Item configs/paths.template.yaml configs/local_paths.yaml
```

Use environment variables or command-line arguments for local data roots, server workspaces, SSH keys, and Python environments.

## Main Tool Groups

- `tools/build_gate1_*`: build static, satellite, forecast, and complete sample files.
- `tools/download_era5_*`: resumable ERA5 forecast and reanalysis-label download helpers.
- `tools/extract_era5_*`: extract ERA5 subsets on the project grid.
- `tools/index_fy_data.py`: index Fengyun satellite files.
- `tools/qc_gate1_complete_samples.py`: validate complete sample files.
- `tools/train_gate1_*`: lightweight deterministic pilot training scripts.

## Data Policy

Do not place large data products in this repository. Keep raw data, processed samples, diagnostics, and checkpoints under external storage and pass paths explicitly at runtime.

