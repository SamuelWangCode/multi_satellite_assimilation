"""Download ERA5 short-range forecast batches for Gate 1.

The script is designed for long-running server jobs. It downloads one request
at a time, writes an .ok marker after each successful file, and can be safely
restarted after SSH disconnects or transient network failures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DATASET = "reanalysis-era5-complete"

PL_PARAM_CODES = {
    "z": 129,
    "t": 130,
    "u": 131,
    "v": 132,
    "q": 133,
}

SFC_PARAM_CODES = {
    "sp": 134,
    "msl": 151,
    "u10": 165,
    "v10": 166,
    "t2m": 167,
    "d2m": 168,
}

DEFAULT_LEVELS = [1000, 925, 850, 700, 500, 300, 200]


@dataclass(frozen=True)
class Task:
    part: str
    init_date: date
    request: dict[str, str]
    request_path: Path
    target_path: Path


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def request_base(init_date: date, steps: str) -> dict[str, str]:
    return {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "fc",
        "date": init_date.isoformat(),
        "time": "06/18",
        "step": steps,
        "area": "60/70/0/150",
        "grid": "0.25/0.25",
    }


def build_tasks(args: argparse.Namespace) -> list[Task]:
    valid_start = parse_date(args.start_date)
    valid_end = parse_date(args.end_date)
    init_start = parse_date(args.init_start_date) if args.init_start_date else valid_start - timedelta(days=1)
    init_end = parse_date(args.init_end_date) if args.init_end_date else valid_end
    if init_start > init_end:
        raise ValueError(f"init_start_date must be <= init_end_date, got {init_start} > {init_end}")
    steps_by_subset = {
        "exact6h": "6",
        "lead12": "12",
        "all_steps": "6/12",
    }
    steps = steps_by_subset[args.subset]
    parts = [item.strip() for item in args.parts.split(",") if item.strip()]
    unknown_parts = sorted(set(parts) - {"pl", "sfc"})
    if unknown_parts:
        raise ValueError(f"unknown parts: {unknown_parts}")

    pl_vars = [item.strip() for item in args.pl_vars.split(",") if item.strip()]
    unknown_pl = sorted(set(pl_vars) - set(PL_PARAM_CODES))
    if unknown_pl:
        raise ValueError(f"unknown pressure-level vars: {unknown_pl}")
    levels = [int(item) for item in args.pl_levels.split(",") if item.strip()]
    sfc_vars = [item.strip() for item in args.sfc_vars.split(",") if item.strip()]
    unknown_sfc = sorted(set(sfc_vars) - set(SFC_PARAM_CODES))
    if unknown_sfc:
        raise ValueError(f"unknown surface vars: {unknown_sfc}")

    output_root = Path(args.output_dir) / args.subset
    request_root = Path(args.request_dir) / args.subset
    tasks: list[Task] = []
    for init_day in date_range(init_start, init_end):
        for part in parts:
            request = request_base(init_day, steps=steps)
            if part == "pl":
                request["levtype"] = "pl"
                request["levelist"] = "/".join(str(item) for item in levels)
                request["param"] = "/".join(str(PL_PARAM_CODES[item]) for item in pl_vars)
            else:
                request["levtype"] = "sfc"
                request["param"] = "/".join(str(SFC_PARAM_CODES[item]) for item in sfc_vars)

            stem = f"era5_forecast_{part}_{args.subset}_init_{init_day:%Y%m%d}"
            request_path = request_root / part / f"{stem}.json"
            target_path = output_root / part / f"{stem}.grib"
            tasks.append(Task(part=part, init_date=init_day, request=request, request_path=request_path, target_path=target_path))
    return tasks


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_ready(path: Path) -> bool:
    ok_path = Path(str(path) + ".ok")
    if ok_path.exists() and path.exists() and path.stat().st_size > 0:
        return True
    if path.exists() and path.stat().st_size > 0:
        write_json(ok_path, {"completed_at_utc": utc_now(), "bytes": path.stat().st_size, "note": "ok marker reconstructed"})
        return True
    return False


def download_task(client: object, task: Task, args: argparse.Namespace, manifest_path: Path) -> bool:
    write_json(task.request_path, task.request)
    if file_ready(task.target_path):
        append_jsonl(
            manifest_path,
            {
                "time_utc": utc_now(),
                "status": "skip_existing",
                "account_label": args.account_label,
                "part": task.part,
                "init_date": task.init_date.isoformat(),
                "target": str(task.target_path),
            },
        )
        return True

    task.target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(task.target_path) + ".part")
    failed_path = Path(str(task.target_path) + ".failed.json")
    if tmp_path.exists():
        tmp_path.unlink()

    for attempt in range(1, args.max_retries + 1):
        append_jsonl(
            manifest_path,
            {
                "time_utc": utc_now(),
                "status": "started",
                "account_label": args.account_label,
                "attempt": attempt,
                "part": task.part,
                "init_date": task.init_date.isoformat(),
                "target": str(task.target_path),
            },
        )
        try:
            client.retrieve(DATASET, task.request, str(tmp_path))
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"download produced an empty file: {tmp_path}")
            os.replace(tmp_path, task.target_path)
            ok_payload = {
                "completed_at_utc": utc_now(),
                "bytes": task.target_path.stat().st_size,
                "dataset": DATASET,
                "part": task.part,
                "init_date": task.init_date.isoformat(),
                "request_path": str(task.request_path),
            }
            write_json(Path(str(task.target_path) + ".ok"), ok_payload)
            if failed_path.exists():
                failed_path.unlink()
            append_jsonl(
                manifest_path,
                {
                "time_utc": utc_now(),
                "status": "completed",
                "account_label": args.account_label,
                "attempt": attempt,
                "part": task.part,
                    "init_date": task.init_date.isoformat(),
                    "target": str(task.target_path),
                    "bytes": task.target_path.stat().st_size,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001 - long-running job must log and continue retrying.
            failed_payload = {
                "time_utc": utc_now(),
                "status": "failed_attempt",
                "account_label": args.account_label,
                "attempt": attempt,
                "part": task.part,
                "init_date": task.init_date.isoformat(),
                "target": str(task.target_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_json(failed_path, failed_payload)
            append_jsonl(manifest_path, failed_payload)
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep_seconds * attempt)

    append_jsonl(
        manifest_path,
        {
            "time_utc": utc_now(),
            "status": "gave_up",
            "account_label": args.account_label,
            "part": task.part,
            "init_date": task.init_date.isoformat(),
            "target": str(task.target_path),
        },
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(os.environ.get("MULTISAT_SERVER_WORKSPACE", "."))
    parser.add_argument("--start-date", default="2024-05-01", help="First valid date to cover.")
    parser.add_argument("--end-date", default="2024-10-31", help="Last valid date to cover.")
    parser.add_argument("--init-start-date", default="", help="Optional first forecast initialization date. Overrides start-date - 1 day.")
    parser.add_argument("--init-end-date", default="", help="Optional last forecast initialization date. Overrides end-date.")
    parser.add_argument("--subset", choices=["exact6h", "lead12", "all_steps"], default="exact6h")
    parser.add_argument("--parts", default="pl,sfc", help="Comma-separated: pl,sfc.")
    parser.add_argument("--pl-vars", default="z,t,u,v,q")
    parser.add_argument("--pl-levels", default=",".join(str(item) for item in DEFAULT_LEVELS))
    parser.add_argument("--sfc-vars", default="sp,msl,u10,v10,t2m,d2m")
    parser.add_argument("--output-dir", default=str(workspace / "data/raw/era5_gate1_forecast"))
    parser.add_argument("--request-dir", default=str(workspace / "data/interim/era5_requests_forecast"))
    parser.add_argument("--manifest", default=str(workspace / "logs/era5_forecast_batch_manifest.jsonl"))
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=int, default=300)
    parser.add_argument("--pause-seconds", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=0, help="Optional cap for smoke/burn-in runs. 0 means no cap.")
    parser.add_argument("--cdsapi-rc", default="", help="Optional .cdsapirc path. Used through CDSAPI_RC.")
    parser.add_argument("--account-label", default="", help="Non-secret account label written to logs/manifests.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    manifest_path = Path(args.manifest)
    print(
        json.dumps(
            {
                "time_utc": utc_now(),
                "event": "job_start",
                "account_label": args.account_label,
                "task_count": len(tasks),
                "subset": args.subset,
                "parts": args.parts,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "init_start_date": args.init_start_date,
                "init_end_date": args.init_end_date,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    append_jsonl(
        manifest_path,
        {
            "time_utc": utc_now(),
            "event": "job_start",
            "account_label": args.account_label,
            "task_count": len(tasks),
            "subset": args.subset,
            "parts": args.parts,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "init_start_date": args.init_start_date,
            "init_end_date": args.init_end_date,
            "dry_run": args.dry_run,
        },
    )
    for task in tasks:
        write_json(task.request_path, task.request)
    if args.dry_run:
        print(f"Prepared {len(tasks)} request files. Manifest: {manifest_path}", flush=True)
        return

    import cdsapi

    if args.cdsapi_rc:
        os.environ["CDSAPI_RC"] = str(Path(args.cdsapi_rc).expanduser())
    client = cdsapi.Client()
    failures = 0
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.part} init={task.init_date} target={task.target_path}", flush=True)
        ok = download_task(client, task, args, manifest_path)
        if not ok:
            failures += 1
        if args.pause_seconds > 0 and index < len(tasks):
            time.sleep(args.pause_seconds)
    append_jsonl(
        manifest_path,
        {
            "time_utc": utc_now(),
            "event": "job_end",
            "task_count": len(tasks),
            "failures": failures,
        },
    )
    print(json.dumps({"event": "job_end", "failures": failures, "task_count": len(tasks)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
