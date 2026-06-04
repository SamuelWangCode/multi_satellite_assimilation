"""Download ERA5 reanalysis label batches for xa.

This is a resumable monthly downloader for ERA5 reanalysis fields used as
training labels. It writes one GRIB per month and vertical part, reconstructs
missing .ok markers for existing files, and supports a per-job CDSAPI rc file
so multiple accounts can work on disjoint month ranges.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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
    year: int
    month: int
    request: dict[str, str]
    request_path: Path
    target_path: Path


def parse_month(value: str) -> tuple[int, int]:
    dt = datetime.strptime(value, "%Y-%m")
    return dt.year, dt.month


def month_iter(start: tuple[int, int], end: tuple[int, int]) -> Iterable[tuple[int, int]]:
    year, month = start
    while (year, month) <= end:
        yield year, month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def month_range(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01/to/{year:04d}-{month:02d}-{last_day:02d}"


def request_base(year: int, month: int, levtype: str) -> dict[str, str]:
    return {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "an",
        "levtype": levtype,
        "date": month_range(year, month),
        "time": "00/06/12/18",
        "area": "60/70/0/150",
        "grid": "0.25/0.25",
    }


def build_tasks(args: argparse.Namespace) -> list[Task]:
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

    output_root = Path(args.output_dir) / "analysis"
    request_root = Path(args.request_dir) / "analysis"
    tasks: list[Task] = []
    for year, month in month_iter(parse_month(args.start_month), parse_month(args.end_month)):
        for part in parts:
            request = request_base(year, month, levtype=part)
            if part == "pl":
                request["levelist"] = "/".join(str(item) for item in levels)
                request["param"] = "/".join(str(PL_PARAM_CODES[item]) for item in pl_vars)
            else:
                request["param"] = "/".join(str(SFC_PARAM_CODES[item]) for item in sfc_vars)

            stem = f"era5_reanalysis_label_{part}_{year:04d}{month:02d}"
            request_path = request_root / part / f"{stem}.json"
            target_path = output_root / part / f"{stem}.grib"
            tasks.append(Task(part=part, year=year, month=month, request=request, request_path=request_path, target_path=target_path))
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
    task_label = f"{task.year:04d}-{task.month:02d}"
    if file_ready(task.target_path):
        append_jsonl(
            manifest_path,
            {
                "time_utc": utc_now(),
                "status": "skip_existing",
                "account_label": args.account_label,
                "part": task.part,
                "month": task_label,
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
                "month": task_label,
                "target": str(task.target_path),
            },
        )
        try:
            client.retrieve(DATASET, task.request, str(tmp_path))
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"download produced an empty file: {tmp_path}")
            os.replace(tmp_path, task.target_path)
            write_json(
                Path(str(task.target_path) + ".ok"),
                {
                    "completed_at_utc": utc_now(),
                    "bytes": task.target_path.stat().st_size,
                    "dataset": DATASET,
                    "part": task.part,
                    "month": task_label,
                    "request_path": str(task.request_path),
                },
            )
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
                    "month": task_label,
                    "target": str(task.target_path),
                    "bytes": task.target_path.stat().st_size,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            failed_payload = {
                "time_utc": utc_now(),
                "status": "failed_attempt",
                "account_label": args.account_label,
                "attempt": attempt,
                "part": task.part,
                "month": task_label,
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
            "month": task_label,
            "target": str(task.target_path),
        },
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(os.environ.get("MULTISAT_SERVER_WORKSPACE", "."))
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument("--parts", default="pl,sfc", help="Comma-separated: pl,sfc.")
    parser.add_argument("--pl-vars", default="z,t,u,v,q")
    parser.add_argument("--pl-levels", default=",".join(str(item) for item in DEFAULT_LEVELS))
    parser.add_argument("--sfc-vars", default="sp,msl,u10,v10,t2m,d2m")
    parser.add_argument("--output-dir", default=str(workspace / "data/raw/era5_reanalysis_label"))
    parser.add_argument("--request-dir", default=str(workspace / "data/interim/era5_requests_reanalysis_label"))
    parser.add_argument("--manifest", default=str(workspace / "logs/era5_reanalysis_label_manifest.jsonl"))
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=int, default=300)
    parser.add_argument("--pause-seconds", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--cdsapi-rc", default="")
    parser.add_argument("--account-label", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = build_tasks(args)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    manifest_path = Path(args.manifest)
    append_jsonl(
        manifest_path,
        {
            "time_utc": utc_now(),
            "event": "job_start",
            "account_label": args.account_label,
            "task_count": len(tasks),
            "parts": args.parts,
            "start_month": args.start_month,
            "end_month": args.end_month,
            "dry_run": args.dry_run,
        },
    )
    print(
        json.dumps(
            {
                "time_utc": utc_now(),
                "event": "job_start",
                "account_label": args.account_label,
                "task_count": len(tasks),
                "parts": args.parts,
                "start_month": args.start_month,
                "end_month": args.end_month,
                "dry_run": args.dry_run,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for task in tasks:
        write_json(task.request_path, task.request)
    if args.dry_run:
        return

    import cdsapi

    if args.cdsapi_rc:
        os.environ["CDSAPI_RC"] = str(Path(args.cdsapi_rc).expanduser())
    client = cdsapi.Client()
    failures = 0
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.part} month={task.year:04d}-{task.month:02d} target={task.target_path}", flush=True)
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
            "account_label": args.account_label,
            "task_count": len(tasks),
            "failures": failures,
        },
    )


if __name__ == "__main__":
    main()
