"""Keep ERA5 Gate 1 forecast downloads moving on the server.

This script does not download data itself. It watches resumable batch-download
jobs, restarts a stopped incomplete job, and starts the lead-12h complement
after the exact-6h subset is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


PARTS = ("pl", "sfc")


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_utc": utc_now(), **payload}
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


def expected_tasks(start_date: str, end_date: str, parts: tuple[str, ...] = PARTS) -> int:
    start = parse_date(start_date) - timedelta(days=1)
    end = parse_date(end_date)
    days = (end - start).days + 1
    return days * len(parts)


def ok_count(workspace: Path, subset: str) -> int:
    root = workspace / "data/raw/era5_gate1_forecast" / subset
    return sum(1 for _ in root.glob("*/*.grib.ok"))


def failed_count(workspace: Path, subset: str) -> int:
    root = workspace / "data/raw/era5_gate1_forecast" / subset
    return sum(1 for _ in root.glob("*/*.failed.json"))


def latest_manifest_event(workspace: Path, subset: str) -> dict[str, object] | None:
    path = workspace / "logs" / f"era5_forecast_{subset}_202405_202410_manifest.jsonl"
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def pidfile(workspace: Path, subset: str) -> Path:
    return workspace / "logs" / f"era5_forecast_{subset}_202405_202410.pid"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def job_alive(workspace: Path, subset: str) -> bool:
    pid = read_pid(pidfile(workspace, subset))
    return bool(pid and pid_alive(pid))


def start_batch_job(workspace: Path, subset: str, args: argparse.Namespace, log_path: Path) -> int:
    stdout_path = workspace / "logs" / f"era5_forecast_{subset}_202405_202410.log"
    manifest_path = workspace / "logs" / f"era5_forecast_{subset}_202405_202410_manifest.jsonl"
    command = [
        str(workspace / "venvs/cdsapi/bin/python"),
        "tools/download_era5_forecast_batches.py",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--subset",
        subset,
        "--parts",
        "pl,sfc",
        "--manifest",
        str(manifest_path),
        "--max-retries",
        str(args.max_retries),
        "--retry-sleep-seconds",
        str(args.retry_sleep_seconds),
        "--pause-seconds",
        str(args.pause_seconds),
    ]
    env = os.environ.copy()
    env["CDSAPI_RC"] = str(workspace / "secure/cdsapirc")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_path.open("ab")
    proc = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pidfile(workspace, subset).write_text(f"{proc.pid}\n", encoding="utf-8")
    append_log(log_path, {"event": "started_batch_job", "subset": subset, "pid": proc.pid, "stdout": str(stdout_path)})
    return proc.pid


def ensure_job(workspace: Path, subset: str, args: argparse.Namespace, expected: int, log_path: Path) -> None:
    count = ok_count(workspace, subset)
    alive = job_alive(workspace, subset)
    append_log(
        log_path,
        {
            "event": "subset_status",
            "subset": subset,
            "ok": count,
            "expected": expected,
            "failed": failed_count(workspace, subset),
            "alive": alive,
            "latest_event": latest_manifest_event(workspace, subset),
        },
    )
    if count >= expected:
        return
    if not alive:
        start_batch_job(workspace, subset, args, log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_workspace = os.environ.get("MULTISAT_SERVER_WORKSPACE", ".")
    parser.add_argument("--workspace", default=default_workspace)
    parser.add_argument("--start-date", default="2024-05-01")
    parser.add_argument("--end-date", default="2024-10-31")
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-sleep-seconds", type=int, default=300)
    parser.add_argument("--pause-seconds", type=int, default=20)
    parser.add_argument(
        "--log",
        default=str(Path(default_workspace) / "logs/era5_download_orchestrator_202405_202410.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    log_path = Path(args.log)
    expected = expected_tasks(args.start_date, args.end_date)
    append_log(
        log_path,
        {
            "event": "orchestrator_start",
            "workspace": str(workspace),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "expected_per_subset": expected,
        },
    )

    lead12_started = ok_count(workspace, "lead12") > 0 or job_alive(workspace, "lead12")
    while True:
        exact_ok = ok_count(workspace, "exact6h")
        ensure_job(workspace, "exact6h", args, expected, log_path)
        if exact_ok >= expected:
            append_log(log_path, {"event": "exact6h_complete", "ok": exact_ok, "expected": expected})
            if not lead12_started:
                start_batch_job(workspace, "lead12", args, log_path)
                lead12_started = True
            ensure_job(workspace, "lead12", args, expected, log_path)
            lead12_ok = ok_count(workspace, "lead12")
            if lead12_ok >= expected:
                ready = workspace / "logs/era5_forecast_202405_202410_all_required.ready.json"
                ready.write_text(
                    json.dumps(
                        {
                            "time_utc": utc_now(),
                            "exact6h_ok": exact_ok,
                            "lead12_ok": lead12_ok,
                            "expected_per_subset": expected,
                            "message": "All required Gate 1 ERA5 forecast subsets are downloaded.",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                append_log(log_path, {"event": "all_required_complete", "ready_marker": str(ready)})
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
