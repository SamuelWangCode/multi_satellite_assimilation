"""Build lightweight file indexes for Gate 1 Fengyun satellite data.

The script only inspects filenames and file sizes. It does not read large
satellite payloads, so it is safe to run on local downloaded archives.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


FY_DATA_ROOT = Path(os.environ.get("MULTISAT_FY_DATA_ROOT", "data/raw/fy"))

DEFAULT_ROOTS = {
    "fy4b_agri": str(FY_DATA_ROOT / "FY4B_AGRI_4KM"),
    "fy4b_clm": str(FY_DATA_ROOT / "FY4B_CLM"),
    "fy3e_mwts_orba": str(FY_DATA_ROOT / "FY3E_MWTS_L1C_ORBA"),
    "fy3e_mwts_orbd": str(FY_DATA_ROOT / "FY3E_MWTS_L1C_ORBD"),
    "fy3e_mwhs_orba": str(FY_DATA_ROOT / "FY3E_MWHS_L1C_ORBA"),
    "fy3e_mwhs_orbd": str(FY_DATA_ROOT / "FY3E_MWHS_L1C_ORBD"),
}

FY4B_TIME_RE = re.compile(r"_(\d{14})_(\d{14})_")
FY3E_TIME_RE = re.compile(
    r"FY3E_(MWHS|MWTS)-_(ORB[AD])_.*?_(\d{8})_(\d{4})_([0-9A-Z]+)_V",
    re.IGNORECASE,
)
RESOLUTION_RE = re.compile(r"_([0-9]{3,5}M|[0-9]{2,3}KM)_V", re.IGNORECASE)


@dataclass(frozen=True)
class FileRecord:
    product: str
    satellite: str
    sensor: str
    orbit_direction: str
    order_id: str
    path: str
    filename: str
    size_bytes: int
    start_time: datetime
    end_time: datetime | None
    center_time: datetime
    resolution: str

    @property
    def start_time_iso(self) -> str:
        return format_dt(self.start_time)

    @property
    def end_time_iso(self) -> str:
        return format_dt(self.end_time) if self.end_time else ""

    @property
    def center_time_iso(self) -> str:
        return format_dt(self.center_time)

    @property
    def is_main_synoptic(self) -> bool:
        return self.start_time.minute == 0 and self.start_time.hour in {0, 6, 12, 18}


def parse_utc_14(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def parse_utc_12(date_value: str, time_value: str) -> datetime:
    return datetime.strptime(date_value + time_value, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def format_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def order_id_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    return rel.parts[0] if rel.parts and rel.parts[0].startswith("A") else ""


def parse_record(path: Path, root: Path, product_key: str) -> FileRecord | None:
    name = path.name
    size = path.stat().st_size
    resolution_match = RESOLUTION_RE.search(name)
    resolution = resolution_match.group(1).upper() if resolution_match else ""
    order_id = order_id_from_path(path, root)

    if product_key.startswith("fy4b"):
        match = FY4B_TIME_RE.search(name)
        if not match:
            return None
        start_time = parse_utc_14(match.group(1))
        end_time = parse_utc_14(match.group(2))
        center_time = start_time + (end_time - start_time) / 2
        sensor = "AGRI"
        product = "FY4B_AGRI_L1_FDI" if product_key == "fy4b_agri" else "FY4B_AGRI_L2_CLM"
        return FileRecord(
            product=product,
            satellite="FY4B",
            sensor=sensor,
            orbit_direction="GEO",
            order_id=order_id,
            path=str(path),
            filename=name,
            size_bytes=size,
            start_time=start_time,
            end_time=end_time,
            center_time=center_time,
            resolution=resolution,
        )

    match = FY3E_TIME_RE.search(name)
    if not match:
        return None
    sensor = match.group(1).upper()
    orbit_direction = match.group(2).upper()
    start_time = parse_utc_12(match.group(3), match.group(4))
    return FileRecord(
        product=f"FY3E_{sensor}_L1C",
        satellite="FY3E",
        sensor=sensor,
        orbit_direction=orbit_direction,
        order_id=order_id,
        path=str(path),
        filename=name,
        size_bytes=size,
        start_time=start_time,
        end_time=None,
        center_time=start_time,
        resolution=resolution,
    )


def scan_root(product_key: str, root: Path) -> list[FileRecord]:
    if not root.exists():
        print(f"[WARN] missing root for {product_key}: {root}")
        return []

    records: list[FileRecord] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        record = parse_record(path, root, product_key)
        if record:
            records.append(record)
    records.sort(key=lambda item: (item.start_time, item.path))
    return records


def write_index(path: Path, records: Iterable[FileRecord]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "product",
        "satellite",
        "sensor",
        "orbit_direction",
        "order_id",
        "start_time_utc",
        "end_time_utc",
        "center_time_utc",
        "hour_utc",
        "minute_utc",
        "is_main_synoptic",
        "resolution",
        "size_bytes",
        "filename",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "product": record.product,
                    "satellite": record.satellite,
                    "sensor": record.sensor,
                    "orbit_direction": record.orbit_direction,
                    "order_id": record.order_id,
                    "start_time_utc": record.start_time_iso,
                    "end_time_utc": record.end_time_iso,
                    "center_time_utc": record.center_time_iso,
                    "hour_utc": record.start_time.hour,
                    "minute_utc": record.start_time.minute,
                    "is_main_synoptic": int(record.is_main_synoptic),
                    "resolution": record.resolution,
                    "size_bytes": record.size_bytes,
                    "filename": record.filename,
                    "path": record.path,
                }
            )


def iter_analysis_times(start_date: str, end_date: str, hours: Iterable[int]) -> list[datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    wanted_hours = sorted(set(hours))
    out: list[datetime] = []
    day = start
    while day <= end:
        for hour in wanted_hours:
            out.append(day.replace(hour=hour))
        day += timedelta(days=1)
    return out


def nearest_record(
    records: list[FileRecord], target: datetime, window_minutes: int
) -> tuple[FileRecord | None, float | None]:
    if not records:
        return None, None
    times = [item.start_time for item in records]
    pos = bisect.bisect_left(times, target)
    candidates = []
    if pos < len(records):
        candidates.append(records[pos])
    if pos > 0:
        candidates.append(records[pos - 1])
    best = min(candidates, key=lambda item: abs((item.start_time - target).total_seconds()))
    offset_minutes = (best.start_time - target).total_seconds() / 60.0
    if abs(offset_minutes) <= window_minutes:
        return best, offset_minutes
    return None, None


def count_in_window(records: list[FileRecord], target: datetime, window_minutes: int) -> list[FileRecord]:
    if not records:
        return []
    times = [item.start_time for item in records]
    left = bisect.bisect_left(times, target - timedelta(minutes=window_minutes))
    right = bisect.bisect_right(times, target + timedelta(minutes=window_minutes))
    return records[left:right]


def write_fy4b_matches(
    path: Path,
    analysis_times: list[datetime],
    agri_records: list[FileRecord],
    clm_records: list[FileRecord],
) -> None:
    fieldnames = [
        "analysis_time_utc",
        "agri_available",
        "agri_time_utc",
        "agri_offset_min",
        "agri_path",
        "clm_available",
        "clm_time_utc",
        "clm_offset_min",
        "clm_path",
        "both_available",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for target in analysis_times:
            agri, agri_offset = nearest_record(agri_records, target, 30)
            clm, clm_offset = nearest_record(clm_records, target, 30)
            writer.writerow(
                {
                    "analysis_time_utc": format_dt(target),
                    "agri_available": int(agri is not None),
                    "agri_time_utc": agri.start_time_iso if agri else "",
                    "agri_offset_min": f"{agri_offset:.1f}" if agri_offset is not None else "",
                    "agri_path": agri.path if agri else "",
                    "clm_available": int(clm is not None),
                    "clm_time_utc": clm.start_time_iso if clm else "",
                    "clm_offset_min": f"{clm_offset:.1f}" if clm_offset is not None else "",
                    "clm_path": clm.path if clm else "",
                    "both_available": int(agri is not None and clm is not None),
                }
            )


def summarize_orbits(records: list[FileRecord]) -> tuple[int, int]:
    orba = sum(1 for item in records if item.orbit_direction == "ORBA")
    orbd = sum(1 for item in records if item.orbit_direction == "ORBD")
    return orba, orbd


def write_fy3e_matches(
    path: Path,
    analysis_times: list[datetime],
    mwts_records: list[FileRecord],
    mwhs_records: list[FileRecord],
) -> None:
    fieldnames = [
        "analysis_time_utc",
        "mwts_primary_count_pm90",
        "mwts_orba_primary_count_pm90",
        "mwts_orbd_primary_count_pm90",
        "mwts_sensitivity_count_pm180",
        "mwts_nearest_time_utc_pm90",
        "mwts_nearest_offset_min_pm90",
        "mwhs_primary_count_pm90",
        "mwhs_orba_primary_count_pm90",
        "mwhs_orbd_primary_count_pm90",
        "mwhs_sensitivity_count_pm180",
        "mwhs_nearest_time_utc_pm90",
        "mwhs_nearest_offset_min_pm90",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for target in analysis_times:
            mwts90 = count_in_window(mwts_records, target, 90)
            mwts180 = count_in_window(mwts_records, target, 180)
            mwhs90 = count_in_window(mwhs_records, target, 90)
            mwhs180 = count_in_window(mwhs_records, target, 180)
            mwts_nearest, mwts_offset = nearest_record(mwts_records, target, 90)
            mwhs_nearest, mwhs_offset = nearest_record(mwhs_records, target, 90)
            mwts_orba, mwts_orbd = summarize_orbits(mwts90)
            mwhs_orba, mwhs_orbd = summarize_orbits(mwhs90)
            writer.writerow(
                {
                    "analysis_time_utc": format_dt(target),
                    "mwts_primary_count_pm90": len(mwts90),
                    "mwts_orba_primary_count_pm90": mwts_orba,
                    "mwts_orbd_primary_count_pm90": mwts_orbd,
                    "mwts_sensitivity_count_pm180": len(mwts180),
                    "mwts_nearest_time_utc_pm90": mwts_nearest.start_time_iso if mwts_nearest else "",
                    "mwts_nearest_offset_min_pm90": f"{mwts_offset:.1f}" if mwts_offset is not None else "",
                    "mwhs_primary_count_pm90": len(mwhs90),
                    "mwhs_orba_primary_count_pm90": mwhs_orba,
                    "mwhs_orbd_primary_count_pm90": mwhs_orbd,
                    "mwhs_sensitivity_count_pm180": len(mwhs180),
                    "mwhs_nearest_time_utc_pm90": mwhs_nearest.start_time_iso if mwhs_nearest else "",
                    "mwhs_nearest_offset_min_pm90": f"{mwhs_offset:.1f}" if mwhs_offset is not None else "",
                }
            )


def write_summary(path: Path, counts: dict[str, int], analysis_times: list[datetime]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["item", "count"])
        writer.writeheader()
        writer.writerow({"item": "analysis_times", "count": len(analysis_times)})
        for key in sorted(counts):
            writer.writerow({"item": key, "count": counts[key]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/interim/indexes")
    parser.add_argument("--start-date", default="2024-03-05")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--analysis-hours", default="0,6,12,18")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Override a data root as key=path. Keys: " + ", ".join(DEFAULT_ROOTS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = dict(DEFAULT_ROOTS)
    for item in args.root:
        if "=" not in item:
            raise ValueError(f"--root must use key=path format, got: {item}")
        key, value = item.split("=", 1)
        if key not in roots:
            raise ValueError(f"unknown root key: {key}")
        roots[key] = value

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hours = [int(item) for item in args.analysis_hours.split(",") if item.strip()]
    analysis_times = iter_analysis_times(args.start_date, args.end_date, hours)

    records_by_key = {key: scan_root(key, Path(root)) for key, root in roots.items()}

    agri = records_by_key["fy4b_agri"]
    clm = records_by_key["fy4b_clm"]
    mwts = records_by_key["fy3e_mwts_orba"] + records_by_key["fy3e_mwts_orbd"]
    mwhs = records_by_key["fy3e_mwhs_orba"] + records_by_key["fy3e_mwhs_orbd"]
    mwts.sort(key=lambda item: (item.start_time, item.path))
    mwhs.sort(key=lambda item: (item.start_time, item.path))
    all_records = agri + clm + mwts + mwhs
    all_records.sort(key=lambda item: (item.product, item.start_time, item.path))

    write_index(output_dir / "fy4b_agri_index.csv", agri)
    write_index(output_dir / "fy4b_clm_index.csv", clm)
    write_index(output_dir / "fy3e_mwts_index.csv", mwts)
    write_index(output_dir / "fy3e_mwhs_index.csv", mwhs)
    write_index(output_dir / "fy_file_index_gate1.csv", all_records)
    write_fy4b_matches(output_dir / "fy4b_agri_clm_matches.csv", analysis_times, agri, clm)
    write_fy3e_matches(output_dir / "fy3e_mwts_mwhs_matches.csv", analysis_times, mwts, mwhs)
    write_summary(
        output_dir / "fy_gate1_index_summary.csv",
        {
            "fy4b_agri": len(agri),
            "fy4b_clm": len(clm),
            "fy3e_mwts": len(mwts),
            "fy3e_mwhs": len(mwhs),
            "fy_all_gate1": len(all_records),
        },
        analysis_times,
    )

    print("Gate 1 FY index completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Analysis times: {len(analysis_times)}")
    print(f"FY4B AGRI: {len(agri)}")
    print(f"FY4B CLM: {len(clm)}")
    print(f"FY3E MWTS: {len(mwts)}")
    print(f"FY3E MWHS: {len(mwhs)}")


if __name__ == "__main__":
    main()
