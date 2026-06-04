"""Prepare or execute ERA5 Gate 1 CDS/MARS requests.

By default this script is a dry run: it writes monthly JSON request files and
does not download data. Use --execute for actual CDS API retrieval after the
request manifest has been reviewed.
"""

from __future__ import annotations

import argparse
import calendar
import json
from datetime import date, datetime
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

PL_LEVELS = [1000, 925, 850, 700, 500, 300, 200]


def month_iter(start: date, end: date) -> Iterable[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def month_range(year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01/to/{year:04d}-{month:02d}-{last_day:02d}"


def mars_base(kind: str, levtype: str, year: int, month: int, date_range: str | None = None) -> dict[str, str]:
    dates = date_range if date_range else month_range(year, month)
    if kind == "analysis":
        return {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levtype": levtype,
            "date": dates,
            "time": "00/06/12/18",
            "area": "60/70/0/150",
            "grid": "0.25/0.25",
        }
    if kind == "forecast":
        return {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "fc",
            "levtype": levtype,
            "date": dates,
            "time": "06/18",
            "step": "6/12",
            "area": "60/70/0/150",
            "grid": "0.25/0.25",
        }
    raise ValueError(f"unsupported kind: {kind}")


def request_for(
    kind: str,
    levtype: str,
    year: int,
    month: int,
    date_range: str | None = None,
    pl_vars: list[str] | None = None,
    pl_levels: list[int] | None = None,
    sfc_vars: list[str] | None = None,
) -> dict[str, str]:
    request = mars_base(kind, levtype, year, month, date_range=date_range)
    if levtype == "pl":
        selected_levels = pl_levels if pl_levels else PL_LEVELS
        selected_vars = pl_vars if pl_vars else list(PL_PARAM_CODES)
        request["levelist"] = "/".join(str(item) for item in selected_levels)
        request["param"] = "/".join(str(PL_PARAM_CODES[item]) for item in selected_vars)
    elif levtype == "sfc":
        selected_vars = sfc_vars if sfc_vars else list(SFC_PARAM_CODES)
        request["param"] = "/".join(str(SFC_PARAM_CODES[item]) for item in selected_vars)
    else:
        raise ValueError(f"unsupported levtype: {levtype}")
    return request


def write_request_json(path: Path, request: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_request(request: dict[str, str], target: Path) -> None:
    import cdsapi

    target.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(DATASET, request, str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default="2024-03")
    parser.add_argument("--end-month", default="2025-12")
    parser.add_argument(
        "--date-range",
        default=None,
        help="Optional MARS date range, e.g. 2024-03-05 or 2024-03-05/to/2024-03-07. Overrides month dates while preserving monthly output names.",
    )
    parser.add_argument("--pl-vars", default=",".join(PL_PARAM_CODES), help="Comma-separated pressure-level variable names.")
    parser.add_argument("--pl-levels", default=",".join(str(item) for item in PL_LEVELS), help="Comma-separated pressure levels.")
    parser.add_argument("--sfc-vars", default=",".join(SFC_PARAM_CODES), help="Comma-separated surface variable names.")
    parser.add_argument("--output-dir", default="data/raw/era5_gate1")
    parser.add_argument("--request-dir", default="data/interim/era5_requests")
    parser.add_argument(
        "--kind",
        choices=["analysis", "forecast", "both"],
        default="both",
        help="Which ERA5 request family to prepare.",
    )
    parser.add_argument(
        "--levtype",
        choices=["pl", "sfc", "both"],
        default="both",
        help="Which vertical type to prepare.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually download through CDS API.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.strptime(args.start_month, "%Y-%m").date().replace(day=1)
    end = datetime.strptime(args.end_month, "%Y-%m").date().replace(day=1)
    kinds = ["analysis", "forecast"] if args.kind == "both" else [args.kind]
    levtypes = ["pl", "sfc"] if args.levtype == "both" else [args.levtype]
    pl_vars = [item.strip() for item in args.pl_vars.split(",") if item.strip()]
    unknown_pl = sorted(set(pl_vars) - set(PL_PARAM_CODES))
    if unknown_pl:
        raise ValueError(f"unknown pressure-level variables: {unknown_pl}")
    pl_levels = [int(item) for item in args.pl_levels.split(",") if item.strip()]
    sfc_vars = [item.strip() for item in args.sfc_vars.split(",") if item.strip()]
    unknown_sfc = sorted(set(sfc_vars) - set(SFC_PARAM_CODES))
    if unknown_sfc:
        raise ValueError(f"unknown surface variables: {unknown_sfc}")
    request_dir = Path(args.request_dir)
    output_dir = Path(args.output_dir)

    rows = []
    for year, month in month_iter(start, end):
        for kind in kinds:
            for levtype in levtypes:
                request = request_for(
                    kind,
                    levtype,
                    year,
                    month,
                    date_range=args.date_range,
                    pl_vars=pl_vars,
                    pl_levels=pl_levels,
                    sfc_vars=sfc_vars,
                )
                stem = f"era5_{kind}_{levtype}_{year:04d}{month:02d}"
                request_path = request_dir / f"{stem}.json"
                target_path = output_dir / kind / f"{stem}.grib"
                write_request_json(request_path, request)
                rows.append((kind, levtype, year, month, request_path, target_path))
                if args.execute:
                    execute_request(request, target_path)

    print("ERA5 Gate 1 request manifest prepared.")
    print(f"Dataset: {DATASET}")
    print(f"Requests: {len(rows)}")
    print(f"Request directory: {request_dir.resolve()}")
    print(f"Target directory: {output_dir.resolve()}")
    print(f"Execute: {args.execute}")


if __name__ == "__main__":
    main()
