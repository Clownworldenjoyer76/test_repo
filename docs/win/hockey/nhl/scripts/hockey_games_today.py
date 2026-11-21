#!/usr/bin/env python3
"""Combine dated NHL market CSVs into one CSV per date.

Run from the repository root. All paths are relative to the working directory.
Dates without inputs are skipped, and games missing from a market are retained.
"""

import csv
import re
import sys
from pathlib import Path

SOURCE_DIR = Path("docs/win/hockey/nhl/02_juice")
OUTPUT_DIR = Path("frontend/data/index_home/hockey/NHL")
MARKETS = ("moneyline", "total", "puck_line")
FILE_PATTERN = re.compile(r"^(\d{4}_\d{2}_\d{2})_NHL_(moneyline|total|puck_line)\.csv$")
KEY = "game_id"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or KEY not in headers:
            raise ValueError(f"{path}: missing required '{KEY}' column")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        rows = {}
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: inconsistent column count")
            game_id = row[KEY].strip()
            if not game_id:
                raise ValueError(f"{path}:{line}: empty '{KEY}'")
            if game_id in rows:
                raise ValueError(f"{path}:{line}: duplicate game_id {game_id}")
            rows[game_id] = row
    return headers, rows


def combine_date(date: str, source_files: dict) -> tuple[list[str], list[dict[str, str]]]:
    headers = []
    games = {}

    for market in MARKETS:
        path = source_files.get(market)
        if path is None:
            continue
        market_headers, market_rows = read_csv(path)
        for header in market_headers:
            if header not in headers:
                headers.append(header)

        for game_id, incoming in market_rows.items():
            record = games.setdefault(game_id, {})
            for header, raw_value in incoming.items():
                value = raw_value or ""
                previous = record.get(header, "")
                if previous and value and previous != value:
                    raise ValueError(
                        f"{date} game_id={game_id}: conflicting values for "
                        f"'{header}' ({previous!r} vs {value!r}) in {path}"
                    )
                if value or header not in record:
                    record[header] = value

    return headers, list(games.values())


def main() -> int:
    if not SOURCE_DIR.is_dir():
        print(f"No source directory: {SOURCE_DIR}; nothing to do.")
        return 0

    dates = {}
    for path in SOURCE_DIR.iterdir():
        if not path.is_file():
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            date, market = match.groups()
            dates.setdefault(date, {})[market] = path

    if not dates:
        print(f"No NHL market CSVs found in {SOURCE_DIR}; nothing to do.")
        return 0

    written = 0
    for date in sorted(dates):
        headers, rows = combine_date(date, dates[date])
        if not rows:
            print(f"{date}: no game rows; skipped.")
            continue
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"{date}_NHL_games_today.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {target} ({len(rows)} games)")
        written += 1

    print(f"Created or updated {written} NHL games today CSV(s).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
