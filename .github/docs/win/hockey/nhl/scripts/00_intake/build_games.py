#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/build_games.py

from __future__ import annotations

import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path


RECONCILED_DIR = Path("docs/win/hockey/nhl/00_intake/reconciled")
GAMES_DIR = Path("docs/win/hockey/nhl/00_intake/games")
LOG_PATH = Path("docs/win/hockey/nhl/errors/00_intake/build_games.txt")

INPUT_SUFFIX = ".csv"
OUTPUT_SUFFIX = "_nhl_games.csv"

REQUIRED_COLUMNS = [
    "game_id",
    "sportsbook_event_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

OUTPUT_COLUMNS = [
    "game_id",
    "sportsbook_event_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_log(lines: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_log(lines: list[str]) -> None:
    print("\n".join(lines), file=sys.stderr)


def fail(lines: list[str], message: str) -> None:
    lines.append(f"ERROR: {message}")
    lines.append(f"Finished: {now_stamp()}")
    write_log(lines)
    print_log(lines)
    raise SystemExit(1)


def is_valid_input_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != INPUT_SUFFIX:
        return False

    return path.name.startswith("NHL_")


def extract_date_from_input_name(path: Path) -> str:
    name = path.name

    if not name.startswith("NHL_") or not name.endswith(INPUT_SUFFIX):
        raise ValueError(f"Invalid reconciled filename format: {name}")

    date_value = name[len("NHL_") : -len(INPUT_SUFFIX)]

    if not date_value:
        raise ValueError(f"Missing date in reconciled filename: {name}")

    return date_value


def read_reconciled_file(
    path: Path,
    log_lines: list[str],
) -> tuple[str, list[dict[str, str]]]:
    file_date = extract_date_from_input_name(path)

    with path.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            fail(log_lines, f"{path} has no header row.")

        missing_columns = [
            col
            for col in REQUIRED_COLUMNS
            if col not in reader.fieldnames
        ]

        if missing_columns:
            fail(
                log_lines,
                f"{path} missing required columns: "
                f"{', '.join(missing_columns)}",
            )

        rows: list[dict[str, str]] = []
        game_dates_seen: set[str] = set()
        game_ids_seen: set[str] = set()
        sportsbook_ids_seen: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            output_row = {
                col: (row.get(col) or "").strip()
                for col in OUTPUT_COLUMNS
            }

            missing_values = [
                col
                for col in OUTPUT_COLUMNS
                if col != "sportsbook_event_id"
                and output_row[col] == ""
            ]

            if missing_values:
                fail(
                    log_lines,
                    f"{path} row {row_number} missing values for: "
                    f"{', '.join(missing_values)}",
                )

            game_id = output_row["game_id"]
            sportsbook_event_id = output_row["sportsbook_event_id"]

            if game_id in game_ids_seen:
                fail(
                    log_lines,
                    f"{path} has duplicate official game_id: {game_id}",
                )

            game_ids_seen.add(game_id)

            if (
                sportsbook_event_id
                and sportsbook_event_id in sportsbook_ids_seen
            ):
                fail(
                    log_lines,
                    f"{path} has duplicate sportsbook_event_id: "
                    f"{sportsbook_event_id}",
                )

            if sportsbook_event_id:
                sportsbook_ids_seen.add(sportsbook_event_id)

            game_dates_seen.add(output_row["game_date"])
            rows.append(output_row)

    if not rows:
        fail(
            log_lines,
            f"{path} contains no reconciled game rows.",
        )

    if len(game_dates_seen) != 1:
        fail(
            log_lines,
            f"{path} contains multiple game_date values: "
            f"{', '.join(sorted(game_dates_seen))}",
        )

    game_date = next(iter(game_dates_seen))

    if game_date != file_date:
        fail(
            log_lines,
            f"{path} filename date {file_date} does not match "
            f"game_date column {game_date}.",
        )

    return game_date, rows


def write_games_file(
    game_date: str,
    rows: list[dict[str, str]],
) -> Path:
    GAMES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = GAMES_DIR / f"{game_date}{OUTPUT_SUFFIX}"

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    log_lines = [
        "NHL build_games.py summary",
        f"Started: {now_stamp()}",
        f"Input directory: {RECONCILED_DIR}",
        f"Output directory: {GAMES_DIR}",
        f"Log path: {LOG_PATH}",
        "Canonical game_id source: official NHL schedule via "
        "reconcile_game_ids.py",
        "Historical games files are retained independently of the "
        "currently available reconciled/sportsbook input files.",
        "",
    ]

    if not RECONCILED_DIR.exists():
        fail(
            log_lines,
            f"Reconciled directory does not exist: {RECONCILED_DIR}",
        )

    reconciled_files = sorted(
        path
        for path in RECONCILED_DIR.iterdir()
        if is_valid_input_file(path)
    )

    if not reconciled_files:
        fail(
            log_lines,
            f"No reconciled NHL files found in {RECONCILED_DIR}",
        )

    written_files: list[Path] = []
    total_rows = 0

    for reconciled_path in reconciled_files:
        log_lines.append(
            f"Processing reconciled file: {reconciled_path}"
        )

        game_date, rows = read_reconciled_file(
            reconciled_path,
            log_lines,
        )

        total_rows += len(rows)

        output_path = write_games_file(
            game_date,
            rows,
        )

        written_files.append(output_path)

        log_lines.append(f"Date: {game_date}")
        log_lines.append(f"Rows written: {len(rows)}")
        log_lines.append(f"Output file: {output_path}")
        log_lines.append("")

    log_lines.extend(
        [
            "Completed successfully.",
            f"Reconciled files processed: {len(reconciled_files)}",
            f"Games files written: {len(written_files)}",
            f"Total official NHL game rows written: {total_rows}",
            "Historical games files deleted: 0",
            f"Finished: {now_stamp()}",
        ]
    )

    write_log(log_lines)
    print_log(log_lines)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        error_lines = [
            "NHL build_games.py summary",
            f"Started: {now_stamp()}",
            f"Input directory: {RECONCILED_DIR}",
            f"Output directory: {GAMES_DIR}",
            f"Log path: {LOG_PATH}",
            "Canonical game_id source: official NHL schedule via "
            "reconcile_game_ids.py",
            "",
            f"ERROR: Unhandled exception: {exc}",
            "",
            "TRACEBACK:",
            traceback.format_exc(),
            f"Finished: {now_stamp()}",
        ]
        write_log(error_lines)
        print_log(error_lines)
        raise SystemExit(1)