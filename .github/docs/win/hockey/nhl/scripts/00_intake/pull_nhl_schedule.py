#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/pull_nhl_schedule.py

from __future__ import annotations

import csv
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


BASE_DIR = Path("docs/win/hockey/nhl")
SPORTSBOOK_DIR = BASE_DIR / "00_intake" / "sportsbook"
PREDICTIONS_DIR = BASE_DIR / "00_intake" / "predictions"
NHL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "pull_nhl_schedule.txt"

ET = ZoneInfo("America/New_York")
NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"
REQUEST_TIMEOUT = 30
REQUEST_ATTEMPTS = 3

OUTPUT_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_team_abbrev",
    "away_team_abbrev",
    "home_team_id",
    "away_team_id",
    "season",
    "game_type",
    "game_state",
    "schedule_state",
    "start_time_utc",
]

NHL_SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_FILE.write_text(
        f"=== pull_nhl_schedule RUN {datetime.now(ET).isoformat()} ===\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {message}\n")


def normalize_date(value: str) -> str:
    text = str(value).strip().replace("-", "_").replace("/", "_")

    for fmt in ("%Y_%m_%d", "%m_%d_%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y_%m_%d")
        except ValueError:
            continue

    return ""


def date_from_sportsbook_filename(path: Path) -> str:
    name = path.stem
    if not name.startswith("NHL_"):
        return ""
    return normalize_date(name[len("NHL_") :])


def date_from_prediction_filename(path: Path) -> str:
    name = path.stem
    if not name.startswith("hockey_"):
        return ""
    return normalize_date(name[len("hockey_") :])


def discover_target_dates() -> list[str]:
    dates: set[str] = set()

    if SPORTSBOOK_DIR.exists():
        for path in SPORTSBOOK_DIR.glob("NHL_*.csv"):
            date_value = date_from_sportsbook_filename(path)
            if date_value:
                dates.add(date_value)

    if PREDICTIONS_DIR.exists():
        for path in PREDICTIONS_DIR.glob("hockey_*.csv"):
            date_value = date_from_prediction_filename(path)
            if date_value:
                dates.add(date_value)

    dates.add(datetime.now(ET).strftime("%Y_%m_%d"))

    return sorted(dates)


def get_localized_text(value) -> str:
    if isinstance(value, str):
        return value.strip()

    if not isinstance(value, dict):
        return ""

    for key in ("default", "en"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()

    for text in value.values():
        if isinstance(text, str) and text.strip():
            return text.strip()

    return ""


def team_full_name(team: dict) -> str:
    if not isinstance(team, dict):
        return ""

    direct_name = get_localized_text(team.get("name"))
    if direct_name:
        return direct_name

    place_name = get_localized_text(team.get("placeName"))
    common_name = get_localized_text(team.get("commonName"))

    if place_name and common_name:
        if common_name.lower().startswith(place_name.lower()):
            return common_name
        return f"{place_name} {common_name}".strip()

    return place_name or common_name


def to_et_time(start_time_utc: str) -> str:
    if not start_time_utc:
        return ""

    try:
        dt = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
        return dt.astimezone(ET).strftime("%H:%M")
    except (TypeError, ValueError, OverflowError):
        return ""


def request_schedule(session: requests.Session, date_value: str) -> dict:
    api_date = date_value.replace("_", "-")
    url = NHL_SCHEDULE_URL.format(date=api_date)
    last_error = None

    for attempt in range(1, REQUEST_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict):
                raise ValueError("NHL schedule response was not a JSON object")

            return payload

        except Exception as exc:
            last_error = exc
            log(
                f"WARNING request failed date={date_value} "
                f"attempt={attempt}/{REQUEST_ATTEMPTS}: {exc}"
            )

            if attempt < REQUEST_ATTEMPTS:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Failed to retrieve NHL schedule for {date_value}: {last_error}"
    )


def games_for_target_date(payload: dict, target_date: str) -> list[dict]:
    target_iso = target_date.replace("_", "-")
    games: list[dict] = []

    game_week = payload.get("gameWeek", [])
    if isinstance(game_week, list):
        for day in game_week:
            if not isinstance(day, dict):
                continue

            day_date = str(day.get("date", "")).strip()
            day_games = day.get("games", [])

            if day_date != target_iso or not isinstance(day_games, list):
                continue

            games.extend(game for game in day_games if isinstance(game, dict))

    if games:
        return games

    top_level_games = payload.get("games", [])
    if isinstance(top_level_games, list):
        for game in top_level_games:
            if not isinstance(game, dict):
                continue

            game_date = str(game.get("gameDate", "")).strip()
            if game_date == target_iso:
                games.append(game)

    return games


def build_schedule_row(game: dict, target_date: str) -> dict[str, str]:
    home_team = game.get("homeTeam", {})
    away_team = game.get("awayTeam", {})

    if not isinstance(home_team, dict):
        home_team = {}
    if not isinstance(away_team, dict):
        away_team = {}

    game_id = str(game.get("id", "")).strip()
    start_time_utc = str(game.get("startTimeUTC", "")).strip()
    game_date = normalize_date(str(game.get("gameDate", ""))) or target_date

    return {
        "game_id": game_id,
        "sport": "hockey",
        "league": "nhl",
        "game_date": game_date,
        "game_time": to_et_time(start_time_utc),
        "home_team": team_full_name(home_team),
        "away_team": team_full_name(away_team),
        "home_team_abbrev": str(home_team.get("abbrev", "")).strip(),
        "away_team_abbrev": str(away_team.get("abbrev", "")).strip(),
        "home_team_id": str(home_team.get("id", "")).strip(),
        "away_team_id": str(away_team.get("id", "")).strip(),
        "season": str(game.get("season", "")).strip(),
        "game_type": str(game.get("gameType", "")).strip(),
        "game_state": str(game.get("gameState", "")).strip(),
        "schedule_state": str(game.get("gameScheduleState", "")).strip(),
        "start_time_utc": start_time_utc,
    }


def validate_rows(rows: list[dict[str, str]], date_value: str) -> None:
    seen_game_ids: set[str] = set()
    seen_matchups: set[tuple[str, str, str]] = set()

    for row_number, row in enumerate(rows, start=2):
        required = ["game_id", "game_date", "home_team", "away_team"]
        missing = [field for field in required if not str(row.get(field, "")).strip()]

        if missing:
            raise ValueError(
                f"NHL schedule {date_value} row {row_number} missing: {missing}"
            )

        game_id = row["game_id"]
        if game_id in seen_game_ids:
            raise ValueError(
                f"NHL schedule {date_value} contains duplicate game_id={game_id}"
            )
        seen_game_ids.add(game_id)

        matchup_key = (
            row["game_date"],
            row["home_team"].strip().lower(),
            row["away_team"].strip().lower(),
        )
        if matchup_key in seen_matchups:
            raise ValueError(
                f"NHL schedule {date_value} contains duplicate matchup={matchup_key}"
            )
        seen_matchups.add(matchup_key)


def write_schedule_file(date_value: str, rows: list[dict[str, str]]) -> Path:
    output_path = NHL_SCHEDULE_DIR / f"NHL_{date_value}.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    reset_log()

    target_dates = discover_target_dates()
    log(f"Target dates discovered: {len(target_dates)}")
    for date_value in target_dates:
        log(f"Target date: {date_value}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "NHL_for_mat/1.0",
            "Accept": "application/json",
        }
    )

    files_written = 0
    games_written = 0

    try:
        for date_value in target_dates:
            payload = request_schedule(session, date_value)
            games = games_for_target_date(payload, date_value)
            rows = [build_schedule_row(game, date_value) for game in games]

            rows.sort(key=lambda row: (row["game_time"], row["game_id"]))
            validate_rows(rows, date_value)

            output_path = write_schedule_file(date_value, rows)
            files_written += 1
            games_written += len(rows)

            log(
                f"WROTE {output_path} rows={len(rows)} "
                "using official NHL game_id values"
            )

        log("--- SUMMARY ---")
        log(f"Schedule files written: {files_written}")
        log(f"Official NHL games written: {games_written}")
        log("STATUS: SUCCESS")
        print("NHL official schedule pull complete.")

    except Exception as exc:
        log(f"FATAL ERROR: {exc}")
        log(traceback.format_exc())
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()