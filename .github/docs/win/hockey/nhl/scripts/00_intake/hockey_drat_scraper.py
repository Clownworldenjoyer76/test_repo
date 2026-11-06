# docs/win/hockey/nhl/scripts/00_intake/hockey_drat_scraper.py

import csv
import sys
import json
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from team_map_common import (
    normalize_team_alias_key,
    parse_nhl_team_map_row,
    register_team_identity,
)
from datetime import datetime

import pandas as pd
import pytz
from playwright.sync_api import sync_playwright


URLS = {
    "nhl": "https://www.dratings.com/predictor/nhl-hockey-predictions/",
}

UTC = pytz.utc
ET = pytz.timezone("America/New_York")

BASE_DIR = Path("docs/win/hockey/nhl")
NHL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"

ERROR_DIR = BASE_DIR / "errors" / "00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "hockey_drat_scraper.txt"

EXPECTED_GAME_TYPES = {"2", "3"}

with open(LOG_FILE, "w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== hockey_drat_scraper RUN {datetime.now(ET).isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {msg}\n")


def convert_utc_to_et(date_time_str: str) -> str:
    try:
        dt = datetime.strptime(date_time_str.strip(), "%m/%d/%Y %I:%M %p")
        dt_utc = UTC.localize(dt)
        dt_et = dt_utc.astimezone(ET)
        return dt_et.strftime("%m/%d/%Y %I:%M %p")
    except (TypeError, ValueError, OverflowError):
        return date_time_str


def parse_et_datetime(date_time_str: str) -> datetime:
    dt = datetime.strptime(str(date_time_str).strip(), "%m/%d/%Y %I:%M %p")
    return ET.localize(dt)


def load_team_map() -> dict:
    if not TEAM_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing team mapping file: {TEAM_MAP_PATH}"
        )

    by_source: dict[str, dict[str, dict[str, str]]] = {}
    by_id: dict[str, dict[str, str]] = {}

    with TEAM_MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "league",
            "source",
            "alias",
            "canonical_team",
            "nhl_team_id",
            "nhl_abbrev",
        }
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)

        if missing:
            raise ValueError(
                f"{TEAM_MAP_PATH} missing required columns: {missing}"
            )

        for row_number, row in enumerate(reader, start=2):
            parsed = parse_nhl_team_map_row(
                row,
                row_number,
                TEAM_MAP_PATH,
            )
            if parsed is None:
                continue

            (
                source,
                alias,
                team_id,
                _abbrev,
                identity,
            ) = parsed

            register_team_identity(
                by_id,
                team_id,
                identity,
                TEAM_MAP_PATH,
            )

            key = normalize_team_alias_key(alias)
            source_map = by_source.setdefault(
                source,
                {},
            )
            prior = source_map.get(key)

            if prior is not None and prior != identity:
                raise ValueError(
                    f"{TEAM_MAP_PATH} has conflicting mapping for "
                    f"source={source} alias={alias!r}: "
                    f"{prior} != {identity}"
                )

            source_map[key] = identity

    if not by_source.get("dratings"):
        raise ValueError(
            f"No dratings mappings loaded from {TEAM_MAP_PATH}"
        )

    if not by_source.get("official_nhl"):
        raise ValueError(
            f"No official_nhl mappings loaded from {TEAM_MAP_PATH}"
        )

    if len(by_id) != 32:
        raise ValueError(
            f"Expected 32 stable NHL team IDs in {TEAM_MAP_PATH}; "
            f"found {len(by_id)}"
        )

    return {
        "by_source": by_source,
        "by_id": by_id,
    }


def resolve_team_identity(
    value: str,
    source: str,
    team_map: dict,
) -> dict[str, str] | None:
    key = normalize_team_alias_key(value)

    for candidate_source in (source, "shared", "official_nhl"):
        identity = (
            team_map["by_source"]
            .get(candidate_source, {})
            .get(key)
        )
        if identity is not None:
            return identity

    return None


def matchup_key_from_dratings(
    game: dict,
    team_map: dict,
) -> tuple[str, str]:
    team1 = resolve_team_identity(
        game.get("team1", ""),
        "dratings",
        team_map,
    )
    team2 = resolve_team_identity(
        game.get("team2", ""),
        "dratings",
        team_map,
    )

    if team1 is None or team2 is None:
        raise ValueError(
            "Unmapped D-Ratings team identity: "
            f"team1={game.get('team1', '')!r} "
            f"team2={game.get('team2', '')!r}"
        )

    return tuple(sorted((team1["nhl_team_id"], team2["nhl_team_id"])))


def matchup_key_from_schedule(
    row: dict[str, str],
    team_map: dict,
) -> tuple[str, str]:
    home_id = str(row.get("home_team_id", "")).strip()
    away_id = str(row.get("away_team_id", "")).strip()

    if home_id not in team_map["by_id"] or away_id not in team_map["by_id"]:
        raise ValueError(
            "Official schedule contains unmapped team ID: "
            f"home_team_id={home_id!r} away_team_id={away_id!r}"
        )

    return tuple(sorted((home_id, away_id)))


def load_expected_schedule(
    date_value: str,
    team_map: dict,
) -> dict[tuple[str, str], dict[str, str]]:
    schedule_path = NHL_SCHEDULE_DIR / f"NHL_{date_value}.csv"

    if not schedule_path.exists():
        raise FileNotFoundError(
            f"Missing official NHL schedule for intake date: {schedule_path}"
        )

    with schedule_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_team_id",
            "away_team_id",
            "game_type",
        }
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)

        if missing:
            raise ValueError(
                f"{schedule_path} missing required columns: {missing}"
            )

        expected: dict[tuple[str, str], dict[str, str]] = {}
        seen_game_ids: set[str] = set()

        for row_number, row in enumerate(reader, start=2):
            game_date = str(row.get("game_date", "")).strip()
            game_type = str(row.get("game_type", "")).strip()

            if game_date != date_value:
                continue

            if game_type not in EXPECTED_GAME_TYPES:
                continue

            game_id = str(row.get("game_id", "")).strip()
            if not game_id:
                raise ValueError(
                    f"{schedule_path} row {row_number} missing game_id"
                )

            if game_id in seen_game_ids:
                raise ValueError(
                    f"{schedule_path} contains duplicate game_id={game_id}"
                )
            seen_game_ids.add(game_id)

            key = matchup_key_from_schedule(row, team_map)

            if key in expected:
                raise ValueError(
                    f"{schedule_path} contains duplicate matchup={key}"
                )

            expected[key] = dict(row)

    return expected


def is_complete_probability(value: str) -> bool:
    text = str(value).strip().replace("%", "")
    if not text:
        return False

    try:
        number = float(text)
    except ValueError:
        return False

    return 0.0 <= number <= 100.0


def is_complete_projected_score(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False

    try:
        number = float(text)
    except ValueError:
        return False

    return number >= 0.0


def validate_prediction_slate(
    games: list[dict],
    date_value: str,
    expected_schedule: dict[tuple[str, str], dict[str, str]],
    team_map: dict,
    parse_errors: int,
) -> list[dict]:
    if parse_errors:
        raise ValueError(
            f"D-Ratings parse errors detected: {parse_errors}"
        )

    current_date = datetime.strptime(date_value, "%Y_%m_%d").date()
    current_games: list[dict] = []
    seen_matchups: set[tuple[str, str]] = set()
    ignored_other_date_games = 0

    for game_number, game in enumerate(games, start=1):
        try:
            game_dt = parse_et_datetime(game.get("date_time", ""))
        except Exception as exc:
            raise ValueError(
                f"D-Ratings game {game_number} has invalid date_time="
                f"{game.get('date_time', '')!r}: {exc}"
            ) from exc

        if game_dt.date() != current_date:
            ignored_other_date_games += 1
            continue

        key = matchup_key_from_dratings(game, team_map)

        if key in seen_matchups:
            raise ValueError(
                f"Duplicate D-Ratings game detected for {date_value}: matchup={key}"
            )

        seen_matchups.add(key)
        current_games.append(game)

        if game.get("game_status") == "upcoming":
            missing_fields = []

            if not is_complete_probability(game.get("team1_win_pct", "")):
                missing_fields.append("team1_win_pct")

            if not is_complete_probability(game.get("team2_win_pct", "")):
                missing_fields.append("team2_win_pct")

            if not is_complete_projected_score(game.get("proj_score_1", "")):
                missing_fields.append("proj_score_1")

            if not is_complete_projected_score(game.get("proj_score_2", "")):
                missing_fields.append("proj_score_2")

            if missing_fields:
                raise ValueError(
                    "Incomplete D-Ratings prediction row for "
                    f"{date_value} matchup={key}: "
                    f"{', '.join(missing_fields)}"
                )

    if ignored_other_date_games:
        log(
            "Ignored D-Ratings games outside intake date | "
            f"date={date_value} "
            f"ignored_games={ignored_other_date_games}"
        )

    scraped_keys = {
        matchup_key_from_dratings(game, team_map)
        for game in current_games
    }
    expected_keys = set(expected_schedule)

    missing_expected = sorted(expected_keys - scraped_keys)
    unexpected = sorted(scraped_keys - expected_keys)

    if missing_expected:
        details = []
        for key in missing_expected:
            row = expected_schedule[key]
            details.append(
                f"{row.get('away_team', '')} at {row.get('home_team', '')} "
                f"(game_id={row.get('game_id', '')})"
            )

        raise ValueError(
            "D-Ratings is missing NHL schedule games expected for "
            f"{date_value}: " + "; ".join(details)
        )

    if unexpected:
        raise ValueError(
            "D-Ratings contains games not present in the official NHL "
            f"schedule for {date_value}: {unexpected}"
        )

    if len(scraped_keys) != len(expected_keys):
        raise ValueError(
            "D-Ratings game count does not match the official NHL schedule "
            f"for {date_value}: scraped={len(scraped_keys)} "
            f"expected={len(expected_keys)}"
        )

    upcoming = [
        game
        for game in current_games
        if game.get("game_status") == "upcoming"
    ]

    log(
        "INTAKE GATE PASSED | "
        f"date={date_value} "
        f"expected_games={len(expected_keys)} "
        f"dratings_games={len(scraped_keys)} "
        f"upcoming_predictions={len(upcoming)}"
    )

    return upcoming


def is_game_row(row):
    return len(row) >= 6 and "\n" in row[1]


def split_pair(
    value: str,
    *,
    strip: bool = False,
) -> tuple[str, str]:
    parts = str(value).split("\n")
    first, second = parts[0], parts[1]

    if strip:
        return first.strip(), second.strip()

    return first, second


def parse_nhl(row):
    if not is_game_row(row):
        return None

    try:
        if len(row) == 11:
            date_time = convert_utc_to_et(row[0].replace("\n", " "))
            team1, team2 = split_pair(row[1], strip=True)
            wp1, wp2 = split_pair(row[3])
            ml1, ml2 = split_pair(row[4])
            sp1, sp2 = split_pair(row[5])
            proj1, proj2 = split_pair(row[6])
            total = row[7]
            over_line, under_line = split_pair(row[8])

            return {
                "sport": "NHL",
                "date_time": date_time,
                "team1": team1,
                "team2": team2,
                "team1_win_pct": wp1,
                "team2_win_pct": wp2,
                "team1_moneyline": ml1,
                "team2_moneyline": ml2,
                "team1_spread": sp1,
                "team2_spread": sp2,
                "proj_score_1": proj1,
                "proj_score_2": proj2,
                "total": total,
                "over_line": over_line,
                "under_line": under_line,
                "score1": "",
                "score2": "",
                "game_status": "upcoming",
            }

        if len(row) == 8:
            date_time = convert_utc_to_et(row[0].replace("\n", " "))
            team1, team2 = split_pair(row[1], strip=True)
            wp1, wp2 = split_pair(row[2])
            ml1, ml2 = split_pair(row[3])
            sp1, sp2 = split_pair(row[4])
            score1, score2 = split_pair(row[5], strip=True)

            return {
                "sport": "NHL",
                "date_time": date_time,
                "team1": team1,
                "team2": team2,
                "team1_win_pct": wp1,
                "team2_win_pct": wp2,
                "team1_moneyline": ml1,
                "team2_moneyline": ml2,
                "team1_spread": sp1,
                "team2_spread": sp2,
                "proj_score_1": "",
                "proj_score_2": "",
                "total": "",
                "over_line": "",
                "under_line": "",
                "score1": score1,
                "score2": score2,
                "game_status": "completed",
            }

    except Exception as e:
        log(f"WARNING: parse_nhl failed on row (len={len(row)}): {e}")

    return None


def scrape_page(page, url):
    page.goto(url)
    page.wait_for_selector("table")
    rows = page.query_selector_all("table tbody tr")
    return [[c.inner_text().strip() for c in r.query_selector_all("td")] for r in rows]


def main():
    files_written = []
    parse_errors = 0

    try:
        date = datetime.now(ET).strftime("%Y_%m_%d")

        raw_rows_dir = BASE_DIR / "00_intake" / "drat_raw" / "rows"
        raw_rows_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = BASE_DIR / "00_intake" / "drat_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        scraper_dir = BASE_DIR / "00_intake" / "predictions" / "scraper"
        scraper_dir.mkdir(parents=True, exist_ok=True)

        scraper_path = scraper_dir / f"{date}_nhl_predictions.csv"

        team_map = load_team_map()
        expected_schedule = load_expected_schedule(date, team_map)

        log(
            "Official NHL schedule gate loaded: "
            f"date={date} "
            f"regular_or_playoff_games={len(expected_schedule)}"
        )
        log("D-Ratings date_time is interpreted as UTC and converted to ET.")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            try:
                page = browser.new_page()

                page.set_extra_http_headers(
                    {
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/122.0.0.0 Safari/537.36"
                        )
                    }
                )

                raw = scrape_page(page, URLS["nhl"])

                raw_rows_path = raw_rows_dir / f"{date}_nhl_raw_rows.json"
                with open(raw_rows_path, "w", encoding="utf-8") as f:
                    json.dump(raw, f, indent=2)
                files_written.append((str(raw_rows_path), len(raw)))

                col_counts = {}
                for r in raw:
                    n = len(r)
                    col_counts[n] = col_counts.get(n, 0) + 1
                log(f"Column count distribution: {col_counts}")

                games = []
                for r in raw:
                    result = parse_nhl(r)
                    if result:
                        games.append(result)
                    elif is_game_row(r):
                        parse_errors += 1

                raw_path = raw_dir / f"{date}_nhl_raw.json"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(games, f, indent=2)
                files_written.append((str(raw_path), len(games)))

                completed = [
                    g for g in games
                    if g["game_status"] == "completed"
                ]

                upcoming = validate_prediction_slate(
                    games=games,
                    date_value=date,
                    expected_schedule=expected_schedule,
                    team_map=team_map,
                    parse_errors=parse_errors,
                )

                log(f"Upcoming games after intake gates: {len(upcoming)}")
                log(
                    "Completed games retained in raw JSON only: "
                    f"{len(completed)}"
                )

                if scraper_path.exists():
                    scraper_path.unlink()
                    log(
                        "Removed existing scraper output before rewrite: "
                        f"{scraper_path}"
                    )

                if upcoming:
                    df_up = pd.DataFrame(upcoming)
                    df_up.to_csv(scraper_path, index=False)
                    files_written.append((str(scraper_path), len(df_up)))
                    log(
                        f"WROTE validated upcoming scraper copy -> "
                        f"{scraper_path} ({len(df_up)} rows)"
                    )
                else:
                    log(
                        "No upcoming regular-season or playoff predictions "
                        "for the intake date."
                    )

            finally:
                browser.close()

        log("--- SUMMARY ---")
        log(f"Raw rows scraped: {len(raw)}")
        log(f"Games parsed: {len(games)}")
        log(f"Parse errors: {parse_errors}")
        log(
            "Expected regular-season/playoff NHL games: "
            f"{len(expected_schedule)}"
        )
        log(f"Validated upcoming predictions: {len(upcoming)}")
        log(f"Completed retained in raw JSON only: {len(completed)}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
