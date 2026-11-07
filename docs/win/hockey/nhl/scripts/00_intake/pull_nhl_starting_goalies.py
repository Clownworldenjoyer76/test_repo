#!/usr/bin/env python3
"""Scrape Daily Faceoff starting goalies, joining official NHL game/team IDs and SDV player IDs.

Run from GitHub Actions or the repository root:
    python docs/win/hockey/nhl/scripts/00_intake/pull_nhl_starting_goalies.py

Requires the existing dated NHL schedule, NHL team map, and SDV goalie roster
files in this repository. Writes only the dated starting-goalies CSV, after
all source records have been matched and validated.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.dailyfaceoff.com/starting-goalies"
BASE_DIR = Path(__file__).resolve().parents[2]  # docs/win/hockey/nhl
OUTPUT_DIR = BASE_DIR / "00_intake" / "lineups"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
ROSTER_DIR = BASE_DIR / "sdv" / "goalie"
TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"

HEADERS = [
    "game_id",
    "CapturedUTC",
    "date",
    "time",
    "dateGmt",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "homeTeamAbbreviation",
    "awayTeamAbbreviation",
    "home_player_id",
    "away_player_id",
    "homeGoalieWins",
    "awayGoalieWins",
    "homeGoalieLosses",
    "awayGoalieLosses",
    "homeGoalieOvertimeLosses",
    "awayGoalieOvertimeLosses",
    "homeGoalieShutouts",
    "awayGoalieShutouts",
    "homeGoalieSavePercentage",
    "awayGoalieSavePercentage",
    "homeGoalieGoalsAgainstAvg",
    "awayGoalieGoalsAgainstAvg",
    "homeNewsStrengthId",
    "awayNewsStrengthId",
    "homeNewsStrengthName",
    "awayNewsStrengthName",
    "homeNewsDetails",
    "awayNewsDetails",
]

REQUIRED_DFO_FIELDS = (
    "date", "time", "dateGmt", "homeTeamName", "awayTeamName",
    "homeTeamAbbreviation", "awayTeamAbbreviation",
    "homeGoalieName", "awayGoalieName",
)
SCHEDULE_FIELDS = (
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "home_team_abbrev", "away_team_abbrev", "home_team_id",
    "away_team_id", "start_time_utc",
)


class NextDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_next_data = False
        self.next_data_parts = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.in_next_data = True
            self.found = True

    def handle_data(self, data):
        if self.in_next_data:
            self.next_data_parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self.in_next_data = False


def read_games(html):
    parser = NextDataParser()
    parser.feed(html)
    if not parser.found:
        raise ValueError("Daily Faceoff __NEXT_DATA__ script not found")

    document = json.loads("".join(parser.next_data_parts))
    data = document["props"]["pageProps"]["data"]
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        data = data.get("games")
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a nonempty game list in pageProps.data")
    if not all(isinstance(game, dict) for game in data):
        raise ValueError("Game list contains a non-object record")
    return data


def read_csv(path, required_fields):
    if not path.is_file():
        raise FileNotFoundError(f"Required input file missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError(f"Required input file is empty: {path}")
    if any(None in row for row in records):
        raise ValueError(f"Malformed CSV row with unexpected extra cells: {path}")
    return records


def norm_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def require_text(record, field, context):
    value = str(record.get(field) or "").strip()
    if not value:
        raise ValueError(f"{context}: missing {field}")
    return value


def nhl_id(value, context):
    text = str(value or "").strip()
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"{context}: invalid NHL ID {value!r}")
    return text


def normalized_date(value):
    raw = str(value or "").strip().replace("_", "-")
    return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()


def parse_utc(value, context):
    try:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timestamp has no timezone")
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid UTC timestamp {value!r}") from exc


def load_team_map(path=TEAM_MAP_PATH):
    rows = read_csv(
        path,
        ("league", "source", "alias", "canonical_team", "nhl_team_id", "nhl_abbrev"),
    )
    by_abbrev = {}
    by_official_alias = {}
    for row in rows:
        if row["league"].strip().casefold() != "nhl" or row["source"].strip() != "official_nhl":
            continue
        abbrev = require_text(row, "nhl_abbrev", str(path)).upper()
        team = {
            "id": nhl_id(row["nhl_team_id"], str(path)),
            "canonical": require_text(row, "canonical_team", str(path)),
            "alias": require_text(row, "alias", str(path)),
        }
        if abbrev in by_abbrev:
            raise ValueError(f"Duplicate official NHL team abbreviation {abbrev} in {path}")
        by_abbrev[abbrev] = team
        key = norm_name(team["alias"])
        if key in by_official_alias:
            raise ValueError(f"Duplicate official NHL team alias {team['alias']!r} in {path}")
        by_official_alias[key] = abbrev
    if not by_abbrev:
        raise ValueError(f"No official NHL team entries in {path}")
    return by_abbrev


def load_schedule(game_date, team_map, schedule_dir=SCHEDULE_DIR):
    path = schedule_dir / f"NHL_{game_date.replace('-', '_')}.csv"
    records = read_csv(path, SCHEDULE_FIELDS)
    index = {}
    game_ids = set()
    for number, row in enumerate(records, start=2):
        context = f"{path} row {number}"
        date = normalized_date(require_text(row, "game_date", context))
        if date != game_date:
            raise ValueError(f"{context}: wrong game_date {date} (expected {game_date})")
        home = require_text(row, "home_team_abbrev", context).upper()
        away = require_text(row, "away_team_abbrev", context).upper()
        if home == away or home not in team_map or away not in team_map:
            raise ValueError(f"{context}: invalid team abbreviations {home}/{away}")
        for side, abbr in (("home", home), ("away", away)):
            team = team_map[abbr]
            source_id = nhl_id(row[f"{side}_team_id"], context)
            if source_id != team["id"]:
                raise ValueError(f"{context}: {abbr} schedule team ID {source_id} differs from mapped NHL ID {team['id']}")
            name = require_text(row, f"{side}_team", context)
            if norm_name(name) != norm_name(team["alias"]):
                raise ValueError(f"{context}: unexpected schedule team name {name!r} for {abbr}")
        game_id = nhl_id(require_text(row, "game_id", context), context)
        if game_id in game_ids:
            raise ValueError(f"{context}: duplicate game_id {game_id}")
        game_ids.add(game_id)
        require_text(row, "game_time", context)
        parse_utc(require_text(row, "start_time_utc", context), context)
        key = (date, home, away)
        if key in index:
            raise ValueError(f"{context}: duplicate schedule matchup {key}")
        index[key] = row
    return index


def load_roster(abbrev, roster_dir=ROSTER_DIR):
    path = roster_dir / f"latest_{abbrev}_goalies_roster.csv"
    records = read_csv(path, ("id", "first_name_default", "last_name_default", "position_group"))
    names = {}
    ids = set()
    for number, row in enumerate(records, start=2):
        context = f"{path} row {number}"
        if require_text(row, "position_group", context).lower() != "goalies":
            raise ValueError(f"{context}: non-goalie in goalie roster")
        player_id = nhl_id(require_text(row, "id", context), context)
        if player_id in ids:
            raise ValueError(f"{context}: duplicate NHL player ID {player_id}")
        ids.add(player_id)
        full_name = " ".join((require_text(row, "first_name_default", context),
                              require_text(row, "last_name_default", context)))
        key = norm_name(full_name)
        if key in names:
            raise ValueError(f"{context}: ambiguous goalie name {full_name!r} in {abbrev} roster")
        names[key] = player_id
    return names


def csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def build_rows(games, schedule_dir=SCHEDULE_DIR, roster_dir=ROSTER_DIR, team_map_path=TEAM_MAP_PATH):
    dates = set()
    keys = set()
    for number, game in enumerate(games, start=1):
        context = f"Daily Faceoff game {number}"
        for field in REQUIRED_DFO_FIELDS:
            require_text(game, field, context)
        dates.add(normalized_date(game["date"]))
        key = (str(game["date"]), str(game["homeTeamAbbreviation"]).strip().upper(),
               str(game["awayTeamAbbreviation"]).strip().upper())
        if key in keys:
            raise ValueError(f"{context}: duplicate source matchup {key}")
        keys.add(key)
    if len(dates) != 1:
        raise ValueError(f"Daily Faceoff contains multiple game dates: {sorted(dates)}")
    game_date = dates.pop()

    team_map = load_team_map(team_map_path)
    schedule = load_schedule(game_date, team_map, schedule_dir)
    roster_cache = {}
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = []
    matched_game_ids = set()

    for number, game in enumerate(games, start=1):
        context = f"Daily Faceoff game {number}"
        home = str(game["homeTeamAbbreviation"]).strip().upper()
        away = str(game["awayTeamAbbreviation"]).strip().upper()
        key = (normalized_date(game["date"]), home, away)
        match = schedule.get(key)
        if match is None:
            raise ValueError(f"{context}: no unique official NHL schedule matchup {key}")
        game_id = match["game_id"].strip()
        if game_id in matched_game_ids:
            raise ValueError(f"{context}: duplicate matched NHL game_id {game_id}")
        matched_game_ids.add(game_id)
        if str(game["time"]).strip() != match["game_time"].strip():
            raise ValueError(f"{context} game_id={game_id}: game time differs from official schedule")
        if parse_utc(game["dateGmt"], context) != parse_utc(match["start_time_utc"], context):
            raise ValueError(f"{context} game_id={game_id}: UTC start differs from official schedule")

        player_ids = {}
        for side, abbrev in (("home", home), ("away", away)):
            expected_name = team_map[abbrev]["canonical"]
            supplied_name = str(game[f"{side}TeamName"]).strip()
            if norm_name(supplied_name) != norm_name(expected_name):
                raise ValueError(f"{context}: DFO {side} team name {supplied_name!r} doesn't match {abbrev} ({expected_name})")
            if abbrev not in roster_cache:
                roster_cache[abbrev] = load_roster(abbrev, roster_dir)
            goalie_name = str(game[f"{side}GoalieName"]).strip()
            player_id = roster_cache[abbrev].get(norm_name(goalie_name))
            if player_id is None:
                raise ValueError(f"{context} game_id={game_id}: unmatched {side} goalie {goalie_name!r} for {abbrev}")
            player_ids[side] = player_id

        row = {field: csv_cell(game.get(field)) for field in HEADERS}
        row.update({
            "game_id": game_id,
            "CapturedUTC": captured,
            "home_team_id": team_map[home]["id"],
            "away_team_id": team_map[away]["id"],
            "home_team": team_map[home]["canonical"],
            "away_team": team_map[away]["canonical"],
            "homeTeamAbbreviation": home,
            "awayTeamAbbreviation": away,
            "home_player_id": player_ids["home"],
            "away_player_id": player_ids["away"],
        })
        rows.append(row)

    return rows, game_date


def write_csv(rows, game_date, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{game_date}_goalies.csv"
    temp_name = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_dir,
            prefix=".goalies_", suffix=".tmp", delete=False,
        ) as temp:
            temp_name = temp.name
            writer = csv.DictWriter(temp, fieldnames=HEADERS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination


def main():
    request = Request(
        SOURCE_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=45) as response:
        html = response.read().decode(response.headers.get_content_charset() or "utf-8")

    games = read_games(html)
    rows, game_date = build_rows(games)
    destination = write_csv(rows, game_date)
    print(f"CSV: {destination}")
    print(f"Games: {len(rows)} | Columns: {len(HEADERS)} | NHL IDs matched: {2 * len(rows)}")


if __name__ == "__main__":
    main()
