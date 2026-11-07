#!/usr/bin/env python3
"""Export Daily Faceoff starting goalies, one row per game, to NHL intake CSV.

Run from the repository root:
    python docs/win/hockey/nhl/scripts/00_intake/pull_nhl_starting_goalies.py

Uses only the Python standard library. No repository files are modified other
than the date-specific CSV in docs/win/hockey/nhl/00_intake/lineups/.
"""

import csv
import json
import os
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


SOURCE_URL = "https://www.dailyfaceoff.com/starting-goalies"
OUTPUT_DIR = Path("docs/win/hockey/nhl/00_intake/lineups")

# Exactly the 38 columns selected from the inspected 89-column CSV, in order.
HEADERS = [
    "CapturedUTC",
    "GameNumber",
    "date",
    "time",
    "pointSpread",
    "dateGmt",
    "homeTeamId",
    "awayTeamId",
    "homeTeamName",
    "awayTeamName",
    "homeTeamAbbreviation",
    "awayTeamAbbreviation",
    "homeTeamSlug",
    "awayTeamSlug",
    "homeGoalieId",
    "awayGoalieId",
    "homeGoalieLegacyId",
    "awayGoalieLegacyId",
    "homeGoalieSlug",
    "awayGoalieSlug",
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
REQUIRED_FIELDS = (
    "date",
    "time",
    "homeTeamId",
    "awayTeamId",
    "homeGoalieId",
    "awayGoalieId",
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


def build_rows(games):
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = []
    dates = set()
    game_keys = set()

    for index, game in enumerate(games, start=1):
        for field in REQUIRED_FIELDS:
            if game.get(field) is None or str(game[field]).strip() == "":
                raise ValueError(f"Game {index} missing required field: {field}")

        game_date = str(game["date"])
        datetime.strptime(game_date, "%Y-%m-%d")
        dates.add(game_date)
        key = (game_date, str(game["homeTeamId"]), str(game["awayTeamId"]))
        if key in game_keys:
            raise ValueError(f"Duplicate game in source: {key}")
        game_keys.add(key)

        row = {}
        for field in HEADERS:
            if field == "CapturedUTC":
                value = captured
            elif field == "GameNumber":
                value = index
            else:
                value = game.get(field)
            if value is None:
                value = ""
            elif isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            row[field] = value
        rows.append(row)

    if len(dates) != 1:
        raise ValueError(f"Source contains multiple game dates: {sorted(dates)}")
    return rows, dates.pop()


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
    print(f"Games: {len(rows)} | Columns: {len(HEADERS)}")


if __name__ == "__main__":
    main()
