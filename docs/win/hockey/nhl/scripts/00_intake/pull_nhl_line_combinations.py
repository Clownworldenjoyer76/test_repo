#!/usr/bin/env python3
"""Daily Faceoff NHL line combinations -> team/player-ID-normalized assignment CSV.

Run from repository root:
    python docs/win/hockey/nhl/scripts/00_intake/pull_nhl_line_combinations.py

Requires existing official NHL schedule and team mapping and SportsDataverse
team rosters in this repository. Standard-library Python only.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[2]  # docs/win/hockey/nhl
LINEUPS_DIR = BASE_DIR / "00_intake" / "lineups"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
ROSTER_DIR = BASE_DIR / "sdv" / "lineup-strength"
TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"
SOURCE_ROOT = "https://www.dailyfaceoff.com/teams"
EASTERN = ZoneInfo("America/New_York")

TEAMS = {
    "ANA": "anaheim-ducks",
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars",
    "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers",
    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens",
    "NSH": "nashville-predators",
    "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken",
    "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "UTA": "utah-mammoth",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals",
    "WPG": "winnipeg-jets",
}

HEADERS = [
    "game_id", "CapturedUTC", "date", "team_id", "team", "team_abbrev",
    "player_id", "player_name", "lineup_group", "lineup_category", "position",
    "injury_status", "game_time_decision", "lineup_updated_utc",
    "lineup_source_name", "lineup_source_url", "lineup_page_url",
]

SCHEDULE_REQUIRED = (
    "game_id", "game_date", "home_team", "away_team",
    "home_team_abbrev", "away_team_abbrev", "home_team_id", "away_team_id",
)
PLAYER_REQUIRED = ("playerId", "name", "groupIdentifier", "categoryIdentifier", "positionIdentifier")
GROUP_CATEGORIES = {
    **{f"f{i}": "ev" for i in range(1, 5)},
    **{f"d{i}": "ev" for i in range(1, 4)},
    "g": "ev", "pp1": "pp", "pp2": "pp", "pk1": "pk", "pk2": "pk", "ir": "oi",
}


class NextDataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []
        self.found = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            if self.found:
                raise ValueError("Duplicate __NEXT_DATA__ script")
            self.active = True
            self.found = True

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "script":
            self.active = False


def parse_page(html, abbrev, slug):
    parser = NextDataParser()
    parser.feed(html)
    if not parser.found or not parser.parts:
        raise ValueError(f"{abbrev}: missing __NEXT_DATA__")
    doc = json.loads("".join(parser.parts))
    page = doc["props"]["pageProps"]
    combinations = page["combinations"]
    if isinstance(combinations, str):
        combinations = json.loads(combinations)
    if not isinstance(combinations, dict):
        raise ValueError(f"{abbrev}: invalid combinations payload")
    for field, expected in (("teamAbbreviation", abbrev), ("teamSlug", slug)):
        if str(combinations.get(field) or "").strip() != expected:
            raise ValueError(f"{abbrev}: {field} doesn't match requested team ({combinations.get(field)!r})")
    if not isinstance(combinations.get("players"), list) or not combinations["players"]:
        raise ValueError(f"{abbrev}: missing/empty players array")
    return combinations


def get_page(abbrev, slug, retries=3):
    url = f"{SOURCE_ROOT}/{slug}/line-combinations"
    error = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
            })
            with urlopen(req, timeout=50) as response:
                html = response.read().decode(response.headers.get_content_charset() or "utf-8")
            return parse_page(html, abbrev, slug)
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(attempt + 2)
    raise RuntimeError(f"{abbrev} {url}: failed after {retries} requests: {error}")


def read_csv(path, required, allow_empty=False):
    if not path.is_file():
        raise FileNotFoundError(f"Required repository input missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV header")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not allow_empty and not rows:
        raise ValueError(f"{path}: no data rows")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def required(row, field, context):
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{context}: missing {field}")
    return value


def normalized(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def numeric_id(value, context):
    text = str(value or "").strip()
    if not text.isdecimal() or int(text) <= 0:
        raise ValueError(f"{context}: invalid positive NHL/player ID {value!r}")
    return text


def read_team_map(path=TEAM_MAP_PATH):
    teams = {}
    for row in read_csv(path, ("league", "source", "canonical_team", "nhl_team_id", "nhl_abbrev", "alias")):
        if row["league"].strip().lower() != "nhl" or row["source"].strip() != "official_nhl":
            continue
        abbrev = required(row, "nhl_abbrev", str(path)).upper()
        if abbrev in teams:
            raise ValueError(f"Duplicate official team mapping for {abbrev}")
        teams[abbrev] = {
            "team_id": numeric_id(row["nhl_team_id"], f"Team {abbrev}"),
            "team": required(row, "canonical_team", f"Team {abbrev}"),
            "official_alias": required(row, "alias", f"Team {abbrev}"),
        }
    if set(teams) != set(TEAMS):
        raise ValueError(f"NHL team map team coverage != 32: missing={sorted(set(TEAMS)-set(teams))} extra={sorted(set(teams)-set(TEAMS))}")
    return teams


def read_schedule(date_value, teams, schedule_dir=SCHEDULE_DIR):
    path = schedule_dir / f"NHL_{date_value.replace('-', '_')}.csv"
    games = read_csv(path, SCHEDULE_REQUIRED, allow_empty=True)
    schedule = {}
    seen_games = set()
    for line, game in enumerate(games, start=2):
        context = f"{path} row {line}"
        game_date = required(game, "game_date", context).replace("_", "-")
        if game_date != date_value:
            raise ValueError(f"{context}: wrong game date {game_date}")
        gid = numeric_id(required(game, "game_id", context), context)
        if gid in seen_games:
            raise ValueError(f"{context}: duplicate game_id {gid}")
        seen_games.add(gid)
        for side in ("home", "away"):
            abbr = required(game, f"{side}_team_abbrev", context).upper()
            if abbr not in teams:
                raise ValueError(f"{context}: unknown {side} team {abbr}")
            mapped = teams[abbr]
            if numeric_id(game[f"{side}_team_id"], context) != mapped["team_id"]:
                raise ValueError(f"{context}: {abbr} team ID differs from official NHL team map")
            if normalized(required(game, f"{side}_team", context)) != normalized(mapped["official_alias"]):
                raise ValueError(f"{context}: {abbr} official team name mismatch")
            if abbr in schedule:
                raise ValueError(f"{context}: multiple games for {abbr} on {date_value}; team-only game_id is ambiguous")
            schedule[abbr] = gid
        if game["home_team_abbrev"].upper() == game["away_team_abbrev"].upper():
            raise ValueError(f"{context}: team playing itself")
    return schedule


def roster_index(path):
    records = read_csv(path, ("id", "first_name_default", "last_name_default", "position_group"))
    result = {}
    for line, item in enumerate(records, start=2):
        context = f"{path} row {line}"
        player_id = numeric_id(required(item, "id", context), context)
        name = required(item, "first_name_default", context) + " " + required(item, "last_name_default", context)
        key = normalized(name)
        if not key:
            raise ValueError(f"{context}: no normalized player name")
        result.setdefault(key, set()).add(player_id)
    return result


class PlayerResolver:
    """Resolve a DFO player using same-team SDV rosters only; never fuzzy-match IDs."""
    def __init__(self, roster_dir=ROSTER_DIR):
        self.dir = Path(roster_dir)
        self.current = {}
        self.history = {}
        self.stats = {"current": 0, "historical": 0}

    def _load(self, team):
        path = self.dir / f"latest_{team}_roster.csv"
        self.current[team] = roster_index(path)

    def _historical(self, team):
        combined = {}
        paths = sorted(self.dir.glob(f"*_{team}_roster.csv"))
        paths = [path for path in paths if not path.name.startswith("latest_")]
        for path in paths:
            for name, ids in roster_index(path).items():
                combined.setdefault(name, set()).update(ids)
        self.history[team] = combined

    def resolve(self, team, player_name):
        if team not in self.current:
            self._load(team)
        key = normalized(player_name)
        matches = self.current[team].get(key, set())
        source = "current"
        if not matches:
            if team not in self.history:
                self._historical(team)
            matches = self.history[team].get(key, set())
            source = "historical"
        if len(matches) != 1:
            raise ValueError(f"{team}: {player_name!r} SDV {source} roster match count={len(matches)}; cannot assign NHL player_id")
        self.stats[source] += 1
        return next(iter(matches))


def timestamp(value, context):
    raw = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("missing timezone")
        return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid lineup timestamp {value!r}") from exc


def validate_group(player, context):
    group = required(player, "groupIdentifier", context).lower()
    category = required(player, "categoryIdentifier", context).lower()
    position = required(player, "positionIdentifier", context).lower()
    if group not in GROUP_CATEGORIES or category != GROUP_CATEGORIES[group]:
        raise ValueError(f"{context}: unknown/conflicting lineup group/category {group}/{category}")
    if group.startswith("f") and position not in {"lw", "c", "rw"}:
        raise ValueError(f"{context}: forward with unexpected position {position}")
    if group.startswith("d") and position not in {"ld", "rd"}:
        raise ValueError(f"{context}: defensive pair with unexpected position {position}")
    if group == "g" and position not in {"g1", "g2"}:
        raise ValueError(f"{context}: goalie slot unexpected {position}")
    if group.startswith("pp") and position not in {f"sk{j}" for j in range(1, 6)}:
        raise ValueError(f"{context}: power play slot unexpected {position}")
    if group.startswith("pk") and position not in {f"sk{j}" for j in range(1, 5)}:
        raise ValueError(f"{context}: penalty kill slot unexpected {position}")
    if group == "ir" and not re.fullmatch(r"ir[1-9][0-9]*", position):
        raise ValueError(f"{context}: injury/off-ice slot unexpected {position}")
    return group, category, position


def build_rows(pages, date_value, captured, team_map_path=TEAM_MAP_PATH,
               schedule_dir=SCHEDULE_DIR, roster_dir=ROSTER_DIR):
    if set(pages) != set(TEAMS):
        raise ValueError(f"Expected 32 teams, received {len(pages)}: missing={sorted(set(TEAMS)-set(pages))}")
    teams = read_team_map(team_map_path)
    scheduled = read_schedule(date_value, teams, schedule_dir)
    resolver = PlayerResolver(roster_dir)
    rows = []
    crosswalk = {}

    for team in TEAMS:
        page = pages[team]
        context = f"Daily Faceoff {team}"
        if not isinstance(page, dict):
            raise ValueError(f"{context}: invalid page payload")
        if required(page, "teamAbbreviation", context).upper() != team or required(page, "teamSlug", context) != TEAMS[team]:
            raise ValueError(f"{context}: wrong team/slug in payload")
        if normalized(required(page, "teamName", context)) != normalized(teams[team]["team"]):
            raise ValueError(f"{context}: team name doesn't match official NHL mapping")
        last_updated = timestamp(required(page, "updatedAt", context), context)
        players = page.get("players")
        if not isinstance(players, list) or not players:
            raise ValueError(f"{context}: missing player assignments")
        group_counts = {group: 0 for group in GROUP_CATEGORIES}
        used_slots = set()

        for i, player in enumerate(players):
            context_player = f"{context} player[{i}]"
            if not isinstance(player, dict):
                raise ValueError(f"{context_player}: not a JSON object")
            for field in PLAYER_REQUIRED:
                required(player, field, context_player)
            dfo_id = numeric_id(player["playerId"], context_player)
            name = required(player, "name", context_player)
            group, category, position = validate_group(player, context_player)
            identity_key = (team, dfo_id)
            if identity_key in crosswalk and crosswalk[identity_key][0] != normalized(name):
                raise ValueError(f"{context_player}: DFO player ID {dfo_id} used with conflicting names")
            if identity_key not in crosswalk:
                crosswalk[identity_key] = (normalized(name), resolver.resolve(team, name))
            player_id = crosswalk[identity_key][1]
            assignment_key = (team, group, position)
            if assignment_key in used_slots:
                raise ValueError(f"{context_player}: duplicate lineup slot {assignment_key}")
            used_slots.add(assignment_key)
            group_counts[group] += 1
            gtd = player.get("gameTimeDecision")
            if not isinstance(gtd, bool):
                raise ValueError(f"{context_player}: gameTimeDecision must be true/false")
            rows.append({
                "game_id": scheduled.get(team, ""),
                "CapturedUTC": captured,
                "date": date_value,
                "team_id": teams[team]["team_id"],
                "team": teams[team]["team"],
                "team_abbrev": team,
                "player_id": player_id,
                "player_name": name,
                "lineup_group": group,
                "lineup_category": category,
                "position": position,
                "injury_status": str(player.get("injuryStatus") or "").strip().lower(),
                "game_time_decision": "true" if gtd else "false",
                "lineup_updated_utc": last_updated,
                "lineup_source_name": str(page.get("sourceName") or "").strip(),
                "lineup_source_url": str(page.get("source") or "").strip(),
                "lineup_page_url": f"{SOURCE_ROOT}/{TEAMS[team]}/line-combinations",
            })
        expected_counts = {
            **{f"f{i}": 3 for i in range(1, 5)},
            **{f"d{i}": 2 for i in range(1, 4)},
            "g": 2, "pp1": 5, "pp2": 5, "pk1": 4, "pk2": 4,
        }
        unexpected = {g: (group_counts[g], required_count) for g, required_count in expected_counts.items()
                      if group_counts[g] != required_count}
        if unexpected:
            raise ValueError(f"{context}: incomplete lineup assignments: {unexpected}")

    if not rows or len({r["team_abbrev"] for r in rows}) != 32:
        raise ValueError("Lineup export incomplete")
    return rows, resolver.stats


def write_csv(rows, date_value, output_dir=LINEUPS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{date_value}_line_combinations.csv"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir,
                                prefix=".lineups_", suffix=".tmp", delete=False) as f:
            temporary = f.name
            writer = csv.DictWriter(f, fieldnames=HEADERS, lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return target


def main():
    if len(TEAMS) != 32 or len(set(TEAMS.values())) != 32:
        raise ValueError("Team list must contain 32 unique NHL team URLs")
    now = datetime.now(timezone.utc)
    date_value = now.astimezone(EASTERN).date().isoformat()
    captured = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    pages = {}
    for index, (abbrev, slug) in enumerate(TEAMS.items(), 1):
        pages[abbrev] = get_page(abbrev, slug)
        print(f"Fetched {index}/32: {abbrev} ({len(pages[abbrev]['players'])} assignments)", flush=True)
        if index < len(TEAMS):
            time.sleep(0.3)
    rows, stats = build_rows(pages, date_value, captured)
    path = write_csv(rows, date_value)
    print(f"CSV: {path}")
    print(f"Teams: 32/32 | Rows: {len(rows)} | Columns: {len(HEADERS)}")
    print(f"Player crosswalks (unique DFO player IDs): {sum(stats.values())} | latest={stats['current']} historical={stats['historical']}")
    print(f"Rows with scheduled game_id: {sum(bool(row['game_id']) for row in rows)}")


if __name__ == "__main__":
    main()
