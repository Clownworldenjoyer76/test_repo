#!/usr/bin/env python3
"""Daily Faceoff NHL line combinations -> team/player-ID-normalized assignment CSV.

Run from repository root:
    python docs/win/hockey/nhl/scripts/00_intake/pull_nhl_line_combinations.py

Uses repository NHL schedule when present and the NHL daily scoreboard otherwise.
Normalizes IDs from team rosters and official NHL records; unresolved players
are retained with blank IDs and exported in a separate audit CSV. Python stdlib only.
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
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parents[2]  # docs/win/hockey/nhl
LINEUPS_DIR = BASE_DIR / "00_intake" / "lineups"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
ROSTER_DIR = BASE_DIR / "sdv" / "lineup-strength"
TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"
SOURCE_ROOT = "https://www.dailyfaceoff.com/teams"
NHL_API_ROOT = "https://api-web.nhle.com/v1"
NHL_PLAYER_SEARCH = "https://search.d3.nhle.com/api/v1/search/player"
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

# Verifiable nickname/full-name variations, never generic fuzzy matching.
# A candidate is accepted ONLY when it is unique within the same team's
# NHL/SDV roster, with the expected even-strength position group.
NAME_VARIANTS = (
    {"alex", "alexander"}, {"gabe", "gabriel"}, {"jacob", "jake"},
    {"joe", "joseph"}, {"josh", "joshua"}, {"matt", "matthew"},
    {"nick", "nicholas"}, {"sam", "samuel"},
    {"zach", "zack", "zachary"},
)

UNRESOLVED_HEADERS = (
    "CapturedUTC", "date", "team_abbrev", "dfo_player_id",
    "player_name", "position_group", "reason",
)
MISSING_SLOT_HEADERS = (
    "CapturedUTC", "date", "team_abbrev", "lineup_group",
    "lineup_category", "position", "reason",
)
FIXED_LINEUP_SLOTS = {
    **{f"f{i}": ("lw", "c", "rw") for i in range(1, 5)},
    **{f"d{i}": ("ld", "rd") for i in range(1, 4)},
    "g": ("g1", "g2"),
    "pp1": tuple(f"sk{i}" for i in range(1, 6)),
    "pp2": tuple(f"sk{i}" for i in range(1, 6)),
    "pk1": tuple(f"sk{i}" for i in range(1, 5)),
    "pk2": tuple(f"sk{i}" for i in range(1, 5)),
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


def nhl_daily_schedule(date_value, teams, nhl_fetch=None):
    """Get game IDs from the NHL's DATE-specific scoreboard (no guessing dates)."""
    url = f"{NHL_API_ROOT}/score/{date_value}"
    fetch = nhl_fetch or fetch_nhl_json
    payload = fetch(url)
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
        raise ValueError(f"{url}: missing NHL games array")
    if payload.get("currentDate") and payload["currentDate"] != date_value:
        raise ValueError(f"{url}: response date differs from requested {date_value}")
    schedule = {}
    seen_games = set()
    for index, game in enumerate(payload["games"]):
        context = f"{url} games[{index}]"
        if not isinstance(game, dict):
            raise ValueError(f"{context}: game is not an object")
        if game.get("gameDate") and game["gameDate"] != date_value:
            raise ValueError(f"{context}: wrong game date {game['gameDate']}")
        gid = numeric_id(game.get("id"), context)
        if gid in seen_games:
            raise ValueError(f"{context}: duplicate game ID {gid}")
        seen_games.add(gid)
        abbrs = []
        for side in ("home", "away"):
            team_data = game.get(f"{side}Team")
            if not isinstance(team_data, dict):
                raise ValueError(f"{context}: missing {side} team")
            abbr = required(team_data, "abbrev", context).upper()
            if abbr not in teams:
                raise ValueError(f"{context}: unknown NHL {side} team {abbr}")
            if team_data.get("id") is not None:
                if numeric_id(team_data["id"], context) != teams[abbr]["team_id"]:
                    raise ValueError(f"{context}: {abbr} team ID mismatches official mapping")
            if abbr in schedule:
                raise ValueError(f"{context}: multiple games for {abbr} on {date_value}; game_id ambiguous")
            abbrs.append(abbr)
        if abbrs[0] == abbrs[1]:
            raise ValueError(f"{context}: team playing itself")
        for abbr in abbrs:
            schedule[abbr] = gid
    return schedule


def read_schedule(date_value, teams, schedule_dir=SCHEDULE_DIR, nhl_fetch=None):
    path = schedule_dir / f"NHL_{date_value.replace('-', '_')}.csv"
    if not path.is_file():
        print(f"Schedule CSV absent: {path.name}; checking official NHL scoreboard", flush=True)
        try:
            schedule = nhl_daily_schedule(date_value, teams, nhl_fetch=nhl_fetch)
        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
            # Scraping all 32 team lineups does not depend on there being a local
            # schedule. Never assign unverifiable game IDs; expose degradation.
            print(f"WARNING: Official NHL schedule unavailable/invalid: {exc}", flush=True)
            print("WARNING: Exporting all 32 teams with blank game_id values", flush=True)
            return {}
        print(f"Official NHL scoreboard: {len(schedule)//2} games; "
              f"{len(schedule)} teams matched to game_id", flush=True)
        return schedule

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
    print(f"Local NHL schedule: {len(schedule)//2} games; "
          f"{len(schedule)} teams matched to game_id", flush=True)
    return schedule


def roster_index(path, allow_empty=False, include_roles=False):
    records = read_csv(path, ("id", "first_name_default", "last_name_default", "position_group"),
                       allow_empty=allow_empty)
    result = {}
    roles = {}
    for line, item in enumerate(records, start=2):
        context = f"{path} row {line}"
        player_id = numeric_id(required(item, "id", context), context)
        name = required(item, "first_name_default", context) + " " + required(item, "last_name_default", context)
        key = normalized(name)
        if not key:
            raise ValueError(f"{context}: no normalized player name")
        role = required(item, "position_group", context).lower()
        if role not in {"forwards", "defensemen", "goalies"}:
            raise ValueError(f"{context}: unexpected position_group {role}")
        if player_id in roles and roles[player_id] != role:
            raise ValueError(f"{context}: conflicting position group for NHL ID {player_id}")
        roles[player_id] = role
        result.setdefault(key, set()).add(player_id)
    return (result, roles) if include_roles else result


def role_of_lineup_group(group):
    if group in {"f1", "f2", "f3", "f4"}:
        return "forwards"
    if group in {"d1", "d2", "d3"}:
        return "defensemen"
    if group == "g":
        return "goalies"
    return None


def name_variant_match(source, candidate):
    """Exact surname and first-name equivalence; optional extra middle names."""
    a, b = normalized(source).split(), normalized(candidate).split()
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1]:
        return False
    first_equivalent = a[0] == b[0] or any({a[0], b[0]} <= variants for variants in NAME_VARIANTS)
    if not first_equivalent:
        return False
    # A changed given name must be from the explicit nickname list; the
    # remaining names cannot conflict if both source and candidate supply them.
    middle_a, middle_b = a[1:-1], b[1:-1]
    return not (middle_a and middle_b and middle_a != middle_b)


def fetch_nhl_json(url, attempts=2):
    """Fetch only the official NHL roster/search APIs, fail on invalid payloads."""
    last_error = None
    for retry in range(attempts):
        try:
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NHLLineups/2.0)",
                "Accept": "application/json",
            })
            with urlopen(req, timeout=20) as response:
                return json.load(response)
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, UnicodeError) as exc:
            last_error = exc
            if retry + 1 < attempts:
                time.sleep(1 + retry)
    raise RuntimeError(f"NHL API request failed: {url}: {last_error}")


def nhl_roster_index(payload, context, include_roles=False):
    """Index NHL-issued player IDs from a roster scoped to a specific team."""
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: expected NHL roster object")
    result = {}
    roles = {}
    present_groups = 0
    for group in ("forwards", "defensemen", "goalies"):
        entries = payload.get(group)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"{context}: invalid {group} array")
        present_groups += 1
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{context}: unexpected player record")
            first, last = entry.get("firstName"), entry.get("lastName")
            if isinstance(first, dict):
                first = first.get("default")
            if isinstance(last, dict):
                last = last.get("default")
            key = normalized(f"{first or ''} {last or ''}")
            if not key:
                raise ValueError(f"{context}: NHL player missing full name")
            player_id = numeric_id(entry.get("id"), f"{context}: {key}")
            result.setdefault(key, set()).add(player_id)
            roles[player_id] = group
    if not present_groups:
        raise ValueError(f"{context}: response contained no NHL roster groups")
    return (result, roles) if include_roles else result


class PlayerResolver:
    """Resolve exact team/player identities; never infer an ID from DFO's ID."""
    def __init__(self, roster_dir=ROSTER_DIR, game_date=None, team_map=None, nhl_fetch=None):
        self.dir = Path(roster_dir)
        self.current = {}
        self.history = {}
        self.current_roles = {}
        self.history_roles = {}
        self.official_current = {}
        self.official_season = {}
        self.official_current_roles = {}
        self.official_season_roles = {}
        self.search_cache = {}
        self.failed_api_requests = []
        self.nhl_fetch = nhl_fetch or fetch_nhl_json
        self.team_map = team_map or {}
        self.season = None
        if game_date:
            parsed = datetime.strptime(game_date, "%Y-%m-%d").date()
            season_start = parsed.year if parsed.month >= 7 else parsed.year - 1
            self.season = f"{season_start}{season_start + 1}"
        self.stats = {"current": 0, "historical": 0, "nhl_current": 0,
                      "nhl_season": 0, "nhl_search": 0, "unresolved": 0}

    def _load(self, team):
        self.current[team], self.current_roles[team] = roster_index(
            self.dir / f"latest_{team}_roster.csv", include_roles=True)

    def _historical(self, team):
        combined = {}
        roles = {}
        paths = sorted(self.dir.glob(f"*_{team}_roster.csv"))
        paths = [path for path in paths if not path.name.startswith("latest_")]
        required_columns = {"id", "first_name_default", "last_name_default", "position_group"}
        skipped = []
        for path in paths:
            # Older SDV snapshots sometimes contain only a newline (no roster
            # schema). Such files are not player-identity evidence. Ignore them
            # without weakening validation of any populated, compatible roster.
            with path.open(encoding="utf-8-sig", newline="") as handle:
                fields = next(csv.reader(handle), [])
            if not required_columns.issubset(set(fields)):
                skipped.append(path.name)
                continue
            index, positions = roster_index(path, allow_empty=True, include_roles=True)
            for name, ids in index.items():
                combined.setdefault(name, set()).update(ids)
            for player_id, group in positions.items():
                if player_id in roles and roles[player_id] != group:
                    raise ValueError(f"{path}: conflicting role for {player_id}")
                roles[player_id] = group
        if skipped:
            print(f"{team}: ignored {len(skipped)} archived SDV roster file(s) without "
                  f"player columns (first: {skipped[0]})", flush=True)
        self.history[team] = combined
        self.history_roles[team] = roles

    def _official_roster(self, team, current):
        store = self.official_current if current else self.official_season
        role_store = self.official_current_roles if current else self.official_season_roles
        if team not in store:
            suffix = "current" if current else self.season
            url = f"{NHL_API_ROOT}/roster/{team}/{suffix}"
            try:
                store[team], role_store[team] = nhl_roster_index(
                    self.nhl_fetch(url), url, include_roles=True)
            except (RuntimeError, ValueError, KeyError) as exc:
                self.failed_api_requests.append(str(exc))
                store[team] = {}
                role_store[team] = {}
        return store[team], role_store[team]

    def _official_player_search(self, team, player_name, key):
        """Search NHL players, only accept exact name PLUS official team verification."""
        cache_key = (team, key)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        url = NHL_PLAYER_SEARCH + "?" + urlencode({
            "culture": "en-us", "limit": 100, "q": player_name,
        })
        try:
            payload = self.nhl_fetch(url)
            if not isinstance(payload, list):
                raise ValueError("NHL search did not return a player array")
            match_ids = set()
            for player in payload:
                if not isinstance(player, dict) or normalized(player.get("name")) != key:
                    continue
                player_id = numeric_id(player.get("playerId"), "NHL player search")
                search_team = str(player.get("teamAbbrev") or "").strip().upper()
                if search_team == team:
                    match_ids.add(player_id)
                    continue
                # A search result may omit a team for an AHL call-up. Confirm
                # the team through the official NHL player profile instead.
                profile = self.nhl_fetch(f"{NHL_API_ROOT}/player/{player_id}/landing")
                if not isinstance(profile, dict) or numeric_id(profile.get("playerId"), "NHL player profile") != player_id:
                    raise ValueError(f"NHL profile failed to verify player {player_id}")
                first = profile.get("firstName")
                last = profile.get("lastName")
                if isinstance(first, dict):
                    first = first.get("default")
                if isinstance(last, dict):
                    last = last.get("default")
                if normalized(f"{first or ''} {last or ''}") != key:
                    continue
                profile_team_id = str(profile.get("currentTeamId") or "").strip()
                if profile_team_id == self.team_map[team]["team_id"]:
                    match_ids.add(player_id)
            result = match_ids
        except (RuntimeError, ValueError, KeyError) as exc:
            self.failed_api_requests.append(str(exc))
            result = set()
        self.search_cache[cache_key] = result
        return result

    def resolve(self, team, player_name, role=None):
        """Resolve a unique NHL ID without guessing across teams or positions."""
        if role is not None and role not in {"forwards", "defensemen", "goalies"}:
            raise ValueError(f"{team} {player_name}: invalid positional role {role}")
        key = normalized(player_name)

        def check(source, index, positions):
            exact = {i for i in index.get(key, ()) if not role or positions.get(i) == role}
            if len(exact) > 1:
                raise ValueError(f"{team}: {player_name!r} ambiguous {source} roster IDs {sorted(exact)}")
            if exact:
                self.stats[source] += 1
                return next(iter(exact))
            variants = set()
            for candidate, ids in index.items():
                if candidate != key and name_variant_match(key, candidate):
                    variants.update(i for i in ids if not role or positions.get(i) == role)
            if len(variants) > 1:
                raise ValueError(f"{team}: {player_name!r} ambiguous {source} name variants {sorted(variants)}")
            if variants:
                self.stats[source] += 1
                return next(iter(variants))
            return None

        if team not in self.current:
            self._load(team)
        matched = check("current", self.current[team], self.current_roles[team])
        if matched:
            return matched

        if team not in self.history:
            self._historical(team)
        matched = check("historical", self.history[team], self.history_roles[team])
        if matched:
            return matched

        for source, current in (("nhl_current", True), ("nhl_season", False)):
            if not current and not self.season:
                continue
            index, positions = self._official_roster(team, current)
            matched = check(source, index, positions)
            if matched:
                return matched

        matches = self._official_player_search(team, player_name, key)
        if len(matches) != 1:
            raise ValueError(f"{team}: {player_name!r} cannot verify unique NHL player ID; "
                             f"matches={sorted(matches)}; API errors={self.failed_api_requests[-3:]}")
        self.stats["nhl_search"] += 1
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
    resolver = PlayerResolver(roster_dir, game_date=date_value, team_map=teams)
    rows = []
    crosswalk = {}
    unresolved = []
    missing_slots = []

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
        used_slots = set()
        roles_by_dfo_id = {}
        for player in players:
            if not isinstance(player, dict):
                continue  # the main validation loop raises below
            dfo_id = str(player.get("playerId") or "")
            role = role_of_lineup_group(str(player.get("groupIdentifier") or "").lower())
            if role and dfo_id in roles_by_dfo_id and roles_by_dfo_id[dfo_id] != role:
                raise ValueError(f"{context}: player {dfo_id} assigned conflicting EV roles")
            if role:
                roles_by_dfo_id[dfo_id] = role

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
                try:
                    player_id = resolver.resolve(team, name, role=roles_by_dfo_id.get(dfo_id))
                except (ValueError, FileNotFoundError, RuntimeError) as exc:
                    player_id = ""
                    resolver.stats["unresolved"] += 1
                    unresolved.append({
                        "CapturedUTC": captured, "date": date_value,
                        "team_abbrev": team, "dfo_player_id": dfo_id,
                        "player_name": name,
                        "position_group": roles_by_dfo_id.get(dfo_id) or "",
                        "reason": str(exc),
                    })
                crosswalk[identity_key] = (normalized(name), player_id)
            player_id = crosswalk[identity_key][1]
            assignment_key = (team, group, position)
            if assignment_key in used_slots:
                raise ValueError(f"{context_player}: duplicate lineup slot {assignment_key}")
            used_slots.add(assignment_key)
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
        # DFO pages may omit expected lineup positions; an empty position
        # means unknown, not an invalid page and not an invented player.
        # Duplicate/unknown slots are already rejected above.
        for group, positions in FIXED_LINEUP_SLOTS.items():
            for position in positions:
                if (team, group, position) not in used_slots:
                    missing_slots.append({
                        "CapturedUTC": captured,
                        "date": date_value,
                        "team_abbrev": team,
                        "lineup_group": group,
                        "lineup_category": GROUP_CATEGORIES[group],
                        "position": position,
                        "reason": "not present in Daily Faceoff source",
                    })

    if not rows or len({r["team_abbrev"] for r in rows}) != 32:
        raise ValueError("Lineup export incomplete")
    return rows, resolver.stats, unresolved, missing_slots


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


def write_unresolved(unresolved, date_value, output_dir=LINEUPS_DIR):
    """Always refresh the audit file, including a header-only file on success."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{date_value}_line_combinations_unresolved.csv"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir,
                                prefix=".lineup_unresolved_", suffix=".tmp", delete=False) as f:
            temporary = f.name
            writer = csv.DictWriter(f, fieldnames=UNRESOLVED_HEADERS, lineterminator="\n",
                                    extrasaction="raise")
            writer.writeheader()
            writer.writerows(unresolved)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_missing_slots(missing_slots, date_value, output_dir=LINEUPS_DIR):
    """Export expected lineup positions absent from DFO, without inventing players."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{date_value}_line_combinations_missing_slots.csv"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir,
                                prefix=".lineup_missing_slots_", suffix=".tmp", delete=False) as f:
            temporary = f.name
            writer = csv.DictWriter(f, fieldnames=MISSING_SLOT_HEADERS,
                                    lineterminator="\n", extrasaction="raise")
            writer.writeheader()
            writer.writerows(missing_slots)
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
    rows, stats, unresolved, missing_slots = build_rows(pages, date_value, captured)
    path = write_csv(rows, date_value)
    report = write_unresolved(unresolved, date_value)
    missing_report = write_missing_slots(missing_slots, date_value)
    print(f"CSV: {path}")
    print(f"Unresolved report: {report} | identities: {len(unresolved)}")
    print(f"Missing lineup slots report: {missing_report} | absent positions: {len(missing_slots)}")
    print(f"Teams: 32/32 | Rows: {len(rows)} | Columns: {len(HEADERS)}")
    print(f"Player crosswalks (unique DFO player IDs): {sum(stats.values())} | "
          + " ".join(f"{key}={value}" for key, value in stats.items()))
    print(f"Rows with scheduled game_id: {sum(bool(row['game_id']) for row in rows)}")
    if missing_slots:
        print(f"WARNING: {len(missing_slots)} expected lineup positions absent from "
              "Daily Faceoff source; existing assignments retained, no players invented", flush=True)
        for slot in missing_slots:
            print(f"  MISSING {slot['team_abbrev']} {slot['lineup_group']} "
                  f"{slot['position']}", flush=True)
    if unresolved:
        print(f"WARNING: {len(unresolved)} team/player identities lack a verified NHL player_id; "
              "their assignment rows were retained with a blank player_id", flush=True)
        for problem in unresolved:
            print(f"  UNRESOLVED {problem['team_abbrev']}: "
                  f"{problem['player_name']} (DFO {problem['dfo_player_id']})", flush=True)



if __name__ == "__main__":
    main()
