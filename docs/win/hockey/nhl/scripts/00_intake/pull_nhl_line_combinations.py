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

# The ONE optional diagnostic CSV: resolved name variants and unresolved IDs.
NAME_MISMATCH_HEADERS = (
    "CapturedUTC", "date", "team_abbrev", "dfo_player_id",
    "dfo_player_name", "position_group", "nhl_roster_name", "nhl_player_id",
    "match_status", "match_source", "match_method", "roster_candidates", "reason",
)

# Exactly the three report types introduced by previous versions of THIS script.
# Clean them up after a successful export; never remove unrelated CSVs.
LEGACY_REPORT_SUFFIXES = ("_unresolved.csv", "_missing_slots.csv", "_source_issues.csv")



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
    """Detect any numbered even-strength line, not just the usual 4F/3D."""
    group = str(group or "").lower().strip()
    if re.fullmatch(r"f[1-9][0-9]*", group):
        return "forwards"
    if re.fullmatch(r"d[1-9][0-9]*", group):
        return "defensemen"
    if group == "g":
        return "goalies"
    return None


def expected_unit(group):
    """Optional expectations used for auditing ONLY; never reject source rows."""
    if re.fullmatch(r"f[1-9][0-9]*", group):
        return "ev", ("lw", "c", "rw")
    if re.fullmatch(r"d[1-9][0-9]*", group):
        return "ev", ("ld", "rd")
    if group == "g":
        return "ev", ("g1", "g2")
    if re.fullmatch(r"pp[1-9][0-9]*", group):
        return "pp", tuple(f"sk{i}" for i in range(1, 6))
    if re.fullmatch(r"pk[1-9][0-9]*", group):
        return "pk", tuple(f"sk{i}" for i in range(1, 5))
    if group == "ir":
        return "oi", ()
    return None, ()


def name_variant_match(source, candidate):
    """Exact surname and first-name equivalence; optional extra middle names."""
    a, b = normalized(source).split(), normalized(candidate).split()
    if len(a) < 2 or len(b) < 2 or a[-1] != b[-1]:
        return False
    first_equivalent = a[0] == b[0] or any({a[0], b[0]} <= variants for variants in NAME_VARIANTS)
    # Generic short/long first names: Max/Maxwell, Chris/Christopher,
    # Dan/Daniel. A minimum three-character prefix plus exact surname is
    # required; same team and same on-ice role are checked by the resolver.
    if not first_equivalent and min(len(a[0]), len(b[0])) >= 3:
        first_equivalent = a[0].startswith(b[0]) or b[0].startswith(a[0])
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
        self.last_match = None
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
            try:
                index, positions = roster_index(path, allow_empty=True, include_roles=True)
            except (ValueError, csv.Error, UnicodeError, OSError) as exc:
                skipped.append(f"{path.name} (invalid data: {exc})")
                continue
            for name, ids in index.items():
                combined.setdefault(name, set()).update(ids)
            for player_id, group in positions.items():
                if player_id in roles and roles[player_id] != group:
                    # Archived roster inconsistency: do not choose an identity.
                    skipped.append(f"{path.name} (conflicting archived role {player_id})")
                    continue
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

    def suggestions(self, team, player_name, role=None):
        """Readable same-surname candidates from loaded, team-scoped rosters.

        These are NOT accepted as NHL IDs. They are diagnostic hints only.
        """
        words = normalized(player_name).split()
        if len(words) < 2:
            return ""
        surname = words[-1]
        options = {}
        for source, roster_attr, role_attr in (
            ("current", "current", "current_roles"),
            ("historical", "history", "history_roles"),
            ("nhl_current", "official_current", "official_current_roles"),
            ("nhl_season", "official_season", "official_season_roles"),
        ):
            roster = getattr(self, roster_attr).get(team, {})
            roles = getattr(self, role_attr).get(team, {})
            for candidate, ids in roster.items():
                if candidate.split()[-1:] != [surname]:
                    continue
                for pid in ids:
                    if role and roles.get(pid) != role:
                        continue
                    options.setdefault(pid, (candidate, source))
        return "; ".join(f"{name.title()} [NHL {pid}, {source}]" for pid, (name, source)
                         in sorted(options.items(), key=lambda item: item[0]))[:900]

    def resolve(self, team, player_name, role=None):
        """Use exact identity first, then a unique same-team, same-role name variant.

        No purely surname-based automatic matches are allowed. The chosen
        NHL roster name/method are retained for the name-mismatch report.
        """
        self.last_match = None
        if role is not None and role not in {"forwards", "defensemen", "goalies"}:
            raise ValueError(f"{team} {player_name}: invalid positional role {role}")
        key = normalized(player_name)

        def check(source, index, positions):
            exact = {i for i in index.get(key, ()) if not role or positions.get(i) == role}
            if len(exact) > 1:
                raise ValueError(f"{team}: {player_name!r} ambiguous {source} roster IDs {sorted(exact)}")
            if exact:
                self.stats[source] += 1
                self.last_match = {"nhl_roster_name": key.title(), "source": source, "method": "exact"}
                return next(iter(exact))
            variants = {}
            for candidate, ids in index.items():
                if candidate != key and name_variant_match(key, candidate):
                    for pid in ids:
                        if not role or positions.get(pid) == role:
                            variants[pid] = candidate
            if len(variants) > 1:
                raise ValueError(f"{team}: {player_name!r} ambiguous {source} name variants "
                                 f"{sorted(variants)}")
            if variants:
                pid, matched_name = next(iter(variants.items()))
                self.stats[source] += 1
                self.last_match = {"nhl_roster_name": matched_name.title(),
                                   "source": source, "method": "verified_name_variant"}
                return pid
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
        self.last_match = {"nhl_roster_name": player_name, "source": "nhl_search", "method": "exact"}
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
    """Keep DFO's raw assignments; unit structure is not a fixed schema.

    A fourth defensive pair (d4), extra forward line, third power-play unit,
    or a newly introduced group must not invalidate the other 32 team pages.
    Empty fields are kept as empty strings and surfaced in source_issues.csv.
    """
    return tuple(str(player.get(field) or "").strip().lower() for field in (
        "groupIdentifier", "categoryIdentifier", "positionIdentifier"))


def build_rows(pages, date_value, captured, team_map_path=TEAM_MAP_PATH,
               schedule_dir=SCHEDULE_DIR, roster_dir=ROSTER_DIR):
    if set(pages) != set(TEAMS):
        raise ValueError(f"Expected 32 teams, received {len(pages)}: missing={sorted(set(TEAMS)-set(pages))}")
    teams = read_team_map(team_map_path)
    scheduled = read_schedule(date_value, teams, schedule_dir)
    resolver = PlayerResolver(roster_dir, game_date=date_value, team_map=teams)
    rows = []
    crosswalk = {}
    name_mismatches = []
    source_issues = []  # In-memory only; never emitted as an extra file.

    for team in TEAMS:
        page = pages[team]
        context = f"Daily Faceoff {team}"
        if not isinstance(page, dict):
            raise ValueError(f"{context}: invalid page payload")
        if required(page, "teamAbbreviation", context).upper() != team or required(page, "teamSlug", context) != TEAMS[team]:
            # Wrong team pages cannot safely be attributed to the requested team.
            raise ValueError(f"{context}: wrong team/slug in payload")
        if normalized(required(page, "teamName", context)) != normalized(teams[team]["team"]):
            raise ValueError(f"{context}: team name doesn't match official NHL mapping")
        players = page.get("players")
        if not isinstance(players, list) or not players:
            # No lineup data is not a complete 32-team scrape.
            raise ValueError(f"{context}: missing player assignments")

        def note(i, player, issue, details):
            source_issues.append({
                "CapturedUTC": captured, "date": date_value, "team_abbrev": team,
                "player_index": i if i is not None else "",
                "player_name": str((player or {}).get("name") or "").strip(),
                "lineup_group": str((player or {}).get("groupIdentifier") or "").strip().lower(),
                "lineup_category": str((player or {}).get("categoryIdentifier") or "").strip().lower(),
                "position": str((player or {}).get("positionIdentifier") or "").strip().lower(),
                "issue": issue, "details": details,
            })

        try:
            last_updated = timestamp(required(page, "updatedAt", context), context)
        except ValueError as exc:
            last_updated = ""
            note(None, None, "invalid_lineup_timestamp", str(exc))

        used_slots = set()
        observed_groups = set()
        roles_by_dfo_id = {}
        conflicting_roles = set()
        source_names_by_dfo_id = {}

        # Even-strength group identifies the player's primary position. Dynamic
        # F/D numbering is intentional: e.g. d4 is a legitimate DFO group.
        for i, player in enumerate(players):
            if not isinstance(player, dict):
                note(i, None, "invalid_player_record", "player array element is not an object")
                continue
            dfo_id = str(player.get("playerId") or "").strip()
            role = role_of_lineup_group(player.get("groupIdentifier"))
            if role and dfo_id:
                previous = roles_by_dfo_id.get(dfo_id)
                if previous and previous != role:
                    conflicting_roles.add(dfo_id)
                    note(i, player, "conflicting_even_strength_roles",
                         f"same DFO player listed as {previous} and {role}; NHL ID must be disambiguated")
                else:
                    roles_by_dfo_id[dfo_id] = role

        for i, player in enumerate(players):
            if not isinstance(player, dict):
                # Preserve source-array coverage with a blank assignment row
                # rather than pretend this slot never existed.
                player = {}
            dfo_id = str(player.get("playerId") or "").strip()
            if not dfo_id.isdecimal() or int(dfo_id) <= 0:
                note(i, player, "missing_or_invalid_dfo_player_id", repr(dfo_id))
                dfo_id = ""
            name = str(player.get("name") or "").strip()
            if not name:
                note(i, player, "missing_player_name", "Cannot resolve player identity")
            group, category, position = validate_group(player, f"{context} player[{i}]")
            if not group or not category or not position:
                note(i, player, "missing_assignment_field",
                     "Source lacks one or more of groupIdentifier/categoryIdentifier/positionIdentifier")
            observed_groups.add(group) if group else None
            expected_category, expected_positions = expected_unit(group)
            if expected_category is None and group:
                note(i, player, "new_source_group", "Unrecognized group retained exactly as supplied")
            elif expected_category and category and category != expected_category:
                note(i, player, "unexpected_category",
                     f"expected {expected_category}, source supplied {category}; retained source value")
            if expected_positions and position and position not in expected_positions:
                note(i, player, "unusual_source_position",
                     f"source position is outside usual slots {expected_positions}; retained")
            assignment_key = (team, group, position)
            if group and position and assignment_key in used_slots:
                note(i, player, "duplicate_source_slot",
                     f"Source repeated group={group!r}, position={position!r}; both rows retained")
            used_slots.add(assignment_key)

            if dfo_id:
                old_name = source_names_by_dfo_id.get(dfo_id)
                if old_name is not None and old_name != normalized(name):
                    note(i, player, "dfo_id_name_conflict",
                         f"DFO ID {dfo_id} appears with more than one player name; resolved independently")
                else:
                    source_names_by_dfo_id[dfo_id] = normalized(name)
            identity_key = (team, dfo_id or f"missing_at_row_{i}", normalized(name))
            if identity_key not in crosswalk:
                if not name:
                    resolver.last_match = None
                    reason = "Cannot resolve NHL ID: source player name missing"
                    player_id = ""
                else:
                    role = None if dfo_id in conflicting_roles else roles_by_dfo_id.get(dfo_id)
                    try:
                        player_id = resolver.resolve(team, name, role=role)
                        reason = ""
                    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                        player_id = ""
                        reason = str(exc)
                match = resolver.last_match or {}
                if not player_id:
                    resolver.stats["unresolved"] += 1
                if not player_id or (match.get("method") == "verified_name_variant"):
                    position_role = "" if dfo_id in conflicting_roles else (roles_by_dfo_id.get(dfo_id) or "")
                    name_mismatches.append({
                        "CapturedUTC": captured, "date": date_value,
                        "team_abbrev": team, "dfo_player_id": dfo_id,
                        "dfo_player_name": name, "position_group": position_role,
                        "nhl_roster_name": match.get("nhl_roster_name", ""),
                        "nhl_player_id": player_id,
                        "match_status": "matched_variant" if player_id else "unresolved",
                        "match_source": match.get("source", ""),
                        "match_method": match.get("method", ""),
                        "roster_candidates": "" if player_id else resolver.suggestions(team, name, role=position_role or None),
                        "reason": reason,
                    })
                crosswalk[identity_key] = player_id
            player_id = crosswalk[identity_key]

            gtd = player.get("gameTimeDecision")
            if isinstance(gtd, bool):
                gtd_value = "true" if gtd else "false"
            elif isinstance(gtd, str) and gtd.strip().lower() in {"true", "false"}:
                gtd_value = gtd.strip().lower()
            else:
                # Missing/null is unknown, not 'false'.
                gtd_value = ""
                if gtd is not None:
                    note(i, player, "invalid_game_time_decision",
                         f"Unexpected value {gtd!r}; kept blank")

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
                "game_time_decision": gtd_value,
                "lineup_updated_utc": last_updated,
                "lineup_source_name": str(page.get("sourceName") or "").strip(),
                "lineup_source_url": str(page.get("source") or "").strip(),
                "lineup_page_url": f"{SOURCE_ROOT}/{TEAMS[team]}/line-combinations",
            })

        # Source lineups may be incomplete; an empty lineup position is not
        # a player record and is not fabricated or reported as an error.

    if not rows or len({r["team_abbrev"] for r in rows}) != 32:
        raise ValueError("Lineup export incomplete: at least one team had no player assignments")
    return rows, resolver.stats, name_mismatches


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


def write_name_mismatches(mismatches, date_value, output_dir=LINEUPS_DIR):
    """Write ONE useful diagnostic file, and only when mismatches exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{date_value}_line_combinations_name_mismatches.csv"
    if not mismatches:
        if target.is_file():
            target.unlink()  # Prevent stale name-mismatch reports.
        return None
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir,
                                prefix=".lineup_name_mismatch_", suffix=".tmp", delete=False) as f:
            temporary = f.name
            writer = csv.DictWriter(f, fieldnames=NAME_MISMATCH_HEADERS, lineterminator="\n",
                                    extrasaction="raise")
            writer.writeheader()
            writer.writerows(mismatches)
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return target


def remove_unwanted_previous_reports(output_dir=LINEUPS_DIR):
    """Remove ONLY CSV report types created by earlier lineup scraper versions.

    Includes November 9 if a later day's scrape performs the cleanup. No
    goalie CSV, schedule CSV, main lineup CSV, or other report is touched.
    """
    removed = []
    for suffix in LEGACY_REPORT_SUFFIXES:
        for path in output_dir.glob(f"????-??-??_line_combinations{suffix}"):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    return sorted(removed)


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
    rows, stats, name_mismatches = build_rows(pages, date_value, captured)
    path = write_csv(rows, date_value)
    mismatch_report = write_name_mismatches(name_mismatches, date_value)
    removed = remove_unwanted_previous_reports()
    print(f"CSV: {path}")
    if mismatch_report:
        print(f"Name mismatches CSV: {mismatch_report} | "
              f"matched={sum(m['match_status'] == 'matched_variant' for m in name_mismatches)} "
              f"unresolved={sum(m['match_status'] == 'unresolved' for m in name_mismatches)}")
    print(f"Teams: 32/32 | Rows: {len(rows)} | Columns: {len(HEADERS)}")
    print(f"Player identities checked: {sum(stats.values())} | "
          + " ".join(f"{key}={value}" for key, value in stats.items()))
    print(f"Rows with scheduled game_id: {sum(bool(row['game_id']) for row in rows)}")
    if removed:
        print(f"Removed {len(removed)} unwanted reports from previous script versions: "
              + ", ".join(removed))
    if any(m["match_status"] == "unresolved" for m in name_mismatches):
        print("WARNING: See the name mismatches CSV for unresolved NHL player IDs.")



if __name__ == "__main__":
    main()
