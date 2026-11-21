#!/usr/bin/env python3
"""Build strictly pregame NHL player, team, and eligible goalie features.

Historical mode: one feature row per (game_id, player_id) in historical targets.
  Every player/team statistic uses games with game_date STRICTLY BEFORE the
  target game's date. All games on the same date are held back together.
  This deliberate rule prevents leakage when exact historical starts are absent.

Slate mode: candidate players from dated pregame line-combinations snapshots;
  candidate goalies from dated Daily Faceoff goalie snapshots. Snapshot fields
  are accepted only when CapturedUTC < official NHL start_time_utc AND
  CapturedUTC <= the actual prediction as-of UTC timestamp. It refuses
  slate production if the target file lacks observations from that NHL season;
  use a complete current-season player-game history before live scoring.

Historical team information is calculated from earlier games, including
past skater offense, defense, and team goaltending outcomes. Dated goalie
names/statuses and time-stamped team ratings are available only when there
is a matched official pregame start timestamp.

No target-game outcomes are placed in the generated feature columns. Inputs
and repository are read-only; files are emitted only to --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

VERSION = "2026-11-17-asof-v2-prop-engine-rolefix1"
TARGET_BASE = "nhl_historical_player_game_targets_2009_2025"
STAT_FIELDS = ("shots_on_goal", "points", "assists", "goals", "blocked_shots", "power_play_goals", "saves", "goals_against")
TEAM_METRICS = ("goals_for", "goals_against", "shots_on_goal", "blocked_shots", "goalie_saves", "goalie_goals_against")
HISTORY_MEASURES = ("prior_career", "prior_season", "last5", "last10")
REQUIRED_TARGET = ("season", "game_id", "game_date", "game_type", "player_id", "team", "opponent_team", "home_away", "position_group", "participated", *STAT_FIELDS)
REQUIRED_DFO = ("game_id", "CapturedUTC", "home_player_id", "away_player_id", "homeNewsStrengthName", "awayNewsStrengthName")
OUTPUT_BASE = "nhl_player_features_asof"
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BASE_COLUMNS = ("season", "game_id", "game_date", "game_type", "game_start_utc", "player_id", "team", "opponent_team", "home_away", "position_group", "candidate_source", "candidate_captured_utc", "prior_player_games_career", "prior_player_games_season", "prior_player_last_game_date", "prior_player_days_since_last_game")
PLAYER_COLUMNS = tuple(f"player_{s}_mean_{h}" for s in STAT_FIELDS for h in HISTORY_MEASURES)
TEAM_COLUMNS = tuple(f"{prefix}_{m}_mean_{scope}" for prefix in ("team", "opponent") for m in TEAM_METRICS for scope in ("prior_season", "last5")) + ("team_games_prior_season", "opponent_games_prior_season")
GOALIE_COLUMNS = ("home_goalie_id_pre", "away_goalie_id_pre", "home_goalie_status_pre", "away_goalie_status_pre", "home_goalie_news_captured_utc", "away_goalie_news_captured_utc", "home_goalie_id_captured_utc", "away_goalie_id_captured_utc", "home_goalie_prior_games", "away_goalie_prior_games", "home_goalie_prior_saves_mean_last5", "away_goalie_prior_saves_mean_last5", "home_goalie_prior_goals_against_mean_last5", "away_goalie_prior_goals_against_mean_last5")
RATING_COLUMNS = tuple(f"{side}_{feature}_asof" for side in ("team", "opponent") for feature in ("adj_xgf", "adj_xga", "net_z")) + tuple(f"{side}_{feature}_captured_utc" for side in ("team", "opponent") for feature in ("adj_xgf", "adj_xga", "net_z")) + ("team_rating_captured_utc", "opponent_rating_captured_utc")
OUTPUT_COLUMNS = (*BASE_COLUMNS, *PLAYER_COLUMNS, *TEAM_COLUMNS, *GOALIE_COLUMNS, *RATING_COLUMNS)


def fail(msg):
    raise ValueError(msg)


def read_csv(path: Path, required=()):
    if not path.is_file():
        fail(f"Required input not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            fail(f"CSV has no header: {path}")
        missing = set(required) - set(rdr.fieldnames)
        if missing:
            fail(f"{path}: required columns missing: {sorted(missing)}")
        yield from rdr


def dt_utc(raw: object, name: str):
    s = str(raw or "").strip()
    if not s:
        fail(f"{name}: UTC timestamp is blank")
    try:
        result = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid timestamp {s!r}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        fail(f"{name}: timestamp without UTC offset: {s!r}")
    return result.astimezone(timezone.utc)


def to_iso(dt: datetime):
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def checked_date(raw, name):
    s = str(raw or "").strip().replace("_", "-")
    if not DATE_PATTERN.fullmatch(s):
        fail(f"{name}: unsupported date {s!r}")
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"{name}: invalid date {s!r}") from exc


def canonical_id(raw, name):
    s = str(raw or "").strip()
    if not s.isdigit() or int(s) == 0:
        fail(f"{name}: invalid ID {s!r}")
    return s


def numeric_stat(raw, context):
    s = str(raw or "").strip()
    if s == "":
        return None
    try:
        n = int(s)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid noninteger statistic {s!r}") from exc
    if n < 0:
        fail(f"{context}: negative statistic {s!r}")
    return n


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class History:
    sums: dict = field(default_factory=lambda: defaultdict(int))
    counts: dict = field(default_factory=lambda: defaultdict(int))
    sums5: dict = field(default_factory=lambda: defaultdict(int))
    counts5: dict = field(default_factory=lambda: defaultdict(int))
    sums10: dict = field(default_factory=lambda: defaultdict(int))
    counts10: dict = field(default_factory=lambda: defaultdict(int))
    n: int = 0
    last5: deque = field(default_factory=deque)
    last10: deque = field(default_factory=deque)
    last_date: date | None = None

    @staticmethod
    def update_window(queue, sums, counts, size, datum):
        if len(queue) == size:
            old = queue.popleft()
            for k, v in old.items():
                if v is not None:
                    sums[k] -= v
                    counts[k] -= 1
        queue.append(datum)
        for k, v in datum.items():
            if v is not None:
                sums[k] += v
                counts[k] += 1

    def add(self, datum: dict, day: date):
        self.n += 1
        for k, v in datum.items():
            if v is not None:
                self.sums[k] += v
                self.counts[k] += 1
        self.update_window(self.last5, self.sums5, self.counts5, 5, datum)
        self.update_window(self.last10, self.sums10, self.counts10, 10, datum)
        self.last_date = day

    def mean(self, key: str, scope: str):
        if scope in ("prior_career", "prior_season"):
            sums, counts = self.sums, self.counts
        elif scope == "last5":
            sums, counts = self.sums5, self.counts5
        else:
            sums, counts = self.sums10, self.counts10
        count = counts.get(key, 0)
        return round(sums[key] / count, 8) if count else ""


class FeatureState:
    def __init__(self):
        self.player_career = defaultdict(History)
        self.player_season = defaultdict(History)
        self.team_season = defaultdict(History)
        self.team_seasons = set()

    def features(self, record: dict, d: date, year: str):
        pid = record["player_id"]
        role = record["position_group"]
        career = self.player_career.get((pid, role))
        season = self.player_season.get((year, pid, role))
        out = {
            "prior_player_games_career": career.n if career else 0,
            "prior_player_games_season": season.n if season else 0,
            "prior_player_last_game_date": career.last_date.isoformat() if career and career.last_date else "",
            "prior_player_days_since_last_game": (d - career.last_date).days if career and career.last_date else "",
        }
        for s in STAT_FIELDS:
            for scope in HISTORY_MEASURES:
                h = season if scope == "prior_season" else career
                out[f"player_{s}_mean_{scope}"] = h.mean(s, scope) if h else ""
        for team_type, team in (("team", record["team"]), ("opponent", record["opponent_team"])):
            h = self.team_season.get((year, team))
            out[f"{team_type}_games_prior_season"] = h.n if h else 0
            for metric in TEAM_METRICS:
                for scope in ("prior_season", "last5"):
                    out[f"{team_type}_{metric}_mean_{scope}"] = h.mean(metric, scope) if h else ""
        return out

    def goalie_history(self, goalie_id: str):
        if not goalie_id:
            return 0, "", ""
        h = self.player_career.get((goalie_id, "GOALIE"))
        return (h.n, h.mean("saves", "last5"), h.mean("goals_against", "last5")) if h else (0, "", "")

    def update_day(self, rows: list[dict], d: date):
        # First aggregate each game's completed skater totals for each team.
        games = defaultdict(lambda: {"teams": defaultdict(lambda: defaultdict(int)), "opponents": {}})
        for rec in rows:
            if rec["participated"] != 1:
                continue
            stats = {s: rec[s] for s in STAT_FIELDS}
            self.player_career[(rec["player_id"], rec["position_group"])].add(stats, d)
            self.player_season[(rec["season"], rec["player_id"], rec["position_group"])].add(stats, d)
            gm = games[(rec["season"], rec["game_id"])]
            gm["opponents"][rec["team"]] = rec["opponent_team"]
            team = gm["teams"][rec["team"]]
            if rec["position_group"] == "SKATER":
                team["goals_for"] += rec["goals"]
                team["shots_on_goal"] += rec["shots_on_goal"]
                team["blocked_shots"] += rec["blocked_shots"]
            elif rec["position_group"] == "GOALIE":
                team["goalie_saves"] += rec["saves"]
                team["goalie_goals_against"] += rec["goals_against"]
        for (season, gid), gm in games.items():
            teams = gm["teams"]
            for team, opponent in gm["opponents"].items():
                if team not in teams or opponent not in teams:
                    fail(f"Game {gid} has no played players for both teams: {team} vs {opponent}")
            for team, opponent in gm["opponents"].items():
                data = {metric: teams[team][metric] for metric in TEAM_METRICS if metric != "goals_against"}
                data["goals_against"] = teams[opponent]["goals_for"]
                self.team_season[(season, team)].add(data, d)


def stream_staged(conn):
    cur = conn.execute("SELECT season,game_id,game_date,game_type,player_id,team,opponent_team,home_away,position_group,participated," + ",".join(STAT_FIELDS) + " FROM targets ORDER BY game_date,game_id,player_id")
    columns = [d[0] for d in cur.description]
    for tup in cur:
        d = dict(zip(columns, tup))
        yield d


def ingest(target_path: Path, conn):
    conn.execute("CREATE TABLE targets(season TEXT NOT NULL, game_id TEXT NOT NULL, game_date TEXT NOT NULL, game_type TEXT NOT NULL, player_id TEXT NOT NULL, team TEXT NOT NULL, opponent_team TEXT NOT NULL, home_away TEXT NOT NULL, position_group TEXT NOT NULL, participated INTEGER NOT NULL," + ",".join(s+" INTEGER" for s in STAT_FIELDS) + ",PRIMARY KEY(game_id,player_id)) WITHOUT ROWID")
    insert = "INSERT INTO targets VALUES(" + ",".join("?" for _ in range(10+len(STAT_FIELDS))) + ")"
    counts = defaultdict(int)
    seasons = set()
    first = last = None
    for i, r in enumerate(read_csv(target_path, REQUIRED_TARGET), 1):
        gid = canonical_id(r["game_id"], f"row {i} game_id")
        pid = canonical_id(r["player_id"], f"row {i} player_id")
        gt = r["game_type"].strip()
        if gt not in ("2", "3") or int(gid[4:6]) != int(gt):
            fail(f"row {i}: game type disagreeing with official game ID")
        if len(gid) != 10:
            fail(f"row {i}: game ID not 10 digits: {gid}")
        d = checked_date(r["game_date"], f"row {i} game_date")
        season = r["season"].strip()
        if not re.fullmatch(r"\d{4}-\d{2}", season) or int(gid[:4]) != int(season[:4]):
            fail(f"row {i}: invalid season/game_id relationship")
        if gid in ("2009021019", "2010020868", "2019030001"):
            fail(f"row {i}: previously excluded game returned: {gid}")
        role = r["position_group"].strip().upper()
        if role not in ("SKATER", "GOALIE"):
            fail(f"row {i}: invalid position_group {role!r}")
        part = r["participated"].strip()
        if part not in ("0", "1"):
            fail(f"row {i}: invalid participated {part!r}")
        played = int(part)
        stats = [numeric_stat(r[k], f"row {i} {k}") for k in STAT_FIELDS]
        if not played and any(v is not None for v in stats):
            fail(f"row {i}: nonparticipant has nonblank outcome")
        # Power-play goals are an optional observed target in Item 3; do not
        # invent them or reject otherwise valid skater histories when blank.
        eligible = STAT_FIELDS[:5] if role == "SKATER" else STAT_FIELDS[6:]
        if played and any(stats[STAT_FIELDS.index(k)] is None for k in eligible):
            fail(f"row {i}: participating {role} missing required outcomes")
        team, opponent = r["team"].strip().upper(), r["opponent_team"].strip().upper()
        if not team or not opponent or team == opponent:
            fail(f"row {i}: invalid teams {team!r}/{opponent!r}")
        orientation = r["home_away"].strip().upper()
        if orientation not in ("HOME", "AWAY"):
            fail(f"row {i}: invalid orientation {orientation!r}")
        try:
            conn.execute(insert, (season, gid, d.isoformat(), gt, pid, team, opponent, orientation, role, played, *stats))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate player/game at input row {i}: {gid}/{pid}") from exc
        counts["rows"] += 1
        counts["played" if played else "not_played"] += 1
        seasons.add(season)
        first = min(first, d.isoformat()) if first else d.isoformat()
        last = max(last, d.isoformat()) if last else d.isoformat()
        if i % 100000 == 0:
            conn.commit()
            print(f"  INGEST | {i:,} player-game records", flush=True)
    conn.commit()
    if not counts["rows"]:
        fail("Historical targets are empty")
    conn.execute("CREATE INDEX targets_date_ix ON targets(game_date, game_id)")
    conn.commit()
    return {**dict(counts), "seasons": sorted(seasons), "first_date": first, "last_date": last}


def find_nhl_root(script: Path):
    for folder in (script.parent, *script.parents):
        if (folder/"00_intake"/"lineups").is_dir() and (folder/"research"/"sdv_history").is_dir():
            return folder
    return None


def find_default_target(output_dir: Path):
    """Select latest Item-3 complete-season target file, not a frozen filename.

    Future Item-3 builds generate suffixes such as 2009_2026.  Explicit
    --targets takes precedence.  Only exact training-data filenames qualify;
    summaries, manifests, and partial files are never eligible.
    """
    pattern = re.compile(r"^nhl_historical_player_game_targets_(\d{4})_(\d{4})\.csv$")
    options = []
    for path in output_dir.glob("nhl_historical_player_game_targets_*.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            first, last = map(int, match.groups())
            if 2000 <= first <= last <= 2100 and path.is_file():
                options.append((last, -first, path))
    if not options:
        fail(f"Historical targets CSV not found under {output_dir}; provide --targets")
    return max(options, key=lambda item: (item[0], item[1]))[2]


def official_slate(root: Path, slate_date: str):
    path = root/"00_intake"/"nhl_schedule"/("NHL_"+slate_date.replace("-", "_")+".csv")
    games = {}
    for r in read_csv(path, ("game_id", "game_date", "start_time_utc", "home_team_abbrev", "away_team_abbrev")):
        gid = canonical_id(r["game_id"], "official slate game_id")
        if gid in games:
            fail(f"Duplicate official game ID {gid}")
        if checked_date(r["game_date"], "official slate date").isoformat() != slate_date:
            fail(f"Wrong date for official slate game {gid}")
        start = dt_utc(r["start_time_utc"], f"official game {gid}")
        home = str(r["home_team_abbrev"] or "").strip().upper()
        away = str(r["away_team_abbrev"] or "").strip().upper()
        if not home or not away or home == away:
            fail(f"Missing official teams for game {gid}")
        games[gid] = {"date": slate_date, "start": start, "home": home, "away": away,
                      "season": f"{int(gid[:4])}-{str((int(gid[:4])+1)%100).zfill(2)}"}
    if not games:
        fail(f"Official slate schedule empty: {path}")
    return games, path


def dfo_snapshots(lineups_dir: Path):
    out = defaultdict(list)
    paths = sorted(lineups_dir.glob("????-??-??_goalies.csv")) if lineups_dir.is_dir() else []
    for path in paths:
        for ix, r in enumerate(read_csv(path, REQUIRED_DFO), 2):
            gid = canonical_id(r["game_id"], f"{path.name} row {ix}")
            stamp = dt_utc(r["CapturedUTC"], f"{path.name} row {ix} CapturedUTC")
            out[gid].append((stamp, r, path.name))
    for gid in out:
        out[gid].sort(key=lambda x: (x[0], x[2]))
    return out, len(paths)


def pregame_goalie_data(gid: str, start: datetime, teams: dict, snapshot_store: dict, observed_at: datetime | None = None):
    state = {side: {"id": "", "status": "", "id_time": "", "status_time": ""} for side in ("home", "away")}
    used = 0
    ignored = 0
    for stamp, row, source in snapshot_store.get(gid, []):
        if stamp >= start or (observed_at is not None and stamp > observed_at):
            ignored += 1
            continue
        # Ensure a snapshot for this game refers to the same official teams.
        for side in ("home", "away"):
            abbrev = row.get("homeTeamAbbreviation" if side == "home" else "awayTeamAbbreviation", "").strip().upper()
            if abbrev and abbrev != teams[side]:
                fail(f"DFO snapshot game {gid} ({source}) has incorrect {side} team {abbrev!r}, expected {teams[side]}")
            new_id = str(row.get(f"{side}_player_id") or "").strip()
            new_status = str(row.get(f"{side}NewsStrengthName") or "").strip()
            current = state[side]
            if new_id:
                new_id = canonical_id(new_id, f"DFO {gid} {side} goalie")
                if new_id != current["id"]:
                    # A changed named goalie invalidates the old goalie's status.
                    current["status"] = ""
                    current["status_time"] = ""
                current["id"] = new_id
                current["id_time"] = to_iso(stamp)
                if new_status:
                    current["status"] = new_status
                    current["status_time"] = to_iso(stamp)
            # No player ID means status cannot safely be attached to anyone.
        used += 1
    return state, used, ignored


def team_rating_snapshots(path: Path | None):
    out = defaultdict(list)
    if path is None or not path.is_file():
        return out, 0
    for ix, r in enumerate(read_csv(path, ("game_id", "team", "pregame_cutoff_utc", "ratings_as_of_utc", "adj_xgf", "adj_xga", "net_z")), 2):
        gid = canonical_id(r["game_id"], f"team rating row {ix}")
        t = str(r["team"] or "").strip().upper()
        stamp = dt_utc(r["ratings_as_of_utc"], f"team rating row {ix}")
        cutoff = dt_utc(r["pregame_cutoff_utc"], f"team rating cutoff row {ix}")
        out[(gid, t)].append((stamp, cutoff, r))
    for k in out:
        out[k].sort(key=lambda x:x[0])
    return out, sum(map(len, out.values()))


def eligible_rating(game_id, team, start, ratings, observed_at=None):
    if not start:
        return None
    values = {}
    times = {}
    for stamp, cutoff, row in ratings.get((game_id, team), []):
        if stamp >= start or (observed_at is not None and stamp > observed_at) or cutoff != start:
            continue
        for col in ("adj_xgf", "adj_xga", "net_z"):
            raw = str(row.get(col) or "").strip()
            if not raw:
                continue  # Newer blank fields must never erase older observations.
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"Team rating for {game_id}/{team} has invalid {col}={raw!r}") from exc
            if not math.isfinite(value):
                fail(f"Team rating for {game_id}/{team} has non-finite {col}={raw!r}")
            values[col] = raw
            times[col] = stamp
    return (values, times) if values else None


def augment_game(out, record, start, goalies, ratings, state, observed_at=None):
    out.update({k:"" for k in GOALIE_COLUMNS})
    out.update({k:"" for k in RATING_COLUMNS})
    if not start:
        return
    home = record["team"] if record["home_away"] == "HOME" else record["opponent_team"]
    away = record["opponent_team"] if record["home_away"] == "HOME" else record["team"]
    sides, _, _ = pregame_goalie_data(record["game_id"], start, {"home":home,"away":away}, goalies, observed_at)
    for side in ("home", "away"):
        s = sides[side]
        out[f"{side}_goalie_id_pre"] = s["id"]
        out[f"{side}_goalie_status_pre"] = s["status"]
        out[f"{side}_goalie_news_captured_utc"] = s["status_time"]
        out[f"{side}_goalie_id_captured_utc"] = s["id_time"]
        games, saves, goals = state.goalie_history(s["id"])
        out[f"{side}_goalie_prior_games"] = games if s["id"] else ""
        out[f"{side}_goalie_prior_saves_mean_last5"] = saves
        out[f"{side}_goalie_prior_goals_against_mean_last5"] = goals
    for prefix, team in (("team",record["team"]),("opponent",record["opponent_team"])):
        entry = eligible_rating(record["game_id"], team, start, ratings, observed_at)
        if entry:
            values, times = entry
            for col, raw in values.items():
                out[f"{prefix}_{col}_asof"] = raw
                out[f"{prefix}_{col}_captured_utc"] = to_iso(times[col])
            out[f"{prefix}_rating_captured_utc"] = to_iso(max(times.values()))


def historical_groups(conn):
    group = []
    current = None
    for r in stream_staged(conn):
        if r["game_date"] != current:
            if group:
                yield current, group
            current = r["game_date"]
            group = []
        group.append(r)
    if group:
        yield current, group


def candidate_slate(root: Path, slate_date: str, schedule: dict, snapshots: dict, observed_at: datetime | None = None):
    path = root/"00_intake"/"lineups"/(slate_date+"_line_combinations.csv")
    candidates = {}
    counts = defaultdict(int)
    if path.is_file():
        for ix, r in enumerate(read_csv(path, ("game_id", "CapturedUTC", "player_id", "team_abbrev", "lineup_group", "lineup_category", "position", "lineup_updated_utc")), 2):
            gid = canonical_id(r["game_id"], f"{path.name} row {ix} game_id")
            if gid not in schedule:
                continue
            stamped = dt_utc(r["CapturedUTC"], f"{path.name} row {ix} CapturedUTC")
            if stamped >= schedule[gid]["start"] or (observed_at is not None and stamped > observed_at):
                counts["late_lineup_rows"] += 1
                continue
            if r["lineup_updated_utc"].strip() and dt_utc(r["lineup_updated_utc"], f"{path.name} row {ix} lineup_updated_utc") > stamped:
                counts["lineup_observed_after_capture"] += 1
                continue
            pid = canonical_id(r["player_id"], f"{path.name} row {ix} player_id")
            team = r["team_abbrev"].strip().upper()
            sc = schedule[gid]
            if team not in (sc["home"],sc["away"]):
                fail(f"Lineup player {pid} in game {gid} has wrong team {team}")
            side = "HOME" if team == sc["home"] else "AWAY"
            # The lineup's 'position' is a SLOT, not an NHL position:
            # goalies are g1/g2 with lineup_group=g, while PP/PK slots
            # are sk1..sk5 and injury-only rows use category=oi.
            # Only even-strength lineup participants are valid candidates.
            group = r["lineup_group"].strip().lower()
            category = r["lineup_category"].strip().lower()
            pos = r["position"].strip().lower()
            if category != "ev":
                counts["non_ev_lineup_rows_skipped"] += 1
                continue
            if group == "g":
                if not re.fullmatch(r"g\d+", pos):
                    fail(f"Invalid goalie lineup slot {pos!r} for game {gid} player {pid}")
                role = "GOALIE"
            elif re.fullmatch(r"[fd]\d+", group):
                if pos not in ("lw", "c", "rw", "ld", "rd"):
                    fail(f"Invalid skater lineup slot {pos!r} for game {gid} player {pid}")
                role = "SKATER"
            else:
                fail(f"Unknown even-strength lineup group {group!r} for game {gid} player {pid}")
            rec = {"season":sc["season"], "game_id":gid,"game_date":slate_date,"game_type":gid[4:6].lstrip("0"),"player_id":pid,
                   "team":team,"opponent_team":sc["away"] if side=="HOME" else sc["home"],
                   "home_away":side,"position_group":role,"candidate_source":"LINE_COMBINATIONS", "candidate_captured_utc":to_iso(stamped)}
            key = (gid,pid)
            old = candidates.get(key)
            if old and (old["position_group"] != role or old["team"] != team):
                fail(f"Conflicting lineup identity for game {gid} player {pid}")
            if not old or stamped > dt_utc(old["candidate_captured_utc"], "stored lineup capture"):
                candidates[key] = rec
        counts["lineup_candidates"] = len(candidates)
    # Include goalie candidates if official pregame snapshot has positive identified goalie ID.
    for gid, sc in schedule.items():
        sides, _, _ = pregame_goalie_data(gid, sc["start"], {"home":sc["home"], "away":sc["away"]}, snapshots, observed_at)
        for side in ("home", "away"):
            goalie = sides[side]
            if not goalie["id"]:
                continue
            team, opp = ((sc["home"], sc["away"]) if side == "home" else (sc["away"],sc["home"]))
            key = (gid, goalie["id"])
            rec = {"season":sc["season"],"game_id":gid,"game_date":slate_date,"game_type":gid[4:6].lstrip("0"),"player_id":goalie["id"],
                   "team":team,"opponent_team":opp,"home_away":side.upper(),"position_group":"GOALIE",
                   "candidate_source":"DAILY_FACEOFF","candidate_captured_utc":goalie["id_time"]}
            if key not in candidates:
                candidates[key]=rec
                counts["goalie_candidates_added"] += 1
            elif candidates[key]["position_group"] != "GOALIE" or candidates[key]["team"] != team:
                fail(f"Inconsistent goalie identity (role/team) for game {gid} player {goalie['id']}")
    for gid, sc in schedule.items():
        for team in (sc["home"], sc["away"]):
            if not any(c["game_id"] == gid and c["team"] == team and c["position_group"] == "SKATER" for c in candidates.values()):
                fail(f"Pregame slate game {gid} lacks prestart skater candidates for {team}; cannot publish a partial player slate")
    return sorted(candidates.values(),key=lambda x:(x["game_id"],x["player_id"])),dict(counts),path


def write_outputs(args, nhl_root, targets, output_dir):
    # Generated artifacts are isolated in the explicitly selected output folder,
    # never the repository unless the caller explicitly chooses that folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    goalie_dir = args.goalie_dir or (nhl_root/"00_intake"/"lineups" if nhl_root else None)
    if args.slate_date and (goalie_dir is None or not goalie_dir.is_dir()):
        fail("Slate requires existing Daily Faceoff snapshot directory; provide --goalie-dir or --nhl-root")
    snapshots, dfo_files = dfo_snapshots(goalie_dir) if goalie_dir and goalie_dir.is_dir() else ({}, 0)
    rating_file = args.team_ratings or (nhl_root/"sdv"/"team-strength"/"latest_team_ratings_asof.csv" if nhl_root else None)
    ratings, rating_count = team_rating_snapshots(rating_file)
    with tempfile.TemporaryDirectory(prefix="nhl_features_build_",dir=output_dir) as tmp:
        work = Path(tmp)
        db = sqlite3.connect(str(work/"targets.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            stats=ingest(targets,db)
            print(f"INPUT | records={stats['rows']} | seasons={len(stats['seasons'])} | dates={stats['first_date']}..{stats['last_date']}",flush=True)
            suffix = (args.slate_date.replace("-", "_") if args.slate_date else
                      f"historical_{stats['seasons'][0][:4]}_{stats['seasons'][-1][:4]}")
            prefix = f"{OUTPUT_BASE}_{suffix}"
            state=FeatureState()
            slate = None
            slate_file = None
            # For a live slate, freeze data availability at the actual request
            # time unless an earlier, explicit as-of timestamp is provided.
            observed_at = (dt_utc(args.observed_at_utc, "--observed-at-utc")
                           if args.observed_at_utc else datetime.now(timezone.utc)) if args.slate_date else None
            if args.slate_date:
                if observed_at > datetime.now(timezone.utc):
                    fail("--observed-at-utc cannot be in the future")
                if not nhl_root:
                    fail("Slate requires NHL repository root; provide --nhl-root")
                slate, slate_file = official_slate(nhl_root,args.slate_date)
                slate_year = next(iter(slate.values()))["season"]
                if len({x["season"] for x in slate.values()}) != 1:
                    fail("Unexpected mixed-season slate")
                if slate_year not in stats["seasons"]:
                    fail(f"Cannot build live slate {args.slate_date}: player history for {slate_year} is missing in {targets}. Refresh current-season completed player-game data first.")
                if stats["last_date"] >= args.slate_date:
                    # For safety, stage may contain future games but strict date processing only sees earlier days.
                    print("INFO | Slate history cut off strictly before requested game date",flush=True)
            data_path=work/(prefix+".csv")
            manifest_path=work/(prefix+"_manifest.json")
            counts=defaultdict(int)
            qa=defaultdict(lambda:defaultdict(int))
            first_feature=None
            last_feature=None
            with data_path.open("w",encoding="utf-8-sig",newline="") as fh:
                writer=csv.DictWriter(fh,fieldnames=OUTPUT_COLUMNS,extrasaction="raise")
                writer.writeheader()
                for day, rows in historical_groups(db):
                    d = checked_date(day,"historical group date")
                    if args.slate_date and day >= args.slate_date:
                        break
                    if not args.slate_date:
                        # Snapshot pregame data is not used without authoritative start UTC.
                        for r in rows:
                            f = {k:"" for k in OUTPUT_COLUMNS}
                            f.update({k:r[k] for k in ("season","game_id","game_date","game_type","player_id","team","opponent_team","home_away","position_group")})
                            f["candidate_source"] = "HISTORICAL_TARGET_ROSTER_POSTGAME"
                            f.update(state.features(r,d,r["season"]))
                            writer.writerow(f)
                            qa[r["season"]]["rows"] += 1
                            if int(f["prior_player_games_career"]) > 0:
                                qa[r["season"]]["with_prior_player_history"] += 1
                            if int(f["team_games_prior_season"]) > 0:
                                qa[r["season"]]["with_prior_team_history"] += 1
                            if int(f["opponent_games_prior_season"]) > 0:
                                qa[r["season"]]["with_prior_opponent_history"] += 1
                            counts["feature_rows"]+=1
                            if counts["feature_rows"] % 100000 == 0:
                                print(f"  FEATURES | {counts['feature_rows']:,} rows generated", flush=True)
                            first_feature=day if first_feature is None else first_feature
                            last_feature=day
                    # Update only AFTER all rows for this exact date were output.
                    state.update_day(rows,d)
                if args.slate_date:
                    candidates,candidate_counts,lineup_file=candidate_slate(nhl_root,args.slate_date,slate,snapshots,observed_at)
                    counts.update(candidate_counts)
                    for c in candidates:
                        d=checked_date(c["game_date"],"candidate date")
                        f={k:"" for k in OUTPUT_COLUMNS}
                        f.update({k:v for k,v in c.items() if k in OUTPUT_COLUMNS})
                        f["game_start_utc"]=to_iso(slate[c["game_id"]]["start"])
                        f.update(state.features(c,d,c["season"]))
                        augment_game(f,c,slate[c["game_id"]]["start"],snapshots,ratings,state,observed_at)
                        writer.writerow(f)
                        qa[c["season"]]["rows"]+=1
                        if int(f["prior_player_games_career"]) > 0:
                            qa[c["season"]]["with_prior_player_history"]+=1
                        if int(f["team_games_prior_season"]) > 0:
                            qa[c["season"]]["with_prior_team_history"]+=1
                        if int(f["opponent_games_prior_season"]) > 0:
                            qa[c["season"]]["with_prior_opponent_history"]+=1
                        if f["home_goalie_id_pre"] or f["away_goalie_id_pre"]:
                            qa[c["season"]]["with_pregame_goalie_id"]+=1
                        if f["team_rating_captured_utc"] or f["opponent_rating_captured_utc"]:
                            qa[c["season"]]["with_asof_team_ratings"]+=1
                        counts["feature_rows"]+=1
                        first_feature=last_feature=args.slate_date
                    if not candidates:
                        fail(f"No eligible pregame lineup/goalie candidates for slate {args.slate_date}")
            qa_path=work/(prefix+"_summary.csv")
            summary_fields=("season","rows","with_prior_player_history","with_prior_team_history", "with_prior_opponent_history", "with_pregame_goalie_id", "with_asof_team_ratings")
            with qa_path.open("w",encoding="utf-8-sig",newline="") as summary_handle:
                qaw=csv.DictWriter(summary_handle,fieldnames=summary_fields)
                qaw.writeheader()
                for season in sorted(qa):
                    qaw.writerow({"season":season,**qa[season]})
            if sum(v["rows"] for v in qa.values()) != counts["feature_rows"]:
                fail("Summary and feature file row counts disagree")
            manifest={"builder_version":VERSION,"mode":"slate" if args.slate_date else "historical",
                      "generated_utc":datetime.now(timezone.utc).isoformat(),"strict_history_policy":"source game_date < target game_date; no same-day target outcomes used",
                      "snapshot_policy":"CapturedUTC strictly before official NHL start_time_utc and no later than requested prediction as-of UTC; blank updates do not override nonblank for the same goalie ID; changing goalie ID resets prior status",
                      "prediction_as_of_utc":to_iso(observed_at) if observed_at else "",
                      "player_candidates_historical":"target-table participation derived postgame; history features themselves use only prior-day games",
                      "no_pregame_history_timestamp":"historical target outcomes have only game_date; date-strict exclusion is deliberately conservative",
                      "power_play_points":"NOT GENERATED; power_play_goals is separate",
                      "historical_source":str(targets),"historical_source_sha256":sha256(targets),"historical_input":stats,
                      "goalie_source_dir":str(goalie_dir),"dated_goalie_files":dfo_files,"team_ratings_asof":str(rating_file) if rating_file and rating_file.is_file() else "",
                      "team_rating_source_rows":rating_count,"slate_date":args.slate_date or "","output_counts":dict(counts),
                      "feature_date_first":first_feature,"feature_date_last":last_feature,"output_file":data_path.name,"summary_file":qa_path.name,"season_summary":{k:dict(v) for k,v in sorted(qa.items())},
                      "snapshot_availability_note":"Historical rows have no verified per-game start UTC in the targets and therefore receive no dated goalie/ratings snapshots." if not args.slate_date else "Verified schedule start, prestart captured goalies and ratings only"}
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
            manifest["output_sha256"]=sha256(data_path)
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
            # Only completed results are published; never overwrite a partly processed result.
            for p in (data_path,qa_path,manifest_path):
                dest=output_dir/p.name
                os.replace(p,dest)
                print(f"REPORT | {dest}",flush=True)
            print(f"STATUS | COMPLETE | rows={counts['feature_rows']} | goalie_files={dfo_files} | ratings={rating_count}",flush=True)
        finally:
            db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhl-root",type=Path,help="NHL repository folder; inferred when installed in the repo")
    p.add_argument("--targets",type=Path,help="Item 3 historical player-game targets CSV (read-only)")
    p.add_argument("--output-dir",type=Path,help="Output directory; defaults to user Downloads")
    p.add_argument("--goalie-dir",type=Path,help="Daily goalie snapshots directory; defaults to NHL_ROOT/00_intake/lineups")
    p.add_argument("--team-ratings",type=Path,help="Optional strictly as-of team ratings CSV")
    p.add_argument("--slate-date",help="Optional YYYY-MM-DD for future pregame candidate features")
    p.add_argument("--observed-at-utc",help="Optional ISO-8601 timestamp of decision time for --slate-date; defaults to actual execution time")
    args=p.parse_args()
    try:
        root=args.nhl_root or find_nhl_root(Path(__file__).resolve())
        out=args.output_dir or (Path.home()/"Downloads")
        targets=args.targets or find_default_target(out)
        if args.slate_date:
            checked_date(args.slate_date,"--slate-date")
        elif args.observed_at_utc:
            fail("--observed-at-utc requires --slate-date")
        write_outputs(args,root,targets,out)
        return 0
    except Exception as exc:
        print(f"FAILED | {exc} | No completed outputs published by this run",file=sys.stderr,flush=True)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
