#!/usr/bin/env python3
"""Manual D-Ratings NHL intake (predictions) and shared parsing/identity helpers.

Use from the repository root. Never assigns an NHL game ID: the existing
transform_hockey.py / reconcile_game_ids.py pipeline owns that contract.

Typical:
  python .../hockey_manual_predictions.py --raw-file /path/to/paste.txt --time-basis et --date 2026_11_17
  python .../hockey_manual_predictions.py --validate-existing --date 2026_11_17
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path("docs/win/hockey/nhl")
RAW_DIR = BASE / "00_intake/drat_raw"
SCRAPER_DIR = BASE / "00_intake/predictions/scraper"
SCHEDULE_DIR = BASE / "00_intake/nhl_schedule"
MAP_FILE = BASE / "config/mapping/team_map_nhl.csv"
ET = ZoneInfo("America/New_York")
FIELDS = (
    "sport", "date_time", "team1", "team2", "team1_win_pct", "team2_win_pct",
    "team1_moneyline", "team2_moneyline", "team1_spread", "team2_spread",
    "proj_score_1", "proj_score_2", "total", "over_line", "under_line",
    "score1", "score2", "game_status",
)
PREDICTION_FIELDS = (
    "team1_win_pct", "team2_win_pct", "team1_moneyline", "team2_moneyline",
    "team1_spread", "team2_spread", "proj_score_1", "proj_score_2",
    "total", "over_line", "under_line",
)
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}(?:\s|$)")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M$", re.I)
DATETIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[AP]M\b", re.I)
PERCENT = re.compile(r"\d+(?:\.\d+)?%")
ML = re.compile(r"[+-]\d{2,5}")
SPREAD = re.compile(r"[+-]\d+(?:½|¼|¾|\.\d+)?[+-]\d{2,4}")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
OU = re.compile(r"[ouOU]\d+(?:½|¼|¾|\.\d+)?[+-]\d{2,4}")
SCORE = re.compile(r"\d+")
LOSS = re.compile(r"[+-]?\d+\.\d+")
RECORD = re.compile(r"\s*\(\d+[-–]\d+(?:[-–]\d+)?\)\s*$")
HEADERS = {
    "time", "teams", "goalies", "win", "best", "ml", "spread", "goals", "total", "o/u",
    "bet", "value", "more", "details", "final", "sportsbook", "dratings", "log", "loss",
    "best ml", "best spread", "total goals", "final goals", "best o/u", "bet value",
    "more details", "upcoming games", "final games", "completed games", "in progress games",
    "volatility bet value active", "svg",
}


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def norm_team(value: str) -> str:
    value = RECORD.sub("", str(value)).strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value)).strip()


def load_team_ids(path: Path | None = None) -> dict[str, str]:
    path = path or MAP_FILE
    if not path.is_file():
        raise ValueError(f"Missing NHL team mapping: {path}")
    result: dict[str, str] = {}
    by_id: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"league", "source", "alias", "canonical_team", "nhl_team_id", "nhl_abbrev"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"NHL team map missing columns: {sorted(needed - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, 2):
            if row["league"].strip().lower() != "nhl":
                continue
            if row["source"].strip().lower() not in ("dratings", "shared", "official_nhl"):
                continue
            key = norm_team(row["alias"])
            tid = str(row["nhl_team_id"]).strip()
            canonical_name = str(row["canonical_team"]).strip()
            abbrev = str(row["nhl_abbrev"]).strip()
            # Shared TBD playoff placeholders intentionally have no NHL team ID.
            # The normal NHL team-map loader accepts these same placeholders.
            # Reject malformed placeholders rather than inventing an identity.
            if canonical_name == "TBD":
                if row["source"].strip().lower() != "shared" or not key or tid or abbrev:
                    raise ValueError(f"Invalid NHL TBD placeholder in {path} row {line}")
                continue
            canonical = norm_team(canonical_name)
            if not key or not tid.isdecimal() or not canonical:
                raise ValueError(f"Invalid NHL identity in {path} row {line}")
            if key in result and result[key] != tid:
                raise ValueError(f"Conflicting NHL alias {row['alias']!r} at row {line}")
            if tid in by_id and by_id[tid] != canonical:
                raise ValueError(f"Conflicting NHL canonical name for ID={tid}")
            result[key] = tid
            by_id[tid] = canonical
            if canonical in result and result[canonical] != tid:
                raise ValueError(f"Conflicting canonical NHL team alias {row['canonical_team']!r}")
            result[canonical] = tid
    if len(by_id) != 32:
        raise ValueError(f"NHL map must identify 32 teams; found {len(by_id)}")
    return result


def team_id(value: str, identities: dict[str, str]) -> str:
    key = norm_team(value)
    if key not in identities:
        raise ValueError(f"Unmapped NHL team: {value!r}; update team_map_nhl.csv")
    return identities[key]


def matchup(row: dict, identities: dict[str, str]) -> tuple[str, str]:
    one, two = team_id(row["team1"], identities), team_id(row["team2"], identities)
    if one == two:
        raise ValueError(f"Same NHL team on both sides: {row['team1']}")
    return tuple(sorted((one, two)))


def when_et(text: str, basis: str) -> datetime:
    text = " ".join(str(text).replace("\n", " ").strip().split())
    try:
        dt = datetime.strptime(text.upper(), "%m/%d/%Y %I:%M %p")
    except ValueError as exc:
        raise ValueError(f"Invalid DRatings date/time: {text!r}") from exc
    if basis == "et":
        return dt.replace(tzinfo=ET)
    if basis == "utc":
        return dt.replace(tzinfo=timezone.utc).astimezone(ET)
    raise ValueError("time_basis must be et or utc")


def clean_markup(text: str) -> str:
    # Preserve HTML cell/row boundaries and the displayed text of Markdown links.
    text = re.sub(r"\\(?=\r?\n)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\n)]*\)", r"\1", text)
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            line = "\t".join(s.strip() for s in line.strip().strip("|").split("|"))
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*t[dh]\s*>", "\t", text)
    text = re.sub(r"(?i)</\s*tr\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).replace("\xa0", " ")


def tokenize(text: str) -> list[str]:
    result: list[str] = []
    for s in re.split(r"[\r\n\t]+", clean_markup(text)):
        s = " ".join(s.strip().split())
        if not s or s.casefold() in HEADERS or re.fullmatch(r"[:|\-\s]+", s):
            continue
        # Preserve team names and goalie names with spaces, split only validated numeric pairs.
        parts = s.split()
        if len(parts) == 2 and any(all(r.fullmatch(p) for p in parts)
                                    for r in (PERCENT, ML, SPREAD, NUMBER, OU, LOSS)):
            result.extend(parts)
        else:
            result.append(s)
    return result


def game_blocks(text: str) -> list[list[str]]:
    tokens = tokenize(text)
    blocks: list[list[str]] = []
    active: list[str] = []
    for token in tokens:
        if DATE_RE.match(token):
            if active:
                blocks.append(active)
            active = [token]
        elif active:
            active.append(token)
    if active:
        blocks.append(active)
    if not blocks:
        raise ValueError("No dated NHL game rows found in pasted table")
    return blocks


def _split_combined_teams(tokens: list[str]) -> list[str]:
    if tokens:
        found = re.findall(r"[^()]+?\(\d+[-–]\d+(?:[-–]\d+)?\)", tokens[0])
        if len(found) == 2 and "".join(found).strip() == tokens[0].strip():
            return [x.strip() for x in found] + tokens[1:]
    return tokens


def _read_header(block: list[str], basis: str) -> tuple[datetime, list[str]]:
    first = block[0]
    tokens = block[1:]
    if not DATETIME_RE.search(first):
        if not tokens or not TIME_RE.fullmatch(tokens[0]):
            raise ValueError(f"Missing time after {first!r}")
        first = f"{first} {tokens.pop(0)}"
    return when_et(first, basis), _split_combined_teams(tokens)


def _pair(tokens: list[str], i: int, pattern: re.Pattern, label: str, *, optional: bool = False) -> tuple[list[str], int]:
    values = []
    while i < len(tokens) and pattern.fullmatch(tokens[i]) and len(values) < 2:
        values.append(tokens[i])
        i += 1
    if optional and not values:
        return ["", ""], i
    if len(values) != 2:
        raise ValueError(f"Expected two {label}; got {values!r}")
    return values, i


def _base_game(when: datetime, team1: str, team2: str) -> dict[str, str]:
    result = dict.fromkeys(FIELDS, "")
    result.update({"sport": "NHL", "date_time": when.strftime("%m/%d/%Y %I:%M %p"),
                   "team1": team1, "team2": team2})
    return result


def parse_predictions(text: str, basis: str = "et") -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for n, block in enumerate(game_blocks(text), 1):
        try:
            when, tokens = _read_header(block, basis)
            if len(tokens) < 4:
                raise ValueError("Missing team names/percentages")
            team1, team2 = tokens[:2]
            i = 2
            # Goalies appear between teams and percentages; they are not raw DRatings fields.
            goalie_count = 0
            while i < len(tokens) and not PERCENT.fullmatch(tokens[i]):
                if goalie_count >= 2 or any(p.fullmatch(tokens[i]) for p in (ML, SPREAD, OU, NUMBER)):
                    raise ValueError(f"Unexpected goalie/field {tokens[i]!r}")
                goalie_count += 1
                i += 1
            if goalie_count not in (0, 2):
                raise ValueError(f"Expected zero or two goalie names; got {goalie_count}")
            pct, i = _pair(tokens, i, PERCENT, "win percentages")
            if abs(sum(float(v[:-1]) for v in pct) - 100) > 0.2:
                raise ValueError("Win percentages must total 100%")
            moneyline, i = _pair(tokens, i, ML, "moneylines", optional=True)
            spreads, i = _pair(tokens, i, SPREAD, "puck lines", optional=True)
            goals, i = _pair(tokens, i, NUMBER, "projected team goals")
            if i >= len(tokens) or not NUMBER.fullmatch(tokens[i]):
                raise ValueError("Missing projected total goals")
            total = tokens[i]
            i += 1
            if abs(float(goals[0]) + float(goals[1]) - float(total)) > 0.055:
                raise ValueError(f"Projected goals do not add up: {goals} / {total}")
            overunder, i = _pair(tokens, i, OU, "over/under", optional=True)
            if overunder[0] and (not overunder[0].lower().startswith("o") or
                                not overunder[1].lower().startswith("u")):
                raise ValueError("O/U must be over then under")
            remaining = tokens[i:]
            if remaining and not all(t.casefold() in HEADERS for t in remaining):
                raise ValueError(f"Unexpected data after O/U: {remaining!r}")
            row = _base_game(when, team1, team2)
            row.update({"team1_win_pct": pct[0], "team2_win_pct": pct[1],
                        "team1_moneyline": moneyline[0], "team2_moneyline": moneyline[1],
                        "team1_spread": spreads[0], "team2_spread": spreads[1],
                        "proj_score_1": goals[0], "proj_score_2": goals[1], "total": total,
                        "over_line": overunder[0], "under_line": overunder[1],
                        "game_status": "upcoming"})
            result.append(row)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Prediction row {n}: {exc}") from exc
    return result


def parse_finals(text: str, basis: str = "et") -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for n, block in enumerate(game_blocks(text), 1):
        try:
            when, tokens = _read_header(block, basis)
            if len(tokens) < 4:
                raise ValueError("Missing team names/percentages")
            team1, team2 = tokens[:2]
            i = 2
            pct, i = _pair(tokens, i, PERCENT, "win percentages")
            if abs(sum(float(v[:-1]) for v in pct) - 100) > 0.2:
                raise ValueError("Win percentages must total 100%")
            moneyline, i = _pair(tokens, i, ML, "moneylines", optional=True)
            spreads, i = _pair(tokens, i, SPREAD, "puck lines", optional=True)
            scores, i = _pair(tokens, i, SCORE, "final goals")
            remaining = tokens[i:]
            if len(remaining) > 2 or any(not LOSS.fullmatch(s) for s in remaining):
                raise ValueError(f"Unexpected final-score fields: {remaining!r}")
            row = _base_game(when, team1, team2)
            row.update({"team1_win_pct": pct[0], "team2_win_pct": pct[1],
                        "team1_moneyline": moneyline[0], "team2_moneyline": moneyline[1],
                        "team1_spread": spreads[0], "team2_spread": spreads[1],
                        "score1": scores[0], "score2": scores[1],
                        "game_status": "completed"})
            result.append(row)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Final row {n}: {exc}") from exc
    return result


def load_raw(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError(f"Invalid raw JSON format: {path}")
    return data


def atomic_json(path: Path, data: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False,
                                     prefix=".nhl-manual-", suffix=".tmp") as fh:
        tmp = Path(fh.name)
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, data: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                     delete=False, prefix=".nhl-manual-", suffix=".tmp") as fh:
        tmp = Path(fh.name)
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in FIELDS} for row in data)
    os.replace(tmp, path)


def read_scraper(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Scraper CSV missing columns: {path}")
        return [dict(row) for row in reader]


def expected_schedule(date: str, identities: dict[str, str]) -> dict[tuple[str, str], dict[str, str]]:
    path = SCHEDULE_DIR / f"NHL_{date}.csv"
    if not path.is_file():
        raise ValueError(f"Missing official NHL schedule {path}; run schedule pull first")
    expected: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"game_id", "game_date", "home_team_id", "away_team_id", "game_type"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f"Official schedule missing columns: {path}")
        for r in reader:
            if r["game_date"].strip() != date or r["game_type"].strip() not in ("2", "3"):
                continue
            home, away = r["home_team_id"].strip(), r["away_team_id"].strip()
            key = tuple(sorted((home, away)))
            if key in expected or home == away or home not in identities.values() or away not in identities.values():
                raise ValueError(f"Invalid/duplicate official schedule matchup: {r}")
            expected[key] = r
    if not expected:
        raise ValueError(f"No regular-season/playoff games in official schedule: {path}")
    return expected


def validate_prediction(row: dict[str, str]) -> None:
    if str(row.get("game_status", "")).strip().lower() != "upcoming":
        raise ValueError("Manual predictions must have game_status=upcoming")
    when_et(row["date_time"], "et")
    probs = []
    for k in ("team1_win_pct", "team2_win_pct"):
        s = str(row.get(k, ""))
        if not PERCENT.fullmatch(s):
            raise ValueError(f"Missing/invalid probability {k}: {s!r}")
        probs.append(float(s[:-1]))
    if not all(0 <= p <= 100 for p in probs) or abs(sum(probs)-100)>0.2:
        raise ValueError("Invalid win probabilities")
    goals = []
    for k in ("proj_score_1", "proj_score_2", "total"):
        s = str(row.get(k, ""))
        if not NUMBER.fullmatch(s):
            raise ValueError(f"Missing/invalid projection {k}: {s!r}")
        goals.append(float(s))
    if abs(goals[0]+goals[1]-goals[2])>0.055:
        raise ValueError("Projected goal total mismatch")
    for keys, regex in ((("team1_moneyline", "team2_moneyline"), ML),
                        (("team1_spread", "team2_spread"), SPREAD),
                        (("over_line", "under_line"), OU)):
        a, b = (str(row.get(k, "")).strip() for k in keys)
        if bool(a)!=bool(b):
            raise ValueError(f"Only one of {keys} is present")
        if a and (not regex.fullmatch(a) or not regex.fullmatch(b)):
            raise ValueError(f"Malformed price fields {keys}: {a} {b}")


def canonicalize_projection(row: dict, expected: dict[tuple[str, str], dict], ids: dict[str, str], date: str) -> tuple[str, str]:
    validate_prediction(row)
    d = when_et(row["date_time"], "et").strftime("%Y_%m_%d")
    if d != date:
        raise ValueError(f"Prediction date {d} is not target date {date}")
    k = matchup(row, ids)
    if k not in expected:
        raise ValueError(f"Game not in official schedule for {date}: {row['team1']} / {row['team2']}")
    one, two = team_id(row["team1"], ids), team_id(row["team2"], ids)
    s = expected[k]
    if one != s["away_team_id"] or two != s["home_team_id"]:
        raise ValueError(f"DRatings orientation does not match official NHL home/away: {row['team1']} / {row['team2']}")
    return k


def verify_prepared(date: str, raw: list[dict], scraper: list[dict], ids: dict, expected: dict) -> None:
    raw_current = {}
    csv_current = {}
    for collection, target in ((raw, raw_current), (scraper, csv_current)):
        for row in collection:
            status = str(row.get("game_status", "")).strip().lower()
            if target is csv_current and status != "upcoming":
                raise ValueError("Scraper CSV must contain upcoming rows only")
            # Raw snapshots also contain completed games from earlier dates.
            d = when_et(row["date_time"], "et").strftime("%Y_%m_%d")
            if d != date:
                if target is csv_current:
                    raise ValueError("Scraper CSV includes a different game date")
                continue
            if target is raw_current and status not in ("upcoming", "completed"):
                raise ValueError(f"Unexpected current raw NHL game status: {status}")
            # A same-day completed DRatings check may have been added after
            # the original pregame snapshot. Validate its immutable pregame
            # fields against the scraper copy without clearing its scores.
            validation_row = dict(row)
            validation_row.update(game_status="upcoming", score1="", score2="")
            k = canonicalize_projection(validation_row, expected, ids, date)
            if k in target:
                raise ValueError(f"Duplicate current NHL matchup: {k}")
            target[k] = row
    missing = set(expected) - set(csv_current)
    excess = set(csv_current) - set(expected)
    if missing or excess or set(raw_current) != set(csv_current):
        raise ValueError(f"Manual NHL slate incomplete: expected={len(expected)} "
                         f"scraper={len(csv_current)} raw={len(raw_current)} missing={sorted(missing)} excess={sorted(excess)}")
    for k, r in csv_current.items():
        raw_row = raw_current[k]
        for key in FIELDS:
            if key in ("game_status", "score1", "score2") and raw_row.get("game_status") == "completed":
                continue
            if str(r.get(key, "")).strip() != str(raw_row.get(key, "")).strip():
                raise ValueError(f"Raw/scraper disagreement for matchup {k}: field={key}")


def publish_predictions(rows: list[dict[str, str]], date: str, ids: dict[str, str]) -> dict[str, int]:
    expected = expected_schedule(date, ids)
    raw_path = RAW_DIR / f"{date}_nhl_raw.json"
    csv_path = SCRAPER_DIR / f"{date}_nhl_predictions.csv"
    raw = load_raw(raw_path)
    existing_csv = read_scraper(csv_path)
    updates = {}
    for r in rows:
        k = canonicalize_projection(r, expected, ids, date)
        if k in updates:
            raise ValueError(f"Duplicate manually supplied matchup: {r['team1']}, {r['team2']}")
        updates[k] = r
    if not updates:
        raise ValueError("No prediction rows provided")
    # A currently valid existing scraper CSV can complete a partial clipboard.
    # Never fabricate missing games or silently borrow unknown records.
    chosen: dict[tuple[str, str], dict] = {}
    for r in existing_csv:
        try:
            k = canonicalize_projection(r, expected, ids, date)
        except ValueError:
            continue
        if k in chosen:
            raise ValueError(f"Duplicate existing scraper matchup: {k}")
        chosen[k] = r
    chosen.update(updates)
    missing = set(expected) - set(chosen)
    if missing or len(chosen) != len(expected):
        details = [f"{expected[k].get('away_team')} at {expected[k].get('home_team')}" for k in sorted(missing)]
        raise ValueError(f"Incomplete manual NHL slate: {len(chosen)}/{len(expected)}; missing={details}")
    # Preserve raw completed/historical rows and refuse to overwrite a completed game.
    old_current = {}
    keep = []
    for r in raw:
        try:
            d = when_et(r["date_time"], "et").strftime("%Y_%m_%d")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Existing raw JSON contains invalid date_time: {exc}") from exc
        if d != date:
            keep.append(r)
            continue
        try:
            k = matchup(r, ids)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Existing current raw JSON has unverified team: {exc}") from exc
        if k in expected:
            if str(r.get("game_status", "")).lower() != "upcoming":
                raise ValueError(f"Existing raw game already completed: {r['team1']} / {r['team2']}")
            if k in old_current:
                raise ValueError(f"Duplicate raw NHL game: {k}")
            old_current[k] = r
        else:
            keep.append(r)
    # Ensure unpublished existing raw rows cannot disagree with authoritative new selection.
    current = [chosen[k] for k in sorted(chosen)]
    new_raw = keep + current
    verify_prepared(date, new_raw, current, ids, expected)
    atomic_json(raw_path, new_raw)
    atomic_csv(csv_path, current)
    return {"expected": len(expected), "manual_updates": len(updates), "retained": len(chosen)-len(updates)}


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-file", type=Path)
    p.add_argument("--time-basis", choices=("et", "utc"), default="et")
    p.add_argument("--date", help="Target Eastern game date YYYY_MM_DD")
    p.add_argument("--validate-existing", action="store_true")
    args = p.parse_args(argv)
    date = args.date or datetime.now(ET).strftime("%Y_%m_%d")
    try:
        datetime.strptime(date, "%Y_%m_%d")
        if len(date) != 10:
            raise ValueError("Date must be YYYY_MM_DD")
        ids = load_team_ids()
        if args.validate_existing:
            e = expected_schedule(date, ids)
            raw = load_raw(RAW_DIR / f"{date}_nhl_raw.json")
            scraper = read_scraper(SCRAPER_DIR / f"{date}_nhl_predictions.csv")
            verify_prepared(date, raw, scraper, ids, e)
            print(f"VERIFIED manual NHL predictions date={date} games={len(e)}")
        else:
            if not args.raw_file:
                raise ValueError("--raw-file required for manual intake")
            rows = parse_predictions(args.raw_file.read_text(encoding="utf-8-sig"), args.time_basis)
            result = publish_predictions(rows, date, ids)
            print(f"SUCCESS: NHL manual predictions date={date} "
                  f"updated={result['manual_updates']} retained={result['retained']} "
                  f"official_games={result['expected']}")
            print(f"WROTE {RAW_DIR / f'{date}_nhl_raw.json'}")
            print(f"WROTE {SCRAPER_DIR / f'{date}_nhl_predictions.csv'}")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
