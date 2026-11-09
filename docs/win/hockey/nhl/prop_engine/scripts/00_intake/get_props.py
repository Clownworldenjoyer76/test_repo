#!/usr/bin/env python3
"""Collect ESPN NHL player-prop odds into one CSV per Eastern Time game date.

Run after the NHL reconciliation step has produced:
  docs/win/hockey/nhl/00_intake/reconciled/NHL_YYYY_MM_DD.csv

ESPN DraftKings (provider 100) does not label both sides of every Total
market. Every observed player/market/betting line is exported, including
lines with no price or no reliably identified Over/Under side. Unknown
Over/Under prices are left blank; they never cause a line to be dropped.

Only the seven requested markets are exported. Goals uses the one-sided
1+ price at Over 0.5; PowerPlayPoints uses each ESPN N+ price at Over N-0.5.
The CSV timestamp is the actual UTC time this script collected the ESPN data.
Each run replaces today's CSV from the current ESPN responses. Previously
saved betting lines are never carried over. Each row has a timestamp.
No non-ESPN odds sources or third-party Python packages are used.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
PROP_ENGINE_ROOT = Path(__file__).resolve().parents[2]
NHL_ROOT = PROP_ENGINE_ROOT.parent
OUTPUT_FIELDS = (
    "date", "game_id", "player", "line", "market",
    "over_odds", "under_odds", "timestamp",
)

ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl"
ESPN_PROVIDER = "100"  # DraftKings; the verified ESPN provider for 2026 NHL props.
TOTAL_MARKETS = {
    "Total Shots on Goal": "SOG",
    "Total Points": "Points",
    "Total Assists": "Assists",
    "Total Blocked Shots": "Blocked_Shots",
    "Total Saves": "Saves",
}
# These are used solely to identify an otherwise unlabelled Over price.
# No rows from these five threshold markets are exported.
EQUIVALENT_PRICE_MARKETS = {
    "Shots on Goal Milestones": "SOG",
    "Points Milestones": "Points",
    "Assists Milestones": "Assists",
    "Blocked Shots Milestones": "Blocked_Shots",
    "Goalkeeper Saves Milestones": "Saves",
}
GOALS_NAME = "Anytime Goalscorer"
POWER_PLAY_NAME = "Powerplay Points Milestones"
ATHLETE_PATH = re.compile(r"/athletes/(\d+)(?:$|[/?])")


def fetch_json(url: str, *, allow_missing: bool = False) -> dict | None:
    """Fetch JSON from ESPN with bounded retries; 404 can mean no props."""
    error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; NHLPropsIntake/1.0)",
                },
            )
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"ESPN returned non-object JSON: {url}")
            return payload
        except HTTPError as exc:
            if exc.code == 404 and allow_missing:
                return None
            error = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, ConnectionError) as exc:
            error = exc
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed: {url}: {error}")


def fetch_props(event_id: str) -> list[dict]:
    """Read every ESPN page for the requested game; missing props => []."""
    url = (
        f"{ESPN_CORE}/events/{event_id}/competitions/{event_id}/"
        f"odds/{ESPN_PROVIDER}/propBets"
    )
    items: list[dict] = []
    for page in range(1, 101):
        query = urlencode({"lang": "en", "region": "us", "page": page, "limit": 100})
        payload = fetch_json(f"{url}?{query}", allow_missing=True)
        if payload is None:
            if page == 1:
                return []
            raise RuntimeError(f"Missing ESPN props page {page} for game {event_id}")
        batch = payload.get("items")
        if not isinstance(batch, list):
            raise RuntimeError(f"Missing ESPN props items for game {event_id}, page {page}")
        items.extend(item for item in batch if isinstance(item, dict))
        try:
            pages = int(payload.get("pageCount", 1))
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid ESPN pageCount for {event_id}") from exc
        if pages < 1 or pages > 100:
            raise RuntimeError(f"Unsupported ESPN pageCount={pages} for {event_id}")
        if page >= pages:
            return items
    raise RuntimeError(f"ESPN props pagination did not finish for {event_id}")


def parse_number(value: object) -> Decimal | None:
    try:
        result = Decimal(str(value).strip())
        return result if result.is_finite() else None
    except (ValueError, InvalidOperation, TypeError):
        return None


def betting_line(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def price(value: object) -> str:
    """Only valid American sportsbook prices; never replace missing odds."""
    if value is None:
        return ""
    raw = str(value).strip().upper()
    if raw in ("EVEN", "EV"):
        return "+100"
    if not re.fullmatch(r"[+-]?\d+", raw):
        return ""
    integer = int(raw)
    if -99 <= integer <= 99:
        return ""
    return f"+{integer}" if integer > 0 else str(integer)


def current_line(item: dict) -> Decimal | None:
    current = item.get("current") or {}
    target = (current.get("target") or {}) if isinstance(current, dict) else {}
    odds = item.get("odds") or {}
    total = (odds.get("total") or {}) if isinstance(odds, dict) else {}
    value = target.get("value")
    if value is None:
        value = total.get("value")
    return parse_number(value)


def side_label(item: dict) -> str:
    """Use explicit side labels only; conflicting labels are rejected."""
    labels = set()
    for field in ("current", "open"):
        block = item.get(field) or {}
        if isinstance(block, dict):
            labels.update(k for k in ("over", "under") if k in block)
    return next(iter(labels)) if len(labels) == 1 else ""


def athlete_identity(item: dict) -> tuple[str, str]:
    athlete = item.get("athlete") or {}
    ref = athlete.get("$ref", "") if isinstance(athlete, dict) else ""
    parsed = urlsplit(ref)
    if (
        parsed.hostname != "sports.core.api.espn.com"
        or not parsed.path.startswith("/v2/sports/hockey/leagues/nhl/")
    ):
        return "", ""
    match = ATHLETE_PATH.search(parsed.path)
    if not match:
        return "", ""
    # ESPN often supplies HTTP references. Fetch the same reference over HTTPS.
    safe_url = urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))
    return match.group(1), safe_url


def source_record(item: dict) -> dict | None:
    athlete_id, athlete_url = athlete_identity(item)
    odds = item.get("odds") or {}
    american = (odds.get("american") or {}) if isinstance(odds, dict) else {}
    odds_price = price(american.get("value"))
    updated = str(item.get("lastUpdated") or "").strip()
    market_name = (item.get("type") or {}).get("name", "")
    if not athlete_id or not isinstance(market_name, str):
        return None
    return {
        "athlete_id": athlete_id,
        "athlete_url": athlete_url,
        "source_market": market_name,
        "line": current_line(item),
        "price": odds_price,
        "side": side_label(item),
        "timestamp": updated,
    }


def normalize_event(event_id: str, game_id: str, date_label: str, raw: list[dict]) -> tuple[list[dict], int]:
    """Build only requested markets; do not guess unlabelled Over/Under."""
    parsed = [record for item in raw if (record := source_record(item)) is not None]
    equivalent: dict[tuple[str, str, Decimal], set[str]] = defaultdict(set)
    totals: dict[tuple[str, str, Decimal], list[dict]] = defaultdict(list)
    singles: dict[tuple[str, str, Decimal], list[dict]] = defaultdict(list)

    for record in parsed:
        name = record["source_market"]
        line = record["line"]
        athlete = record["athlete_id"]
        if name in EQUIVALENT_PRICE_MARKETS and line is not None:
            equivalent[(athlete, EQUIVALENT_PRICE_MARKETS[name], line)].add(record["price"])
        elif name in TOTAL_MARKETS and line is not None:
            totals[(athlete, TOTAL_MARKETS[name], line)].append(record)
        elif name == GOALS_NAME:
            singles[(athlete, "Goals", Decimal("0.5"))].append(record)
        elif (name == POWER_PLAY_NAME and line is not None
              and line > 0 and line == line.to_integral_value()):
            # Preserve every power-play-points betting line ESPN provides.
            singles[(athlete, "PowerPlayPoints", line - Decimal("0.5"))].append(record)

    rows: list[dict] = []
    unresolved = 0

    def append_row(athlete: str, market: str, line: Decimal,
                   over: str, under: str, updated: str, url: str) -> None:
        rows.append({
            "date": date_label,
            "game_id": game_id,
            "player_id": athlete,  # internal lookup only, never written to CSV
            "player_url": url,
            "line": betting_line(line),
            "market": market,
            "over_odds": over,
            "under_odds": under,
            "timestamp": updated,
        })

    for (athlete, market, line), group in totals.items():
        # Export every ESPN betting line, even if zero, one, or several
        # prices were supplied and their Over/Under sides are unknown.
        over = ""
        under = ""
        updated = max((record["timestamp"] for record in group), default="")
        athlete_url = group[0]["athlete_url"]

        if len(group) == 1:
            # An explicit ESPN label identifies the sole available side.
            if group[0]["side"] == "over":
                over = group[0]["price"]
            elif group[0]["side"] == "under":
                under = group[0]["price"]
        elif len(group) == 2:
            explicit_over = [r for r in group if r["side"] == "over"]
            explicit_under = [r for r in group if r["side"] == "under"]
            over_record = None
            under_record = None
            if len(explicit_over) == 1 and len(explicit_under) == 1:
                over_record, under_record = explicit_over[0], explicit_under[0]
            elif len(explicit_over) == 1 and not explicit_under and not group[1 - group.index(explicit_over[0])]["side"]:
                over_record = explicit_over[0]
                under_record = next(r for r in group if r is not over_record)
            elif len(explicit_under) == 1 and not explicit_over and not group[1 - group.index(explicit_under[0])]["side"]:
                under_record = explicit_under[0]
                over_record = next(r for r in group if r is not under_record)
            elif not explicit_over and not explicit_under:
                if group[0]["price"] and group[0]["price"] == group[1]["price"]:
                    # Both sides have the same verified price; order is irrelevant.
                    over_record, under_record = group
                else:
                    threshold = line + Decimal("0.5")
                    matching = equivalent.get((athlete, market, threshold), set())
                    first_matches = bool(group[0]["price"] and group[0]["price"] in matching)
                    second_matches = bool(group[1]["price"] and group[1]["price"] in matching)
                    if first_matches != second_matches:
                        over_record, under_record = (
                            (group[0], group[1]) if first_matches else (group[1], group[0])
                        )
            if over_record is not None and under_record is not None:
                over = over_record["price"]
                under = under_record["price"]
        # Conflicting or additional rows are not assigned guessed prices.
        if not over or not under:
            unresolved += 1
        append_row(athlete, market, line, over, under, updated, athlete_url)

    for (athlete, market, line), group in singles.items():
        # Keep the last timestamp's price if ESPN returns duplicates.
        record = max(group, key=lambda value: value["timestamp"])
        append_row(athlete, market, line, record["price"], "",
                   record["timestamp"], record["athlete_url"])

    return rows, unresolved


def get_player_name(url: str) -> str:
    payload = fetch_json(url)
    if payload is None:
        return ""
    name = payload.get("fullName") or payload.get("displayName")
    if not name:
        name = " ".join(
            str(payload.get(k) or "").strip() for k in ("firstName", "lastName")
        ).strip()
    return str(name).strip() if name else ""


def load_games(day: str, nhl_root: Path) -> list[tuple[str, str]]:
    """Use the repository's verified ESPN-to-official NHL game ID mapping."""
    path = nhl_root / "00_intake" / "reconciled" / f"NHL_{day}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required NHL game-ID reconciliation is missing: {path}")
    games: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"game_id", "sportsbook_event_id", "game_date"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"NHL reconciliation missing required fields: {path}")
        for row in reader:
            if row["game_date"].replace("-", "_") != day:
                continue
            event_id = (row.get("sportsbook_event_id") or "").strip()
            game_id = (row.get("game_id") or "").strip()
            if not event_id.isdecimal() or not game_id.isdecimal():
                raise ValueError(f"Invalid NHL/ESPN game ID in {path}")
            games.append((event_id, game_id))
    if len(games) != len(set(games)) or len({x[0] for x in games}) != len(games):
        raise ValueError(f"Duplicate NHL/ESPN game mapping in {path}")
    return games


def captured_at_utc() -> str:
    """UTC timestamp taken after an ESPN event response was received."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_daily_csv(path: Path, records: list[dict], *, day: str,
                    expected_game_ids: set[str]) -> int:
    """Replace today's entire CSV with the current ESPN pull, never old rows."""
    chosen: dict[tuple[str, ...], dict[str, str]] = {}
    for record in records:
        row = {field: str(record.get(field, "")) for field in OUTPUT_FIELDS}
        if row["date"] != day or row["game_id"] not in expected_game_ids:
            raise ValueError(f"Prop row is not from an expected {day} game: {row}")
        if not row["timestamp"].strip():
            raise ValueError(f"Prop row has no collection timestamp: {row}")
        key = tuple(row[field] for field in (
            "date", "game_id", "player", "line", "market"
        ))
        if key in chosen:
            raise ValueError(f"Duplicate ESPN prop key in current pull: {key}")
        chosen[key] = row

    rows = sorted(chosen.values(), key=lambda row: (
        row["game_id"], row["market"], row["player"], row["line"]
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", dir=path.parent,
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return len(rows)


def run(day: str, *, nhl_root: Path = NHL_ROOT,
        prop_root: Path = PROP_ENGINE_ROOT) -> tuple[Path, int, int, int]:
    games = load_games(day, nhl_root)
    records: list[dict] = []
    unresolved = 0
    failed_games: list[str] = []
    source_counts: Counter[str] = Counter()
    for event_id, game_id in games:
        try:
            raw = fetch_props(event_id)
        except Exception as exc:
            failed_games.append(f"{event_id}: {exc}")
            continue
        # Report what ESPN actually supplied, independently of odds assignment.
        for item in raw:
            source_type = item.get("type")
            name = source_type.get("name") if isinstance(source_type, dict) else None
            if name in TOTAL_MARKETS:
                source_counts[TOTAL_MARKETS[name]] += 1
            elif name == GOALS_NAME:
                source_counts["Goals"] += 1
            elif name == POWER_PLAY_NAME:
                source_counts["PowerPlayPoints"] += 1
        # ESPN's lastUpdated can be later than the collection time. It is
        # not safe to publish as a factual observation timestamp. Record
        # the time this script actually received the odds instead.
        observed_at = captured_at_utc()
        event_rows, missing = normalize_event(event_id, game_id, day, raw)
        for record in event_rows:
            record["timestamp"] = observed_at
        unresolved += missing
        records.extend(event_rows)

    if failed_games:
        raise RuntimeError("ESPN prop retrieval failed for today's game(s); "
                           "refusing to write a partial CSV: "
                           + "; ".join(failed_games))

    player_urls: dict[str, str] = {}
    for record in records:
        player_urls[record["player_id"]] = record["player_url"]

    names: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(get_player_name, url): pid
                   for pid, url in player_urls.items()}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                names[pid] = future.result()
            except Exception:
                names[pid] = ""

    complete = []
    unnamed = 0
    for record in records:
        name = names.get(record["player_id"], "")
        if not name:
            unnamed += 1
            # Preserve the player record; the verified ESPN athlete ID
            # identifies the player when the name endpoint is unavailable.
            name = record["player_id"]
        complete.append({**record, "player": name})

    output = prop_root / "00_intake" / f"{day}_nhl_props_data.csv"
    count = write_daily_csv(
        output, complete, day=day, expected_game_ids={game_id for _, game_id in games}
    )
    if unresolved or unnamed:
        print(f"WARNING: {unresolved} rows have missing Over/Under odds (rows retained); "
              f"{unnamed} names unavailable (ESPN player IDs retained)", file=sys.stderr)
    markets = ("SOG", "Points", "Assists", "Goals", "Blocked_Shots", "Saves", "PowerPlayPoints")
    print("ESPN SOURCE MARKET RECORDS: " + " ".join(
        f"{market}={source_counts[market]}" for market in markets
    ))
    return output, count, len(games), len(complete)


def main() -> None:
    # The workflow is for today's games only; do not accept a historical date.
    if len(sys.argv) != 1:
        raise SystemExit("ERROR: This script collects today's NHL games only; no date options are supported.")
    day = datetime.now(ET).strftime("%Y_%m_%d")
    try:
        output, total, games, fresh = run(day)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"WROTE {output}")
    print(f"GAMES {games} | ESPN ROWS {fresh} | CSV ROWS {total}")


if __name__ == "__main__":
    main()
