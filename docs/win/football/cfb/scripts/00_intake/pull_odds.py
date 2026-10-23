#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_odds.py
"""Pull current CFB odds from ESPN Core and preserve point-in-time snapshots."""

from __future__ import annotations

from http.client import HTTPException

import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    format_american,
    format_number,
    load_current_week_config,
    read_required_csv as read_csv,
    write_csv_rows_durable,
)
from type_support import ScalarValue

CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
ODDS_DIR = CFB_ROOT / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"
SNAPSHOT_DIR = ODDS_DIR / "snapshots"
RAW_SNAPSHOT_DIR = RAW_ODDS_DIR / "snapshots"
REPORT_ROOT = CFB_ROOT / "errors"

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football"
)
ESPN_CORE_HOST = "sports.core.api.espn.com"

SCRIPT_VERSION = "cfb-odds-v2-2026-09-15"

SCHEDULE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "game_timezone",
]

OUTPUT_COLUMNS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]

VALID_MARKET_SIDES = {
    "h2h": {"home", "away"},
    "spreads": {"home", "away"},
    "totals": {"over", "under"},
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def schedule_kickoff_utc(
    row: dict[str, str],
) -> datetime:
    game_id = str(
        row.get("game_id", "")
    ).strip()

    game_date = str(
        row.get("game_date", "")
    ).strip()

    game_time = str(
        row.get("game_time", "")
    ).strip()

    game_timezone = str(
        row.get("game_timezone", "")
    ).strip()

    missing = [
        name
        for name, value in (
            ("game_date", game_date),
            ("game_time", game_time),
            ("game_timezone", game_timezone),
        )
        if not value
    ]

    if missing:
        raise ValueError(
            f"Game {game_id or '<blank>'} missing kickoff fields: "
            + ", ".join(missing)
        )

    try:
        timezone_info = ZoneInfo(
            game_timezone
        )
    except Exception as exc:
        raise ValueError(
            f"Game {game_id or '<blank>'} has invalid "
            f"game_timezone={game_timezone!r}"
        ) from exc

    try:
        local_dt = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError as exc:
        raise ValueError(
            f"Game {game_id or '<blank>'} has invalid kickoff "
            f"date/time: date={game_date!r}, time={game_time!r}"
        ) from exc

    return local_dt.replace(
        tzinfo=timezone_info
    ).astimezone(
        timezone.utc
    )


def schedule_kickoff_iso(
    row: dict[str, str],
) -> str:
    return (
        schedule_kickoff_utc(row)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_game_locked(
    row: dict[str, str],
    now_utc: datetime,
) -> bool:
    return (
        now_utc
        >= schedule_kickoff_utc(row)
    )


def load_target_schedule(
    season: int,
    season_type: int,
    week: int,
) -> tuple[Path, list[dict[str, str]]]:
    schedule_path = (
        SCHEDULE_DIR
        / f"{season}_schedule.csv"
    )

    rows = read_csv(
        schedule_path,
        SCHEDULE_REQUIRED_COLUMNS,
        "CFB schedule CSV",
    )

    target_rows = [
        row
        for row in rows
        if str(
            row.get("season", "")
        ).strip() == str(season)
        and str(
            row.get("season_type", "")
        ).strip() == str(season_type)
        and str(
            row.get("week", "")
        ).strip() == str(week)
    ]

    if not target_rows:
        raise ValueError(
            "Configured CFB schedule group was not found: "
            f"season={season}, "
            f"season_type={season_type}, "
            f"week={week}"
        )

    game_ids: list[str] = []

    for row in target_rows:
        game_id = str(
            row.get("game_id", "")
        ).strip()

        home_team = str(
            row.get("home_team", "")
        ).strip()

        away_team = str(
            row.get("away_team", "")
        ).strip()

        if not game_id:
            raise ValueError(
                "Configured target week contains a blank game_id"
            )

        if not home_team or not away_team:
            raise ValueError(
                f"Game {game_id} has a blank home_team or away_team"
            )

        schedule_kickoff_utc(
            row
        )

        game_ids.append(
            game_id
        )

    counts: dict[str, int] = {}

    for game_id in game_ids:
        counts[game_id] = (
            counts.get(
                game_id,
                0,
            )
            + 1
        )

    duplicates = sorted(
        game_id
        for game_id, count in counts.items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            "Configured target week contains duplicate game_id values: "
            + ", ".join(
                duplicates[:20]
            )
        )

    target_rows.sort(
        key=lambda row: (
            schedule_kickoff_utc(row),
            str(
                row.get("game_id", "")
            ).strip(),
        )
    )

    return (
        schedule_path,
        target_rows,
    )


def build_url(
    path: str,
    params: Optional[dict[str, ScalarValue]] = None,
) -> str:
    url = f"{ESPN_BASE}{path}"

    if params:
        return (
            f"{url}?{urlencode(params)}"
        )

    return url


def http_get_json(
    url: str,
) -> tuple[Optional[int], Optional[object], str]:
    request = Request(
        url,
        headers={
            "User-Agent": "cfb-espn-pull-odds/2.0",
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_CORE_HOST},
            timeout=45,
        ) as response:
            status = response.status
            body = response.read().decode(
                "utf-8"
            )

    except HTTPError as exc:
        body = ""

        try:
            body = exc.read().decode(
                "utf-8"
            )
        except (HTTPException, OSError, UnicodeError, ValueError):
            pass

        return (
            exc.code,
            None,
            body or str(exc),
        )

    except URLError as exc:
        return (
            None,
            None,
            str(exc),
        )

    except Exception as exc:
        return (
            None,
            None,
            str(exc),
        )

    if (
        status < 200
        or status >= 300
    ):
        return (
            status,
            None,
            body,
        )

    try:
        payload = json.loads(
            body
        )
    except Exception as exc:
        return (
            status,
            None,
            f"JSON parse failed: {exc}",
        )

    return (
        status,
        payload,
        "",
    )


def fetch_ref(
    ref: str,
    label: str,
) -> dict:
    if ref.startswith(
        "http://sports.core.api.espn.com/"
    ):
        ref = (
            "https://sports.core.api.espn.com/"
            + ref[
                len("http://sports.core.api.espn.com/"):
            ]
        )

    status, payload, error = (
        http_get_json(ref)
    )

    if (
        status is None
        or status < 200
        or status >= 300
        or payload is None
    ):
        raise RuntimeError(
            f"{label} fetch failed: "
            f"status={status!r}, "
            f"ref={ref}, "
            f"error={error}"
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"{label} fetch returned non-object JSON: {ref}"
        )

    return payload


def to_float(
    value: ScalarValue,
) -> Optional[float]:
    if (
        value is None
        or isinstance(value, bool)
    ):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.casefold() in {
        "even",
        "ev",
        "evens",
    }:
        return 100.0

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def clean_number(
    value: ScalarValue,
) -> str:
    return format_number(
        to_float(value)
    )

def normalize_american(
    value: ScalarValue,
) -> str:
    return format_american(
        to_float(value)
    )

def american_to_decimal(
    value: ScalarValue,
) -> str:
    american = to_float(
        value
    )

    if (
        american is None
        or american == 0
    ):
        return ""

    if american > 0:
        decimal = (
            1
            + american / 100
        )
    else:
        decimal = (
            1
            + 100 / abs(american)
        )

    return clean_number(
        round(
            decimal,
            6,
        )
    )


def provider_info(
    odds_item: dict,
) -> dict[str, ScalarValue]:
    provider = odds_item.get(
        "provider"
    )

    if not isinstance(
        provider,
        dict,
    ):
        return {
            "id": "",
            "name": "",
            "priority": None,
        }

    provider_data = provider

    if provider.get("$ref") and not (
        provider.get("name")
        or provider.get("id")
        or provider.get("priority") is not None
    ):
        provider_data = fetch_ref(
            str(
                provider["$ref"]
            ),
            "provider reference",
        )

    return {
        "id": str(
            provider_data.get(
                "id",
                "",
            )
        ).strip(),
        "name": str(
            provider_data.get("name")
            or provider_data.get(
                "displayName"
            )
            or provider_data.get(
                "shortName"
            )
            or ""
        ).strip(),
        "priority": to_float(
            provider_data.get(
                "priority"
            )
        ),
    }


def resolve_odds_items(
    collection: object,
) -> list[dict]:
    if not isinstance(
        collection,
        dict,
    ):
        raise RuntimeError(
            "ESPN odds response was not a JSON object"
        )

    items = collection.get(
        "items",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        raise RuntimeError(
            "ESPN odds response items field was not a list"
        )

    resolved: list[dict] = []

    for index, item in enumerate(
        items
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise RuntimeError(
                "ESPN odds response contains a non-object "
                f"item at index {index}"
            )

        if item.get("$ref") and not (
            item.get("provider")
            or item.get("homeTeamOdds")
            or item.get("awayTeamOdds")
            or item.get("overUnder") is not None
            or item.get("spread") is not None
        ):
            item = fetch_ref(
                str(
                    item["$ref"]
                ),
                "odds item reference",
            )

        resolved.append(
            item
        )

    return resolved


def select_primary_odds_item(
    items: list[dict],
) -> Optional[dict]:
    if not items:
        return None

    ranked = []

    for index, item in enumerate(
        items
    ):
        info = provider_info(
            item
        )

        priority = info[
            "priority"
        ]

        rank = (
            priority
            if priority is not None
            else 1_000_000
        )

        ranked.append(
            (
                rank,
                index,
                item,
            )
        )

    ranked.sort(
        key=lambda value: (
            value[0],
            value[1],
        )
    )

    return ranked[0][2]


def nested_value(
    data: dict,
    path: tuple[str, ...],
) -> Optional[object]:
    current: object = data

    for key in path:
        if not isinstance(
            current,
            dict,
        ):
            return None

        current = current.get(
            key
        )

    return current


def first_value(
    data: dict,
    paths: list[tuple[str, ...]],
) -> ScalarValue:
    for path in paths:
        value = nested_value(
            data,
            path,
        )

        if value is None:
            continue

        if not isinstance(
            value,
            (str, int, float, bool),
        ):
            continue

        if str(value).strip() != "":
            return value

    return None


def parse_details_line(
    details: ScalarValue,
) -> Optional[float]:
    match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*$",
        str(
            details or ""
        ).strip(),
    )

    if not match:
        return None

    return to_float(
        match.group(1)
    )


def bool_value(
    value: ScalarValue,
) -> bool:
    if value is None:
        return False

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, float),
    ):
        return bool(value)

    text = str(
        value
    ).strip().casefold()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "",
        "none",
        "null",
    }:
        return False

    raise ValueError(
        f"Unsupported boolean value: {value!r}"
    )



def _complete_spread_pair(
    odds_item: dict,
    *,
    home_team_odds: dict,
    away_team_odds: dict,
    home_spread: str,
    away_spread: str,
) -> tuple[str, str]:
    if (
        home_spread == ""
        and away_spread != ""
    ):
        away_num = to_float(
            away_spread
        )

        if away_num is not None:
            home_spread = clean_number(
                -away_num
            )

    if (
        away_spread == ""
        and home_spread != ""
    ):
        home_num = to_float(
            home_spread
        )

        if home_num is not None:
            away_spread = clean_number(
                -home_num
            )

    if (
        home_spread == ""
        and away_spread == ""
    ):
        detail_line = parse_details_line(
            odds_item.get(
                "details",
                "",
            )
        )

        generic_spread = to_float(
            odds_item.get(
                "spread"
            )
        )

        line = (
            detail_line
            if detail_line is not None
            else generic_spread
        )

        home_favorite = bool_value(
            home_team_odds.get(
                "favorite"
            )
        )

        away_favorite = bool_value(
            away_team_odds.get(
                "favorite"
            )
        )

        if line is not None:
            if (
                home_favorite
                and not away_favorite
            ):
                home_spread = clean_number(
                    line
                )
                away_spread = clean_number(
                    -line
                )

            elif (
                away_favorite
                and not home_favorite
            ):
                away_spread = clean_number(
                    line
                )
                home_spread = clean_number(
                    -line
                )

            else:
                home_spread = clean_number(
                    line
                )
                away_spread = clean_number(
                    -line
                )

    return (
        home_spread,
        away_spread,
    )


def extract_market_values(
    odds_item: dict,
) -> dict[str, str]:
    home_team_odds = (
        odds_item.get(
            "homeTeamOdds"
        )
        if isinstance(
            odds_item.get(
                "homeTeamOdds"
            ),
            dict,
        )
        else {}
    )

    away_team_odds = (
        odds_item.get(
            "awayTeamOdds"
        )
        if isinstance(
            odds_item.get(
                "awayTeamOdds"
            ),
            dict,
        )
        else {}
    )

    home_moneyline = normalize_american(
        first_value(
            odds_item,
            [
                (
                    "homeTeamOdds",
                    "moneyLine",
                ),
                (
                    "homeTeamOdds",
                    "moneyline",
                ),
                ("homeMoneyLine",),
                ("homeMoneyline",),
            ],
        )
    )

    away_moneyline = normalize_american(
        first_value(
            odds_item,
            [
                (
                    "awayTeamOdds",
                    "moneyLine",
                ),
                (
                    "awayTeamOdds",
                    "moneyline",
                ),
                ("awayMoneyLine",),
                ("awayMoneyline",),
            ],
        )
    )

    home_spread_odds = normalize_american(
        first_value(
            odds_item,
            [
                (
                    "homeTeamOdds",
                    "spreadOdds",
                ),
                ("homeSpreadOdds",),
            ],
        )
    )

    away_spread_odds = normalize_american(
        first_value(
            odds_item,
            [
                (
                    "awayTeamOdds",
                    "spreadOdds",
                ),
                ("awaySpreadOdds",),
            ],
        )
    )

    total = clean_number(
        first_value(
            odds_item,
            [
                ("overUnder",),
                ("total",),
            ],
        )
    )

    over_american = normalize_american(
        first_value(
            odds_item,
            [
                ("overOdds",),
                ("over",),
            ],
        )
    )

    under_american = normalize_american(
        first_value(
            odds_item,
            [
                ("underOdds",),
                ("under",),
            ],
        )
    )

    direct_home_spread = first_value(
        odds_item,
        [
            (
                "homeTeamOdds",
                "spread",
            ),
            ("homeSpread",),
        ],
    )

    direct_away_spread = first_value(
        odds_item,
        [
            (
                "awayTeamOdds",
                "spread",
            ),
            ("awaySpread",),
        ],
    )

    home_spread = clean_number(
        direct_home_spread
    )

    away_spread = clean_number(
        direct_away_spread
    )

    (
        home_spread,
        away_spread,
    ) = _complete_spread_pair(
        odds_item,
        home_team_odds=home_team_odds,
        away_team_odds=away_team_odds,
        home_spread=home_spread,
        away_spread=away_spread,
    )

    last_update = str(
        first_value(
            odds_item,
            [
                ("lastUpdated",),
                ("lastUpdate",),
                ("updated",),
                ("timestamp",),
                ("date",),
            ],
        )
        or ""
    ).strip()

    return {
        "home_moneyline_american": home_moneyline,
        "away_moneyline_american": away_moneyline,
        "home_spread": home_spread,
        "away_spread": away_spread,
        "home_spread_american": home_spread_odds,
        "away_spread_american": away_spread_odds,
        "total": total,
        "over_american": over_american,
        "under_american": under_american,
        "last_update": last_update,
    }


def add_market_row(
    rows: list[dict[str, str]],
    event: dict[str, str],
    bookmaker: str,
    market_type: str,
    bet_side: str,
    line: ScalarValue,
    odds_american: ScalarValue,
    current_fields: dict[str, str],
    snapshot_id: str,
    snapshot_fetched_at: str,
) -> None:
    rows.append(
        {
            "snapshot_id": snapshot_id,
            "snapshot_fetched_at": snapshot_fetched_at,
            "game_id": event["id"],
            "commence_time": event["date"],
            "home_team": event["home"],
            "away_team": event["away"],
            "bookmaker": bookmaker,
            "market_type": market_type,
            "bet_side": bet_side,
            "line": clean_number(
                line
            ),
            "odds_american": normalize_american(
                odds_american
            ),
            "odds_decimal": american_to_decimal(
                odds_american
            ),
            "last_update": current_fields.get(
                "last_update",
                "",
            ),
            **{
                key: current_fields.get(
                    key,
                    "",
                )
                for key in [
                    "home_moneyline_american",
                    "away_moneyline_american",
                    "home_spread",
                    "away_spread",
                    "home_spread_american",
                    "away_spread_american",
                    "total",
                    "over_american",
                    "under_american",
                ]
            },
        }
    )


def normalize_event_odds(
    event: dict[str, str],
    odds_item: dict,
    snapshot_id: str,
    snapshot_fetched_at: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, ScalarValue],
]:
    info = provider_info(
        odds_item
    )

    bookmaker = str(
        info["name"]
        or info["id"]
        or ""
    ).strip()

    if not bookmaker:
        raise RuntimeError(
            f"Game {event['id']} returned odds "
            "without a provider identity"
        )

    current = extract_market_values(
        odds_item
    )

    rows: list[dict[str, str]] = []

    if (
        current[
            "home_moneyline_american"
        ]
        or current[
            "away_moneyline_american"
        ]
    ):
        add_market_row(
            rows,
            event,
            bookmaker,
            "h2h",
            "home",
            "",
            current[
                "home_moneyline_american"
            ],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

        add_market_row(
            rows,
            event,
            bookmaker,
            "h2h",
            "away",
            "",
            current[
                "away_moneyline_american"
            ],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    if (
        to_float(current["home_spread"]) is not None
        and to_float(current["away_spread"]) is not None
    ):
        add_market_row(
            rows,
            event,
            bookmaker,
            "spreads",
            "home",
            current["home_spread"],
            current[
                "home_spread_american"
            ],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

        add_market_row(
            rows,
            event,
            bookmaker,
            "spreads",
            "away",
            current["away_spread"],
            current[
                "away_spread_american"
            ],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    if to_float(current["total"]) is not None:
        add_market_row(
            rows,
            event,
            bookmaker,
            "totals",
            "over",
            current["total"],
            current["over_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

        add_market_row(
            rows,
            event,
            bookmaker,
            "totals",
            "under",
            current["total"],
            current["under_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    return (
        rows,
        info,
    )


def fetch_game_odds(
    game_id: str,
) -> tuple[
    Optional[dict],
    str,
    int,
    str,
]:
    path = (
        f"/events/{game_id}/competitions/"
        f"{game_id}/odds"
    )

    url = build_url(
        path,
        {
            "limit": 100,
            "lang": "en",
            "region": "us",
        },
    )

    status, collection, error = (
        http_get_json(
            url
        )
    )

    if (
        status is None
        or status < 200
        or status >= 300
        or collection is None
    ):
        raise RuntimeError(
            f"Odds request failed for game_id={game_id}: "
            f"status={status!r}, error={error}"
        )

    items = resolve_odds_items(
        collection
    )

    selected = select_primary_odds_item(
        items
    )

    if selected is None:
        return (
            None,
            url,
            status,
            "EMPTY",
        )

    return (
        selected,
        url,
        status,
        "AVAILABLE",
    )


def parse_aware_iso(
    value: str,
    label: str,
) -> datetime:
    text = str(
        value
    ).strip()

    if not text:
        raise ValueError(
            f"{label} is blank"
        )

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            text
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} is not a valid ISO timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"{label} must include a timezone: {value!r}"
        )

    return parsed.astimezone(
        timezone.utc
    )


def _stage1_validate_odds_row_identity(
    row: dict[str, str],
    *,
    index: int,
    schedule_by_id: dict[str, dict[str, str]],
    snapshot_id: str,
    snapshot_fetched_at: str,
) -> tuple[str, str, str]:
    missing_columns = [
        column for column in OUTPUT_COLUMNS if column not in row
    ]
    if missing_columns:
        raise ValueError(
            f"Normalized odds row {index} missing columns: {missing_columns}"
        )
    if row["snapshot_id"] != snapshot_id:
        raise ValueError(
            f"Normalized odds row {index} has inconsistent snapshot_id"
        )
    if row["snapshot_fetched_at"] != snapshot_fetched_at:
        raise ValueError(
            f"Normalized odds row {index} has inconsistent snapshot_fetched_at"
        )

    game_id = str(row["game_id"]).strip()
    if game_id not in schedule_by_id:
        raise ValueError(
            f"Normalized odds row {index} contains out-of-scope game_id={game_id!r}"
        )

    schedule_row = schedule_by_id[game_id]
    expected_home = str(schedule_row["home_team"]).strip()
    expected_away = str(schedule_row["away_team"]).strip()
    if (
        str(row["home_team"]).strip() != expected_home
        or str(row["away_team"]).strip() != expected_away
    ):
        raise ValueError(
            f"Normalized odds row {index} has team identity mismatch for game_id={game_id}"
        )

    expected_commence = schedule_kickoff_iso(schedule_row)
    if str(row["commence_time"]).strip() != expected_commence:
        raise ValueError(
            f"Normalized odds row {index} has kickoff mismatch for game_id={game_id}"
        )

    bookmaker = str(row["bookmaker"]).strip()
    if not bookmaker:
        raise ValueError(f"Normalized odds row {index} has blank bookmaker")

    market_type = str(row["market_type"]).strip()
    bet_side = str(row["bet_side"]).strip()
    if market_type not in VALID_MARKET_SIDES:
        raise ValueError(
            f"Normalized odds row {index} has invalid market_type={market_type!r}"
        )
    if bet_side not in VALID_MARKET_SIDES[market_type]:
        raise ValueError(
            f"Normalized odds row {index} has invalid bet_side={bet_side!r} "
            f"for market_type={market_type!r}"
        )
    return game_id, market_type, bet_side


def _stage1_validate_market_line(
    *,
    row: dict[str, str],
    line_text: str,
    game_id: str,
    market_type: str,
    bet_side: str,
    home_spread: Optional[float],
    away_spread: Optional[float],
) -> None:
    _stage1_validate_basic_market_line(
        line_text=line_text,
        game_id=game_id,
        market_type=market_type,
    )
    _stage1_validate_specific_market_line(
        row=row,
        line_text=line_text,
        game_id=game_id,
        market_type=market_type,
        bet_side=bet_side,
        home_spread=home_spread,
        away_spread=away_spread,
    )

def _stage1_validate_basic_market_line(
    *,
    line_text: str,
    game_id: str,
    market_type: str,
) -> None:
    if market_type == "h2h":
        if line_text:
            raise ValueError(f"H2H row has nonblank line for game_id={game_id}")
    elif to_float(line_text) is None:
        raise ValueError(
            f"{market_type} row has invalid line for game_id={game_id}"
        )


def _stage1_validate_american_decimal(
    *,
    odds_text: str,
    decimal_text: str,
    game_id: str,
    market_type: str,
    bet_side: str,
) -> None:
    if odds_text:
        american = to_float(odds_text)
        expected_decimal = to_float(american_to_decimal(odds_text))
        actual_decimal = to_float(decimal_text)
        if (
            american is None
            or american == 0
            or expected_decimal is None
            or actual_decimal is None
            or abs(expected_decimal - actual_decimal) > 0.000001
        ):
            raise ValueError(
                "American/decimal odds mismatch for "
                f"game_id={game_id}, market={market_type}, side={bet_side}"
            )
    elif decimal_text:
        raise ValueError(
            "Decimal odds present without American odds for "
            f"game_id={game_id}, market={market_type}, side={bet_side}"
        )


def _stage1_validate_spread_pair(
    row: dict[str, str],
    *,
    game_id: str,
) -> tuple[Optional[float], Optional[float]]:
    home_spread = to_float(row["home_spread"])
    away_spread = to_float(row["away_spread"])
    if (
        home_spread is not None
        and away_spread is not None
        and abs(home_spread + away_spread) > 0.000001
    ):
        raise ValueError(
            f"Home/away spread mismatch for game_id={game_id}: "
            f"home={home_spread}, away={away_spread}"
        )
    return home_spread, away_spread


def _stage1_validate_specific_market_line(
    *,
    row: dict[str, str],
    line_text: str,
    game_id: str,
    market_type: str,
    bet_side: str,
    home_spread: Optional[float],
    away_spread: Optional[float],
) -> None:
    if market_type == "spreads":
        expected_line = home_spread if bet_side == "home" else away_spread
        actual_line = to_float(line_text)
        if (
            expected_line is None
            or actual_line is None
            or abs(expected_line - actual_line) > 0.000001
        ):
            raise ValueError(
                f"Spread row line mismatch for game_id={game_id}, side={bet_side}"
            )
    if market_type == "totals":
        total = to_float(row["total"])
        actual_line = to_float(line_text)
        if (
            total is None
            or actual_line is None
            or abs(total - actual_line) > 0.000001
        ):
            raise ValueError(
                f"Total row line mismatch for game_id={game_id}, side={bet_side}"
            )


def _stage1_validate_odds_row_prices(
    row: dict[str, str],
    *,
    game_id: str,
    market_type: str,
    bet_side: str,
) -> None:
    line_text = str(row["line"]).strip()
    odds_text = str(row["odds_american"]).strip()
    decimal_text = str(row["odds_decimal"]).strip()
    _stage1_validate_basic_market_line(
        line_text=line_text,
        game_id=game_id,
        market_type=market_type,
    )
    _stage1_validate_american_decimal(
        odds_text=odds_text,
        decimal_text=decimal_text,
        game_id=game_id,
        market_type=market_type,
        bet_side=bet_side,
    )
    home_spread, away_spread = _stage1_validate_spread_pair(
        row,
        game_id=game_id,
    )
    _stage1_validate_specific_market_line(
        row=row,
        line_text=line_text,
        game_id=game_id,
        market_type=market_type,
        bet_side=bet_side,
        home_spread=home_spread,
        away_spread=away_spread,
    )




def validate_normalized_rows(
    rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    snapshot_id: str,
    snapshot_fetched_at: str,
) -> None:
    if not rows:
        raise ValueError("Normalized odds output is empty")

    parse_aware_iso(snapshot_fetched_at, "snapshot_fetched_at")
    schedule_by_id = {
        str(row["game_id"]).strip(): row for row in target_rows
    }
    seen_keys: set[tuple[str, str, str]] = set()

    for index, row in enumerate(rows):
        game_id, market_type, bet_side = _stage1_validate_odds_row_identity(
            row,
            index=index,
            schedule_by_id=schedule_by_id,
            snapshot_id=snapshot_id,
            snapshot_fetched_at=snapshot_fetched_at,
        )
        key = (game_id, market_type, bet_side)
        if key in seen_keys:
            raise ValueError(f"Duplicate normalized odds row key: {key}")
        seen_keys.add(key)
        _stage1_validate_odds_row_prices(
            row,
            game_id=game_id,
            market_type=market_type,
            bet_side=bet_side,
        )



def write_csv_file(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    write_csv_rows_durable(
        path,
        rows,
        OUTPUT_COLUMNS,
        project_columns=True,
    )

def write_json_file(
    path: Path,
    payload: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def temporary_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def backup_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.bak"
    )



def _ensure_output_parent_dirs(
    paths: list[Path],
) -> None:
    for path in paths:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


def _ensure_snapshot_paths_absent(
    raw_snapshot_path: Path,
    csv_snapshot_path: Path,
) -> None:
    for snapshot_path in (
        raw_snapshot_path,
        csv_snapshot_path,
    ):
        if snapshot_path.exists():
            raise FileExistsError(
                "Refusing to overwrite immutable odds snapshot: "
                f"{snapshot_path}"
            )


def _restore_current_output(
    *,
    backup: Path,
    final_path: Path,
    had_existing: bool,
) -> None:
    if backup.exists():
        try:
            os.replace(
                backup,
                final_path,
            )
        except OSError:
            pass

    elif not had_existing:
        try:
            final_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass


def publish_output_bundle(
    *,
    raw_path: Path,
    csv_path: Path,
    raw_snapshot_path: Path,
    csv_snapshot_path: Path,
    raw_payload: dict,
    rows: list[dict[str, str]],
) -> None:
    finals = [
        raw_path,
        csv_path,
        raw_snapshot_path,
        csv_snapshot_path,
    ]

    _ensure_output_parent_dirs(finals)

    _ensure_snapshot_paths_absent(
        raw_snapshot_path,
        csv_snapshot_path,
    )

    temp_raw = temporary_path(
        raw_path
    )

    temp_csv = temporary_path(
        csv_path
    )

    temp_raw_snapshot = temporary_path(
        raw_snapshot_path
    )

    temp_csv_snapshot = temporary_path(
        csv_snapshot_path
    )

    temp_paths = [
        temp_raw,
        temp_csv,
        temp_raw_snapshot,
        temp_csv_snapshot,
    ]

    raw_backup = backup_path(
        raw_path
    )

    csv_backup = backup_path(
        csv_path
    )

    raw_had_existing = (
        raw_path.exists()
    )

    csv_had_existing = (
        csv_path.exists()
    )

    snapshot_published: list[
        Path
    ] = []

    current_published: list[
        Path
    ] = []

    published_successfully = False

    try:
        write_json_file(
            temp_raw,
            raw_payload,
        )

        write_csv_file(
            temp_csv,
            rows,
        )

        write_json_file(
            temp_raw_snapshot,
            raw_payload,
        )

        write_csv_file(
            temp_csv_snapshot,
            rows,
        )

        if raw_had_existing:
            os.replace(
                raw_path,
                raw_backup,
            )

        if csv_had_existing:
            os.replace(
                csv_path,
                csv_backup,
            )

        os.replace(
            temp_raw_snapshot,
            raw_snapshot_path,
        )

        snapshot_published.append(
            raw_snapshot_path
        )

        os.replace(
            temp_csv_snapshot,
            csv_snapshot_path,
        )

        snapshot_published.append(
            csv_snapshot_path
        )

        os.replace(
            temp_raw,
            raw_path,
        )

        current_published.append(
            raw_path
        )

        os.replace(
            temp_csv,
            csv_path,
        )

        current_published.append(
            csv_path
        )

        published_successfully = True

    except Exception:
        for path in current_published:
            try:
                path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

        _restore_current_output(
            backup=raw_backup,
            final_path=raw_path,
            had_existing=raw_had_existing,
        )

        _restore_current_output(
            backup=csv_backup,
            final_path=csv_path,
            had_existing=csv_had_existing,
        )

        for path in snapshot_published:
            try:
                path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

        raise

    finally:
        for path in temp_paths:
            try:
                path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

        if published_successfully:
            for path in (
                raw_backup,
                csv_backup,
            ):
                try:
                    path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass



def latest_existing_odds_pair() -> tuple[
    Path,
    Path,
]:
    files = sorted(
        ODDS_DIR.glob(
            "*_CFB_odds.csv"
        ),
        key=lambda path: path.name,
        reverse=True,
    )

    for odds_csv_path in files:
        match = re.fullmatch(
            r"(\d{4}_\d{2}_\d{2})_CFB_odds\.csv",
            odds_csv_path.name,
        )

        if match is None:
            continue

        raw_path = (
            RAW_ODDS_DIR
            / f"{match.group(1)}_cfb_odds.json"
        )

        if raw_path.exists():
            return (
                odds_csv_path,
                raw_path,
            )

    raise FileNotFoundError(
        "All target-week games are locked and no "
        "prior normalized/raw CFB odds pair is available "
        "to preserve."
    )


def main() -> int:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "ESPN Core API",
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.set_detail(
            "output_modified",
            False,
        )

        season, season_type, week = (
            load_current_week_config(CURRENT_WEEK_CONFIG_PATH)
        )

        report.season = season
        report.week = week

        report.set_detail(
            "season_type",
            season_type,
        )

        schedule_path, target_rows = (
            load_target_schedule(
                season,
                season_type,
                week,
            )
        )

        report.add_input(
            schedule_path
        )

        captured_at = utc_now()

        run_date = captured_at.strftime(
            "%Y_%m_%d"
        )

        snapshot_id = captured_at.strftime(
            "%Y_%m_%d_%H%M%S_%f"
        )

        snapshot_fetched_at = (
            captured_at.isoformat()
        )

        raw_path = (
            RAW_ODDS_DIR
            / f"{run_date}_cfb_odds.json"
        )

        csv_path = (
            ODDS_DIR
            / f"{run_date}_CFB_odds.csv"
        )

        raw_snapshot_path = (
            RAW_SNAPSHOT_DIR
            / f"{snapshot_id}_cfb_odds.json"
        )

        csv_snapshot_path = (
            SNAPSHOT_DIR
            / f"{snapshot_id}_CFB_odds.csv"
        )

        for path in (
            raw_path,
            csv_path,
            raw_snapshot_path,
            csv_snapshot_path,
        ):
            report.add_output(
                path
            )

        locked_game_ids: list[
            str
        ] = []

        empty_game_ids: list[
            str
        ] = []

        hard_failures: list[
            dict[str, str]
        ] = []

        raw_events: list[
            dict[str, str]
        ] = []

        raw_odds: list[
            dict
        ] = []

        rows: list[
            dict[str, str]
        ] = []

        request_urls: list[
            dict[str, ScalarValue]
        ] = []

        attempted_count = 0

        for schedule_row in target_rows:
            game_id = str(
                schedule_row[
                    "game_id"
                ]
            ).strip()

            kickoff = (
                schedule_kickoff_iso(
                    schedule_row
                )
            )

            if is_game_locked(
                schedule_row,
                captured_at,
            ):
                locked_game_ids.append(
                    game_id
                )

                request_urls.append(
                    {
                        "game_id": game_id,
                        "url": "",
                        "status": "LOCKED",
                        "result": "LOCKED",
                    }
                )

                continue

            attempted_count += 1

            try:
                (
                    odds_item,
                    odds_url,
                    status,
                    result,
                ) = fetch_game_odds(
                    game_id
                )

                request_urls.append(
                    {
                        "game_id": game_id,
                        "url": odds_url,
                        "status": status,
                        "result": result,
                    }
                )

                if odds_item is None:
                    empty_game_ids.append(
                        game_id
                    )
                    continue

                event = {
                    "id": game_id,
                    "date": kickoff,
                    "home": str(
                        schedule_row[
                            "home_team"
                        ]
                    ).strip(),
                    "away": str(
                        schedule_row[
                            "away_team"
                        ]
                    ).strip(),
                }

                (
                    normalized_rows,
                    provider,
                ) = normalize_event_odds(
                    event,
                    odds_item,
                    snapshot_id,
                    snapshot_fetched_at,
                )

                if not normalized_rows:
                    empty_game_ids.append(
                        game_id
                    )

                    request_urls[-1][
                        "result"
                    ] = (
                        "NO_SUPPORTED_MARKETS"
                    )

                    continue

                raw_events.append(
                    event
                )

                raw_odds.append(
                    {
                        "game_id": game_id,
                        "provider": provider,
                        "odds": odds_item,
                    }
                )

                rows.extend(
                    normalized_rows
                )

            except Exception as exc:
                hard_failures.append(
                    {
                        "game_id": game_id,
                        "error_type": (
                            type(exc).__name__
                        ),
                        "message": str(exc),
                    }
                )

                request_urls.append(
                    {
                        "game_id": game_id,
                        "url": "",
                        "status": "ERROR",
                        "result": "HARD_FAILURE",
                        "error": str(exc),
                    }
                )

        provider_counts: dict[
            str,
            int,
        ] = {}

        for item in raw_odds:
            provider = item.get(
                "provider",
                {},
            )

            provider_name = str(
                provider.get("name")
                or provider.get("id")
                or "unknown"
            ).strip()

            provider_counts[
                provider_name
            ] = (
                provider_counts.get(
                    provider_name,
                    0,
                )
                + 1
            )

        report.set_rows(
            rows_in=len(
                target_rows
            ),
        )

        report.update_details(
            {
                "snapshot_id": snapshot_id,
                "snapshot_fetched_at": (
                    snapshot_fetched_at
                ),
                "target_games": len(
                    target_rows
                ),
                "locked_games": len(
                    locked_game_ids
                ),
                "locked_game_ids": (
                    locked_game_ids
                ),
                "games_requested": (
                    attempted_count
                ),
                "games_with_odds": len(
                    raw_odds
                ),
                "games_without_posted_odds": len(
                    empty_game_ids
                ),
                "games_without_posted_odds_ids": (
                    empty_game_ids
                ),
                "hard_fetch_failures": len(
                    hard_failures
                ),
                "hard_fetch_failure_details": (
                    hard_failures
                ),
                "provider_event_counts": (
                    provider_counts
                ),
                "normalized_rows": len(
                    rows
                ),
            }
        )

        if hard_failures:
            raise RuntimeError(
                "One or more target-week ESPN odds requests failed; "
                "refusing to publish a partial odds snapshot. "
                f"failures={len(hard_failures)}"
            )

        if attempted_count == 0:
            (
                preserved_csv_path,
                preserved_raw_path,
            ) = latest_existing_odds_pair()

            report.warning(
                "All configured target-week games are locked; "
                "preserving the latest existing odds pair."
            )

            report.add_input(
                preserved_csv_path
            )

            report.add_input(
                preserved_raw_path
            )

            report.set_rows(
                rows_out=0,
            )

            report.update_details(
                {
                    "output_modified": False,
                    "no_op_reason": (
                        "all_target_games_locked"
                    ),
                    "preserved_normalized_csv": str(
                        preserved_csv_path
                    ),
                    "preserved_raw_json": str(
                        preserved_raw_path
                    ),
                }
            )

            print(
                "pull_odds.py completed "
                "with no-op: all target-week "
                "games are locked"
            )

            print(
                "preserved_csv="
                f"{preserved_csv_path}"
            )

            print(
                "preserved_raw="
                f"{preserved_raw_path}"
            )

            return 0

        if not rows:
            raise RuntimeError(
                "ESPN returned no supported current odds for any "
                "unlocked configured target-week game. "
                "No odds outputs were modified."
            )

        validate_normalized_rows(
            rows,
            target_rows,
            snapshot_id,
            snapshot_fetched_at,
        )

        raw_payload = {
            "snapshot_id": snapshot_id,
            "fetched_at": snapshot_fetched_at,
            "sport": "football",
            "league": "college-football",
            "source": "ESPN Core API",
            "schedule_input": str(
                schedule_path
            ),
            "selected_schedule_group": {
                "season": str(
                    season
                ),
                "season_type": str(
                    season_type
                ),
                "week": str(
                    week
                ),
            },
            "request_urls": request_urls,
            "target_games_count": len(
                target_rows
            ),
            "locked_games_count": len(
                locked_game_ids
            ),
            "empty_odds_games_count": len(
                empty_game_ids
            ),
            "events_count": len(
                raw_events
            ),
            "odds_events_count": len(
                raw_odds
            ),
            "events": raw_events,
            "odds": raw_odds,
        }

        publish_output_bundle(
            raw_path=raw_path,
            csv_path=csv_path,
            raw_snapshot_path=(
                raw_snapshot_path
            ),
            csv_snapshot_path=(
                csv_snapshot_path
            ),
            raw_payload=raw_payload,
            rows=rows,
        )

        report.set_rows(
            rows_out=len(rows),
        )

        report.update_details(
            {
                "output_modified": True,
                "current_raw_json": str(
                    raw_path
                ),
                "current_normalized_csv": str(
                    csv_path
                ),
                "snapshot_raw_json": str(
                    raw_snapshot_path
                ),
                "snapshot_normalized_csv": str(
                    csv_snapshot_path
                ),
            }
        )

        print(
            "pull_odds.py completed"
        )

        print(
            f"season={season} "
            f"season_type={season_type} "
            f"week={week}"
        )

        print(
            f"snapshot_id={snapshot_id}"
        )

        print(
            f"target_games={len(target_rows)}"
        )

        print(
            f"games_requested={attempted_count}"
        )

        print(
            f"games_with_odds={len(raw_odds)}"
        )

        print(
            "games_without_posted_odds="
            f"{len(empty_game_ids)}"
        )

        print(
            f"normalized_rows={len(rows)}"
        )

        print(
            f"current_csv={csv_path}"
        )

        print(
            f"snapshot_csv={csv_snapshot_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )