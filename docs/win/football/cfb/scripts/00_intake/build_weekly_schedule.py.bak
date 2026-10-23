#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/build_weekly_schedule.py

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo



SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    load_current_week_config,
    write_atomic_csv_rows,
)
from type_support import ScalarValue


SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
WEEKLY_DIR = SCHEDULE_DIR / "weekly"

ODDS_DIR = CFB_ROOT / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"

CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = (
    "cfb-weekly-schedule-v2-2026-10-02-schedule-drift"
)

OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "odds_provider_game_id",
    "game_date",
    "game_time",
    "commence_time",
    "kickoff_utc",
    "game_locked",
    "away_team",
    "home_team",
    "odds_away_team",
    "odds_home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
    "odds_last_update",
    "odds_available",
    "odds_missing_reason",
]

SCHEDULE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
]

ODDS_REQUIRED_COLUMNS = [
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

SUMMARY_FIELDS = [
    "bookmaker",
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

VALID_REQUEST_RESULTS = {
    "AVAILABLE",
    "EMPTY",
    "NO_SUPPORTED_MARKETS",
    "LOCKED",
}

VALID_MISSING_REASONS = {
    "",
    "no_odds_returned",
    "no_supported_markets",
    "locked_before_first_capture",
}


def read_csv(
    path: Path,
    required_columns: list[str],
    label: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    rows: list[dict[str, str]] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        missing = [
            column
            for column
            in required_columns
            if column not in fieldnames
        ]

        if missing:
            raise ValueError(
                f"{label} missing columns: {missing}"
            )

        rows = list(
            reader
        )

    return rows

def schedule_kickoff_utc(
    row: dict[str, str],
) -> datetime:
    game_id = str(
        row.get(
            "game_id",
            "",
        )
    ).strip()

    game_date = str(
        row.get(
            "game_date",
            "",
        )
    ).strip()

    game_time = str(
        row.get(
            "game_time",
            "",
        )
    ).strip()

    game_timezone = str(
        row.get(
            "game_timezone",
            "",
        )
    ).strip()

    missing = [
        field
        for field, value in (
            (
                "game_date",
                game_date,
            ),
            (
                "game_time",
                game_time,
            ),
            (
                "game_timezone",
                game_timezone,
            ),
        )
        if not value
    ]

    if missing:
        raise ValueError(
            f"Game {game_id or '<blank>'} "
            "missing kickoff fields: "
            + ", ".join(
                missing
            )
        )

    try:
        timezone_info = ZoneInfo(
            game_timezone
        )
    except Exception as exc:
        raise ValueError(
            f"Game {game_id or '<blank>'} "
            "has invalid game_timezone="
            f"{game_timezone!r}"
        ) from exc

    try:
        local_dt = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError as exc:
        raise ValueError(
            f"Game {game_id or '<blank>'} "
            "has invalid kickoff date/time: "
            f"date={game_date!r}, "
            f"time={game_time!r}"
        ) from exc

    return local_dt.replace(
        tzinfo=timezone_info
    ).astimezone(
        timezone.utc
    )


def kickoff_iso(
    row: dict[str, str],
) -> str:
    return (
        schedule_kickoff_utc(
            row
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def parse_aware_iso(
    value: ScalarValue,
    label: str,
) -> datetime:
    text = str(
        value
        or ""
    ).strip()

    if not text:
        raise ValueError(
            f"{label} is blank"
        )

    if text.endswith(
        "Z"
    ):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = (
            datetime.fromisoformat(
                text
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} is not a valid "
            f"ISO timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"{label} must include timezone: "
            f"{value!r}"
        )

    return parsed.astimezone(
        timezone.utc
    )


def game_is_locked(
    row: dict[str, str],
    now_utc: datetime,
) -> bool:
    return (
        now_utc
        >= schedule_kickoff_utc(
            row
        )
    )


def load_schedule(
    season: int,
    season_type: int,
    week: int,
) -> tuple[
    Path,
    list[dict[str, str]],
    list[dict[str, str]],
]:
    path = (
        SCHEDULE_DIR
        / f"{season}_schedule.csv"
    )

    schedule_rows = read_csv(
        path,
        SCHEDULE_REQUIRED_COLUMNS,
        "CFB schedule CSV",
    )

    target_rows = [
        row
        for row in schedule_rows
        if str(
            row.get(
                "season",
                "",
            )
        ).strip() == str(
            season
        )
        and str(
            row.get(
                "season_type",
                "",
            )
        ).strip() == str(
            season_type
        )
        and str(
            row.get(
                "week",
                "",
            )
        ).strip() == str(
            week
        )
    ]

    if not target_rows:
        raise ValueError(
            "Configured current week was not "
            "found in schedule: "
            f"season={season}, "
            f"season_type={season_type}, "
            f"week={week}"
        )

    seen_ids: set[str] = set()

    for row in target_rows:
        game_id = str(
            row.get(
                "game_id",
                "",
            )
        ).strip()

        home_team = str(
            row.get(
                "home_team",
                "",
            )
        ).strip()

        away_team = str(
            row.get(
                "away_team",
                "",
            )
        ).strip()

        if not game_id:
            raise ValueError(
                "Target schedule contains "
                "blank game_id"
            )

        if game_id in seen_ids:
            raise ValueError(
                "Target schedule contains "
                "duplicate game_id="
                f"{game_id}"
            )

        seen_ids.add(
            game_id
        )

        if not home_team:
            raise ValueError(
                f"Game {game_id} has "
                "blank home_team"
            )

        if not away_team:
            raise ValueError(
                f"Game {game_id} has "
                "blank away_team"
            )

        schedule_kickoff_utc(
            row
        )

    target_rows.sort(
        key=lambda row: (
            schedule_kickoff_utc(
                row
            ),
            str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip(),
        )
    )

    return (
        path,
        schedule_rows,
        target_rows,
    )


def latest_odds_pair() -> tuple[
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

    if not files:
        raise FileNotFoundError(
            "No current normalized CFB "
            f"odds CSV found in {ODDS_DIR}"
        )

    odds_csv_path = files[0]

    match = re.fullmatch(
        r"(\d{4}_\d{2}_\d{2})_CFB_odds\.csv",
        odds_csv_path.name,
    )

    if not match:
        raise ValueError(
            "Current odds CSV filename does not "
            "match expected convention: "
            f"{odds_csv_path.name}"
        )

    date_key = match.group(
        1
    )

    raw_path = (
        RAW_ODDS_DIR
        / f"{date_key}_cfb_odds.json"
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            "Matching raw odds JSON is missing "
            "for normalized current odds CSV: "
            f"{raw_path}"
        )

    return (
        odds_csv_path,
        raw_path,
    )


def load_raw_odds_payload(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw odds JSON: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Raw odds JSON must contain "
            "a JSON object"
        )

    for field in (
        "snapshot_id",
        "fetched_at",
        "selected_schedule_group",
        "request_urls",
        "events",
        "odds",
    ):
        if field not in payload:
            raise ValueError(
                "Raw odds JSON missing "
                f"required field: {field}"
            )

    if not isinstance(
        payload["selected_schedule_group"],
        dict,
    ):
        raise ValueError(
            "Raw odds selected_schedule_group "
            "must be an object"
        )

    if not isinstance(
        payload["request_urls"],
        list,
    ):
        raise ValueError(
            "Raw odds request_urls "
            "must be a list"
        )

    if not isinstance(
        payload["events"],
        list,
    ):
        raise ValueError(
            "Raw odds events "
            "must be a list"
        )

    if not isinstance(
        payload["odds"],
        list,
    ):
        raise ValueError(
            "Raw odds odds "
            "must be a list"
        )

    return payload


def _stage1_snapshot_header(
    *,
    raw_payload: dict,
    target_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
) -> tuple[str, str, dict[str, dict[str, str]], set[str]]:
    snapshot_id = str(raw_payload.get("snapshot_id", "")).strip()
    snapshot_fetched_at = str(raw_payload.get("fetched_at", "")).strip()
    if not snapshot_id:
        raise ValueError("Raw odds snapshot_id is blank")
    parse_aware_iso(snapshot_fetched_at, "raw odds fetched_at")

    selected_group = raw_payload["selected_schedule_group"]
    expected_group = {
        "season": str(season),
        "season_type": str(season_type),
        "week": str(week),
    }
    actual_group = {
        key: str(selected_group.get(key, "")).strip()
        for key in ("season", "season_type", "week")
    }
    if actual_group != expected_group:
        raise ValueError(
            "Raw odds snapshot targets the wrong configured schedule group. "
            f"expected={expected_group}, actual={actual_group}"
        )

    target_by_id = {
        str(row["game_id"]).strip(): row for row in target_rows
    }
    return snapshot_id, snapshot_fetched_at, target_by_id, set(target_by_id)


def _stage1_validate_snapshot_csv_rows(
    odds_rows: list[dict[str, str]],
    *,
    snapshot_id: str,
    snapshot_fetched_at: str,
    target_by_id: dict[str, dict[str, str]],
    target_ids: set[str],
) -> set[str]:
    normalized_ids: set[str] = set()
    for index, row in enumerate(odds_rows):
        row_snapshot_id = str(row.get("snapshot_id", "")).strip()
        row_fetched_at = str(row.get("snapshot_fetched_at", "")).strip()
        if row_snapshot_id != snapshot_id:
            raise ValueError(
                "Normalized odds CSV snapshot_id does not match raw odds JSON at "
                f"row {index}: {row_snapshot_id!r} != {snapshot_id!r}"
            )
        if row_fetched_at != snapshot_fetched_at:
            raise ValueError(
                "Normalized odds CSV snapshot_fetched_at does not match raw odds JSON "
                f"at row {index}"
            )
        parse_aware_iso(row_fetched_at, "normalized odds snapshot_fetched_at")

        game_id = str(row.get("game_id", "")).strip()
        if not game_id:
            raise ValueError(
                f"Normalized odds CSV contains blank game_id at row {index}"
            )
        if game_id not in target_ids:
            raise ValueError(
                f"Normalized odds CSV contains out-of-scope game_id={game_id}"
            )
        normalized_ids.add(game_id)

        target_row = target_by_id[game_id]
        expected_home = str(target_row["home_team"]).strip()
        expected_away = str(target_row["away_team"]).strip()
        expected_kickoff = kickoff_iso(target_row)
        if str(row.get("home_team", "")).strip() != expected_home:
            raise ValueError(
                "Normalized odds home_team does not match schedule for "
                f"game_id={game_id}"
            )
        if str(row.get("away_team", "")).strip() != expected_away:
            raise ValueError(
                "Normalized odds away_team does not match schedule for "
                f"game_id={game_id}"
            )
        if str(row.get("commence_time", "")).strip() != expected_kickoff:
            raise ValueError(
                "Normalized odds commence_time does not match schedule for "
                f"game_id={game_id}"
            )
    return normalized_ids


def _stage1_request_records(
    raw_payload: dict,
    *,
    target_ids: set[str],
) -> dict[str, dict[str, ScalarValue]]:
    request_by_id: dict[str, dict[str, ScalarValue]] = {}
    for index, request in enumerate(raw_payload["request_urls"]):
        if not isinstance(request, dict):
            raise ValueError(
                "Raw odds request_urls contains a non-object at "
                f"index {index}"
            )
        game_id = str(request.get("game_id", "")).strip()
        if not game_id:
            raise ValueError(
                f"Raw odds request_urls contains blank game_id at index {index}"
            )
        if game_id in request_by_id:
            raise ValueError(
                f"Raw odds request_urls contains duplicate game_id={game_id}"
            )
        if game_id not in target_ids:
            raise ValueError(
                f"Raw odds request_urls contains out-of-scope game_id={game_id}"
            )
        result = str(request.get("result", "")).strip()
        if result not in VALID_REQUEST_RESULTS:
            raise ValueError(
                "Raw odds request result is invalid for game_id="
                f"{game_id}: {result!r}"
            )
        request_by_id[game_id] = request

    missing_request_ids = sorted(target_ids - set(request_by_id))
    if missing_request_ids:
        raise ValueError(
            "Raw odds snapshot is missing target game request records: "
            + ", ".join(missing_request_ids[:20])
        )
    return request_by_id


def _stage1_raw_snapshot_ids(
    values: list,
    *,
    id_field: str,
    object_label: str,
    target_ids: set[str],
) -> list[str]:
    ids: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(
                f"Raw odds {object_label} contains a non-object at index {index}"
            )
        game_id = str(item.get(id_field, "")).strip()
        if not game_id:
            noun = "event" if object_label == "events" else "object"
            raise ValueError(
                f"Raw odds {noun} contains blank {id_field} at index {index}"
            )
        if game_id not in target_ids:
            field_label = "id" if object_label == "events" else "game_id"
            noun = "event" if object_label == "events" else "object"
            raise ValueError(
                f"Raw odds {noun} contains out-of-scope {field_label}={game_id}"
            )
        ids.append(game_id)
    if len(ids) != len(set(ids)):
        suffix = "events" if object_label == "events" else "objects"
        raise ValueError(f"Raw odds {suffix} contains duplicate game IDs")
    return ids


def _stage1_validate_available_snapshot_coverage(
    *,
    normalized_ids: set[str],
    raw_event_ids: list[str],
    raw_odds_ids: list[str],
    request_by_id: dict[str, dict[str, ScalarValue]],
) -> None:
    available_ids = {
        game_id
        for game_id, request in request_by_id.items()
        if str(request.get("result", "")).strip() == "AVAILABLE"
    }
    if set(raw_event_ids) != available_ids:
        raise ValueError(
            "Raw odds events do not exactly match AVAILABLE request results. "
            f"events={sorted(raw_event_ids)}, available={sorted(available_ids)}"
        )
    if set(raw_odds_ids) != available_ids:
        raise ValueError(
            "Raw odds objects do not exactly match AVAILABLE request results. "
            f"odds={sorted(raw_odds_ids)}, available={sorted(available_ids)}"
        )
    if normalized_ids != available_ids:
        raise ValueError(
            "Normalized odds game IDs do not exactly match AVAILABLE request results. "
            f"normalized={sorted(normalized_ids)}, available={sorted(available_ids)}"
        )


def validate_snapshot_provenance(
    *,
    odds_rows: list[dict[str, str]],
    raw_payload: dict,
    target_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
) -> tuple[str, str, dict[str, dict[str, ScalarValue]]]:
    if not odds_rows:
        raise ValueError("Normalized current odds CSV is empty")

    snapshot_id, snapshot_fetched_at, target_by_id, target_ids = (
        _stage1_snapshot_header(
            raw_payload=raw_payload,
            target_rows=target_rows,
            season=season,
            season_type=season_type,
            week=week,
        )
    )
    normalized_ids = _stage1_validate_snapshot_csv_rows(
        odds_rows,
        snapshot_id=snapshot_id,
        snapshot_fetched_at=snapshot_fetched_at,
        target_by_id=target_by_id,
        target_ids=target_ids,
    )
    request_by_id = _stage1_request_records(
        raw_payload,
        target_ids=target_ids,
    )
    raw_event_ids = _stage1_raw_snapshot_ids(
        raw_payload["events"],
        id_field="id",
        object_label="events",
        target_ids=target_ids,
    )
    raw_odds_ids = _stage1_raw_snapshot_ids(
        raw_payload["odds"],
        id_field="game_id",
        object_label="objects",
        target_ids=target_ids,
    )
    _stage1_validate_available_snapshot_coverage(
        normalized_ids=normalized_ids,
        raw_event_ids=raw_event_ids,
        raw_odds_ids=raw_odds_ids,
        request_by_id=request_by_id,
    )
    return snapshot_id, snapshot_fetched_at, request_by_id



def latest_last_update(
    rows: list[dict[str, str]],
) -> str:
    values = sorted(
        {
            str(
                row.get(
                    "last_update",
                    "",
                )
            ).strip()
            for row in rows
            if str(
                row.get(
                    "last_update",
                    "",
                )
            ).strip()
        }
    )

    return (
        values[-1]
        if values
        else ""
    )


def build_odds_summary(
    odds_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    target_by_id = {
        str(
            row[
                "game_id"
            ]
        ).strip(): row
        for row in target_rows
    }

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = {}

    seen_market_keys: set[
        tuple[str, str, str],
    ] = set()

    for row in odds_rows:
        game_id = str(
            row.get(
                "game_id",
                "",
            )
        ).strip()

        market_type = str(
            row.get(
                "market_type",
                "",
            )
        ).strip()

        bet_side = str(
            row.get(
                "bet_side",
                "",
            )
        ).strip()

        key = (
            game_id,
            market_type,
            bet_side,
        )

        if key in seen_market_keys:
            raise ValueError(
                "Normalized odds contains "
                "duplicate market row: "
                f"{key}"
            )

        seen_market_keys.add(
            key
        )

        grouped.setdefault(
            game_id,
            [],
        ).append(
            row
        )

    summaries: dict[
        str,
        dict[str, str],
    ] = {}

    for game_id, rows in grouped.items():
        if game_id not in target_by_id:
            raise ValueError(
                "Odds summary contains "
                "out-of-scope game_id="
                f"{game_id}"
            )

        first = rows[0]

        for field in (
            "commence_time",
            "home_team",
            "away_team",
            *SUMMARY_FIELDS,
        ):
            observed = {
                str(
                    row.get(
                        field,
                        "",
                    )
                ).strip()
                for row in rows
            }

            if len(
                observed
            ) > 1:
                raise ValueError(
                    "Normalized odds rows disagree "
                    f"on {field} for game_id="
                    f"{game_id}: {sorted(observed)}"
                )

        bookmaker = str(
            first.get(
                "bookmaker",
                "",
            )
        ).strip()

        if not bookmaker:
            raise ValueError(
                "Normalized odds contains "
                "blank bookmaker for game_id="
                f"{game_id}"
            )

        summaries[
            game_id
        ] = {
            "bookmaker": bookmaker,
            "home_moneyline_american": (
                first.get(
                    "home_moneyline_american",
                    "",
                )
            ),
            "away_moneyline_american": (
                first.get(
                    "away_moneyline_american",
                    "",
                )
            ),
            "home_spread": first.get(
                "home_spread",
                "",
            ),
            "away_spread": first.get(
                "away_spread",
                "",
            ),
            "home_spread_american": (
                first.get(
                    "home_spread_american",
                    "",
                )
            ),
            "away_spread_american": (
                first.get(
                    "away_spread_american",
                    "",
                )
            ),
            "total": first.get(
                "total",
                "",
            ),
            "over_american": first.get(
                "over_american",
                "",
            ),
            "under_american": first.get(
                "under_american",
                "",
            ),
            "odds_last_update": (
                latest_last_update(
                    rows
                )
            ),
            "commence_time": str(
                first.get(
                    "commence_time",
                    "",
                )
            ).strip(),
            "odds_home_team": str(
                first.get(
                    "home_team",
                    "",
                )
            ).strip(),
            "odds_away_team": str(
                first.get(
                    "away_team",
                    "",
                )
            ).strip(),
        }

    return summaries


def _validate_existing_weekly_rows(
    rows: list[dict[str, str]],
    existing: dict[str, dict[str, str]],
    target_by_id: dict[str, dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
) -> None:
    for index, row in enumerate(
        rows
    ):
        game_id = str(
            row.get(
                "game_id",
                "",
            )
        ).strip()

        if not game_id:
            raise ValueError(
                "Existing weekly schedule "
                "contains blank game_id at "
                f"row {index}"
            )

        if game_id in existing:
            raise ValueError(
                "Existing weekly schedule "
                "contains duplicate game_id="
                f"{game_id}"
            )

        if game_id not in target_by_id:
            raise ValueError(
                "Existing weekly schedule "
                "contains out-of-scope game_id="
                f"{game_id}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ).strip() != str(
            season
        ):
            raise ValueError(
                "Existing weekly schedule "
                "season mismatch for game_id="
                f"{game_id}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ).strip() != str(
            season_type
        ):
            raise ValueError(
                "Existing weekly schedule "
                "season_type mismatch for "
                f"game_id={game_id}"
            )

        if str(
            row.get(
                "week",
                "",
            )
        ).strip() != str(
            week
        ):
            raise ValueError(
                "Existing weekly schedule "
                "week mismatch for game_id="
                f"{game_id}"
            )

        target = (
            target_by_id[
                game_id
            ]
        )

        if str(
            row.get(
                "home_team",
                "",
            )
        ).strip() != str(
            target.get(
                "home_team",
                "",
            )
        ).strip():
            raise ValueError(
                "Existing weekly schedule "
                "home_team mismatch for "
                f"game_id={game_id}"
            )

        if str(
            row.get(
                "away_team",
                "",
            )
        ).strip() != str(
            target.get(
                "away_team",
                "",
            )
        ).strip():
            raise ValueError(
                "Existing weekly schedule "
                "away_team mismatch for "
                f"game_id={game_id}"
            )

        # Schedule timing is mutable until kickoff. Do not reject an
        # existing weekly row solely because ESPN changed game_date,
        # game_time, or the derived kickoff_utc. build_output_rows()
        # refreshes those fields from the current canonical schedule
        # while preserving previously captured odds for locked games.

        locked = str(
            row.get(
                "game_locked",
                "",
            )
        ).strip()

        if locked not in {
            "0",
            "1",
        }:
            raise ValueError(
                "Existing weekly schedule "
                "contains invalid game_locked "
                f"for game_id={game_id}: "
                f"{locked!r}"
            )

        odds_available = str(
            row.get(
                "odds_available",
                "",
            )
        ).strip()

        if odds_available not in {
            "0",
            "1",
        }:
            raise ValueError(
                "Existing weekly schedule "
                "contains invalid "
                "odds_available for game_id="
                f"{game_id}: "
                f"{odds_available!r}"
            )

        existing[
            game_id
        ] = row


def read_existing_weekly(
    path: Path,
    target_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    rows = read_csv(
        path,
        OUTPUT_COLUMNS,
        "existing weekly schedule",
    )

    target_by_id = {
        str(
            row[
                "game_id"
            ]
        ).strip(): row
        for row in target_rows
    }

    existing: dict[
        str,
        dict[str, str],
    ] = {}

    _validate_existing_weekly_rows(
        rows,
        existing,
        target_by_id,
        season=season,
        season_type=season_type,
        week=week,
    )

    return existing


def fresh_schedule_row(
    schedule_row: dict[str, str],
    game_id: str,
    locked: bool,
) -> dict[str, str]:
    return {
        "season": str(
            schedule_row.get(
                "season",
                "",
            )
        ).strip(),
        "season_type": str(
            schedule_row.get(
                "season_type",
                "",
            )
        ).strip(),
        "week": str(
            schedule_row.get(
                "week",
                "",
            )
        ).strip(),
        "game_id": game_id,
        "odds_provider_game_id": game_id,
        "game_date": str(
            schedule_row.get(
                "game_date",
                "",
            )
        ).strip(),
        "game_time": str(
            schedule_row.get(
                "game_time",
                "",
            )
        ).strip(),
        "commence_time": "",
        "kickoff_utc": kickoff_iso(
            schedule_row
        ),
        "game_locked": (
            "1"
            if locked
            else "0"
        ),
        "away_team": str(
            schedule_row.get(
                "away_team",
                "",
            )
        ).strip(),
        "home_team": str(
            schedule_row.get(
                "home_team",
                "",
            )
        ).strip(),
        "odds_away_team": "",
        "odds_home_team": "",
        "neutral_site": str(
            schedule_row.get(
                "neutral_site",
                "",
            )
        ).strip(),
        "stadium": str(
            schedule_row.get(
                "stadium",
                "",
            )
        ).strip(),
        "roof": str(
            schedule_row.get(
                "roof",
                "",
            )
        ).strip(),
        "surface": str(
            schedule_row.get(
                "surface",
                "",
            )
        ).strip(),
        "home_timezone": str(
            schedule_row.get(
                "home_timezone",
                "",
            )
        ).strip(),
        "away_timezone": str(
            schedule_row.get(
                "away_timezone",
                "",
            )
        ).strip(),
        "game_timezone": str(
            schedule_row.get(
                "game_timezone",
                "",
            )
        ).strip(),
        "bookmaker": "",
        "home_moneyline_american": "",
        "away_moneyline_american": "",
        "home_spread": "",
        "away_spread": "",
        "home_spread_american": "",
        "away_spread_american": "",
        "total": "",
        "over_american": "",
        "under_american": "",
        "odds_last_update": "",
        "odds_available": "0",
        "odds_missing_reason": "",
    }


def apply_odds(
    row: dict[str, str],
    odds: dict[str, str],
) -> None:
    for field in (
        "bookmaker",
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
        "odds_last_update",
    ):
        row[field] = str(
            odds.get(
                field,
                "",
            )
        ).strip()

    row["commence_time"] = str(
        odds.get(
            "commence_time",
            "",
        )
    ).strip()

    row["odds_home_team"] = str(
        odds.get(
            "odds_home_team",
            "",
        )
    ).strip()

    row["odds_away_team"] = str(
        odds.get(
            "odds_away_team",
            "",
        )
    ).strip()

    row["odds_available"] = "1"
    row["odds_missing_reason"] = ""


def preserve_locked_row(
    previous: dict[str, str],
    schedule_row: dict[str, str],
) -> dict[str, str]:
    row = {
        column: str(
            previous.get(
                column,
                "",
            )
        )
        for column
        in OUTPUT_COLUMNS
    }

    game_id = str(
        schedule_row[
            "game_id"
        ]
    ).strip()

    row.update(
        {
            "season": str(
                schedule_row[
                    "season"
                ]
            ).strip(),
            "season_type": str(
                schedule_row[
                    "season_type"
                ]
            ).strip(),
            "week": str(
                schedule_row[
                    "week"
                ]
            ).strip(),
            "game_id": game_id,
            "odds_provider_game_id": (
                str(
                    previous.get(
                        "odds_provider_game_id",
                        "",
                    )
                ).strip()
                or game_id
            ),
            "game_date": str(
                schedule_row[
                    "game_date"
                ]
            ).strip(),
            "game_time": str(
                schedule_row[
                    "game_time"
                ]
            ).strip(),
            "kickoff_utc": kickoff_iso(
                schedule_row
            ),
            "game_locked": "1",
            "away_team": str(
                schedule_row[
                    "away_team"
                ]
            ).strip(),
            "home_team": str(
                schedule_row[
                    "home_team"
                ]
            ).strip(),
            "neutral_site": str(
                schedule_row.get(
                    "neutral_site",
                    "",
                )
            ).strip(),
            "stadium": str(
                schedule_row.get(
                    "stadium",
                    "",
                )
            ).strip(),
            "roof": str(
                schedule_row.get(
                    "roof",
                    "",
                )
            ).strip(),
            "surface": str(
                schedule_row.get(
                    "surface",
                    "",
                )
            ).strip(),
            "home_timezone": str(
                schedule_row.get(
                    "home_timezone",
                    "",
                )
            ).strip(),
            "away_timezone": str(
                schedule_row.get(
                    "away_timezone",
                    "",
                )
            ).strip(),
            "game_timezone": str(
                schedule_row.get(
                    "game_timezone",
                    "",
                )
            ).strip(),
        }
    )

    if (
        row["odds_available"]
        == "0"
        and row[
            "odds_missing_reason"
        ] == "no_odds_event_match"
    ):
        row[
            "odds_missing_reason"
        ] = "no_odds_returned"

    return row


def build_output_rows(
    *,
    target_rows: list[dict[str, str]],
    request_by_id: dict[
        str,
        dict[str, ScalarValue],
    ],
    odds_summary: dict[
        str,
        dict[str, str],
    ],
    existing_weekly: dict[
        str,
        dict[str, str],
    ],
    now_utc: datetime,
) -> tuple[
    list[dict[str, str]],
    int,
]:
    output_rows: list[
        dict[str, str]
    ] = []

    locked_preserved = 0

    for schedule_row in target_rows:
        game_id = str(
            schedule_row[
                "game_id"
            ]
        ).strip()

        locked = game_is_locked(
            schedule_row,
            now_utc,
        )

        if (
            locked
            and game_id
            in existing_weekly
        ):
            row = preserve_locked_row(
                existing_weekly[
                    game_id
                ],
                schedule_row,
            )

            locked_preserved += 1

            output_rows.append(
                row
            )

            continue

        row = fresh_schedule_row(
            schedule_row,
            game_id,
            locked,
        )

        request = (
            request_by_id[
                game_id
            ]
        )

        result = str(
            request.get(
                "result",
                "",
            )
        ).strip()

        odds = odds_summary.get(
            game_id
        )

        if result == "AVAILABLE":
            if not odds:
                raise ValueError(
                    "Raw odds request says AVAILABLE "
                    "but normalized odds are missing "
                    f"for game_id={game_id}"
                )

            apply_odds(
                row,
                odds,
            )

        elif result == "EMPTY":
            if odds:
                raise ValueError(
                    "Raw odds request says EMPTY but "
                    "normalized odds exist for "
                    f"game_id={game_id}"
                )

            row[
                "odds_missing_reason"
            ] = "no_odds_returned"

        elif (
            result
            == "NO_SUPPORTED_MARKETS"
        ):
            if odds:
                raise ValueError(
                    "Raw odds request says "
                    "NO_SUPPORTED_MARKETS but "
                    "normalized odds exist for "
                    f"game_id={game_id}"
                )

            row[
                "odds_missing_reason"
            ] = "no_supported_markets"

        elif result == "LOCKED":
            if odds:
                raise ValueError(
                    "Raw odds request says LOCKED "
                    "but normalized odds exist for "
                    f"game_id={game_id}"
                )

            row[
                "odds_missing_reason"
            ] = (
                "locked_before_first_capture"
            )

        else:
            raise ValueError(
                "Unsupported raw odds result "
                f"for game_id={game_id}: "
                f"{result!r}"
            )

        output_rows.append(
            row
        )

    output_rows.sort(
        key=lambda sort_row: (
            sort_row.get(
                "game_date",
                "",
            ),
            sort_row.get(
                "game_time",
                "",
            ),
            sort_row.get(
                "away_team",
                "",
            ),
            sort_row.get(
                "home_team",
                "",
            ),
        )
    )

    return (
        output_rows,
        locked_preserved,
    )


def _stage1_validate_weekly_target_fields(
    row: dict[str, str],
    *,
    game_id: str,
    target: dict[str, str],
    season: int,
    season_type: int,
    week: int,
    now_utc: datetime,
) -> str:
    if str(row.get("season", "")).strip() != str(season):
        raise ValueError(
            f"Weekly schedule output season mismatch for game_id={game_id}"
        )
    if str(row.get("season_type", "")).strip() != str(season_type):
        raise ValueError(
            f"Weekly schedule output season_type mismatch for game_id={game_id}"
        )
    if str(row.get("week", "")).strip() != str(week):
        raise ValueError(
            f"Weekly schedule output week mismatch for game_id={game_id}"
        )

    for field in ("game_date", "game_time", "away_team", "home_team"):
        if str(row.get(field, "")).strip() != str(target.get(field, "")).strip():
            raise ValueError(
                f"Weekly schedule output {field} mismatch for game_id={game_id}"
            )

    expected_kickoff = kickoff_iso(target)
    if str(row.get("kickoff_utc", "")).strip() != expected_kickoff:
        raise ValueError(
            f"Weekly schedule output kickoff_utc mismatch for game_id={game_id}"
        )
    expected_locked = "1" if game_is_locked(target, now_utc) else "0"
    if str(row.get("game_locked", "")).strip() != expected_locked:
        raise ValueError(
            f"Weekly schedule output game_locked mismatch for game_id={game_id}"
        )
    provider_game_id = str(row.get("odds_provider_game_id", "")).strip()
    if provider_game_id != game_id:
        raise ValueError(
            "Weekly schedule output odds_provider_game_id must equal exact game_id "
            f"for game_id={game_id}"
        )
    return expected_kickoff


def _stage1_validate_weekly_odds_fields(
    row: dict[str, str],
    *,
    game_id: str,
    target: dict[str, str],
    expected_kickoff: str,
) -> None:
    odds_available = str(row.get("odds_available", "")).strip()
    if odds_available not in {"0", "1"}:
        raise ValueError(
            "Weekly schedule output has invalid odds_available for "
            f"game_id={game_id}: {odds_available!r}"
        )
    missing_reason = str(row.get("odds_missing_reason", "")).strip()
    if missing_reason not in VALID_MISSING_REASONS:
        raise ValueError(
            "Weekly schedule output has invalid odds_missing_reason for "
            f"game_id={game_id}: {missing_reason!r}"
        )

    bookmaker = str(row.get("bookmaker", "")).strip()
    if odds_available == "1":
        if missing_reason:
            raise ValueError(
                "Weekly schedule row with odds_available=1 has a missing reason "
                f"for game_id={game_id}"
            )
        if not bookmaker:
            raise ValueError(
                "Weekly schedule row with odds_available=1 has blank bookmaker "
                f"for game_id={game_id}"
            )
        if not any(
            str(row.get(field, "")).strip()
            for field in (
                "home_moneyline_american",
                "away_moneyline_american",
                "home_spread",
                "away_spread",
                "total",
            )
        ):
            raise ValueError(
                "Weekly schedule row marked odds_available=1 has no market values "
                f"for game_id={game_id}"
            )
        expected_home = str(target["home_team"]).strip()
        expected_away = str(target["away_team"]).strip()
        if str(row.get("odds_home_team", "")).strip() != expected_home:
            raise ValueError(
                f"Weekly schedule odds_home_team mismatch for game_id={game_id}"
            )
        if str(row.get("odds_away_team", "")).strip() != expected_away:
            raise ValueError(
                f"Weekly schedule odds_away_team mismatch for game_id={game_id}"
            )
        if str(row.get("commence_time", "")).strip() != expected_kickoff:
            raise ValueError(
                f"Weekly schedule commence_time mismatch for game_id={game_id}"
            )
    elif not missing_reason:
        raise ValueError(
            "Weekly schedule row with odds_available=0 has blank "
            f"odds_missing_reason for game_id={game_id}"
        )


def validate_output_rows(
    *,
    output_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
    now_utc: datetime,
) -> None:
    if len(output_rows) != len(target_rows):
        raise ValueError(
            "Weekly schedule output row count does not match target schedule. "
            f"output={len(output_rows)}, target={len(target_rows)}"
        )

    target_by_id = {
        str(row["game_id"]).strip(): row for row in target_rows
    }
    output_ids: set[str] = set()
    for index, row in enumerate(output_rows):
        missing_columns = [
            column for column in OUTPUT_COLUMNS if column not in row
        ]
        if missing_columns:
            raise ValueError(
                f"Weekly schedule output row {index} missing columns: {missing_columns}"
            )
        game_id = str(row.get("game_id", "")).strip()
        if not game_id:
            raise ValueError(
                f"Weekly schedule output row {index} has blank game_id"
            )
        if game_id in output_ids:
            raise ValueError(
                f"Weekly schedule output contains duplicate game_id={game_id}"
            )
        output_ids.add(game_id)
        if game_id not in target_by_id:
            raise ValueError(
                f"Weekly schedule output contains foreign game_id={game_id}"
            )
        target = target_by_id[game_id]
        expected_kickoff = _stage1_validate_weekly_target_fields(
            row,
            game_id=game_id,
            target=target,
            season=season,
            season_type=season_type,
            week=week,
            now_utc=now_utc,
        )
        _stage1_validate_weekly_odds_fields(
            row,
            game_id=game_id,
            target=target,
            expected_kickoff=expected_kickoff,
        )

    if output_ids != set(target_by_id):
        raise ValueError(
            "Weekly schedule output game IDs do not exactly match configured "
            "target schedule"
        )



def write_csv_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    write_atomic_csv_rows(
        path,
        rows,
        OUTPUT_COLUMNS,
    )

def read_all_locked_weekly(
    path: Path,
    target_rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "All target-week games are locked, but the "
            "previous weekly schedule does not exist: "
            f"{path}"
        )

    rows = read_csv(
        path,
        OUTPUT_COLUMNS,
        "existing locked weekly schedule",
    )

    target_by_id = {
        str(
            row["game_id"]
        ).strip(): row
        for row in target_rows
    }

    existing: dict[
        str,
        dict[str, str],
    ] = {}

    for index, row in enumerate(
        rows
    ):
        game_id = str(
            row.get(
                "game_id",
                "",
            )
        ).strip()

        if not game_id:
            raise ValueError(
                "Existing locked weekly schedule "
                "contains blank game_id at "
                f"row {index}"
            )

        if game_id in existing:
            raise ValueError(
                "Existing locked weekly schedule "
                "contains duplicate game_id="
                f"{game_id}"
            )

        target = target_by_id.get(
            game_id
        )

        if target is None:
            raise ValueError(
                "Existing locked weekly schedule "
                "contains out-of-scope game_id="
                f"{game_id}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ).strip() != str(season):
            raise ValueError(
                "Existing locked weekly schedule "
                "season mismatch for "
                f"game_id={game_id}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ).strip() != str(season_type):
            raise ValueError(
                "Existing locked weekly schedule "
                "season_type mismatch for "
                f"game_id={game_id}"
            )

        if str(
            row.get(
                "week",
                "",
            )
        ).strip() != str(week):
            raise ValueError(
                "Existing locked weekly schedule "
                "week mismatch for "
                f"game_id={game_id}"
            )

        for field in (
            "away_team",
            "home_team",
        ):
            if str(
                row.get(
                    field,
                    "",
                )
            ).strip() != str(
                target.get(
                    field,
                    "",
                )
            ).strip():
                raise ValueError(
                    "Existing locked weekly schedule "
                    f"{field} mismatch for "
                    f"game_id={game_id}"
                )

        odds_available = str(
            row.get(
                "odds_available",
                "",
            )
        ).strip()

        if odds_available not in {
            "0",
            "1",
        }:
            raise ValueError(
                "Existing locked weekly schedule "
                "contains invalid odds_available "
                f"for game_id={game_id}: "
                f"{odds_available!r}"
            )

        existing[
            game_id
        ] = row

    if set(existing) != set(target_by_id):
        missing = sorted(
            set(target_by_id)
            - set(existing)
        )

        extra = sorted(
            set(existing)
            - set(target_by_id)
        )

        raise ValueError(
            "All target-week games are locked, but "
            "the previous weekly schedule does not "
            "exactly cover the current target set. "
            f"missing={missing[:20]}, "
            f"extra={extra[:20]}"
        )

    return existing


def validate_all_locked_output(
    *,
    output_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
    now_utc: datetime,
) -> None:
    validation_rows: list[
        dict[str, str]
    ] = []

    for row in output_rows:
        candidate = dict(
            row
        )

        if candidate.get(
            "odds_available"
        ) == "1":
            parse_aware_iso(
                candidate.get(
                    "commence_time",
                    "",
                ),
                (
                    "preserved locked odds "
                    "commence_time for "
                    f"game_id={candidate.get('game_id', '')}"
                ),
            )

            candidate[
                "commence_time"
            ] = str(
                candidate.get(
                    "kickoff_utc",
                    "",
                )
            ).strip()

        validation_rows.append(
            candidate
        )

    validate_output_rows(
        output_rows=validation_rows,
        target_rows=target_rows,
        season=season,
        season_type=season_type,
        week=week,
        now_utc=now_utc,
    )


def all_locked_metrics(
    rows: list[dict[str, str]],
) -> dict[str, int]:
    return {
        "rows_with_odds": sum(
            row.get(
                "odds_available"
            ) == "1"
            for row in rows
        ),
        "rows_no_odds_returned": sum(
            row.get(
                "odds_missing_reason"
            ) == "no_odds_returned"
            for row in rows
        ),
        "rows_no_supported_markets": sum(
            row.get(
                "odds_missing_reason"
            ) == "no_supported_markets"
            for row in rows
        ),
        "rows_locked_before_first_capture": sum(
            row.get(
                "odds_missing_reason"
            ) == "locked_before_first_capture"
            for row in rows
        ),
    }


def maybe_publish_all_locked_week(
    *,
    report: PipelineReporter,
    target_rows: list[dict[str, str]],
    schedule_rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
    now_utc: datetime,
    output_path: Path,
) -> int | None:
    all_locked = bool(
        target_rows
    ) and all(
        game_is_locked(
            row,
            now_utc,
        )
        for row in target_rows
    )

    if not all_locked:
        return None

    report.warning(
        "All configured target-week games are locked; "
        "preserving previously published weekly market "
        "data and refreshing schedule metadata only."
    )

    report.add_input(
        output_path
    )

    report.add_output(
        output_path
    )

    existing_weekly = (
        read_all_locked_weekly(
            output_path,
            target_rows,
            season=season,
            season_type=season_type,
            week=week,
        )
    )

    (
        output_rows,
        locked_preserved,
    ) = build_output_rows(
        target_rows=target_rows,
        request_by_id={},
        odds_summary={},
        existing_weekly=existing_weekly,
        now_utc=now_utc,
    )

    if locked_preserved != len(
        target_rows
    ):
        raise RuntimeError(
            "All-locked weekly preservation did not "
            "preserve every target game. "
            f"preserved={locked_preserved}, "
            f"target={len(target_rows)}"
        )

    validate_all_locked_output(
        output_rows=output_rows,
        target_rows=target_rows,
        season=season,
        season_type=season_type,
        week=week,
        now_utc=now_utc,
    )

    metrics = all_locked_metrics(
        output_rows
    )

    report.set_rows(
        rows_in=len(
            target_rows
        ),
    )

    report.update_details(
        {
            "snapshot_id": "",
            "snapshot_fetched_at": "",
            "schedule_rows_loaded": len(
                schedule_rows
            ),
            "target_schedule_rows": len(
                target_rows
            ),
            "raw_odds_events_loaded": 0,
            "raw_odds_objects_loaded": 0,
            "odds_csv_rows_loaded": 0,
            "existing_weekly_rows": len(
                existing_weekly
            ),
            "request_result_counts": {},
            "rows_with_odds": metrics[
                "rows_with_odds"
            ],
            "rows_no_odds_returned": metrics[
                "rows_no_odds_returned"
            ],
            "rows_no_supported_markets": metrics[
                "rows_no_supported_markets"
            ],
            "rows_locked_before_first_capture": metrics[
                "rows_locked_before_first_capture"
            ],
            "locked_games": len(
                target_rows
            ),
            "locked_rows_preserved": (
                locked_preserved
            ),
            "odds_snapshot_skipped": True,
            "odds_snapshot_skip_reason": (
                "all_target_games_locked"
            ),
            "output_rows": len(
                output_rows
            ),
            "output_modified": False,
        }
    )

    write_csv_atomic(
        output_path,
        output_rows,
    )

    report.set_rows(
        rows_out=len(
            output_rows
        ),
    )

    report.update_details(
        {
            "output_modified": True,
            "output_path": str(
                output_path
            ),
        }
    )

    print(
        "build_weekly_schedule.py completed "
        "with all-locked preservation"
    )

    print(
        f"season={season} "
        f"season_type={season_type} "
        f"week={week}"
    )

    print(
        f"rows_written={len(output_rows)}"
    )

    print(
        "locked_rows_preserved="
        f"{locked_preserved}"
    )

    print(
        "odds_snapshot_skipped=true"
    )

    print(
        f"output={output_path}"
    )

    return 0


def main() -> int:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.set_detail(
            "output_modified",
            False,
        )

        (
            season,
            season_type,
            week,
        ) = load_current_week_config(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.season = season
        report.week = week

        report.set_detail(
            "season_type",
            season_type,
        )

        (
            schedule_path,
            schedule_rows,
            target_rows,
        ) = load_schedule(
            season,
            season_type,
            week,
        )

        report.add_input(
            schedule_path
        )

        now_utc = datetime.now(
            timezone.utc
        )

        output_path = (
            WEEKLY_DIR
            / (
                f"week_{week}_"
                "CFB_weekly_schedule.csv"
            )
        )

        locked_result = (
            maybe_publish_all_locked_week(
                report=report,
                target_rows=target_rows,
                schedule_rows=schedule_rows,
                season=season,
                season_type=season_type,
                week=week,
                now_utc=now_utc,
                output_path=output_path,
            )
        )

        if locked_result is not None:
            return locked_result

        (
            odds_csv_path,
            raw_odds_path,
        ) = latest_odds_pair()

        report.add_input(
            odds_csv_path
        )

        report.add_input(
            raw_odds_path
        )

        odds_rows = read_csv(
            odds_csv_path,
            ODDS_REQUIRED_COLUMNS,
            "current normalized CFB odds CSV",
        )

        raw_payload = (
            load_raw_odds_payload(
                raw_odds_path
            )
        )

        (
            snapshot_id,
            snapshot_fetched_at,
            request_by_id,
        ) = validate_snapshot_provenance(
            odds_rows=odds_rows,
            raw_payload=raw_payload,
            target_rows=target_rows,
            season=season,
            season_type=season_type,
            week=week,
        )

        odds_summary = (
            build_odds_summary(
                odds_rows,
                target_rows,
            )
        )

        output_path = (
            WEEKLY_DIR
            / (
                f"week_{week}_"
                "CFB_weekly_schedule.csv"
            )
        )

        if output_path.exists():
            report.add_input(
                output_path
            )

        report.add_output(
            output_path
        )

        existing_weekly = (
            read_existing_weekly(
                output_path,
                target_rows,
                season,
                season_type,
                week,
            )
        )

        now_utc = datetime.now(
            timezone.utc
        )

        (
            output_rows,
            locked_preserved,
        ) = build_output_rows(
            target_rows=target_rows,
            request_by_id=request_by_id,
            odds_summary=odds_summary,
            existing_weekly=existing_weekly,
            now_utc=now_utc,
        )

        validate_output_rows(
            output_rows=output_rows,
            target_rows=target_rows,
            season=season,
            season_type=season_type,
            week=week,
            now_utc=now_utc,
        )

        matched_with_odds = sum(
            1
            for row in output_rows
            if row.get(
                "odds_available"
            ) == "1"
        )

        no_odds_returned = sum(
            1
            for row in output_rows
            if row.get(
                "odds_missing_reason"
            ) == "no_odds_returned"
        )

        no_supported_markets = sum(
            1
            for row in output_rows
            if row.get(
                "odds_missing_reason"
            ) == "no_supported_markets"
        )

        locked_before_capture = sum(
            1
            for row in output_rows
            if row.get(
                "odds_missing_reason"
            ) == (
                "locked_before_first_capture"
            )
        )

        locked_games = sum(
            1
            for row in output_rows
            if row.get(
                "game_locked"
            ) == "1"
        )

        request_result_counts: dict[
            str,
            int,
        ] = {}

        for request in request_by_id.values():
            result = str(
                request.get(
                    "result",
                    "",
                )
            ).strip()

            request_result_counts[
                result
            ] = (
                request_result_counts.get(
                    result,
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
                "schedule_rows_loaded": len(
                    schedule_rows
                ),
                "target_schedule_rows": len(
                    target_rows
                ),
                "raw_odds_events_loaded": len(
                    raw_payload[
                        "events"
                    ]
                ),
                "raw_odds_objects_loaded": len(
                    raw_payload[
                        "odds"
                    ]
                ),
                "odds_csv_rows_loaded": len(
                    odds_rows
                ),
                "existing_weekly_rows": len(
                    existing_weekly
                ),
                "request_result_counts": (
                    request_result_counts
                ),
                "rows_with_odds": (
                    matched_with_odds
                ),
                "rows_no_odds_returned": (
                    no_odds_returned
                ),
                "rows_no_supported_markets": (
                    no_supported_markets
                ),
                "rows_locked_before_first_capture": (
                    locked_before_capture
                ),
                "locked_games": (
                    locked_games
                ),
                "locked_rows_preserved": (
                    locked_preserved
                ),
                "output_rows": len(
                    output_rows
                ),
                "output_modified": False,
            }
        )

        write_csv_atomic(
            output_path,
            output_rows,
        )

        report.set_rows(
            rows_out=len(
                output_rows
            ),
        )

        report.update_details(
            {
                "output_modified": True,
                "output_path": str(
                    output_path
                ),
            }
        )

        print(
            "build_weekly_schedule.py "
            "completed"
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
            f"rows_written="
            f"{len(output_rows)}"
        )

        print(
            f"rows_with_odds="
            f"{matched_with_odds}"
        )

        print(
            "rows_no_odds_returned="
            f"{no_odds_returned}"
        )

        print(
            "rows_no_supported_markets="
            f"{no_supported_markets}"
        )

        print(
            f"locked_games={locked_games}"
        )

        print(
            "locked_rows_preserved="
            f"{locked_preserved}"
        )

        print(
            f"output={output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )