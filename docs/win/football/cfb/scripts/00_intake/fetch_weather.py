#!/usr/bin/env python3
"""
Fetch kickoff weather for the configured CFB target week.

build_travel.py is authoritative for the resolved game venue. Weather provider
failure is degradable: a previously valid forecast is retained when possible.
Structural schedule/travel/output corruption is fatal.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from pipeline_shared import write_csv_rows_durable
from type_support import ScalarValue

CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
TRAVEL_DIR = CFB_ROOT / "data" / "travel"
STADIUM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "stadium_map.csv"
OUTPUT_DIR = CFB_ROOT / "data" / "weather"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-fetch-weather-v2-2026-09-16"

METNO_URL = (
    "https://api.met.no/weatherapi/"
    "locationforecast/2.0/complete"
)
METNO_HOST = "api.met.no"

METNO_USER_AGENT = os.environ.get(
    "METNO_USER_AGENT",
    (
        "cfb_for_mat/1.0 "
        "(+https://github.com/"
        "Clownworldenjoyer76/cfb_for_mat)"
    ),
).strip()

REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECONDS = 1.25

MAX_FORECAST_OFFSET_SECONDS = 3 * 3600
MIN_FORECAST_TOLERANCE_SECONDS = 30 * 60
MAX_PROVIDER_CADENCE_SECONDS = 6 * 3600

OUTPUT_HEADERS = [
    "game_id",
    "stadium",
    "venue_resolution_status",
    "latitude",
    "longitude",
    "game_time",
    "game_timezone",
    "kickoff_utc",
    "weather_timestep_utc",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
    "roof",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_fetched_at",
]

WEATHER_COLUMNS = [
    "weather_timestep_utc",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
]

SCHEDULE_REQUIRED_COLUMNS = {
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "game_timezone",
    "kickoff_utc",
    "commence_time",
    "game_locked",
    "stadium",
    "roof",
}

TRAVEL_REQUIRED_COLUMNS = {
    "game_id",
    "stadium",
    "venue_resolution_status",
    "venue_lat",
    "venue_lon",
    "venue_timezone",
}

STADIUM_REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "timezone",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
}

STADIUM_ROOF_COLUMNS = [
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
]

RESOLVED_VENUE_STATUSES = {
    "resolved_schedule_stadium",
    "resolved_schedule_stadium_timezone",
    "resolved_schedule_stadium_home_match",
}

PROVIDER_FAILURE_STATUSES = {
    "http_error",
    "network_error",
    "timeout",
    "invalid_json",
    "invalid_response",
}


class WeatherValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderResult:
    status: str
    payload: dict[str, Any] | None = None
    http_status: int | None = None
    error: str = ""


@dataclass(frozen=True)
class TimestepResult:
    status: str
    entry: dict[str, Any] | None = None
    timestep: datetime | None = None
    offset_seconds: float | None = None
    cadence_seconds: float | None = None


def clean(value: ScalarValue) -> str:
    return "" if value is None else str(value).strip()


def positive_int(
    value: ScalarValue,
    *,
    label: str,
) -> int:
    value_text = clean(value)

    if not value_text.isdigit():
        raise WeatherValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(value_text)

    if parsed <= 0:
        raise WeatherValidationError(
            f"{label} must be positive: {parsed}"
        )

    return parsed


def finite_float(
    value: ScalarValue,
    *,
    label: str,
) -> float:
    value_text = clean(value)

    if not value_text:
        raise WeatherValidationError(
            f"{label} is blank"
        )

    try:
        number = float(value_text)
    except (TypeError, ValueError) as exc:
        raise WeatherValidationError(
            f"{label} is not numeric: {value_text!r}"
        ) from exc

    if not math.isfinite(number):
        raise WeatherValidationError(
            f"{label} is not finite: {value_text!r}"
        )

    return number


def validate_coordinate_pair(
    latitude_value: ScalarValue,
    longitude_value: ScalarValue,
    *,
    label: str,
) -> tuple[float, float]:
    latitude = finite_float(
        latitude_value,
        label=f"{label} latitude",
    )

    longitude = finite_float(
        longitude_value,
        label=f"{label} longitude",
    )

    if not -90 <= latitude <= 90:
        raise WeatherValidationError(
            f"{label} latitude outside [-90, 90]: {latitude}"
        )

    if not -180 <= longitude <= 180:
        raise WeatherValidationError(
            f"{label} longitude outside [-180, 180]: {longitude}"
        )

    return latitude, longitude


def validate_timezone(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    timezone_name = clean(value)

    if not timezone_name:
        raise WeatherValidationError(
            f"{label} is blank"
        )

    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise WeatherValidationError(
            f"{label} is invalid: {timezone_name!r}"
        ) from exc

    return timezone_name


def parse_iso_utc(
    value: ScalarValue,
) -> datetime | None:
    value_text = clean(value)

    if not value_text:
        return None

    try:
        if value_text.endswith("Z"):
            value_text = (
                value_text[:-1]
                + "+00:00"
            )

        parsed = datetime.fromisoformat(
            value_text
        )

        if parsed.tzinfo is None:
            return None

        return parsed.astimezone(
            timezone.utc
        )

    except (TypeError, ValueError):
        return None


def require_iso_utc(
    value: ScalarValue,
    *,
    label: str,
) -> datetime:
    parsed = parse_iso_utc(
        value
    )

    if parsed is None:
        raise WeatherValidationError(
            f"{label} is not a valid timezone-aware timestamp: {value!r}"
        )

    return parsed


def load_csv(
    path: Path,
    *,
    label: str,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(
                handle,
                strict=True,
            )

            fieldnames = list(
                reader.fieldnames or []
            )

            rows: list[
                dict[str, str]
            ] = []

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                if (
                    None in row
                    or any(
                        value is None
                        for value in row.values()
                    )
                ):
                    raise WeatherValidationError(
                        f"{label} malformed row at CSV line {line_number}"
                    )

                rows.append({
                    key: clean(value)
                    for key, value in row.items()
                })

    except csv.Error as exc:
        raise WeatherValidationError(
            f"Malformed CSV in {label}: {path}: {exc}"
        ) from exc

    if not fieldnames:
        raise WeatherValidationError(
            f"{label} has no CSV header: {path}"
        )

    return fieldnames, rows


def require_columns(
    fieldnames: list[str],
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(
        required - set(fieldnames)
    )

    if missing:
        raise WeatherValidationError(
            f"{label} missing required columns: {missing}"
        )


def load_config() -> tuple[int, int, int]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing current-week config: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise WeatherValidationError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[
        str,
        int,
    ] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise WeatherValidationError(
                f"current_week.yaml missing required key: {key}"
            )

        values[key] = positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise WeatherValidationError(
            f"Invalid configured season: {values['season']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def target_paths(
    season: int,
    week: int,
) -> tuple[
    Path,
    Path,
    Path,
]:
    schedule_path = (
        SCHEDULE_DIR
        / f"week_{week}_CFB_weekly_schedule.csv"
    )

    travel_path = (
        TRAVEL_DIR
        / f"{season}_week_{week}_travel.csv"
    )

    output_path = (
        OUTPUT_DIR
        / f"week_{week}_CFB_weekly_weather.csv"
    )

    return (
        schedule_path,
        travel_path,
        output_path,
    )



def schedule_game_locked(
    game: dict[str, str],
) -> bool:
    game_id = clean(
        game.get("game_id")
    )

    value = clean(
        game.get("game_locked")
    )

    if value not in {
        "0",
        "1",
    }:
        raise WeatherValidationError(
            "game_locked must be 0/1 for "
            f"game_id={game_id}: {value!r}"
        )

    return value == "1"


def resolve_kickoff_utc(
    game: dict[str, str],
) -> datetime:
    game_id = clean(
        game.get("game_id")
    )

    authoritative = require_iso_utc(
        game.get("kickoff_utc"),
        label=f"kickoff_utc for game_id={game_id}",
    )

    commence_text = clean(
        game.get("commence_time")
    )

    if commence_text:
        commence = require_iso_utc(
            commence_text,
            label=f"commence_time for game_id={game_id}",
        )

        if (
            abs(
                (
                    authoritative
                    - commence
                ).total_seconds()
            ) > 1
            and not schedule_game_locked(
                game
            )
        ):
            raise WeatherValidationError(
                "kickoff_utc and commence_time disagree "
                "for unlocked game "
                f"game_id={game_id}: "
                f"kickoff_utc={authoritative.isoformat()} "
                f"commence_time={commence.isoformat()}"
            )

    game_date = clean(
        game.get("game_date")
    )

    game_time = clean(
        game.get("game_time")
    )

    game_timezone = validate_timezone(
        game.get("game_timezone"),
        label=f"game_timezone for game_id={game_id}",
    )

    if not game_date or not game_time:
        raise WeatherValidationError(
            f"Missing local kickoff date/time for game_id={game_id}"
        )

    try:
        local_naive = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        )
    except ValueError as exc:
        raise WeatherValidationError(
            f"Invalid local kickoff for game_id={game_id}: "
            f"date={game_date!r} time={game_time!r}"
        ) from exc

    local_utc = local_naive.replace(
        tzinfo=ZoneInfo(
            game_timezone
        )
    ).astimezone(
        timezone.utc
    )

    if abs(
        (
            authoritative
            - local_utc
        ).total_seconds()
    ) > 60:
        raise WeatherValidationError(
            "Authoritative kickoff does not agree "
            "with local date/time/timezone for "
            f"game_id={game_id}: "
            f"authoritative={authoritative.isoformat()} "
            f"local_derived={local_utc.isoformat()}"
        )

    return authoritative


def load_schedule(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    fieldnames, rows = load_csv(
        path,
        label="target weekly schedule",
    )

    require_columns(
        fieldnames,
        SCHEDULE_REQUIRED_COLUMNS,
        label="target weekly schedule",
    )

    if not rows:
        raise WeatherValidationError(
            "Target weekly schedule contains no data rows"
        )

    lookup: dict[
        str,
        dict[str, str],
    ] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        target = (
            positive_int(
                row.get("season"),
                label=f"schedule season line {line_number}",
            ),
            positive_int(
                row.get("season_type"),
                label=f"schedule season_type line {line_number}",
            ),
            positive_int(
                row.get("week"),
                label=f"schedule week line {line_number}",
            ),
        )

        if target != (
            season,
            season_type,
            week,
        ):
            raise WeatherValidationError(
                "Weekly schedule target mismatch at "
                f"CSV line {line_number}: "
                f"expected={season}/{season_type}/{week}, "
                f"actual={target[0]}/{target[1]}/{target[2]}"
            )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"schedule game_id line {line_number}",
            )
        )

        if game_id in lookup:
            raise WeatherValidationError(
                f"Duplicate schedule game_id={game_id}"
            )

        stadium = clean(
            row.get("stadium")
        )

        if not stadium:
            raise WeatherValidationError(
                f"Blank schedule stadium for game_id={game_id}"
            )

        normalized = dict(
            row
        )

        normalized[
            "game_id"
        ] = game_id

        resolve_kickoff_utc(
            normalized
        )

        lookup[
            game_id
        ] = normalized

    return (
        list(
            lookup.values()
        ),
        lookup,
    )


def load_travel(
    path: Path,
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    fieldnames, rows = load_csv(
        path,
        label="target weekly travel",
    )

    require_columns(
        fieldnames,
        TRAVEL_REQUIRED_COLUMNS,
        label="target weekly travel",
    )

    if len(rows) != len(
        schedule_lookup
    ):
        raise WeatherValidationError(
            "Travel row-count mismatch: "
            f"expected={len(schedule_lookup)}, "
            f"actual={len(rows)}"
        )

    lookup: dict[
        str,
        dict[str, str],
    ] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"travel game_id line {line_number}",
            )
        )

        if game_id in lookup:
            raise WeatherValidationError(
                f"Duplicate travel game_id={game_id}"
            )

        game = schedule_lookup.get(
            game_id
        )

        if game is None:
            raise WeatherValidationError(
                f"Travel contains foreign game_id={game_id}"
            )

        stadium = clean(
            row.get("stadium")
        )

        if stadium != clean(
            game.get("stadium")
        ):
            raise WeatherValidationError(
                "Travel stadium mismatch for "
                f"game_id={game_id}: "
                f"schedule={clean(game.get('stadium'))!r} "
                f"travel={stadium!r}"
            )

        status = clean(
            row.get(
                "venue_resolution_status"
            )
        )

        if status not in RESOLVED_VENUE_STATUSES:
            raise WeatherValidationError(
                "Travel contains unresolved venue "
                f"for game_id={game_id}: status={status!r}"
            )

        validate_coordinate_pair(
            row.get("venue_lat"),
            row.get("venue_lon"),
            label=f"travel venue for game_id={game_id}",
        )

        travel_timezone = (
            validate_timezone(
                row.get(
                    "venue_timezone"
                ),
                label=(
                    "travel venue_timezone "
                    f"for game_id={game_id}"
                ),
            )
        )

        schedule_timezone = (
            validate_timezone(
                game.get(
                    "game_timezone"
                ),
                label=(
                    "schedule game_timezone "
                    f"for game_id={game_id}"
                ),
            )
        )

        if travel_timezone != schedule_timezone:
            raise WeatherValidationError(
                "Travel/schedule timezone mismatch "
                f"for game_id={game_id}: "
                f"travel={travel_timezone!r} "
                f"schedule={schedule_timezone!r}"
            )

        normalized = dict(
            row
        )

        normalized[
            "game_id"
        ] = game_id

        lookup[
            game_id
        ] = normalized

    if set(lookup) != set(
        schedule_lookup
    ):
        raise WeatherValidationError(
            "Travel game coverage does not match target schedule"
        )

    return (
        list(
            lookup.values()
        ),
        lookup,
    )


def coordinate_key(
    latitude_value: ScalarValue,
    longitude_value: ScalarValue,
) -> tuple[float, float] | None:
    latitude_text = clean(
        latitude_value
    )

    longitude_text = clean(
        longitude_value
    )

    if (
        not latitude_text
        and not longitude_text
    ):
        return None

    if (
        not latitude_text
        or not longitude_text
    ):
        raise WeatherValidationError(
            "Partial stadium-map coordinate pair"
        )

    latitude, longitude = (
        validate_coordinate_pair(
            latitude_text,
            longitude_text,
            label="stadium map",
        )
    )

    return (
        round(
            latitude,
            6,
        ),
        round(
            longitude,
            6,
        ),
    )


def roof_metadata_score(
    row: dict[str, str],
) -> int:
    return sum(
        bool(
            clean(
                row.get(column)
            )
        )
        for column
        in STADIUM_ROOF_COLUMNS
    )


def load_stadium_coordinate_lookup() -> tuple[
    list[dict[str, str]],
    dict[
        tuple[float, float],
        dict[str, str],
    ],
]:
    fieldnames, rows = load_csv(
        STADIUM_MAP_PATH,
        label="stadium map",
    )

    require_columns(
        fieldnames,
        STADIUM_REQUIRED_COLUMNS,
        label="stadium map",
    )

    if not rows:
        raise WeatherValidationError(
            "stadium_map.csv contains no data rows"
        )

    lookup: dict[
        tuple[float, float],
        dict[str, str],
    ] = {}

    for row in rows:
        key = coordinate_key(
            row.get("latitude"),
            row.get("longitude"),
        )

        if key is None:
            continue

        timezone_text = clean(
            row.get("timezone")
        )

        if timezone_text:
            validate_timezone(
                timezone_text,
                label="stadium map timezone",
            )

        for column in (
            "dome_flag",
            "retractable_roof_flag",
            "open_air_flag",
        ):
            value = clean(
                row.get(column)
            )

            if (
                value
                and value not in {
                    "0",
                    "1",
                }
            ):
                raise WeatherValidationError(
                    f"stadium map {column} must be 0/1/blank: {value!r}"
                )

        existing = lookup.get(
            key
        )

        if (
            existing is None
            or roof_metadata_score(row)
            > roof_metadata_score(existing)
        ):
            lookup[
                key
            ] = row

    return (
        rows,
        lookup,
    )


def venue_from_travel(
    game: dict[str, str],
    travel_lookup: dict[
        str,
        dict[str, str],
    ],
    stadium_coordinate_lookup: dict[
        tuple[float, float],
        dict[str, str],
    ],
) -> tuple[
    dict[str, str],
    bool,
]:
    game_id = clean(
        game.get("game_id")
    )

    travel_row = travel_lookup.get(
        game_id
    )

    if travel_row is None:
        raise WeatherValidationError(
            f"Missing travel row for game_id={game_id}"
        )

    venue_status = clean(
        travel_row.get(
            "venue_resolution_status"
        )
    )

    if venue_status not in RESOLVED_VENUE_STATUSES:
        raise WeatherValidationError(
            "Travel venue is not resolved for "
            f"game_id={game_id}: {venue_status!r}"
        )

    latitude = clean(
        travel_row.get("venue_lat")
    )

    longitude = clean(
        travel_row.get("venue_lon")
    )

    venue_timezone = (
        validate_timezone(
            travel_row.get(
                "venue_timezone"
            ),
            label=(
                "travel venue timezone "
                f"for game_id={game_id}"
            ),
        )
    )

    validate_coordinate_pair(
        latitude,
        longitude,
        label=f"travel venue for game_id={game_id}",
    )

    venue_row = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": venue_timezone,
        "roof_type": "",
        "dome_flag": "",
        "retractable_roof_flag": "",
        "open_air_flag": "",
    }

    key = coordinate_key(
        latitude,
        longitude,
    )

    stadium_row = (
        stadium_coordinate_lookup.get(
            key
        )
        if key is not None
        else None
    )

    roof_metadata_missing = (
        stadium_row is None
    )

    if stadium_row is not None:
        for column in STADIUM_ROOF_COLUMNS:
            venue_row[
                column
            ] = clean(
                stadium_row.get(
                    column
                )
            )

    return (
        venue_row,
        roof_metadata_missing,
    )


def truncate_coordinate(
    value: float,
) -> str:
    truncated = (
        math.trunc(
            value * 10000.0
        )
        / 10000.0
    )

    if truncated == 0:
        truncated = 0.0

    return f"{truncated:.4f}"


def canonical_request_coordinates(
    latitude_value: ScalarValue,
    longitude_value: ScalarValue,
) -> tuple[str, str]:
    latitude, longitude = (
        validate_coordinate_pair(
            latitude_value,
            longitude_value,
            label="weather request",
        )
    )

    return (
        truncate_coordinate(
            latitude
        ),
        truncate_coordinate(
            longitude
        ),
    )


def fetch_weather_json(
    latitude: str,
    longitude: str,
) -> ProviderResult:
    query = urllib.parse.urlencode({
        "lat": latitude,
        "lon": longitude,
    })

    url = (
        f"{METNO_URL}?{query}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                METNO_USER_AGENT
            ),
            "Accept": (
                "application/json"
            ),
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={METNO_HOST},
            timeout=REQUEST_TIMEOUT,
        ) as response:
            http_status = int(
                getattr(
                    response,
                    "status",
                    200,
                )
            )

            raw = response.read().decode(
                "utf-8"
            )

    except urllib.error.HTTPError as exc:
        return ProviderResult(
            status="http_error",
            http_status=exc.code,
            error=str(exc),
        )

    except TimeoutError as exc:
        return ProviderResult(
            status="timeout",
            error=str(exc),
        )

    except urllib.error.URLError as exc:
        return ProviderResult(
            status="network_error",
            error=str(exc),
        )

    except OSError as exc:
        return ProviderResult(
            status="network_error",
            error=str(exc),
        )

    try:
        payload = json.loads(
            raw
        )
    except json.JSONDecodeError as exc:
        return ProviderResult(
            status="invalid_json",
            http_status=http_status,
            error=str(exc),
        )

    if not isinstance(
        payload,
        dict,
    ):
        return ProviderResult(
            status="invalid_response",
            http_status=http_status,
            error="Provider response root is not an object",
        )

    timeseries = (
        payload
        .get("properties", {})
        .get("timeseries")
    )

    if not isinstance(
        timeseries,
        list,
    ):
        return ProviderResult(
            status="invalid_response",
            http_status=http_status,
            error="Provider response has no timeseries list",
        )

    return ProviderResult(
        status="request_success",
        payload=payload,
        http_status=http_status,
    )


def parsed_timeseries(
    payload: dict[str, Any],
) -> list[
    tuple[
        datetime,
        dict[str, Any],
    ]
]:
    raw_series = (
        payload
        .get("properties", {})
        .get("timeseries", [])
    )

    parsed: list[
        tuple[
            datetime,
            dict[str, Any],
        ]
    ] = []

    if not isinstance(
        raw_series,
        list,
    ):
        return parsed

    for entry in raw_series:
        if not isinstance(
            entry,
            dict,
        ):
            continue

        entry_time = parse_iso_utc(
            entry.get("time")
        )

        if entry_time is None:
            continue

        parsed.append((
            entry_time,
            entry,
        ))

    parsed.sort(
        key=lambda item: item[0]
    )

    return parsed


def local_cadence_seconds(
    series: list[
        tuple[
            datetime,
            dict[str, Any],
        ]
    ],
    index: int,
) -> float:
    gaps: list[float] = []

    if index > 0:
        gaps.append(
            (
                series[index][0]
                - series[index - 1][0]
            ).total_seconds()
        )

    if index + 1 < len(
        series
    ):
        gaps.append(
            (
                series[index + 1][0]
                - series[index][0]
            ).total_seconds()
        )

    positive = [
        gap
        for gap in gaps
        if gap > 0
    ]

    if not positive:
        return 3600.0

    return min(
        max(positive),
        float(
            MAX_PROVIDER_CADENCE_SECONDS
        ),
    )


def select_kickoff_timestep(
    payload: dict[str, Any],
    kickoff_utc: datetime,
) -> TimestepResult:
    series = parsed_timeseries(
        payload
    )

    if not series:
        return TimestepResult(
            status="invalid_response"
        )

    closest_index = min(
        range(
            len(series)
        ),
        key=lambda index: abs(
            (
                series[index][0]
                - kickoff_utc
            ).total_seconds()
        ),
    )

    closest_time, closest_entry = (
        series[
            closest_index
        ]
    )

    offset_seconds = abs(
        (
            closest_time
            - kickoff_utc
        ).total_seconds()
    )

    cadence_seconds = (
        local_cadence_seconds(
            series,
            closest_index,
        )
    )

    tolerance_seconds = max(
        float(
            MIN_FORECAST_TOLERANCE_SECONDS
        ),
        min(
            float(
                MAX_FORECAST_OFFSET_SECONDS
            ),
            cadence_seconds / 2.0,
        ),
    )

    first_time = series[
        0
    ][0]

    last_time = series[
        -1
    ][0]

    if (
        (
            kickoff_utc < first_time
            or kickoff_utc > last_time
        )
        and (
            offset_seconds
            > tolerance_seconds
        )
    ):
        return TimestepResult(
            status=(
                "outside_forecast_range"
            ),
            offset_seconds=(
                offset_seconds
            ),
            cadence_seconds=(
                cadence_seconds
            ),
        )

    if (
        offset_seconds
        > tolerance_seconds
    ):
        return TimestepResult(
            status=(
                "no_acceptable_timestep"
            ),
            offset_seconds=(
                offset_seconds
            ),
            cadence_seconds=(
                cadence_seconds
            ),
        )

    return TimestepResult(
        status="forecast_available",
        entry=closest_entry,
        timestep=closest_time,
        offset_seconds=offset_seconds,
        cadence_seconds=cadence_seconds,
    )


def optional_numeric(
    value: ScalarValue,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> ScalarValue:
    if value is None:
        return ""

    value_text = clean(
        value
    )

    if not value_text:
        return ""

    number = finite_float(
        value_text,
        label=label,
    )

    if (
        minimum is not None
        and number < minimum
    ):
        raise WeatherValidationError(
            f"{label} below minimum {minimum}: {number}"
        )

    if (
        maximum is not None
        and number > maximum
    ):
        raise WeatherValidationError(
            f"{label} above maximum {maximum}: {number}"
        )

    return value


def extract_precip_probability(
    entry: dict[str, Any],
) -> ScalarValue:
    data = entry.get(
        "data",
        {}
    )

    if not isinstance(
        data,
        dict,
    ):
        return ""

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        period = data.get(
            period_key,
            {}
        )

        if not isinstance(
            period,
            dict,
        ):
            continue

        details = period.get(
            "details",
            {}
        )

        if not isinstance(
            details,
            dict,
        ):
            continue

        probability = details.get(
            "probability_of_precipitation"
        )

        if probability is not None:
            return probability

    return ""


def extract_symbol_code(
    entry: dict[str, Any],
) -> str:
    data = entry.get(
        "data",
        {}
    )

    if not isinstance(
        data,
        dict,
    ):
        return ""

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        period = data.get(
            period_key,
            {}
        )

        if not isinstance(
            period,
            dict,
        ):
            continue

        summary = period.get(
            "summary",
            {}
        )

        if not isinstance(
            summary,
            dict,
        ):
            continue

        symbol = clean(
            summary.get(
                "symbol_code"
            )
        )

        if symbol:
            return symbol

    return ""


def derive_rain_snow_flags(
    symbol_code: str,
) -> tuple[
    ScalarValue,
    ScalarValue,
]:
    code = clean(
        symbol_code
    ).casefold()

    if not code:
        return "", ""

    return (
        int(
            "rain" in code
            or "sleet" in code
        ),
        int(
            "snow" in code
        ),
    )


def extract_weather_values(
    timestep_result: TimestepResult,
) -> dict[str, object]:
    if (
        timestep_result.status
        != "forecast_available"
        or timestep_result.entry
        is None
        or timestep_result.timestep
        is None
    ):
        raise WeatherValidationError(
            "Cannot extract weather from unavailable forecast"
        )

    entry = timestep_result.entry

    data = entry.get(
        "data",
        {}
    )

    if not isinstance(
        data,
        dict,
    ):
        raise WeatherValidationError(
            "Forecast entry data is not an object"
        )

    instant = data.get(
        "instant",
        {}
    )

    if not isinstance(
        instant,
        dict,
    ):
        raise WeatherValidationError(
            "Forecast entry instant block is not an object"
        )

    details = instant.get(
        "details",
        {}
    )

    if not isinstance(
        details,
        dict,
    ):
        raise WeatherValidationError(
            "Forecast instant details are not an object"
        )

    wind_speed = finite_float(
        details.get(
            "wind_speed"
        ),
        label="provider wind_speed",
    )

    if wind_speed < 0:
        raise WeatherValidationError(
            f"provider wind_speed is negative: {wind_speed}"
        )

    temperature = optional_numeric(
        details.get(
            "air_temperature"
        ),
        label="provider air_temperature",
    )

    wind_gust = optional_numeric(
        details.get(
            "wind_speed_of_gust"
        ),
        label="provider wind_speed_of_gust",
        minimum=0,
    )

    humidity = optional_numeric(
        details.get(
            "relative_humidity"
        ),
        label="provider relative_humidity",
        minimum=0,
        maximum=100,
    )

    precip_probability = optional_numeric(
        extract_precip_probability(
            entry
        ),
        label=(
            "provider probability_of_precipitation"
        ),
        minimum=0,
        maximum=100,
    )

    (
        rain_flag,
        snow_flag,
    ) = derive_rain_snow_flags(
        extract_symbol_code(
            entry
        )
    )

    return {
        "weather_timestep_utc": (
            timestep_result
            .timestep
            .isoformat()
        ),
        "temperature": temperature,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "precip_probability": (
            precip_probability
        ),
        "rain_flag": rain_flag,
        "snow_flag": snow_flag,
        "humidity": humidity,
    }


def blank_weather() -> dict[str, str]:
    return {
        column: ""
        for column
        in WEATHER_COLUMNS
    }


def build_base_row(
    game: dict[str, str],
    venue_row: dict[str, str],
    kickoff_utc: datetime,
) -> dict[str, object]:
    return {
        "game_id": clean(
            game.get("game_id")
        ),
        "stadium": clean(
            game.get("stadium")
        ),
        "venue_resolution_status": clean(
            game.get(
                "_venue_resolution_status"
            )
        ),
        "latitude": clean(
            venue_row.get("latitude")
        ),
        "longitude": clean(
            venue_row.get("longitude")
        ),
        "game_time": clean(
            game.get("game_time")
        ),
        "game_timezone": clean(
            game.get("game_timezone")
        ),
        "kickoff_utc": (
            kickoff_utc.isoformat()
        ),
        **blank_weather(),
        "roof": clean(
            game.get("roof")
        ),
        "roof_type": clean(
            venue_row.get("roof_type")
        ),
        "dome_flag": clean(
            venue_row.get("dome_flag")
        ),
        "retractable_roof_flag": clean(
            venue_row.get(
                "retractable_roof_flag"
            )
        ),
        "open_air_flag": clean(
            venue_row.get(
                "open_air_flag"
            )
        ),
        "weather_fetched_at": "",
    }


def weather_fields_present(
    row: dict[str, object],
) -> bool:
    return any(
        clean(
            row.get(column)
        )
        for column
        in WEATHER_COLUMNS
    )


def has_usable_weather(
    row: dict[str, object] | None,
) -> bool:
    if row is None:
        return False

    return bool(
        clean(
            row.get(
                "weather_timestep_utc"
            )
        )
        and clean(
            row.get(
                "wind_speed"
            )
        )
        and clean(
            row.get(
                "weather_fetched_at"
            )
        )
    )



def prior_weather_usable_for_kickoff(
    row: dict[str, object] | None,
    kickoff_utc: datetime,
) -> bool:
    if not has_usable_weather(
        row
    ):
        return False

    timestep = parse_iso_utc(
        row.get(
            "weather_timestep_utc"
        )
    )

    if timestep is None:
        return False

    return (
        abs(
            (
                timestep
                - kickoff_utc
            ).total_seconds()
        )
        <= MAX_FORECAST_OFFSET_SECONDS
    )


def copy_prior_weather(
    target_row: dict[str, object],
    prior_row: dict[str, str],
) -> None:
    for column in WEATHER_COLUMNS:
        target_row[
            column
        ] = clean(
            prior_row.get(
                column
            )
        )

    target_row[
        "weather_fetched_at"
    ] = clean(
        prior_row.get(
            "weather_fetched_at"
        )
    )



def _validate_weather_flags(
    row: dict[str, str],
    *,
    game_id: str,
) -> None:
    for field in (
        "rain_flag",
        "snow_flag",
    ):
        value = clean(
            row.get(field)
        )

        if value and value not in {
            "0",
            "1",
        }:
            raise WeatherValidationError(
                f"{field} must be 0/1/blank "
                f"for game_id={game_id}: {value!r}"
            )


def validate_weather_values(
    row: dict[str, str],
    *,
    game_id: str,
    kickoff_utc: datetime,
    strict_blank_timestamp: bool,
) -> float | None:
    any_weather = weather_fields_present(
        row
    )

    fetched_at = clean(
        row.get(
            "weather_fetched_at"
        )
    )

    if not any_weather:
        if (
            strict_blank_timestamp
            and fetched_at
        ):
            raise WeatherValidationError(
                "weather_fetched_at populated without "
                f"weather values for game_id={game_id}"
            )

        return None

    timestep = require_iso_utc(
        row.get(
            "weather_timestep_utc"
        ),
        label=(
            "weather_timestep_utc "
            f"for game_id={game_id}"
        ),
    )

    offset_seconds = abs(
        (
            timestep
            - kickoff_utc
        ).total_seconds()
    )

    if (
        offset_seconds
        > MAX_FORECAST_OFFSET_SECONDS
    ):
        raise WeatherValidationError(
            "Weather timestep exceeds maximum "
            f"kickoff offset for game_id={game_id}: "
            f"offset_seconds={offset_seconds}"
        )

    wind_speed = finite_float(
        row.get("wind_speed"),
        label=f"wind_speed for game_id={game_id}",
    )

    if wind_speed < 0:
        raise WeatherValidationError(
            f"Negative wind_speed for game_id={game_id}"
        )

    temperature = clean(
        row.get("temperature")
    )

    if temperature:
        finite_float(
            temperature,
            label=f"temperature for game_id={game_id}",
        )

    wind_gust = clean(
        row.get("wind_gust")
    )

    if wind_gust:
        gust = finite_float(
            wind_gust,
            label=f"wind_gust for game_id={game_id}",
        )

        if gust < 0:
            raise WeatherValidationError(
                f"Negative wind_gust for game_id={game_id}"
            )

    precip = clean(
        row.get(
            "precip_probability"
        )
    )

    if precip:
        precip_value = finite_float(
            precip,
            label=(
                "precip_probability "
                f"for game_id={game_id}"
            ),
        )

        if not 0 <= precip_value <= 100:
            raise WeatherValidationError(
                "precip_probability outside [0,100] "
                f"for game_id={game_id}: {precip_value}"
            )

    humidity = clean(
        row.get("humidity")
    )

    if humidity:
        humidity_value = finite_float(
            humidity,
            label=f"humidity for game_id={game_id}",
        )

        if not 0 <= humidity_value <= 100:
            raise WeatherValidationError(
                f"humidity outside [0,100] for game_id={game_id}"
            )

    _validate_weather_flags(
        row,
        game_id=game_id,
    )

    if not fetched_at:
        raise WeatherValidationError(
            "weather_fetched_at missing for populated "
            f"weather row game_id={game_id}"
        )

    require_iso_utc(
        fetched_at,
        label=(
            "weather_fetched_at "
            f"for game_id={game_id}"
        ),
    )

    return (
        offset_seconds
        / 3600.0
    )


def _validate_weather_output_flags(
    row: dict[str, str],
    *,
    game_id: str,
) -> None:
    for field in (
        "dome_flag",
        "retractable_roof_flag",
        "open_air_flag",
    ):
        value = clean(
            row.get(field)
        )

        if (
            value
            and value not in {
                "0",
                "1",
            }
        ):
            raise WeatherValidationError(
                f"{field} must be 0/1/blank "
                f"for game_id={game_id}"
            )


def _validate_weather_output_coverage(
    seen: set[str],
    schedule_lookup: dict[str, dict[str, str]],
) -> None:
    if seen != set(
        schedule_lookup
    ):
        raise WeatherValidationError(
            "Weather game coverage does not match target schedule"
        )


def _validate_weather_output_row_count(
    rows: list[dict[str, str]],
    schedule_lookup: dict[str, dict[str, str]],
) -> None:
    if len(rows) != len(
        schedule_lookup
    ):
        raise WeatherValidationError(
            "Weather row-count mismatch: "
            f"expected={len(schedule_lookup)}, "
            f"actual={len(rows)}"
        )



def validate_weather_schedule_time(
    row: dict[str, str],
    game: dict[str, str],
    *,
    game_id: str,
    strict_blank_timestamp: bool,
) -> datetime:
    allow_existing_drift = (
        not strict_blank_timestamp
    )

    if (
        clean(
            row.get("game_time")
        )
        != clean(
            game.get("game_time")
        )
        and not allow_existing_drift
    ):
        raise WeatherValidationError(
            "Weather game_time mismatch "
            f"for game_id={game_id}"
        )

    expected_kickoff = (
        resolve_kickoff_utc(
            game
        )
    )

    actual_kickoff = (
        require_iso_utc(
            row.get(
                "kickoff_utc"
            ),
            label=(
                "weather kickoff_utc "
                f"for game_id={game_id}"
            ),
        )
    )

    if (
        abs(
            (
                actual_kickoff
                - expected_kickoff
            ).total_seconds()
        ) > 1
        and not allow_existing_drift
    ):
        raise WeatherValidationError(
            "Weather kickoff mismatch "
            f"for game_id={game_id}"
        )

    if allow_existing_drift:
        return actual_kickoff

    return expected_kickoff


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
    travel_lookup: dict[
        str,
        dict[str, str],
    ],
    strict_blank_timestamp: bool,
) -> list[float]:
    _validate_weather_output_row_count(
        rows,
        schedule_lookup,
    )

    seen: set[str] = set()
    offsets: list[float] = []

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        if list(row.keys()) != OUTPUT_HEADERS:
            raise WeatherValidationError(
                f"Weather schema mismatch at row {line_number}"
            )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=(
                    f"weather game_id row {line_number}"
                ),
            )
        )

        if game_id in seen:
            raise WeatherValidationError(
                f"Duplicate weather game_id={game_id}"
            )

        game = schedule_lookup.get(
            game_id
        )

        travel_row = travel_lookup.get(
            game_id
        )

        if (
            game is None
            or travel_row is None
        ):
            raise WeatherValidationError(
                f"Foreign weather game_id={game_id}"
            )

        seen.add(
            game_id
        )

        if clean(
            row.get("stadium")
        ) != clean(
            game.get("stadium")
        ):
            raise WeatherValidationError(
                f"Weather stadium mismatch for game_id={game_id}"
            )

        expected_status = clean(
            travel_row.get(
                "venue_resolution_status"
            )
        )

        if clean(
            row.get(
                "venue_resolution_status"
            )
        ) != expected_status:
            raise WeatherValidationError(
                "Weather venue-resolution status mismatch "
                f"for game_id={game_id}"
            )

        expected_latitude = clean(
            travel_row.get("venue_lat")
        )

        expected_longitude = clean(
            travel_row.get("venue_lon")
        )

        if clean(
            row.get("latitude")
        ) != expected_latitude:
            raise WeatherValidationError(
                f"Weather latitude mismatch for game_id={game_id}"
            )

        if clean(
            row.get("longitude")
        ) != expected_longitude:
            raise WeatherValidationError(
                f"Weather longitude mismatch for game_id={game_id}"
            )

        validate_coordinate_pair(
            row.get("latitude"),
            row.get("longitude"),
            label=f"weather venue for game_id={game_id}",
        )

        expected_timezone = clean(
            game.get(
                "game_timezone"
            )
        )

        if clean(
            row.get(
                "game_timezone"
            )
        ) != expected_timezone:
            raise WeatherValidationError(
                f"Weather timezone mismatch for game_id={game_id}"
            )

        validate_timezone(
            expected_timezone,
            label=(
                "weather game_timezone "
                f"for game_id={game_id}"
            ),
        )

        expected_kickoff = (
            validate_weather_schedule_time(
                row,
                game,
                game_id=game_id,
                strict_blank_timestamp=(
                    strict_blank_timestamp
                ),
            )
        )

        if clean(
            row.get("roof")
        ) != clean(
            game.get("roof")
        ):
            raise WeatherValidationError(
                f"Weather roof mismatch for game_id={game_id}"
            )

        _validate_weather_output_flags(
            row,
            game_id=game_id,
        )

        offset = validate_weather_values(
            row,
            game_id=game_id,
            kickoff_utc=expected_kickoff,
            strict_blank_timestamp=(
                strict_blank_timestamp
            ),
        )

        if offset is not None:
            offsets.append(
                offset
            )

    _validate_weather_output_coverage(
        seen,
        schedule_lookup,
    )

    return offsets


def load_existing_output(
    path: Path,
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
    travel_lookup: dict[
        str,
        dict[str, str],
    ],
) -> dict[
    str,
    dict[str, str],
]:
    if not path.exists():
        return {}

    fieldnames, rows = load_csv(
        path,
        label="existing target weather output",
    )

    if fieldnames != OUTPUT_HEADERS:
        raise WeatherValidationError(
            "Existing weather output header mismatch: "
            f"expected={OUTPUT_HEADERS}, actual={fieldnames}"
        )

    validate_output_rows(
        rows,
        schedule_lookup=schedule_lookup,
        travel_lookup=travel_lookup,
        strict_blank_timestamp=False,
    )

    return {
        clean(
            row.get("game_id")
        ): row
        for row in rows
    }


def weather_is_exposed(
    row: dict[str, object],
) -> bool:
    roof = clean(
        row.get("roof")
    ).casefold()

    roof_type = clean(
        row.get("roof_type")
    ).casefold()

    dome_flag = clean(
        row.get("dome_flag")
    )

    open_air_flag = clean(
        row.get("open_air_flag")
    )

    if (
        dome_flag == "1"
        or "dome" in roof
        or "indoor" in roof
        or "closed" in roof
        or "dome" in roof_type
        or "indoor" in roof_type
        or "closed" in roof_type
    ):
        return False

    if open_air_flag == "1":
        return True

    return (
        "open_air" in roof
        or "open air" in roof
        or "outdoor" in roof
    )


def _stage1_weather_state() -> dict[str, object]:
    return {
        "provider_cache": {},
        "provider_status_counts": Counter(),
        "forecast_status_counts": Counter(),
        "provider_http_status_counts": Counter(),
        "provider_request_count": 0,
        "provider_success_count": 0,
        "provider_failure_count": 0,
        "future_game_count": 0,
        "completed_game_count": 0,
        "newly_fetched_count": 0,
        "reused_prior_count": 0,
        "reused_completed_count": 0,
        "reused_after_refresh_failure_count": 0,
        "blank_weather_count": 0,
        "roof_metadata_missing_count": 0,
        "exposed_game_count": 0,
        "refresh_failure_game_ids": [],
        "blank_weather_game_ids": [],
        "new_offset_hours": [],
    }


def _stage1_completed_weather(
    row: dict[str, object],
    *,
    prior: dict[str, str] | None,
    prior_usable: bool,
    game_id: str,
    state: dict[str, object],
) -> dict[str, object]:
    state["completed_game_count"] += 1
    forecast_counts = state["forecast_status_counts"]
    if prior is not None and prior_usable:
        copy_prior_weather(row, prior)
        state["reused_prior_count"] += 1
        state["reused_completed_count"] += 1
        forecast_counts["reused_completed_forecast"] += 1
    else:
        state["blank_weather_count"] += 1
        state["blank_weather_game_ids"].append(game_id)
        forecast_counts["completed_without_stored_weather"] += 1
    return row


def _stage1_refresh_failure(
    row: dict[str, object],
    *,
    prior: dict[str, str] | None,
    prior_usable: bool,
    game_id: str,
    status: str,
    state: dict[str, object],
) -> dict[str, object]:
    state["forecast_status_counts"][status] += 1
    state["refresh_failure_game_ids"].append(game_id)
    if prior is not None and prior_usable:
        copy_prior_weather(row, prior)
        state["reused_prior_count"] += 1
        state["reused_after_refresh_failure_count"] += 1
    else:
        state["blank_weather_count"] += 1
        state["blank_weather_game_ids"].append(game_id)
    return row


def _stage1_provider_for_venue(
    venue_row: dict[str, str],
    *,
    state: dict[str, object],
    fetcher: Callable[[str, str], ProviderResult],
    sleep_fn: Callable[[float], None],
) -> ProviderResult:
    request_latitude, request_longitude = canonical_request_coordinates(
        venue_row.get("latitude"),
        venue_row.get("longitude"),
    )
    cache_key = (request_latitude, request_longitude)
    provider_cache = state["provider_cache"]
    provider_result = provider_cache.get(cache_key)
    if provider_result is not None:
        return provider_result

    state["provider_request_count"] += 1
    provider_result = fetcher(request_latitude, request_longitude)
    provider_cache[cache_key] = provider_result
    state["provider_status_counts"][provider_result.status] += 1
    if provider_result.http_status is not None:
        state["provider_http_status_counts"][str(provider_result.http_status)] += 1
    if provider_result.status == "request_success":
        state["provider_success_count"] += 1
    else:
        state["provider_failure_count"] += 1
    sleep_fn(REQUEST_SLEEP_SECONDS)
    return provider_result


def _stage1_build_weather_game(
    game: dict[str, str],
    *,
    travel_lookup: dict[str, dict[str, str]],
    stadium_coordinate_lookup: dict[tuple[float, float], dict[str, str]],
    existing_rows: dict[str, dict[str, str]],
    fetched_at: datetime,
    now_utc: datetime,
    state: dict[str, object],
    fetcher: Callable[[str, str], ProviderResult],
    sleep_fn: Callable[[float], None],
) -> dict[str, object]:
    game_id = clean(game.get("game_id"))
    travel_row = travel_lookup[game_id]
    venue_row, roof_metadata_missing = venue_from_travel(
        game,
        travel_lookup,
        stadium_coordinate_lookup,
    )
    if roof_metadata_missing:
        state["roof_metadata_missing_count"] += 1

    game["_venue_resolution_status"] = clean(
        travel_row.get("venue_resolution_status")
    )
    kickoff_utc = resolve_kickoff_utc(game)
    row = build_base_row(game, venue_row, kickoff_utc)
    if weather_is_exposed(row):
        state["exposed_game_count"] += 1

    prior = existing_rows.get(game_id)
    prior_usable = (
        prior_weather_usable_for_kickoff(
            prior,
            kickoff_utc,
        )
    )
    if kickoff_utc <= now_utc:
        return _stage1_completed_weather(
            row,
            prior=prior,
            prior_usable=prior_usable,
            game_id=game_id,
            state=state,
        )

    state["future_game_count"] += 1
    provider_result = _stage1_provider_for_venue(
        venue_row,
        state=state,
        fetcher=fetcher,
        sleep_fn=sleep_fn,
    )
    if provider_result.status != "request_success" or provider_result.payload is None:
        return _stage1_refresh_failure(
            row,
            prior=prior,
            prior_usable=prior_usable,
            game_id=game_id,
            status=provider_result.status,
            state=state,
        )

    timestep_result = select_kickoff_timestep(
        provider_result.payload,
        kickoff_utc,
    )
    if timestep_result.status != "forecast_available":
        return _stage1_refresh_failure(
            row,
            prior=prior,
            prior_usable=prior_usable,
            game_id=game_id,
            status=timestep_result.status,
            state=state,
        )

    try:
        values = extract_weather_values(timestep_result)
    except WeatherValidationError:
        return _stage1_refresh_failure(
            row,
            prior=prior,
            prior_usable=prior_usable,
            game_id=game_id,
            status="invalid_forecast_values",
            state=state,
        )

    row.update(values)
    row["weather_fetched_at"] = fetched_at.isoformat()
    state["newly_fetched_count"] += 1
    state["forecast_status_counts"]["forecast_available"] += 1
    if timestep_result.offset_seconds is not None:
        state["new_offset_hours"].append(
            timestep_result.offset_seconds / 3600.0
        )
    return row


def _stage1_weather_offsets(
    output_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, str]],
) -> list[float]:
    all_offsets: list[float] = []
    schedule_by_game_id = {
        clean(game.get("game_id")): game for game in schedule_rows
    }
    for row in output_rows:
        if not has_usable_weather(row):
            continue
        game_id = clean(row.get("game_id"))
        kickoff = resolve_kickoff_utc(schedule_by_game_id[game_id])
        timestep = require_iso_utc(
            row.get("weather_timestep_utc"),
            label=f"weather_timestep_utc for game_id={game_id}",
        )
        all_offsets.append(
            abs((timestep - kickoff).total_seconds()) / 3600.0
        )
    return all_offsets


def _stage1_weather_metrics(
    output_rows: list[dict[str, object]],
    schedule_rows: list[dict[str, str]],
    state: dict[str, object],
) -> dict[str, Any]:
    weather_available_count = sum(has_usable_weather(row) for row in output_rows)
    wind_speed_count = sum(
        bool(clean(row.get("wind_speed"))) for row in output_rows
    )
    temperature_count = sum(
        bool(clean(row.get("temperature"))) for row in output_rows
    )
    wind_gust_count = sum(
        bool(clean(row.get("wind_gust"))) for row in output_rows
    )
    precip_count = sum(
        bool(clean(row.get("precip_probability"))) for row in output_rows
    )
    humidity_count = sum(
        bool(clean(row.get("humidity"))) for row in output_rows
    )
    all_offsets = _stage1_weather_offsets(output_rows, schedule_rows)
    new_offsets = state["new_offset_hours"]
    return {
        "future_game_count": state["future_game_count"],
        "completed_game_count": state["completed_game_count"],
        "provider_request_count": state["provider_request_count"],
        "provider_request_success_count": state["provider_success_count"],
        "provider_request_failure_count": state["provider_failure_count"],
        "provider_request_status_counts": dict(
            sorted(state["provider_status_counts"].items())
        ),
        "provider_http_status_counts": dict(
            sorted(state["provider_http_status_counts"].items())
        ),
        "forecast_result_status_counts": dict(
            sorted(state["forecast_status_counts"].items())
        ),
        "newly_fetched_game_count": state["newly_fetched_count"],
        "reused_prior_forecast_count": state["reused_prior_count"],
        "reused_completed_forecast_count": state["reused_completed_count"],
        "reused_after_refresh_failure_count": state[
            "reused_after_refresh_failure_count"
        ],
        "blank_weather_game_count": state["blank_weather_count"],
        "blank_weather_game_ids": sorted(set(state["blank_weather_game_ids"])),
        "refresh_failure_game_ids": sorted(set(state["refresh_failure_game_ids"])),
        "weather_available_game_count": weather_available_count,
        "wind_speed_coverage": wind_speed_count,
        "temperature_coverage": temperature_count,
        "wind_gust_coverage": wind_gust_count,
        "precip_probability_coverage": precip_count,
        "humidity_coverage": humidity_count,
        "roof_metadata_missing_count": state["roof_metadata_missing_count"],
        "weather_exposed_game_count": state["exposed_game_count"],
        "new_forecast_offset_hours_max": max(new_offsets) if new_offsets else None,
        "new_forecast_offset_hours_median": (
            statistics.median(new_offsets) if new_offsets else None
        ),
        "output_forecast_offset_hours_max": max(all_offsets) if all_offsets else None,
        "output_forecast_offset_hours_median": (
            statistics.median(all_offsets) if all_offsets else None
        ),
    }


def build_weather_rows(
    schedule_rows: list[dict[str, str]],
    *,
    travel_lookup: dict[str, dict[str, str]],
    stadium_coordinate_lookup: dict[tuple[float, float], dict[str, str]],
    existing_rows: dict[str, dict[str, str]],
    fetched_at: datetime,
    now_utc: datetime,
    fetcher: Callable[[str, str], ProviderResult] = fetch_weather_json,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    state = _stage1_weather_state()
    output_rows = [
        _stage1_build_weather_game(
            game,
            travel_lookup=travel_lookup,
            stadium_coordinate_lookup=stadium_coordinate_lookup,
            existing_rows=existing_rows,
            fetched_at=fetched_at,
            now_utc=now_utc,
            state=state,
            fetcher=fetcher,
            sleep_fn=sleep_fn,
        )
        for game in schedule_rows
    ]
    metrics = _stage1_weather_metrics(
        output_rows,
        schedule_rows,
        state,
    )
    return output_rows, metrics



def read_staged_rows(
    path: Path,
) -> list[
    dict[str, str]
]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle,
            strict=True,
        )

        if reader.fieldnames != OUTPUT_HEADERS:
            raise WeatherValidationError(
                "Staged weather header mismatch: "
                f"{reader.fieldnames}"
            )

        rows: list[
            dict[str, str]
        ] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if (
                None in row
                or any(
                    value is None
                    for value in row.values()
                )
            ):
                raise WeatherValidationError(
                    "Malformed staged weather row "
                    f"at CSV line {line_number}"
                )

            rows.append({
                key: clean(value)
                for key, value in row.items()
            })

    return rows


def publish_atomic(
    rows: list[
        dict[str, object]
    ],
    path: Path,
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
    travel_lookup: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    bool,
    int,
    list[float],
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        write_csv_rows_durable(
            temp_path,
            rows,
            OUTPUT_HEADERS,
        )

        staged_rows = (
            read_staged_rows(
                temp_path
            )
        )

        offsets = (
            validate_output_rows(
                staged_rows,
                schedule_lookup=schedule_lookup,
                travel_lookup=travel_lookup,
                strict_blank_timestamp=True,
            )
        )

        output_modified = (
            not path.exists()
            or path.read_bytes()
            != temp_path.read_bytes()
        )

        if output_modified:
            os.replace(
                temp_path,
                path,
            )
        else:
            temp_path.unlink()

        return (
            output_modified,
            len(staged_rows),
            offsets,
        )

    except Exception:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise


def main() -> None:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "provider": "api.met.no Locationforecast 2.0 complete",
        },
    ) as report:
        report.add_input(
            CONFIG_PATH
        )

        (
            season,
            season_type,
            week,
        ) = load_config()

        report.season = season
        report.week = week

        (
            schedule_path,
            travel_path,
            output_path,
        ) = target_paths(
            season,
            week,
        )

        report.add_input(
            schedule_path
        )

        report.add_input(
            travel_path
        )

        report.add_input(
            STADIUM_MAP_PATH
        )

        report.add_output(
            output_path
        )

        report.update_details({
            "season_type": season_type,
            "schedule_path": schedule_path,
            "travel_path": travel_path,
            "stadium_map_path": STADIUM_MAP_PATH,
            "output_path": output_path,
            "provider_url": METNO_URL,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "request_sleep_seconds": REQUEST_SLEEP_SECONDS,
            "coordinate_precision_decimals": 4,
            "max_forecast_offset_hours": (
                MAX_FORECAST_OFFSET_SECONDS
                / 3600
            ),
            "output_modified": False,
        })

        (
            schedule_rows,
            schedule_lookup,
        ) = load_schedule(
            schedule_path,
            season=season,
            season_type=season_type,
            week=week,
        )

        (
            travel_rows,
            travel_lookup,
        ) = load_travel(
            travel_path,
            schedule_lookup=schedule_lookup,
        )

        (
            stadium_rows,
            stadium_coordinate_lookup,
        ) = (
            load_stadium_coordinate_lookup()
        )

        existing_rows = (
            load_existing_output(
                output_path,
                schedule_lookup=schedule_lookup,
                travel_lookup=travel_lookup,
            )
        )

        report.set_rows(
            rows_in=len(
                schedule_rows
            )
        )

        fetched_at = datetime.now(
            timezone.utc
        )

        (
            output_rows,
            metrics,
        ) = build_weather_rows(
            schedule_rows,
            travel_lookup=travel_lookup,
            stadium_coordinate_lookup=(
                stadium_coordinate_lookup
            ),
            existing_rows=existing_rows,
            fetched_at=fetched_at,
            now_utc=fetched_at,
        )

        serialized = [
            {
                key: clean(value)
                for key, value in row.items()
            }
            for row in output_rows
        ]

        validate_output_rows(
            serialized,
            schedule_lookup=schedule_lookup,
            travel_lookup=travel_lookup,
            strict_blank_timestamp=True,
        )

        (
            output_modified,
            staged_row_count,
            staged_offsets,
        ) = publish_atomic(
            output_rows,
            output_path,
            schedule_lookup=schedule_lookup,
            travel_lookup=travel_lookup,
        )

        report.set_rows(
            rows_out=staged_row_count
        )

        report.update_details({
            "target_schedule_rows": (
                len(schedule_rows)
            ),
            "target_travel_rows": (
                len(travel_rows)
            ),
            "stadium_map_rows": (
                len(stadium_rows)
            ),
            "existing_weather_rows": (
                len(existing_rows)
            ),
            "output_rows": (
                staged_row_count
            ),
            "output_modified": (
                output_modified
            ),
            "staged_forecast_offset_hours_max": (
                max(staged_offsets)
                if staged_offsets
                else None
            ),
            "staged_forecast_offset_hours_median": (
                statistics.median(
                    staged_offsets
                )
                if staged_offsets
                else None
            ),
            **metrics,
        })

        degraded = (
            int(
                metrics[
                    "provider_request_failure_count"
                ]
            )
            > 0
            or int(
                metrics[
                    "blank_weather_game_count"
                ]
            )
            > 0
            or int(
                metrics[
                    "reused_after_refresh_failure_count"
                ]
            )
            > 0
            or any(
                status
                in {
                    "outside_forecast_range",
                    "no_acceptable_timestep",
                    "invalid_response",
                    "invalid_forecast_values",
                }
                and count
                for (
                    status,
                    count,
                )
                in dict(
                    metrics[
                        "forecast_result_status_counts"
                    ]
                ).items()
            )
        )

        if degraded:
            report.warning(
                "Weather refresh completed with degraded provider coverage",
                provider_request_failures=(
                    metrics[
                        "provider_request_failure_count"
                    ]
                ),
                reused_prior_forecasts=(
                    metrics[
                        "reused_after_refresh_failure_count"
                    ]
                ),
                blank_weather_games=(
                    metrics[
                        "blank_weather_game_count"
                    ]
                ),
                forecast_status_counts=(
                    metrics[
                        "forecast_result_status_counts"
                    ]
                ),
            )

        if int(
            metrics[
                "roof_metadata_missing_count"
            ]
        ) > 0:
            report.warning(
                "Resolved travel coordinates lacked stadium-map roof metadata",
                count=(
                    metrics[
                        "roof_metadata_missing_count"
                    ]
                ),
            )

        print(
            "fetch_weather.py "
            f"version={SCRIPT_VERSION} "
            f"target={season}/{season_type}/{week} "
            f"rows={staged_row_count} "
            "weather_available="
            f"{metrics['weather_available_game_count']} "
            "provider_requests="
            f"{metrics['provider_request_count']} "
            "provider_failures="
            f"{metrics['provider_request_failure_count']} "
            "reused_prior="
            f"{metrics['reused_prior_forecast_count']} "
            f"output_modified={output_modified}"
        )


if __name__ == "__main__":
    main()
