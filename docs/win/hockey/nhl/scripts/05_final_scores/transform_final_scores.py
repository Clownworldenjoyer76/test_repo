#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/transform_final_scores.py

from __future__ import annotations

import json
import re
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never
from zoneinfo import ZoneInfo

import pandas as pd
import requests


SPORT = "hockey"
LEAGUE = "nhl"
LEAGUE_OUT = "NHL"

BASE_DIR = Path(__file__).resolve().parents[2]

SELECT_DIR = BASE_DIR / "04_select"
RAW_DIR = BASE_DIR / "00_intake" / "drat_raw"
OUT_DIR = BASE_DIR / "05_final_scores" / "final_scores"
STATUS_DIR = BASE_DIR / "05_final_scores" / "intermediate"
STATUS_FILE = STATUS_DIR / "nhl_game_status.csv"
ERROR_DIR = BASE_DIR / "errors" / "05_final_scores"
LOG_FILE = ERROR_DIR / "transform_final_scores.txt"

SELECT_PATTERN = "*_NHL.csv"
RAW_PATTERN = "*_nhl_raw.json"

NHL_SCORE_URL = "https://api-web.nhle.com/v1/score/{date}"
REQUEST_TIMEOUT = 30

ET = ZoneInfo("America/New_York")
FINAL_GAME_STATES = {"FINAL", "OFF"}
GAME_ID_RE = re.compile(r"^\d{10}$")

STATUS_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "game_state",
    "game_schedule_state",
    "is_final",
    "status_observed_at",
]

OUTPUT_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "total_score",
    "away_puck_line_result",
    "home_puck_line_result",
]


# ============================================================
# LOGGING / FILE HELPERS
# ============================================================

def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    stamp = datetime.now(UTC).isoformat()

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{stamp} | {msg}\n")


def fail(msg: str) -> Never:
    log(f"ERROR: {msg}")
    raise RuntimeError(msg)


def atomic_write_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    tmp_path = output_path.with_suffix(".tmp")

    df.to_csv(
        tmp_path,
        index=False,
    )

    tmp_path.replace(output_path)


# ============================================================
# GENERIC NORMALIZATION
# ============================================================

def norm_key(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return re.sub(r"\s+", " ", value).strip()


def first_present(
    row: dict[str, Any],
    names: list[str],
) -> Any:
    for name in names:
        if name in row and row[name] not in [None, ""]:
            return row[name]

    lower_map = {
        str(key).lower(): key
        for key in row.keys()
    }

    for name in names:
        key = lower_map.get(name.lower())

        if key is not None and row[key] not in [None, ""]:
            return row[key]

    return None


def parse_int_score(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if pd.isna(value):
            return None

        if value.is_integer():
            return int(value)

        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None

    if not number.is_integer():
        return None

    return int(number)


def normalize_game_date(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    formats = [
        "%Y_%m_%d",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%Y %I:%M %p",
        "%m/%d/%Y %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                text,
                fmt,
            ).strftime("%Y_%m_%d")
        except ValueError:
            pass

    match = re.search(
        r"(\d{4})[-_](\d{2})[-_](\d{2})",
        text,
    )

    if match:
        year, month, day = match.groups()
        return f"{year}_{month}_{day}"

    match = re.search(
        r"(\d{2})/(\d{2})/(\d{4})",
        text,
    )

    if match:
        month, day, year = match.groups()
        return f"{year}_{month}_{day}"

    return ""


def api_date_from_game_date(game_date: str) -> str:
    return datetime.strptime(
        game_date,
        "%Y_%m_%d",
    ).strftime("%Y-%m-%d")


def parse_date_from_filename(path: Path) -> str:
    match = re.search(
        r"(\d{4})[_-](\d{2})[_-](\d{2})",
        path.stem,
    )

    if not match:
        return ""

    year, month, day = match.groups()

    return f"{year}_{month}_{day}"


def current_game_date() -> str:
    return datetime.now(ET).strftime("%Y_%m_%d")


# ============================================================
# STAGE 04 — TARGET DATE SOURCE
# ============================================================

def discover_target_dates() -> set[str]:
    if not SELECT_DIR.exists():
        fail(
            f"Stage 04 select folder does not exist: {SELECT_DIR}"
        )

    select_files = sorted(
        SELECT_DIR.glob(SELECT_PATTERN)
    )

    if not select_files:
        fail(
            f"No Stage 04 select files found matching "
            f"{SELECT_PATTERN} in {SELECT_DIR}"
        )

    target_dates: set[str] = set()

    for select_path in select_files:
        game_date = parse_date_from_filename(
            select_path
        )

        if not game_date:
            log(
                "SELECT DATE SKIP: could not determine date from "
                f"{select_path}"
            )
            continue

        target_dates.add(game_date)

    if not target_dates:
        fail(
            "Could not determine any target dates "
            "from Stage 04 select filenames"
        )

    return target_dates


# ============================================================
# EXISTING OFFICIAL HISTORY — INCREMENTAL FETCH STATE
# ============================================================

def final_score_path(game_date: str) -> Path:
    return (
        OUT_DIR
        / f"{game_date}_{LEAGUE_OUT}_final_scores.csv"
    )


def validate_game_ids(
    df: pd.DataFrame,
    *,
    label: str,
) -> None:
    invalid = [
        str(value).strip()
        for value in df["game_id"]
        if not GAME_ID_RE.fullmatch(
            str(value).strip()
        )
    ]

    if invalid:
        fail(
            f"{label} contains invalid canonical game_id values: "
            f"{invalid[:10]}"
        )


def load_existing_status_snapshot() -> pd.DataFrame:
    if not STATUS_FILE.exists():
        log(
            "No existing official NHL status snapshot found; "
            "all target dates require evaluation"
        )
        return pd.DataFrame(
            columns=STATUS_COLUMNS,
        )

    try:
        df = pd.read_csv(
            STATUS_FILE,
            dtype=str,
        ).fillna("")
    except Exception as e:
        return fail(
            f"Failed reading existing status snapshot "
            f"{STATUS_FILE}: {e}"
        )

    missing = [
        col
        for col in STATUS_COLUMNS
        if col not in df.columns
    ]

    if missing:
        fail(
            "Existing official NHL status snapshot is missing "
            f"required columns: {missing}"
        )

    df = df[STATUS_COLUMNS].copy()

    if df.empty:
        return df

    df["game_date"] = df["game_date"].map(
        normalize_game_date
    )
    df["game_id"] = (
        df["game_id"]
        .astype(str)
        .str.strip()
    )
    df["game_state"] = (
        df["game_state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["game_schedule_state"] = (
        df["game_schedule_state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if df["game_date"].eq("").any():
        fail(
            "Existing official NHL status snapshot contains "
            "an invalid or blank game_date"
        )

    validate_game_ids(
        df,
        label=str(STATUS_FILE),
    )

    duplicate_ids = df.duplicated(
        subset=["game_id"],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            df.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        fail(
            "Existing official NHL status snapshot contains "
            f"duplicate game_id values: {duplicate_values}"
        )

    log(
        "Loaded existing official NHL status history: "
        f"rows={len(df)}"
    )

    return df


def split_status_history_by_date(
    status_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if status_df.empty:
        return {}

    history: dict[str, pd.DataFrame] = {}

    for game_date, group in status_df.groupby(
        "game_date",
        sort=False,
    ):
        if not isinstance(game_date, str):
            fail(
                "Existing NHL status history contains "
                "a non-string game_date key."
            )

        history[game_date] = (
            group.copy().reset_index(drop=True)
        )

    return history


def load_existing_final_scores_for_date(
    game_date: str,
) -> pd.DataFrame:
    path = final_score_path(game_date)

    if not path.exists():
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS,
        )

    try:
        df = pd.read_csv(
            path,
            dtype=str,
        ).fillna("")
    except Exception as e:
        return fail(
            f"Failed reading existing official final-score file "
            f"{path}: {e}"
        )

    missing = [
        col
        for col in OUTPUT_COLUMNS
        if col not in df.columns
    ]

    if missing:
        fail(
            f"Existing official final-score file {path} "
            f"is missing required columns: {missing}"
        )

    df = df[OUTPUT_COLUMNS].copy()

    if df.empty:
        return df

    df["game_date"] = df["game_date"].map(
        normalize_game_date
    )
    df["game_id"] = (
        df["game_id"]
        .astype(str)
        .str.strip()
    )

    bad_dates = sorted(
        set(
            df.loc[
                df["game_date"] != game_date,
                "game_date",
            ].astype(str)
        )
    )

    if bad_dates:
        fail(
            f"Existing official final-score file {path} "
            f"contains unexpected game_date values: {bad_dates}"
        )

    validate_game_ids(
        df,
        label=str(path),
    )

    duplicate_ids = df.duplicated(
        subset=["game_id"],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            df.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        fail(
            f"Existing official final-score file {path} "
            f"contains duplicate game_id values: {duplicate_values}"
        )

    for column in [
        "away_score",
        "home_score",
        "total_score",
        "away_puck_line_result",
        "home_puck_line_result",
    ]:
        invalid_rows = [
            value
            for value in df[column]
            if parse_int_score(value) is None
        ]

        if invalid_rows:
            fail(
                f"Existing official final-score file {path} "
                f"contains invalid values in {column}"
            )

    return df


def completed_historical_date(
    game_date: str,
    status_by_date: dict[str, pd.DataFrame],
    today_game_date: str,
) -> bool:
    if game_date >= today_game_date:
        return False

    status_df = status_by_date.get(game_date)

    if status_df is None or status_df.empty:
        return False

    states = (
        status_df["game_state"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not states.isin(FINAL_GAME_STATES).all():
        return False

    existing_scores = load_existing_final_scores_for_date(
        game_date
    )

    if existing_scores.empty:
        return False

    status_ids = set(
        status_df["game_id"]
        .astype(str)
        .str.strip()
    )
    score_ids = set(
        existing_scores["game_id"]
        .astype(str)
        .str.strip()
    )

    return status_ids == score_ids


def merge_official_final_history(
    game_date: str,
    existing_df: pd.DataFrame,
    fetched_df: pd.DataFrame,
) -> pd.DataFrame:
    existing = normalize_output_df(
        existing_df
    )
    fetched = normalize_output_df(
        fetched_df
    )

    if existing.empty:
        return fetched

    if fetched.empty:
        return existing

    fetched_ids = set(
        fetched["game_id"]
        .astype(str)
        .str.strip()
    )

    preserved = existing[
        ~existing["game_id"]
        .astype(str)
        .str.strip()
        .isin(fetched_ids)
    ].copy()

    merged = pd.concat(
        [
            preserved,
            fetched,
        ],
        ignore_index=True,
    )

    duplicate_ids = merged.duplicated(
        subset=["game_id"],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            merged.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        fail(
            "Merged official final-score history contains duplicate "
            f"game_id values for {game_date}: {duplicate_values}"
        )

    return merged


# ============================================================
# D-RATINGS — OPTIONAL RECONCILIATION/CHECK SOURCE ONLY
# ============================================================

def flatten_raw_payload(
    payload: Any,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload

    elif isinstance(payload, dict):
        rows = None

        for key in [
            "rows",
            "data",
            "games",
            "events",
            "raw_rows",
        ]:
            if key in payload and isinstance(payload[key], list):
                rows = payload[key]
                break

        if rows is None:
            rows = [payload]

    else:
        raise ValueError(
            "Unsupported raw JSON payload type: "
            f"{type(payload).__name__}"
        )

    return [
        row
        for row in rows
        if isinstance(row, dict)
    ]


def parse_dratings_game_date(
    row: dict[str, Any],
) -> str:
    raw_date = first_present(
        row,
        [
            "game_date",
            "date",
            "date_time",
            "datetime",
            "game_time",
            "start_time",
        ],
    )

    return normalize_game_date(raw_date)


def dratings_is_completed(
    row: dict[str, Any],
) -> bool:
    status = first_present(
        row,
        [
            "game_status",
            "status",
            "status_type",
            "event_status",
            "state",
            "game_state",
        ],
    )

    completed_values = {
        "completed",
        "complete",
        "final",
        "final ot",
        "final so",
        "ended",
        "closed",
    }

    return norm_key(status) in completed_values


def get_dratings_team(
    row: dict[str, Any],
    side: str,
) -> str:
    if side == "away":
        names = [
            "away_team",
            "away",
            "visitor_team",
            "visitor",
            "road_team",
            "team_away",
            "away_name",
            "team1",
        ]

    elif side == "home":
        names = [
            "home_team",
            "home",
            "team_home",
            "home_name",
            "team2",
        ]

    else:
        raise ValueError(
            f"Invalid side: {side}"
        )

    value = first_present(
        row,
        names,
    )

    if value is None:
        return ""

    return str(value).strip()


def get_dratings_score(
    row: dict[str, Any],
    side: str,
) -> int | None:
    if side == "away":
        names = [
            "away_score",
            "away_team_score",
            "visitor_score",
            "road_score",
            "away_points",
            "score_away",
            "score1",
        ]

    elif side == "home":
        names = [
            "home_score",
            "home_team_score",
            "home_points",
            "score_home",
            "score2",
        ]

    else:
        raise ValueError(
            f"Invalid side: {side}"
        )

    return parse_int_score(
        first_present(
            row,
            names,
        )
    )


def collect_dratings_checks(
    raw_files: list[Path],
) -> dict[
    tuple[str, str, str],
    list[dict[str, Any]],
]:
    checks: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = {}

    for raw_path in raw_files:
        log(
            f"Reading D-Ratings check source: {raw_path}"
        )

        try:
            payload = json.loads(
                raw_path.read_text(
                    encoding="utf-8"
                )
            )

            raw_rows = flatten_raw_payload(
                payload
            )

        except Exception as e:
            log(
                "DRAT CHECK FILE ERROR: "
                f"{raw_path} | {e}"
            )
            continue

        filename_date = parse_date_from_filename(
            raw_path
        )

        for raw in raw_rows:
            try:
                game_date = (
                    parse_dratings_game_date(raw)
                    or filename_date
                )

                if not dratings_is_completed(raw):
                    continue

                away_team = get_dratings_team(
                    raw,
                    "away",
                )

                home_team = get_dratings_team(
                    raw,
                    "home",
                )

                away_score = get_dratings_score(
                    raw,
                    "away",
                )

                home_score = get_dratings_score(
                    raw,
                    "home",
                )

                if (
                    not game_date
                    or not away_team
                    or not home_team
                    or away_score is None
                    or home_score is None
                ):
                    log(
                        "DRAT CHECK SKIP: "
                        f"{raw_path} | "
                        "completed row missing date/team/score"
                    )
                    continue

                key = (
                    game_date,
                    norm_key(away_team),
                    norm_key(home_team),
                )

                checks.setdefault(
                    key,
                    [],
                ).append(
                    {
                        "game_date": game_date,
                        "away_team": away_team,
                        "home_team": home_team,
                        "away_score": away_score,
                        "home_score": home_score,
                    }
                )

            except Exception as e:
                log(
                    "DRAT CHECK ROW ERROR: "
                    f"{raw_path} | {e}"
                )

    return checks


# ============================================================
# OFFICIAL NHL API — AUTHORITATIVE FINAL-SCORE SOURCE
# ============================================================

def localized_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        default = value.get("default")

        if default not in [None, ""]:
            return str(default).strip()

        for candidate in value.values():
            if candidate not in [None, ""]:
                return str(candidate).strip()

    return ""


def official_team_name(
    team: dict[str, Any],
) -> str:
    direct_name = localized_text(
        team.get("name")
    )

    if direct_name:
        return direct_name

    place_name = localized_text(
        team.get("placeName")
    )

    common_name = localized_text(
        team.get("commonName")
    )

    if place_name and common_name:
        return (
            f"{place_name} {common_name}"
        ).strip()

    if place_name:
        return place_name

    if common_name:
        return common_name

    return str(
        team.get(
            "abbrev",
            "",
        )
    ).strip()


def official_team_payloads(
    game: dict[str, Any],
    game_id: str,
    context: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
]:
    away_team_data = game.get(
        "awayTeam",
        {},
    )
    home_team_data = game.get(
        "homeTeam",
        {},
    )

    if not isinstance(
        away_team_data,
        dict,
    ):
        fail(
            f"{context} {game_id} "
            "has invalid awayTeam data"
        )

    if not isinstance(
        home_team_data,
        dict,
    ):
        fail(
            f"{context} {game_id} "
            "has invalid homeTeam data"
        )

    return (
        away_team_data,
        home_team_data,
        official_team_name(away_team_data),
        official_team_name(home_team_data),
    )


def fetch_official_score_payload(
    game_date: str,
) -> dict[str, Any]:
    api_date = api_date_from_game_date(
        game_date
    )

    url = NHL_SCORE_URL.format(
        date=api_date
    )

    log(
        f"Fetching official NHL scores: {url}"
    )

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "NHL_for_mat/transform_final_scores",
                "Accept": "application/json",
            },
        )

    except Exception as e:
        return fail(
            f"Official NHL API request failed "
            f"for {game_date}: {e}"
        )

    if response.status_code != 200:
        fail(
            "Official NHL API returned "
            f"HTTP {response.status_code} "
            f"for {game_date}"
        )

    try:
        payload = response.json()

    except Exception as e:
        return fail(
            f"Official NHL API returned invalid JSON "
            f"for {game_date}: {e}"
        )

    if not isinstance(payload, dict):
        fail(
            "Official NHL API payload was not an object "
            f"for {game_date}"
        )

    return payload


def is_official_final(
    game: dict[str, Any],
) -> bool:
    game_state = str(
        game.get(
            "gameState",
            "",
        )
    ).strip().upper()

    return (
        game_state
        in FINAL_GAME_STATES
    )


def build_official_status_rows(
    game_date: str,
    payload: dict[str, Any],
) -> pd.DataFrame:
    games = payload.get(
        "games",
        [],
    )

    if not isinstance(games, list):
        fail(
            f"Official NHL API games field was not a list "
            f"for {game_date}"
        )

    observed_at = datetime.now(UTC).isoformat()

    rows: list[
        dict[str, Any]
    ] = []

    for game in games:
        if not isinstance(game, dict):
            continue

        official_date = normalize_game_date(
            game.get("gameDate")
        )

        if (
            official_date
            and official_date != game_date
        ):
            continue

        game_id = str(
            game.get(
                "id",
                "",
            )
        ).strip()

        if not GAME_ID_RE.fullmatch(game_id):
            fail(
                "Official NHL game has invalid canonical game_id: "
                f"date={game_date} | game_id={game_id}"
            )

        game_state = str(
            game.get(
                "gameState",
                "",
            )
        ).strip().upper()

        if not game_state:
            fail(
                "Official NHL game is missing gameState: "
                f"game_id={game_id}"
            )

        game_schedule_state = str(
            game.get(
                "gameScheduleState",
                "",
            )
        ).strip().upper()

        (
            away_team_data,
            home_team_data,
            away_team,
            home_team,
        ) = official_team_payloads(
            game,
            game_id,
            'Official NHL game',
        )

        if (
            not away_team
            or not home_team
        ):
            fail(
                "Official NHL game is missing team names: "
                f"game_id={game_id}"
            )

        rows.append(
            {
                "sport": SPORT,
                "league": LEAGUE,
                "game_date": game_date,
                "game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
                "game_state": game_state,
                "game_schedule_state": game_schedule_state,
                "is_final": game_state in FINAL_GAME_STATES,
                "status_observed_at": observed_at,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=STATUS_COLUMNS,
    )

    if df.empty:
        return df

    duplicate_ids = df.duplicated(
        subset=[
            "game_id",
        ],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            df.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        fail(
            "Duplicate official NHL status game_id values: "
            f"{duplicate_values}"
        )

    return df


def build_official_final_rows(
    game_date: str,
    payload: dict[str, Any],
) -> pd.DataFrame:
    games = payload.get(
        "games",
        [],
    )

    if not isinstance(games, list):
        fail(
            f"Official NHL API games field was not a list "
            f"for {game_date}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for game in games:
        if not isinstance(game, dict):
            continue

        official_date = normalize_game_date(
            game.get("gameDate")
        )

        if (
            official_date
            and official_date != game_date
        ):
            continue

        if not is_official_final(game):
            continue

        game_id = str(
            game.get(
                "id",
                "",
            )
        ).strip()

        if not GAME_ID_RE.fullmatch(game_id):
            fail(
                "Official final NHL game has invalid canonical game_id: "
                f"date={game_date} | game_id={game_id}"
            )

        (
            away_team_data,
            home_team_data,
            away_team,
            home_team,
        ) = official_team_payloads(
            game,
            game_id,
            'Official final game',
        )

        away_score = parse_int_score(
            away_team_data.get("score")
        )

        home_score = parse_int_score(
            home_team_data.get("score")
        )

        if (
            not away_team
            or not home_team
        ):
            fail(
                "Official final NHL game is missing team names: "
                f"game_id={game_id}"
            )

        if (
            away_score is None
            or home_score is None
        ):
            fail(
                "Official final NHL game is missing final score: "
                f"game_id={game_id}"
            )

        rows.append(
            {
                "sport": SPORT,
                "league": LEAGUE,
                "game_date": game_date,
                "game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
                "away_score": away_score,
                "home_score": home_score,
                "total_score": (
                    away_score
                    + home_score
                ),
                "away_puck_line_result": (
                    away_score
                    - home_score
                ),
                "home_puck_line_result": (
                    home_score
                    - away_score
                ),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if df.empty:
        return df

    blank_ids = (
        df["game_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_ids.any():
        fail(
            "Blank official NHL game_id "
            "reached final-score dataset"
        )

    duplicate_ids = df.duplicated(
        subset=[
            "game_id",
        ],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            df.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        fail(
            "Duplicate official NHL game_id values: "
            f"{duplicate_values}"
        )

    duplicate_matchups = df.duplicated(
        subset=[
            "game_date",
            "away_team",
            "home_team",
        ],
        keep=False,
    )

    if duplicate_matchups.any():
        fail(
            "Duplicate official NHL date/team matchups "
            f"for {game_date}"
        )

    return df


# ============================================================
# D-RATINGS RECONCILIATION CHECK
# ============================================================

def compare_with_dratings(
    official_df: pd.DataFrame,
    dratings_checks: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ],
) -> None:
    if official_df.empty:
        return

    for _, row in official_df.iterrows():
        game_date = str(
            row["game_date"]
        ).strip()

        away_team = str(
            row["away_team"]
        ).strip()

        home_team = str(
            row["home_team"]
        ).strip()

        away_score = int(
            row["away_score"]
        )

        home_score = int(
            row["home_score"]
        )

        exact_key = (
            game_date,
            norm_key(away_team),
            norm_key(home_team),
        )

        reverse_key = (
            game_date,
            norm_key(home_team),
            norm_key(away_team),
        )

        exact_matches = dratings_checks.get(
            exact_key,
            [],
        )

        reverse_matches = dratings_checks.get(
            reverse_key,
            [],
        )

        if (
            len(exact_matches) == 1
            and len(reverse_matches) == 0
        ):
            check = exact_matches[0]

            drat_away_score = int(
                check["away_score"]
            )

            drat_home_score = int(
                check["home_score"]
            )

        elif (
            len(exact_matches) == 0
            and len(reverse_matches) == 1
        ):
            check = reverse_matches[0]

            drat_away_score = int(
                check["home_score"]
            )

            drat_home_score = int(
                check["away_score"]
            )

            log(
                "DRAT CHECK ORIENTATION REVERSED: "
                f"{game_date} | "
                f"{away_team} at {home_team}"
            )

        elif (
            len(exact_matches) == 0
            and len(reverse_matches) == 0
        ):
            log(
                "DRAT CHECK MISSING: "
                f"{game_date} | "
                f"{away_team} at {home_team} "
                f"| official_game_id={row['game_id']}"
            )

            continue

        else:
            log(
                "DRAT CHECK AMBIGUOUS: "
                f"{game_date} | "
                f"{away_team} at {home_team} "
                f"| exact={len(exact_matches)} "
                f"| reverse={len(reverse_matches)}"
            )

            continue

        if (
            drat_away_score == away_score
            and drat_home_score == home_score
        ):
            log(
                "DRAT CHECK MATCH: "
                f"{game_date} | "
                f"{away_team} {away_score} at "
                f"{home_team} {home_score} "
                f"| official_game_id={row['game_id']}"
            )

        else:
            log(
                "DRAT CHECK SCORE MISMATCH: "
                f"{game_date} | "
                f"{away_team} at {home_team} "
                f"| NHL={away_score}-{home_score} "
                f"| DRAT={drat_away_score}-{drat_home_score} "
                f"| official_game_id={row['game_id']}"
            )


# ============================================================
# OUTPUT
# ============================================================

def normalize_output_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[
        OUTPUT_COLUMNS
    ]

    return df.fillna("")


def write_status_snapshot(
    status_parts: list[pd.DataFrame],
) -> int:
    non_empty_parts = [
        df
        for df in status_parts
        if not df.empty
    ]

    if non_empty_parts:
        df = pd.concat(
            non_empty_parts,
            ignore_index=True,
        )
    else:
        df = pd.DataFrame(
            columns=STATUS_COLUMNS,
        )

    if not df.empty:
        df = df[STATUS_COLUMNS].copy()
        df["game_date"] = df["game_date"].map(
            normalize_game_date
        )
        df["game_id"] = (
            df["game_id"]
            .astype(str)
            .str.strip()
        )

        validate_game_ids(
            df,
            label="status snapshot before write",
        )

        duplicate_ids = df.duplicated(
            subset=[
                "game_id",
            ],
            keep=False,
        )

        if duplicate_ids.any():
            duplicate_values = sorted(
                df.loc[
                    duplicate_ids,
                    "game_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            fail(
                "Duplicate official NHL game_id values "
                "in status snapshot: "
                f"{duplicate_values}"
            )

        df = df.sort_values(
            [
                "game_date",
                "game_id",
                "away_team",
                "home_team",
            ],
            kind="stable",
        )

    atomic_write_csv(
        df,
        STATUS_FILE,
    )

    log(
        "WROTE OFFICIAL NHL GAME STATUS: "
        f"{STATUS_FILE} | rows={len(df)}"
    )

    return len(df)


def write_official_scores_for_date(
    game_date: str,
    df: pd.DataFrame,
) -> int:
    if df.empty:
        log(
            f"No officially final NHL games for {game_date}"
        )

        return 0

    df = normalize_output_df(
        df
    )

    if (
        df["game_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        fail(
            "Refusing to write final-score file "
            f"with blank game_id for {game_date}"
        )

    validate_game_ids(
        df,
        label=f"final scores for {game_date}",
    )

    duplicate_ids = df.duplicated(
        subset=["game_id"],
        keep=False,
    )

    if duplicate_ids.any():
        duplicate_values = sorted(
            df.loc[
                duplicate_ids,
                "game_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        fail(
            f"Refusing to write duplicate final-score game_id values "
            f"for {game_date}: {duplicate_values}"
        )

    df = df.sort_values(
        [
            "game_date",
            "game_id",
            "away_team",
            "home_team",
        ],
        kind="stable",
    )

    output_path = final_score_path(
        game_date
    )

    atomic_write_csv(
        df,
        output_path,
    )

    log(
        "WROTE OFFICIAL FINAL SCORES: "
        f"{output_path} "
        f"| rows={len(df)}"
    )

    return len(df)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ensure_dirs()
    reset_log()

    log(
        "=== transform_final_scores START ==="
    )

    log(
        f"TARGET_DATE_SOURCE={SELECT_DIR}"
    )

    log(
        f"DRAT_CHECK_DIR={RAW_DIR}"
    )

    log(
        f"OUT_DIR={OUT_DIR}"
    )

    log(
        f"STATUS_FILE={STATUS_FILE}"
    )

    target_dates = discover_target_dates()

    log(
        f"Stage 04 target dates discovered: "
        f"{len(target_dates)}"
    )

    today_game_date = current_game_date()
    log(
        f"CURRENT_GAME_DATE_ET={today_game_date}"
    )

    existing_status = load_existing_status_snapshot()
    status_by_date = split_status_history_by_date(
        existing_status
    )

    if RAW_DIR.exists():
        raw_files = sorted(
            RAW_DIR.glob(
                RAW_PATTERN
            )
        )

    else:
        raw_files = []

        log(
            "DRAT CHECK SOURCE UNAVAILABLE: "
            f"{RAW_DIR}"
        )

    if raw_files:
        dratings_checks = collect_dratings_checks(
            raw_files
        )

    else:
        dratings_checks = {}

        log(
            "No D-Ratings check files found; "
            "official NHL final-score processing will continue"
        )

    total_official_games = 0
    files_written = 0
    queried_dates = 0
    skipped_completed_dates = 0
    preserved_existing_rows = 0

    for game_date in sorted(
        target_dates
    ):
        if completed_historical_date(
            game_date,
            status_by_date,
            today_game_date,
        ):
            existing_df = load_existing_final_scores_for_date(
                game_date
            )
            preserved_existing_rows += len(
                existing_df
            )
            skipped_completed_dates += 1

            log(
                "INCREMENTAL SKIP: completed historical date "
                f"{game_date} | preserved_final_rows={len(existing_df)}"
            )
            continue

        queried_dates += 1

        log(
            f"Processing official NHL final scores "
            f"for {game_date}"
        )

        payload = fetch_official_score_payload(
            game_date
        )

        status_df = build_official_status_rows(
            game_date,
            payload,
        )

        if not status_df.empty:
            status_by_date[game_date] = status_df
        elif game_date in status_by_date:
            log(
                "INCREMENTAL STATUS PRESERVE: fetched status set was empty; "
                f"keeping existing status history for {game_date}"
            )
        else:
            log(
                "No official NHL status rows returned for "
                f"{game_date}"
            )

        fetched_official_df = build_official_final_rows(
            game_date,
            payload,
        )

        compare_with_dratings(
            fetched_official_df,
            dratings_checks,
        )

        existing_official_df = load_existing_final_scores_for_date(
            game_date
        )

        merged_official_df = merge_official_final_history(
            game_date,
            existing_official_df,
            fetched_official_df,
        )

        preserved_existing_rows += max(
            0,
            len(merged_official_df)
            - len(fetched_official_df),
        )

        rows_written = write_official_scores_for_date(
            game_date,
            merged_official_df,
        )

        total_official_games += (
            rows_written
        )

        if rows_written > 0:
            files_written += 1

    status_parts = [
        status_by_date[game_date]
        for game_date in sorted(
            status_by_date
        )
        if not status_by_date[game_date].empty
    ]

    status_rows_written = write_status_snapshot(
        status_parts
    )

    log(
        f"Stage 04 target dates considered: "
        f"{len(target_dates)}"
    )

    log(
        f"Official NHL dates queried: "
        f"{queried_dates}"
    )

    log(
        f"Completed historical dates skipped: "
        f"{skipped_completed_dates}"
    )

    log(
        f"Existing final-score rows preserved without refetch/replacement: "
        f"{preserved_existing_rows}"
    )

    log(
        f"Official NHL status rows written: "
        f"{status_rows_written}"
    )

    log(
        f"D-Ratings check files processed: "
        f"{len(raw_files)}"
    )

    log(
        f"Official final-score rows written on queried dates: "
        f"{total_official_games}"
    )

    log(
        f"Official final-score files written on queried dates: "
        f"{files_written}"
    )

    log(
        "=== transform_final_scores END ==="
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as main_exc:
        ensure_dirs()

        log(
            f"FATAL: {main_exc}\n"
            f"{traceback.format_exc()}"
        )

        print(
            f"transform_final_scores failed: {main_exc}",
            file=sys.stderr,
        )

        raise