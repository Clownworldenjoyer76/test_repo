#!/usr/bin/env python3
"""
Build the local SportsDataverse NHL historical research store.

Output:
    docs/win/hockey/nhl/research/sdv_history/

Season convention
-----------------
Command-line seasons use NHL/project START YEAR semantics.

Examples:
    --season 2019 = 2019-20 NHL season
    --season 2024 = 2024-25 NHL season

SportsDataverse labels NHL seasons by END YEAR, so this script translates:

    project 2019 -> SDV 2020
    project 2024 -> SDV 2025

Leakage policy
--------------
Historical tables are source facts, not target-game pregame features.

Postgame/game-derived information is eligible for a target game only when:

    source_game_date < target_game_date

Timestamped observations must additionally satisfy:

    observed_at_utc < pregame_cutoff_utc

Historical SportsDataverse schedule score fields through 2023 are not treated
as grading ground truth. Use PBP GOAL events or another independently validated
official NHL score source for grading.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


# IMPORTANT:
# This must be set BEFORE Polars or SportsDataverse is imported.
# Some SDV Parquet assets contain R Arrow extension metadata.
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import polars as pl
import sportsdataverse.nhl as nhl


SCRIPT_VERSION = "SDV-P5-2026-08-30-v4"
PINNED_SDV_VERSION = "0.0.75"

REPO_ROOT = Path(__file__).resolve().parents[6]
BASE_DIR = REPO_ROOT / "docs" / "win" / "hockey" / "nhl"

OUTPUT_DIR = BASE_DIR / "research" / "sdv_history"
OFFICIAL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"

GITIGNORE_PATH = REPO_ROOT / ".gitignore"
GITIGNORE_ENTRY = "docs/win/hockey/nhl/research/sdv_history/"

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader_name: str

    # SportsDataverse END-YEAR minimum.
    min_sdv_season: int

    # Whether rows should ultimately carry official NHL game_id.
    game_keyed: bool


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "schedules",
        "load_nhl_schedule",
        2010,
        True,
    ),
    DatasetSpec(
        "play_by_play",
        "load_nhl_pbp",
        2010,
        True,
    ),
    DatasetSpec(
        "team_boxscores",
        "load_nhl_team_boxscores",
        2010,
        True,
    ),
    DatasetSpec(
        "player_boxscores",
        "load_nhl_player_boxscores",
        2010,
        True,
    ),
    DatasetSpec(
        "rosters",
        "load_nhl_rosters",
        2010,
        False,
    ),
    DatasetSpec(
        "game_info",
        "load_nhl_game_info",
        2024,
        True,
    ),
    DatasetSpec(
        "game_rosters",
        "load_nhl_game_rosters",
        2024,
        True,
    ),
    DatasetSpec(
        "goalie_boxscores",
        "load_nhl_goalie_boxscores",
        2024,
        True,
    ),
    DatasetSpec(
        "skater_boxscores",
        "load_nhl_skater_boxscores",
        2024,
        True,
    ),
    DatasetSpec(
        "linescores",
        "load_nhl_linescore",
        2024,
        True,
    ),
    DatasetSpec(
        "penalties",
        "load_nhl_penalties",
        2024,
        True,
    ),
    DatasetSpec(
        "scoring",
        "load_nhl_scoring",
        2024,
        True,
    ),
    DatasetSpec(
        "scratches",
        "load_nhl_scratches",
        2024,
        True,
    ),
    DatasetSpec(
        "shifts",
        "load_nhl_shifts",
        2025,
        True,
    ),
    DatasetSpec(
        "shots_by_period",
        "load_nhl_shots_by_period",
        2025,
        True,
    ),
)

SPEC_BY_NAME = {
    spec.name: spec
    for spec in DATASET_SPECS
}


GAME_ID_ALIASES = (
    "gameId",
    "gameID",
    "gameid",
    "game_pk",
    "gamePk",
)

DATE_ALIASES = (
    "game_date",
    "gameDate",
    "date",
    "game_date_time",
    "start_date",
)

TEAM_ALIASES = (
    "team_abbrev",
    "team_abbr",
    "teamAbbrev",
    "team_abbreviation",
)

HOME_AWAY_ALIASES = (
    "home_away",
    "homeAway",
    "homeaway",
)


# Older SDV boxscore assets may be missing official game_id.
# These are the only datasets for which deterministic recovery is attempted.
LEGACY_GAME_ID_RECOVERY_DATASETS = {
    "team_boxscores",
    "player_boxscores",
}


PBP_REQUIRED_RESEARCH_COLUMNS = {
    "game_id",
    "source_game_date",
    "event_type",
    "xg",
    "x",
    "y",
    "shot_distance",
    "shot_angle",
    "strength_state",
    "home_skaters",
    "away_skaters",
    "event_goalie_id",
    "home_goalie_id",
    "away_goalie_id",
}


LEAKAGE_POLICY = {
    "status": "REQUIRED_ENFORCED",

    "game_id": (
        "Official 10-digit NHL game ID only. "
        "Provider identifiers remain separate."
    ),

    "historical_postgame_source_rule": (
        "source_game_date < target_game_date"
    ),

    "same_day_postgame_sources": "excluded",

    "timestamped_observation_rule": (
        "observed_at_utc < pregame_cutoff_utc"
    ),

    "pregame_cutoff_preference": (
        "official NHL start_time_utc"
    ),

    "pregame_cutoff_fallback": (
        "start of target game date in America/New_York"
    ),

    "target_game_postgame_data": (
        "never eligible for that target game's pregame features"
    ),

    "schedule_score_guardrail": (
        "SportsDataverse schedule score fields through 2023 "
        "must not be used as grading ground truth without "
        "independent validation."
    ),

    "lineup_goalie_guardrail": (
        "Final participation, scratches, starter results, or later "
        "confirmations must not be backfilled into a target game's "
        "pregame feature state."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build local SportsDataverse NHL historical Parquet data."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        action="append",
        help=(
            "NHL season START YEAR. "
            "Example: --season 2024 means 2024-25."
        ),
    )

    parser.add_argument(
        "--start-season",
        type=int,
    )

    parser.add_argument(
        "--end-season",
        type=int,
    )

    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "Comma-separated dataset names or 'all'. "
            "Schedules are always loaded."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing season Parquet files.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat unavailable eligible datasets as fatal."
        ),
    )

    return parser.parse_args()


def sdv_season_year(
    nhl_season_start_year: int,
) -> int:
    """
    Translate NHL/project START YEAR to SportsDataverse END YEAR.

    2019 -> 2020
    2024 -> 2025
    """
    return nhl_season_start_year + 1


def requested_seasons(
    args: argparse.Namespace,
) -> list[int]:
    seasons: list[int] = []

    if args.season:
        seasons.extend(args.season)

    if (
        args.start_season is not None
        or args.end_season is not None
    ):
        if (
            args.start_season is None
            or args.end_season is None
        ):
            raise SystemExit(
                "--start-season and --end-season must be used together"
            )

        if args.end_season < args.start_season:
            raise SystemExit(
                "--end-season cannot be less than --start-season"
            )

        seasons.extend(
            range(
                args.start_season,
                args.end_season + 1,
            )
        )

    seasons = sorted(
        set(seasons)
    )

    if not seasons:
        raise SystemExit(
            "Specify --season or "
            "--start-season/--end-season"
        )

    # SDV starts at end-year 2010 = NHL 2009-10.
    invalid = [
        season
        for season in seasons
        if season < 2009
    ]

    if invalid:
        raise SystemExit(
            f"Unsupported NHL season start year(s): {invalid}"
        )

    return seasons


def requested_datasets(
    args: argparse.Namespace,
) -> set[str]:
    raw = str(
        args.datasets
    ).strip()

    if raw.lower() == "all":
        return set(
            SPEC_BY_NAME
        )

    selected = {
        item.strip()
        for item in raw.split(",")
        if item.strip()
    }

    unknown = sorted(
        selected
        - set(SPEC_BY_NAME)
    )

    if unknown:
        raise SystemExit(
            f"Unknown datasets: {unknown}"
        )

    # Schedule is mandatory because it is the canonical game map.
    selected.add(
        "schedules"
    )

    return selected


def installed_sdv_version() -> str:
    try:
        return version(
            "sportsdataverse"
        )

    except PackageNotFoundError as exc:
        raise SystemExit(
            "sportsdataverse is not installed"
        ) from exc


def enforce_sdv_pin() -> str:
    installed = installed_sdv_version()

    if installed != PINNED_SDV_VERSION:
        raise SystemExit(
            "Refusing to build with "
            f"sportsdataverse=={installed}. "
            f"Required version is {PINNED_SDV_VERSION}."
        )

    return installed


def enforce_gitignored_output() -> None:
    if not GITIGNORE_PATH.exists():
        raise SystemExit(
            f"Missing .gitignore: {GITIGNORE_PATH}"
        )

    entries = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            line.strip()
            and not line.lstrip().startswith("#")
        )
    }

    if GITIGNORE_ENTRY not in entries:
        raise SystemExit(
            "Refusing to materialize SDV history because "
            "the output directory is not Git-ignored. "
            f"Add: {GITIGNORE_ENTRY}"
        )


def repo_relative(
    path: Path,
) -> str:
    try:
        return (
            path.resolve()
            .relative_to(
                REPO_ROOT.resolve()
            )
            .as_posix()
        )

    except ValueError:
        return str(
            path.resolve()
        )


def first_existing_column(
    df: pl.DataFrame,
    names: tuple[str, ...],
) -> str | None:
    columns = set(
        df.columns
    )

    for name in names:
        if name in columns:
            return name

    return None


def normalized_date_expr(
    column: str,
) -> pl.Expr:
    """
    Convert common SDV date/datetime strings to YYYY-MM-DD.
    """
    text = (
        pl.col(column)
        .cast(
            pl.String,
            strict=False,
        )
        .str.strip_chars()
        .str.slice(
            0,
            10,
        )
    )

    return (
        text
        .str.strptime(
            pl.Date,
            "%Y-%m-%d",
            strict=False,
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


def normalized_team_expr(
    column: str,
) -> pl.Expr:
    return (
        pl.col(column)
        .cast(
            pl.String,
            strict=False,
        )
        .str.strip_chars()
        .str.to_uppercase()
    )


def normalized_home_away_expr(
    column: str,
) -> pl.Expr:
    value = (
        pl.col(column)
        .cast(
            pl.String,
            strict=False,
        )
        .str.strip_chars()
        .str.to_lowercase()
    )

    return (
        pl.when(
            value.str.starts_with("h")
        )
        .then(
            pl.lit("home")
        )
        .when(
            value.str.starts_with("a")
        )
        .then(
            pl.lit("away")
        )
        .otherwise(
            None
        )
    )


def canonical_game_id_expr(
    column: str = "game_id",
) -> pl.Expr:
    return (
        pl.col(column)
        .cast(
            pl.Int64,
            strict=False,
        )
        .cast(
            pl.String
        )
        .alias(
            column
        )
    )


def promote_game_id_alias(
    df: pl.DataFrame,
    *,
    season: int,
) -> tuple[
    pl.DataFrame,
    str | None,
]:
    """
    Promote a known alternate game-ID column only if all non-null values
    are valid official 10-digit NHL IDs for the requested season.
    """
    if "game_id" in df.columns:
        return df, None

    for alias in GAME_ID_ALIASES:
        if alias not in df.columns:
            continue

        candidate = (
            df
            .with_columns(
                pl.col(alias)
                .cast(
                    pl.Int64,
                    strict=False,
                )
                .cast(
                    pl.String
                )
                .alias(
                    "_candidate_game_id"
                )
            )
        )

        non_null_count = int(
            candidate.select(
                pl.col(
                    "_candidate_game_id"
                )
                .is_not_null()
                .sum()
            ).item()
        )

        if non_null_count == 0:
            continue

        valid_count = int(
            candidate.select(
                (
                    pl.col(
                        "_candidate_game_id"
                    )
                    .str.contains(
                        r"^\d{10}$"
                    )
                    & (
                        pl.col(
                            "_candidate_game_id"
                        )
                        .str.slice(
                            0,
                            4,
                        )
                        == str(season)
                    )
                )
                .fill_null(False)
                .sum()
            ).item()
        )

        if valid_count != non_null_count:
            continue

        return (
            df.with_columns(
                pl.col(alias)
                .cast(
                    pl.Int64,
                    strict=False,
                )
                .alias(
                    "game_id"
                )
            ),
            f"promoted_alias:{alias}",
        )

    return df, None


def canonicalize_game_id(
    df: pl.DataFrame,
    *,
    season: int,
    dataset: str,
) -> pl.DataFrame:
    if "game_id" not in df.columns:
        raise RuntimeError(
            f"{dataset}: expected game_id column is missing"
        )

    original_non_null = int(
        df.select(
            pl.col(
                "game_id"
            )
            .is_not_null()
            .sum()
        ).item()
    )

    out = df.with_columns(
        canonical_game_id_expr()
    )

    converted_non_null = int(
        out.select(
            pl.col(
                "game_id"
            )
            .is_not_null()
            .sum()
        ).item()
    )

    if (
        converted_non_null
        != original_non_null
    ):
        raise RuntimeError(
            f"{dataset}: one or more non-null game_id "
            "values could not be converted"
        )

    invalid_count = int(
        out.select(
            (
                pl.col(
                    "game_id"
                )
                .is_not_null()
                & ~pl.col(
                    "game_id"
                )
                .str.contains(
                    r"^\d{10}$"
                )
            )
            .fill_null(False)
            .sum()
        ).item()
    )

    if invalid_count:
        raise RuntimeError(
            f"{dataset}: found {invalid_count} "
            "non-canonical game_id values"
        )

    wrong_season_count = int(
        out.select(
            (
                pl.col(
                    "game_id"
                )
                .is_not_null()
                & (
                    pl.col(
                        "game_id"
                    )
                    .str.slice(
                        0,
                        4,
                    )
                    != str(season)
                )
            )
            .fill_null(False)
            .sum()
        ).item()
    )

    if wrong_season_count:
        raise RuntimeError(
            f"{dataset}: found {wrong_season_count} "
            "game_id values outside NHL season "
            f"start year {season}"
        )

    return out


def prepare_schedule(
    schedule: pl.DataFrame,
    season: int,
) -> pl.DataFrame:
    """
    Normalize SDV schedule and establish canonical source_game_date.
    """
    schedule = canonicalize_game_id(
        schedule,
        season=season,
        dataset="schedules",
    )

    if "game_date" not in schedule.columns:
        raise RuntimeError(
            "schedules: required game_date column is missing"
        )

    schedule = (
        schedule
        .with_columns(
            normalized_date_expr(
                "game_date"
            ).alias(
                "source_game_date"
            )
        )
    )

    missing_dates = int(
        schedule.select(
            pl.col(
                "source_game_date"
            )
            .is_null()
            .sum()
        ).item()
    )

    if missing_dates:
        raise RuntimeError(
            "schedules: "
            f"{missing_dates} rows have no parseable game_date"
        )

    duplicate_ids = (
        schedule
        .group_by(
            "game_id"
        )
        .len()
        .filter(
            pl.col(
                "len"
            )
            > 1
        )
        .height
    )

    if duplicate_ids:
        raise RuntimeError(
            "schedules: duplicate official game_id values found"
        )

    return (
        schedule
        .with_columns(
            pl.lit(
                season
            ).alias(
                "nhl_season_start_year"
            ),

            pl.lit(
                sdv_season_year(
                    season
                )
            ).alias(
                "sdv_season_year"
            ),

            pl.lit(
                True
            ).alias(
                "pregame_source_eligible"
            ),
        )
    )


def schedule_date_lookup(
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    return (
        schedule
        .select(
            "game_id",

            pl.col(
                "source_game_date"
            ).alias(
                "_schedule_source_game_date"
            ),
        )
    )


def schedule_team_game_map(
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build deterministic:
        date + team + home/away -> official NHL game_id
    map.
    """
    home_col = first_existing_column(
        schedule,
        (
            "home_team_abbr",
            "home_team_abbrev",
            "home_abbr",
        ),
    )

    away_col = first_existing_column(
        schedule,
        (
            "away_team_abbr",
            "away_team_abbrev",
            "away_abbr",
        ),
    )

    if (
        home_col is None
        or away_col is None
    ):
        raise RuntimeError(
            "schedules: team abbreviation columns "
            "are missing"
        )

    home = schedule.select(
        "game_id",

        pl.col(
            "source_game_date"
        ).alias(
            "_match_date"
        ),

        normalized_team_expr(
            home_col
        ).alias(
            "_match_team"
        ),

        pl.lit(
            "home"
        ).alias(
            "_match_home_away"
        ),
    )

    away = schedule.select(
        "game_id",

        pl.col(
            "source_game_date"
        ).alias(
            "_match_date"
        ),

        normalized_team_expr(
            away_col
        ).alias(
            "_match_team"
        ),

        pl.lit(
            "away"
        ).alias(
            "_match_home_away"
        ),
    )

    return pl.concat(
        [
            home,
            away,
        ],
        how="vertical",
    )


def recover_legacy_game_id(
    df: pl.DataFrame,
    *,
    spec: DatasetSpec,
    season: int,
    schedule: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    str | None,
]:
    """
    Recover official NHL game_id from legacy SportsDataverse assets.

    Recovery is allowed only when deterministic.

    Never infer game IDs from:
    - row number
    - ordering
    - sequential assumptions
    """

    promoted, method = (
        promote_game_id_alias(
            df,
            season=season,
        )
    )

    if "game_id" in promoted.columns:
        return promoted, method

    if (
        spec.name
        not in LEGACY_GAME_ID_RECOVERY_DATASETS
    ):
        return promoted, None

    date_col = first_existing_column(
        promoted,
        DATE_ALIASES,
    )

    team_col = first_existing_column(
        promoted,
        TEAM_ALIASES,
    )

    home_away_col = first_existing_column(
        promoted,
        HOME_AWAY_ALIASES,
    )

    if (
        date_col is None
        or team_col is None
    ):
        return promoted, None

    work = (
        promoted
        .with_columns(
            normalized_date_expr(
                date_col
            ).alias(
                "_match_date"
            ),

            normalized_team_expr(
                team_col
            ).alias(
                "_match_team"
            ),
        )
    )

    mapping = (
        schedule_team_game_map(
            schedule
        )
    )

    join_keys = [
        "_match_date",
        "_match_team",
    ]

    recovery_method = (
        "schedule_date_team"
    )

    if home_away_col is not None:
        work = work.with_columns(
            normalized_home_away_expr(
                home_away_col
            ).alias(
                "_match_home_away"
            )
        )

        null_home_away = int(
            work.select(
                pl.col(
                    "_match_home_away"
                )
                .is_null()
                .sum()
            ).item()
        )

        if null_home_away == 0:
            join_keys.append(
                "_match_home_away"
            )

            recovery_method = (
                "schedule_date_team_homeaway"
            )

    ambiguous = (
        mapping
        .group_by(
            join_keys
        )
        .len()
        .filter(
            pl.col(
                "len"
            )
            > 1
        )
    )

    if ambiguous.height:
        return promoted, None

    unique_map = (
        mapping
        .select(
            join_keys
            + [
                "game_id"
            ]
        )
        .unique()
    )

    recovered = (
        work
        .join(
            unique_map,
            on=join_keys,
            how="left",
        )
    )

    missing_count = int(
        recovered.select(
            pl.col(
                "game_id"
            )
            .is_null()
            .sum()
        ).item()
    )

    if missing_count:
        return promoted, None

    cleanup = [
        column
        for column in (
            "_match_date",
            "_match_team",
            "_match_home_away",
        )
        if column in recovered.columns
    ]

    if cleanup:
        recovered = (
            recovered
            .drop(
                cleanup
            )
        )

    return (
        recovered,
        recovery_method,
    )


def attach_source_dates(
    df: pl.DataFrame,
    *,
    spec: DatasetSpec,
    season: int,
    schedule: pl.DataFrame,
    schedule_dates: pl.DataFrame,
) -> tuple[
    pl.DataFrame,
    int,
    str | None,
]:
    """
    Add leakage-safe source date metadata.

    For game-keyed datasets:
        source_game_date comes primarily from canonical schedule using game_id.

    This specifically fixes legacy PBP assets that have game_id but no game_date.
    """

    if spec.game_keyed:
        recovered, recovery_method = (
            recover_legacy_game_id(
                df,
                spec=spec,
                season=season,
                schedule=schedule,
            )
        )

        out = canonicalize_game_id(
            recovered,
            season=season,
            dataset=spec.name,
        )

        out = out.join(
            schedule_dates,
            on="game_id",
            how="left",
        )

        if "game_date" in out.columns:
            out = out.with_columns(
                pl.coalesce(
                    pl.col(
                        "_schedule_source_game_date"
                    ),

                    normalized_date_expr(
                        "game_date"
                    ),
                ).alias(
                    "source_game_date"
                )
            )

        else:
            # Legacy PBP path:
            # no game_date required in source asset.
            # Recover it using official game_id -> schedule date.
            out = out.with_columns(
                pl.col(
                    "_schedule_source_game_date"
                ).alias(
                    "source_game_date"
                )
            )

        out = out.drop(
            "_schedule_source_game_date"
        )

        out = out.with_columns(
            pl.col(
                "source_game_date"
            )
            .is_not_null()
            .alias(
                "pregame_source_eligible"
            ),

            pl.lit(
                season
            ).alias(
                "nhl_season_start_year"
            ),

            pl.lit(
                sdv_season_year(
                    season
                )
            ).alias(
                "sdv_season_year"
            ),
        )

        ineligible_count = int(
            out.select(
                (
                    ~pl.col(
                        "pregame_source_eligible"
                    )
                )
                .fill_null(True)
                .sum()
            ).item()
        )

        return (
            out,
            ineligible_count,
            recovery_method,
        )

    # Non-game-keyed seasonal/daily table such as rosters.
    out = df

    if "game_date" in out.columns:
        out = out.with_columns(
            normalized_date_expr(
                "game_date"
            ).alias(
                "source_observed_date"
            )
        )

        out = out.with_columns(
            pl.col(
                "source_observed_date"
            )
            .is_not_null()
            .alias(
                "pregame_source_eligible"
            )
        )

        ineligible_count = int(
            out.select(
                (
                    ~pl.col(
                        "pregame_source_eligible"
                    )
                )
                .fill_null(True)
                .sum()
            ).item()
        )

    else:
        out = out.with_columns(
            pl.lit(
                False
            ).alias(
                "pregame_source_eligible"
            )
        )

        ineligible_count = out.height

    out = out.with_columns(
        pl.lit(
            season
        ).alias(
            "nhl_season_start_year"
        ),

        pl.lit(
            sdv_season_year(
                season
            )
        ).alias(
            "sdv_season_year"
        ),
    )

    return (
        out,
        ineligible_count,
        None,
    )


def validate_pbp_research_columns(
    df: pl.DataFrame,
) -> None:
    """
    Validate the columns needed for SDV-P5 research.

    IMPORTANT:
    We require source_game_date, not a raw game_date column.
    Older SDV PBP assets can legitimately lack game_date.
    """

    missing = sorted(
        PBP_REQUIRED_RESEARCH_COLUMNS
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "play_by_play is missing required "
            "SDV-P5 research columns: "
            + ", ".join(
                missing
            )
        )


def parse_utc_timestamp(
    value: Any,
) -> datetime | None:
    text = str(
        value or ""
    ).strip()

    if not text:
        return None

    try:
        parsed = (
            datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )
        )

    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=UTC
        )

    return parsed.astimezone(
        UTC
    )


def read_official_start_times() -> dict[
    str,
    datetime,
]:
    lookup: dict[
        str,
        datetime,
    ] = {}

    if not OFFICIAL_SCHEDULE_DIR.exists():
        return lookup

    for path in sorted(
        OFFICIAL_SCHEDULE_DIR.glob(
            "*.csv"
        )
    ):
        try:
            with path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as handle:

                reader = csv.DictReader(
                    handle
                )

                for row in reader:
                    game_id = str(
                        row.get(
                            "game_id",
                            "",
                        )
                    ).strip()

                    if (
                        not game_id.isdigit()
                        or len(game_id) != 10
                    ):
                        continue

                    start = parse_utc_timestamp(
                        row.get(
                            "start_time_utc",
                            "",
                        )
                    )

                    if start is not None:
                        lookup[
                            game_id
                        ] = start

        except (
            OSError,
            csv.Error,
        ):
            continue

    return lookup


def utc_iso(
    value: datetime,
) -> str:
    return (
        value.astimezone(
            UTC
        )
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )


def build_pregame_index(
    schedule: pl.DataFrame,
    official_start_times: dict[
        str,
        datetime,
    ],
) -> pl.DataFrame:

    candidate_columns = (
        "game_id",
        "season",
        "nhl_season_start_year",
        "sdv_season_year",
        "game_type",
        "source_game_date",
        "home_team_abbr",
        "home_team_abbrev",
        "away_team_abbr",
        "away_team_abbrev",
    )

    selected_columns = [
        column
        for column in candidate_columns
        if column in schedule.columns
    ]

    base = schedule.select(
        selected_columns
    )

    rows: list[
        dict[str, Any]
    ] = []

    for row in base.iter_rows(
        named=True
    ):
        game_id = str(
            row["game_id"]
        )

        game_day = date.fromisoformat(
            str(
                row[
                    "source_game_date"
                ]
            )
        )

        cutoff = (
            official_start_times.get(
                game_id
            )
        )

        if cutoff is not None:
            cutoff_source = (
                "official_nhl_schedule:start_time_utc"
            )

        else:
            cutoff = (
                datetime.combine(
                    game_day,
                    dt_time.min,
                    tzinfo=NY,
                )
                .astimezone(
                    UTC
                )
            )

            cutoff_source = (
                "conservative_game_date_start_et"
            )

        output_row = dict(
            row
        )

        output_row[
            "game_date"
        ] = output_row.pop(
            "source_game_date"
        )

        output_row[
            "pregame_cutoff_utc"
        ] = utc_iso(
            cutoff
        )

        output_row[
            "pregame_cutoff_source"
        ] = cutoff_source

        output_row[
            "historical_postgame_source_rule"
        ] = (
            "source_game_date < target_game_date"
        )

        output_row[
            "timestamped_observation_rule"
        ] = (
            "observed_at_utc < pregame_cutoff_utc"
        )

        rows.append(
            output_row
        )

    if not rows:
        return pl.DataFrame()

    return (
        pl.DataFrame(
            rows
        )
        .sort(
            [
                "game_date",
                "game_id",
            ]
        )
    )


def dataset_path(
    spec: DatasetSpec,
    season: int,
) -> Path:
    return (
        OUTPUT_DIR
        / spec.name
        / f"season_{season}.parquet"
    )


def pregame_index_path(
    season: int,
) -> Path:
    return (
        OUTPUT_DIR
        / "pregame_index"
        / f"season_{season}.parquet"
    )


def write_parquet_atomic(
    df: pl.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    df.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
    )

    temporary.replace(
        path
    )


def write_json_atomic(
    obj: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    temporary.write_text(
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def load_sdv_frame(
    spec: DatasetSpec,
    season: int,
) -> pl.DataFrame:
    loader = getattr(
        nhl,
        spec.loader_name,
        None,
    )

    if loader is None:
        raise RuntimeError(
            f"SportsDataverse {PINNED_SDV_VERSION} "
            f"does not provide {spec.loader_name}"
        )

    requested_sdv_season = (
        sdv_season_year(
            season
        )
    )

    frame = loader(
        seasons=requested_sdv_season
    )

    if not isinstance(
        frame,
        pl.DataFrame,
    ):
        raise TypeError(
            f"{spec.loader_name} returned "
            f"{type(frame).__name__}; "
            "expected polars.DataFrame"
        )

    return frame


def result_entry(
    *,
    spec: DatasetSpec,
    season: int,
    status: str,
    path: Path | None = None,
    rows: int | None = None,
    columns: int | None = None,
    ineligible_rows: int | None = None,
    game_id_recovery: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:

    entry: dict[
        str,
        Any,
    ] = {
        "dataset": spec.name,
        "loader": spec.loader_name,

        "nhl_season_start_year": season,

        "sdv_season_year": (
            sdv_season_year(
                season
            )
        ),

        "minimum_supported_sdv_season": (
            spec.min_sdv_season
        ),

        "minimum_supported_nhl_start_season": (
            spec.min_sdv_season - 1
        ),

        "status": status,
    }

    if path is not None:
        entry[
            "path"
        ] = repo_relative(
            path
        )

    if rows is not None:
        entry[
            "rows"
        ] = rows

    if columns is not None:
        entry[
            "columns"
        ] = columns

    if ineligible_rows is not None:
        entry[
            "pregame_source_ineligible_rows"
        ] = ineligible_rows

    if game_id_recovery:
        entry[
            "game_id_recovery"
        ] = game_id_recovery

    if message:
        entry[
            "message"
        ] = message

    return entry


def materialize_schedule(
    *,
    season: int,
    overwrite: bool,
) -> tuple[
    pl.DataFrame,
    dict[str, Any],
]:
    spec = SPEC_BY_NAME[
        "schedules"
    ]

    path = dataset_path(
        spec,
        season,
    )

    if (
        path.exists()
        and not overwrite
    ):
        frame = pl.read_parquet(
            path
        )

        frame = prepare_schedule(
            frame,
            season,
        )

        return (
            frame,
            result_entry(
                spec=spec,
                season=season,
                status="existing_validated",
                path=path,
                rows=frame.height,
                columns=frame.width,
                ineligible_rows=0,
            ),
        )

    frame = load_sdv_frame(
        spec,
        season,
    )

    if frame.is_empty():
        raise RuntimeError(
            "SportsDataverse returned no schedule rows "
            f"for NHL {season}-{str(season + 1)[-2:]} "
            f"(SDV season {sdv_season_year(season)})"
        )

    frame = prepare_schedule(
        frame,
        season,
    )

    write_parquet_atomic(
        frame,
        path,
    )

    return (
        frame,
        result_entry(
            spec=spec,
            season=season,
            status="written",
            path=path,
            rows=frame.height,
            columns=frame.width,
            ineligible_rows=0,
        ),
    )


def materialize_dataset(
    *,
    spec: DatasetSpec,
    season: int,
    schedule: pl.DataFrame,
    schedule_dates: pl.DataFrame,
    overwrite: bool,
) -> dict[str, Any]:

    path = dataset_path(
        spec,
        season,
    )

    requested_sdv_season = (
        sdv_season_year(
            season
        )
    )

    if (
        requested_sdv_season
        < spec.min_sdv_season
    ):
        return result_entry(
            spec=spec,
            season=season,
            status=(
                "unavailable_before_minimum_season"
            ),
        )

    if (
        path.exists()
        and not overwrite
    ):
        return result_entry(
            spec=spec,
            season=season,
            status="skipped_existing",
            path=path,
        )

    frame = load_sdv_frame(
        spec,
        season,
    )

    if frame.is_empty():
        return result_entry(
            spec=spec,
            season=season,
            status="empty_or_unpublished",
        )

    probe_method: str | None = None

    if spec.game_keyed:
        probe, probe_method = (
            recover_legacy_game_id(
                frame,
                spec=spec,
                season=season,
                schedule=schedule,
            )
        )

        # This is deliberate.
        # Old SDV boxscores that cannot be safely keyed are not fatal.
        if "game_id" not in probe.columns:
            return result_entry(
                spec=spec,
                season=season,
                status="legacy_unkeyed_unavailable",
                rows=frame.height,
                columns=frame.width,
                message=(
                    "Legacy SportsDataverse asset has no "
                    "deterministically recoverable official game_id. "
                    "Dataset was not written. "
                    "No row-order ID inference was attempted. "
                    "Available columns: "
                    + ", ".join(
                        frame.columns
                    )
                ),
            )

        frame = probe

    frame, ineligible_rows, recovery_method = (
        attach_source_dates(
            frame,
            spec=spec,
            season=season,
            schedule=schedule,
            schedule_dates=schedule_dates,
        )
    )

    if spec.name == "play_by_play":
        validate_pbp_research_columns(
            frame
        )

    write_parquet_atomic(
        frame,
        path,
    )

    return result_entry(
        spec=spec,
        season=season,
        status="written",
        path=path,
        rows=frame.height,
        columns=frame.width,
        ineligible_rows=ineligible_rows,
        game_id_recovery=(
            recovery_method
            or probe_method
        ),
    )


def main() -> int:
    args = parse_args()

    seasons = requested_seasons(
        args
    )

    selected = requested_datasets(
        args
    )

    sdv_version = enforce_sdv_pin()

    enforce_gitignored_output()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"BUILD_SDV_HISTORY_VERSION | {SCRIPT_VERSION}"
    )

    print(
        f"SPORTSDATAVERSE_VERSION | {sdv_version}"
    )

    print(
        "SEASON_CONVENTION | "
        "CLI=NHL_START_YEAR | "
        "SDV=END_YEAR"
    )

    write_json_atomic(
        LEAKAGE_POLICY,
        OUTPUT_DIR
        / "leakage_policy.json",
    )

    official_start_times = (
        read_official_start_times()
    )

    run_started = datetime.now(
        UTC
    )

    results: list[
        dict[str, Any]
    ] = []

    errors: list[
        str
    ] = []

    strict_failures: list[
        str
    ] = []

    for season in seasons:
        print(
            f"SEASON {season} "
            f"| SDV {sdv_season_year(season)}"
        )

        try:
            schedule, schedule_result = (
                materialize_schedule(
                    season=season,
                    overwrite=args.overwrite,
                )
            )

            results.append(
                schedule_result
            )

            print(
                "  schedules: "
                f"{schedule_result['status']}"
            )

        except Exception as exc:
            message = (
                f"season {season} schedules: {exc}"
            )

            errors.append(
                message
            )

            print(
                f"ERROR | {message}",
                file=sys.stderr,
            )

            continue

        schedule_dates = (
            schedule_date_lookup(
                schedule
            )
        )

        pregame_index = (
            build_pregame_index(
                schedule,
                official_start_times,
            )
        )

        if pregame_index.is_empty():
            message = (
                f"season {season}: "
                "failed to build pregame index"
            )

            errors.append(
                message
            )

            print(
                f"ERROR | {message}",
                file=sys.stderr,
            )

            continue

        index_path = (
            pregame_index_path(
                season
            )
        )

        write_parquet_atomic(
            pregame_index,
            index_path,
        )

        print(
            "  pregame_index: written "
            f"({pregame_index.height} games)"
        )

        for spec in DATASET_SPECS:
            if spec.name == "schedules":
                continue

            if spec.name not in selected:
                continue

            try:
                entry = materialize_dataset(
                    spec=spec,
                    season=season,
                    schedule=schedule,
                    schedule_dates=schedule_dates,
                    overwrite=args.overwrite,
                )

                results.append(
                    entry
                )

                print(
                    f"  {spec.name}: "
                    f"{entry['status']}"
                )

                if (
                    args.strict
                    and (
                        sdv_season_year(
                            season
                        )
                        >= spec.min_sdv_season
                    )
                    and entry[
                        "status"
                    ]
                    in {
                        "empty_or_unpublished",
                        "legacy_unkeyed_unavailable",
                    }
                ):
                    strict_failures.append(
                        f"season {season} "
                        f"{spec.name}: "
                        f"{entry['status']}"
                    )

            except Exception as exc:
                message = (
                    f"season {season} "
                    f"{spec.name}: {exc}"
                )

                errors.append(
                    message
                )

                results.append(
                    result_entry(
                        spec=spec,
                        season=season,
                        status="error",
                        message=str(exc),
                    )
                )

                print(
                    f"ERROR | {message}",
                    file=sys.stderr,
                )

    run_finished = datetime.now(
        UTC
    )

    manifest = {
        "script_version": SCRIPT_VERSION,

        "builder": (
            "docs/win/hockey/nhl/scripts/"
            "research/build_sdv_history.py"
        ),

        "output_root": repo_relative(
            OUTPUT_DIR
        ),

        "storage_format": "parquet",

        "compression": "zstd",

        "git_tracked": False,

        "required_gitignore_entry": (
            GITIGNORE_ENTRY
        ),

        "sportsdataverse_version": (
            sdv_version
        ),

        "nhl_season_start_years": (
            seasons
        ),

        "sportsdataverse_season_years": [
            sdv_season_year(
                season
            )
            for season in seasons
        ],

        "datasets_requested": sorted(
            selected
        ),

        "run_started_utc": utc_iso(
            run_started
        ),

        "run_finished_utc": utc_iso(
            run_finished
        ),

        "official_start_time_rows_available": (
            len(
                official_start_times
            )
        ),

        "pbp_required_research_columns": (
            sorted(
                PBP_REQUIRED_RESEARCH_COLUMNS
            )
        ),

        "leakage_policy": (
            LEAKAGE_POLICY
        ),

        "results": (
            results
        ),

        "errors": (
            errors
        ),

        "strict_failures": (
            strict_failures
        ),
    }

    stamp = (
        run_finished.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    write_json_atomic(
        manifest,
        OUTPUT_DIR
        / "manifests"
        / f"run_{stamp}.json",
    )

    write_json_atomic(
        manifest,
        OUTPUT_DIR
        / "manifest_latest.json",
    )

    if errors:
        print(
            f"FAILED | "
            f"{len(errors)} loader/build error(s)",
            file=sys.stderr,
        )

        return 1

    if strict_failures:
        print(
            "FAILED | "
            f"{len(strict_failures)} "
            "strict unavailable dataset(s)",
            file=sys.stderr,
        )

        return 1

    print(
        "DONE | "
        f"{repo_relative(OUTPUT_DIR)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
