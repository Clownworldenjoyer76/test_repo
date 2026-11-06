#!/usr/bin/env python3
"""
SDV-P9 — NHL EDGE and microstat research evaluator.

Research-only. No production pipeline outputs are modified.

What this script does
---------------------
1. Audits local P5 historical play-by-play coverage for:
   - shot characteristics
   - shot location
   - goalie identity fields
   - zone-transition prerequisites
2. Probes SportsDataverse 0.0.75 NHL EDGE endpoints on a small deterministic
   historical sample to establish which seasons/endpoints currently respond.
3. Explicitly blocks same-season EDGE aggregate features from historical
   pregame promotion when an observation timestamp cannot be reconstructed.
4. Builds strict prior-date team microstats from PBP:
       source_game_date < target_game_date
5. Runs season-walk-forward out-of-sample tests for:
   - baseline SportsDataverse predictions
   - baseline + shot/location microstats
   - baseline + zone-transition microstats, when reconstructable
   - baseline + all safe microstats
6. Writes a research summary and detailed audit CSVs.

Season convention
-----------------
CLI seasons use NHL/project START YEAR semantics.

    --season 2024 = 2024-25 NHL season
    SportsDataverse season argument = 2025

Historical score caveat
-----------------------
Schedule score columns are NOT used for outcomes. Realized goals are obtained
from SportsDataverse team_game_xg_rates, which derives goals from PBP GOAL
events and therefore avoids the historical schedule placeholder-score caveat.

Outputs
-------
docs/win/hockey/nhl/research/sdv_edge_microstats/
    coverage_by_season.csv
    edge_endpoint_probe.csv
    game_feature_matrix.csv
    oos_comparison.csv
    oos_metrics.csv
    summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

# Must be set before Polars / SportsDataverse import.
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
import sportsdataverse.nhl as nhl


RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

# noinspection PyPep8
from sdv_ratings_common import prior_adjusted_ratings


SCRIPT_VERSION = "SDV-P9-2026-08-30-v1"
PINNED_SDV_VERSION = "0.0.75"

REPO_ROOT = Path(__file__).resolve().parents[6]
BASE_DIR = REPO_ROOT / "docs" / "win" / "hockey" / "nhl"
HISTORY_ROOT = BASE_DIR / "research" / "sdv_history"
OUTPUT_ROOT = BASE_DIR / "research" / "sdv_edge_microstats"

REGULAR_GAME_CODE = "02"

SHOT_FEATURES = [
    "shots_with_xg",
    "xg_sum",
    "xg_per_shot",
    "avg_shot_distance",
    "avg_abs_shot_angle",
    "close_shot_share",
    "central_shot_share",
    "high_xg_shot_share",
]

ZONE_FEATURES = [
    "controlled_entries",
    "dump_entries",
    "exits",
    "controlled_entry_rate",
]

BASELINE_MODEL_FEATURES = [
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
]

SHOT_MODEL_FEATURES = [
    f"{feature}_{suffix}"
    for feature in SHOT_FEATURES
    for suffix in ("diff", "sum")
]

ZONE_MODEL_FEATURES = [
    f"{feature}_{suffix}"
    for feature in ZONE_FEATURES
    for suffix in ("diff", "sum")
]

PBP_RATING_REQUIRED = {
    "game_id",
    "event_team_abbr",
    "home_abbr",
    "away_abbr",
    "home_skaters",
    "away_skaters",
    "home_goalie_in",
    "away_goalie_in",
    "xg",
    "event_type",
}

ZONE_REQUIRED = {
    "game_id",
    "event_idx",
    "type_desc_key",
    "zone_code",
    "event_owner_team_id",
    "time_in_period",
    "winning_player_id",
    "shooting_player_id",
}

EDGE_SAFETY = {
    "historical_same_season_status": "BLOCKED",
    "reason": (
        "SportsDataverse NHL EDGE detail endpoints expose season/current aggregates "
        "without a historical observation timestamp that can reconstruct the exact "
        "pregame information state. Same-season EDGE aggregates therefore cannot be "
        "backfilled into historical T-60/pregame predictions."
    ),
    "safe_alternative": (
        "Use only prior-game PBP-derived microstats with source_game_date < "
        "target_game_date, or separately validated timestamped EDGE snapshots."
    ),
}

EDGE_ENDPOINTS = (
    ("skater_detail", "nhl_edge_skater_detail", "skater"),
    ("skating_speed", "nhl_edge_skater_skating_speed_detail", "skater"),
    ("skating_distance", "nhl_edge_skater_skating_distance_detail", "skater"),
    ("shot_speed", "nhl_edge_skater_shot_speed_detail", "skater"),
    ("shot_location", "nhl_edge_skater_shot_location_detail", "skater"),
    ("zone_time", "nhl_edge_skater_zone_time", "skater"),
    ("goalie_detail", "nhl_edge_goalie_detail", "goalie"),
    ("goalie_5v5", "nhl_edge_goalie_5v5_detail", "goalie"),
    ("goalie_save_percentage", "nhl_edge_goalie_save_percentage_detail", "goalie"),
    ("goalie_shot_location", "nhl_edge_goalie_shot_location_detail", "goalie"),
)


def log(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, action="append")
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument(
        "--phase",
        choices=("coverage", "oos", "all"),
        default="all",
    )
    parser.add_argument(
        "--endpoint-sample-size",
        type=int,
        default=3,
        help="Deterministic skater/goalie IDs probed per season per endpoint.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip live EDGE endpoint probes; local coverage/OOS still run.",
    )
    parser.add_argument(
        "--min-team-games",
        type=int,
        default=10,
        help="Minimum prior games per team for SDV baseline prediction eligibility.",
    )
    parser.add_argument(
        "--min-train-games",
        type=int,
        default=500,
        help="Minimum strictly earlier-season games required for an OOS fold.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="L2 regularization strength for research-only meta models.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def enforce_sdv_pin() -> str:
    try:
        installed = version("sportsdataverse")
    except PackageNotFoundError as exc:
        raise SystemExit("sportsdataverse is not installed") from exc
    if installed != PINNED_SDV_VERSION:
        raise SystemExit(
            f"Refusing sportsdataverse=={installed}; required {PINNED_SDV_VERSION}."
        )
    return installed


def discovered_seasons() -> list[int]:
    schedule_dir = HISTORY_ROOT / "schedules"
    pbp_dir = HISTORY_ROOT / "play_by_play"
    schedules = {
        int(path.stem.replace("season_", ""))
        for path in schedule_dir.glob("season_*.parquet")
        if path.stem.replace("season_", "").isdigit()
    }
    pbp = {
        int(path.stem.replace("season_", ""))
        for path in pbp_dir.glob("season_*.parquet")
        if path.stem.replace("season_", "").isdigit()
    }
    return sorted(schedules & pbp)


def requested_seasons(args: argparse.Namespace) -> list[int]:
    available = discovered_seasons()
    if not available:
        raise SystemExit(
            f"No P5 schedule+PBP seasons found under {HISTORY_ROOT}"
        )

    explicit: set[int] = set(args.season or [])
    if args.start_season is not None or args.end_season is not None:
        if args.start_season is None or args.end_season is None:
            raise SystemExit("--start-season and --end-season must be used together")
        if args.end_season < args.start_season:
            raise SystemExit("--end-season cannot be less than --start-season")
        explicit.update(range(args.start_season, args.end_season + 1))

    seasons = sorted(explicit) if explicit else available
    missing = [season for season in seasons if season not in available]
    if missing:
        raise SystemExit(
            "Requested season(s) missing P5 schedule/PBP history: "
            + ", ".join(map(str, missing))
        )
    return seasons


def sdv_season_year(project_start_year: int) -> int:
    return project_start_year + 1


def first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def canonical_game_id_expr(column: str = "game_id") -> pl.Expr:
    return pl.col(column).cast(pl.Int64, strict=False).cast(pl.String)


def validate_official_game_ids(values: Iterable[Any], *, label: str) -> None:
    bad: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if len(text) != 10 or not text.isdigit():
            bad.append(text)
            if len(bad) >= 10:
                break
    if bad:
        raise RuntimeError(f"{label}: non-canonical NHL game_id values: {bad}")


def season_paths(season: int) -> tuple[Path, Path]:
    return (
        HISTORY_ROOT / "schedules" / f"season_{season}.parquet",
        HISTORY_ROOT / "play_by_play" / f"season_{season}.parquet",
    )


def read_season(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    schedule_path, pbp_path = season_paths(season)
    return pl.read_parquet(schedule_path), pl.read_parquet(pbp_path)


def normalize_schedule(schedule: pl.DataFrame, season: int) -> pl.DataFrame:
    required = {"game_id", "source_game_date"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise RuntimeError(f"season {season}: schedule missing {missing}")

    home_col = first_column(
        schedule.columns,
        ("home_team_abbr", "home_team_abbrev", "home_abbr"),
    )
    away_col = first_column(
        schedule.columns,
        ("away_team_abbr", "away_team_abbrev", "away_abbr"),
    )
    if not home_col or not away_col:
        raise RuntimeError(f"season {season}: schedule missing home/away abbreviations")

    out = schedule.with_columns(
        canonical_game_id_expr(),
        pl.col("source_game_date")
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("date"),
        pl.col(home_col).cast(pl.String).alias("home_abbr"),
        pl.col(away_col).cast(pl.String).alias("away_abbr"),
        pl.lit(False).alias("neutral_site"),
        pl.lit(sdv_season_year(season), dtype=pl.Int64).alias("season"),
    )

    validate_official_game_ids(
        out.get_column("game_id").drop_nulls().unique().to_list(),
        label=f"season {season} schedule",
    )
    if out.get_column("date").null_count():
        raise RuntimeError(f"season {season}: schedule has unparseable source_game_date")

    return (
        out.select(
            "game_id",
            "date",
            "home_abbr",
            "away_abbr",
            "neutral_site",
            "season",
        )
        .unique(subset=["game_id"], keep="first")
        .filter(pl.col("game_id").str.slice(4, 2) == REGULAR_GAME_CODE)
        .sort(["date", "game_id"])
    )


def normalize_pbp_for_ratings(pbp: pl.DataFrame, season: int) -> pl.DataFrame:
    missing = sorted(PBP_RATING_REQUIRED - set(pbp.columns))
    if missing:
        raise RuntimeError(
            f"season {season}: P5 PBP missing SDV rating fields: {missing}"
        )
    out = pbp.with_columns(canonical_game_id_expr())
    validate_official_game_ids(
        out.get_column("game_id").drop_nulls().unique().to_list(),
        label=f"season {season} PBP",
    )
    return out


def finite_numeric_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Float64, strict=False)
        .is_not_null()
        & pl.col(column).cast(pl.Float64, strict=False).is_finite()
    )


def non_null_rate(frame: pl.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.height == 0:
        return None
    return float(1.0 - frame.get_column(column).null_count() / frame.height)


def coverage_record(season: int, schedule: pl.DataFrame, pbp: pl.DataFrame) -> dict[str, Any]:
    shot_cols = [
        "xg",
        "x",
        "y",
        "shot_distance",
        "shot_angle",
        "event_goalie_id",
        "home_goalie_id",
        "away_goalie_id",
    ]
    zone_candidates = {
        "type_desc_key": ("type_desc_key", "event_type"),
        "zone_code": ("zone_code",),
        "event_idx": ("event_idx", "event_index"),
        "event_owner_team_id": ("event_owner_team_id", "event_team_id"),
        "time_in_period": ("time_in_period",),
        "winning_player_id": ("winning_player_id",),
        "shooting_player_id": ("shooting_player_id", "shooter_player_id"),
    }

    row: dict[str, Any] = {
        "season_start_year": season,
        "sdv_season_year": sdv_season_year(season),
        "schedule_games": schedule.height,
        "pbp_rows": pbp.height,
        "pbp_games": (
            pbp.get_column("game_id").drop_nulls().n_unique()
            if "game_id" in pbp.columns
            else 0
        ),
        "teams": (
            pbp.get_column("event_team_abbr").drop_nulls().n_unique()
            if "event_team_abbr" in pbp.columns
            else 0
        ),
    }

    for column in shot_cols:
        row[f"{column}_present"] = column in pbp.columns
        row[f"{column}_non_null_rate"] = non_null_rate(pbp, column)

    zone_resolved: dict[str, str | None] = {}
    for canonical, aliases in zone_candidates.items():
        zone_resolved[canonical] = first_column(pbp.columns, aliases)
        row[f"zone_{canonical}_present"] = zone_resolved[canonical] is not None

    row["zone_transition_schema_available"] = all(
        value is not None for value in zone_resolved.values()
    )

    player_col = first_column(
        pbp.columns,
        ("shooting_player_id", "shooter_player_id", "player_id"),
    )
    goalie_col = first_column(
        pbp.columns,
        ("event_goalie_id", "goalie_id"),
    )
    row["skater_ids_observed"] = (
        pbp.get_column(player_col).drop_nulls().n_unique() if player_col else 0
    )
    row["goalie_ids_observed"] = (
        pbp.get_column(goalie_col).drop_nulls().n_unique() if goalie_col else 0
    )
    return row


def sample_ids(
    pbp: pl.DataFrame,
    *,
    role: str,
    n: int,
) -> list[int]:
    if n <= 0:
        return []

    if role == "goalie":
        candidates = (
            "event_goalie_id",
            "home_goalie_id",
            "away_goalie_id",
            "goalie_id",
        )
    else:
        candidates = (
            "shooting_player_id",
            "winning_player_id",
            "player_id",
            "shooter_player_id",
        )

    values: set[int] = set()
    for column in candidates:
        if column not in pbp.columns:
            continue
        for raw in pbp.get_column(column).drop_nulls().unique().to_list():
            try:
                value = int(float(raw))
            except (TypeError, ValueError, OverflowError):
                continue
            # NHL player ids are currently 7 digits; keep this permissive but sane.
            if value > 0:
                values.add(value)
        if len(values) >= n:
            break
    return sorted(values)[:n]


def raw_payload_nonempty(payload: Any) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        return bool(payload)
    if isinstance(payload, (list, tuple)):
        return len(payload) > 0
    if hasattr(payload, "height"):
        return bool(getattr(payload, "height"))
    if hasattr(payload, "empty"):
        return not bool(getattr(payload, "empty"))
    return True


def payload_signature(payload: Any) -> str:
    if isinstance(payload, dict):
        return ",".join(sorted(map(str, payload.keys()))[:30])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return ",".join(sorted(map(str, payload[0].keys()))[:30])
    if hasattr(payload, "columns"):
        return ",".join(map(str, list(payload.columns)[:30]))
    return type(payload).__name__


def call_edge_endpoint(
    function_name: str,
    player_id: int,
    season: int | None,
) -> Any:
    fn = getattr(nhl, function_name, None)
    if fn is None:
        raise AttributeError(f"sportsdataverse.nhl.{function_name} not exported")

    kwargs: dict[str, Any] = {
        "player_id": player_id,
        "return_parsed": False,
    }
    if season is not None:
        kwargs["season"] = season
        kwargs["game_type"] = 2

    try:
        return fn(timeout=15, **kwargs)
    except TypeError:
        # Some runtime wrappers do not forward requests-style timeout.
        return fn(**kwargs)


def probe_edge_endpoints(
    seasons: list[int],
    pbp_by_season: dict[int, pl.DataFrame],
    sample_size: int,
    *,
    skip_network: bool,
) -> pd.DataFrame:
    columns = [
        "checked_at_utc",
        "season_start_year",
        "sdv_season_year",
        "endpoint_group",
        "function_name",
        "role",
        "player_id",
        "status",
        "nonempty",
        "response_signature",
        "error",
        "historical_pregame_safe",
        "historical_pregame_reason",
    ]
    if skip_network:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for season in seasons:
        pbp = pbp_by_season[season]
        role_ids = {
            "skater": sample_ids(pbp, role="skater", n=sample_size),
            "goalie": sample_ids(pbp, role="goalie", n=sample_size),
        }

        for endpoint_group, function_name, role in EDGE_ENDPOINTS:
            ids = role_ids[role]
            if not ids:
                rows.append(
                    {
                        "checked_at_utc": utc_now(),
                        "season_start_year": season,
                        "sdv_season_year": sdv_season_year(season),
                        "endpoint_group": endpoint_group,
                        "function_name": function_name,
                        "role": role,
                        "player_id": "",
                        "status": "no_sample_id",
                        "nonempty": False,
                        "response_signature": "",
                        "error": "",
                        "historical_pregame_safe": False,
                        "historical_pregame_reason": EDGE_SAFETY["reason"],
                    }
                )
                continue

            for player_id in ids:
                try:
                    payload = call_edge_endpoint(
                        function_name,
                        player_id,
                        sdv_season_year(season),
                    )
                    rows.append(
                        {
                            "checked_at_utc": utc_now(),
                            "season_start_year": season,
                            "sdv_season_year": sdv_season_year(season),
                            "endpoint_group": endpoint_group,
                            "function_name": function_name,
                            "role": role,
                            "player_id": player_id,
                            "status": "success",
                            "nonempty": raw_payload_nonempty(payload),
                            "response_signature": payload_signature(payload),
                            "error": "",
                            "historical_pregame_safe": False,
                            "historical_pregame_reason": EDGE_SAFETY["reason"],
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "checked_at_utc": utc_now(),
                            "season_start_year": season,
                            "sdv_season_year": sdv_season_year(season),
                            "endpoint_group": endpoint_group,
                            "function_name": function_name,
                            "role": role,
                            "player_id": player_id,
                            "status": "error",
                            "nonempty": False,
                            "response_signature": "",
                            "error": f"{type(exc).__name__}: {exc}",
                            "historical_pregame_safe": False,
                            "historical_pregame_reason": EDGE_SAFETY["reason"],
                        }
                    )
                time.sleep(0.05)

    return pd.DataFrame(rows, columns=columns)


def normalize_zone_pbp(pbp: pl.DataFrame) -> tuple[pl.DataFrame | None, str]:
    aliases = {
        "type_desc_key": ("type_desc_key", "event_type"),
        "zone_code": ("zone_code",),
        "event_idx": ("event_idx", "event_index"),
        "event_owner_team_id": ("event_owner_team_id", "event_team_id"),
        "time_in_period": ("time_in_period",),
        "winning_player_id": ("winning_player_id",),
        "shooting_player_id": ("shooting_player_id", "shooter_player_id"),
    }

    out = pbp
    missing: list[str] = []
    for canonical, candidates in aliases.items():
        if canonical in out.columns:
            continue
        source = first_column(out.columns, candidates)
        if source is None:
            missing.append(canonical)
        else:
            out = out.with_columns(pl.col(source).alias(canonical))

    if missing:
        return None, "missing_columns:" + ",".join(sorted(missing))

    required = {"game_id", *aliases.keys()}
    still_missing = sorted(required - set(out.columns))
    if still_missing:
        return None, "missing_columns:" + ",".join(still_missing)

    return out, "available"


def expand_team_game_features(
    schedule: pl.DataFrame,
    agg: pl.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    schedule_pd = schedule.select(
        "game_id",
        "date",
        "home_abbr",
        "away_abbr",
    ).to_pandas()
    agg_pd = agg.to_pandas()

    by_key = {
        (str(row.game_id), str(row.team)): row
        for row in agg_pd.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []

    for game in schedule_pd.itertuples(index=False):
        game_id = str(game.game_id)

        for team, opponent in (
            (
                str(game.home_abbr),
                str(game.away_abbr),
            ),
            (
                str(game.away_abbr),
                str(game.home_abbr),
            ),
        ):
            rec = by_key.get(
                (
                    game_id,
                    team,
                )
            )

            row = {
                "game_id": game_id,
                "date": game.date,
                "team": team,
                "opponent": opponent,
            }

            for feature in features:
                row[feature] = (
                    getattr(rec, feature)
                    if rec is not None
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


def shot_team_game_features(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pd.DataFrame:
    required = {
        "game_id",
        "event_team_abbr",
        "xg",
        "shot_distance",
        "shot_angle",
    }
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise RuntimeError(f"shot microstats missing PBP fields: {missing}")

    regular_ids = schedule.get_column("game_id")
    shots = (
        pbp.filter(pl.col("game_id").is_in(regular_ids))
        .with_columns(
            pl.col("xg").cast(pl.Float64, strict=False).alias("_xg"),
            pl.col("shot_distance").cast(pl.Float64, strict=False).alias("_distance"),
            pl.col("shot_angle").cast(pl.Float64, strict=False).abs().alias("_angle"),
            pl.col("event_team_abbr").cast(pl.String).alias("_team"),
        )
        .filter(pl.col("_xg").is_not_null() & pl.col("_team").is_not_null())
    )

    if shots.is_empty():
        return pd.DataFrame()

    agg = (
        shots.group_by(["game_id", "_team"])
        .agg(
            pl.len().alias("shots_with_xg"),
            pl.col("_xg").sum().alias("xg_sum"),
            pl.col("_xg").mean().alias("xg_per_shot"),
            pl.col("_distance").mean().alias("avg_shot_distance"),
            pl.col("_angle").mean().alias("avg_abs_shot_angle"),
            (pl.col("_distance") <= 20.0).mean().alias("close_shot_share"),
            (pl.col("_angle") <= 30.0).mean().alias("central_shot_share"),
            (pl.col("_xg") >= 0.20).mean().alias("high_xg_shot_share"),
        )
        .rename({"_team": "team"})
    )

    return expand_team_game_features(
        schedule,
        agg,
        SHOT_FEATURES,
    )


def zone_team_game_features(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> tuple[pd.DataFrame, str]:
    normalized, status = normalize_zone_pbp(pbp)
    if normalized is None:
        return pd.DataFrame(), status

    try:
        from sportsdataverse.nhl.nhl_zone_transitions import infer_zone_transitions
    except Exception as exc:
        return pd.DataFrame(), f"import_error:{type(exc).__name__}:{exc}"

    regular_ids = schedule.get_column("game_id")
    zone_input = normalized.filter(pl.col("game_id").is_in(regular_ids))

    try:
        transitions = infer_zone_transitions(zone_input, league="nhl")
    except Exception as exc:
        return pd.DataFrame(), f"runtime_error:{type(exc).__name__}:{exc}"

    if transitions.is_empty():
        return pd.DataFrame(), "zero_transitions"

    # Map per-game event-owner team ids back to canonical abbreviations.
    mapping = (
        normalized.select(
            canonical_game_id_expr(),
            pl.col("event_owner_team_id").cast(pl.String).alias("team_id"),
            pl.col("event_team_abbr").cast(pl.String).alias("team"),
        )
        .drop_nulls(["game_id", "team_id", "team"])
        .unique(["game_id", "team_id", "team"])
    )

    transitions = transitions.with_columns(
        canonical_game_id_expr(),
        pl.col("team_id").cast(pl.String),
    ).join(
        mapping,
        on=["game_id", "team_id"],
        how="left",
    )

    agg = (
        transitions.drop_nulls(["team"])
        .group_by(["game_id", "team"])
        .agg(
            (
                (pl.col("transition_type") == "entry")
                & (pl.col("controlled") == True)  # noqa: E712
            ).sum().alias("controlled_entries"),
            (
                (pl.col("transition_type") == "entry")
                & (pl.col("controlled") == False)  # noqa: E712
            ).sum().alias("dump_entries"),
            (pl.col("transition_type") == "exit").sum().alias("exits"),
        )
        .with_columns(
            (
                pl.col("controlled_entries")
                / (pl.col("controlled_entries") + pl.col("dump_entries")).cast(pl.Float64)
            ).alias("controlled_entry_rate")
        )
    )

    return (
        expand_team_game_features(
            schedule,
            agg,
            ZONE_FEATURES,
        ),
        "available",
    )


def build_team_game_rates(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    from sportsdataverse.nhl.nhl_team_ratings import team_game_xg_rates

    regular_ids = schedule.get_column("game_id")
    regular_pbp = pbp.filter(pl.col("game_id").is_in(regular_ids))
    game_rates = team_game_xg_rates(regular_pbp, schedule)
    if game_rates.is_empty():
        raise RuntimeError("SportsDataverse team_game_xg_rates returned zero rows")
    return game_rates


def ratings_from_prior_games(
    game_rates: pl.DataFrame,
    target_date: date,
) -> pl.DataFrame:
    return prior_adjusted_ratings(
        game_rates,
        target_date,
    )


def prior_team_means(
    team_games: pd.DataFrame,
    team: str,
    target_date: date,
    features: Sequence[str],
    min_games: int,
) -> dict[str, float] | None:
    prior = team_games[
        (team_games["team"] == team)
        & (team_games["date"] < target_date)
    ]
    if len(prior) < min_games:
        return None
    values = prior[list(features)].apply(pd.to_numeric, errors="coerce")
    means = values.mean(axis=0, skipna=True)
    return {name: float(means[name]) if pd.notna(means[name]) else np.nan for name in features}


def outcome_lookup(game_rates: pl.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    pdf = game_rates.select("game_id", "team", "gf", "ga").to_pandas()
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in pdf.itertuples(index=False):
        out[(str(row.game_id), str(row.team))] = {
            "gf": float(row.gf),
            "ga": float(row.ga),
        }
    return out


def build_season_feature_matrix(
    season: int,
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
    *,
    min_team_games: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sportsdataverse.nhl.nhl_market import nhl_predict_games

    pbp_ratings = normalize_pbp_for_ratings(pbp, season)
    game_rates = build_team_game_rates(schedule, pbp_ratings)
    shot_team_games = shot_team_game_features(schedule, pbp)
    zone_team_games, zone_status = zone_team_game_features(schedule, pbp)

    shot_team_games["date"] = pd.to_datetime(shot_team_games["date"]).dt.date
    if not zone_team_games.empty:
        zone_team_games["date"] = pd.to_datetime(zone_team_games["date"]).dt.date

    outcomes = outcome_lookup(game_rates)
    rows: list[dict[str, Any]] = []

    for target_day in schedule.get_column("date").unique().sort().to_list():
        day_games = schedule.filter(pl.col("date") == pl.lit(target_day))
        ratings = ratings_from_prior_games(game_rates, target_day)
        if ratings.is_empty():
            continue
        if "games" in ratings.columns and min_team_games > 0:
            ratings = ratings.filter(pl.col("games") >= min_team_games)
        if ratings.is_empty():
            continue

        rated_teams = set(ratings.get_column("team").to_list())
        games = (
            day_games.select(
                "game_id",
                pl.col("home_abbr").alias("home_team"),
                pl.col("away_abbr").alias("away_team"),
                "neutral_site",
            )
            .filter(
                pl.col("home_team").is_in(rated_teams)
                & pl.col("away_team").is_in(rated_teams)
            )
        )
        if games.is_empty():
            continue

        preds = nhl_predict_games(games, ratings)
        if preds.is_empty():
            continue

        for pred in preds.to_dicts():
            game_id = str(pred["game_id"])
            home = str(pred["home_team"])
            away = str(pred["away_team"])
            home_out = outcomes.get((game_id, home))
            away_out = outcomes.get((game_id, away))
            if home_out is None or away_out is None:
                continue

            home_goals = home_out["gf"]
            away_goals = away_out["gf"]

            # PBP-derived GOAL counts can remain tied for shootout-decided games.
            # Do not guess the shootout winner or use historically unsafe schedule
            # score columns; exclude those games from the OOS target set.
            if home_goals == away_goals:
                continue

            row: dict[str, Any] = {
                "season_start_year": season,
                "game_date": target_day.isoformat(),
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_win": float(home_goals > away_goals),
                "goal_margin": home_goals - away_goals,
                "goal_total": home_goals + away_goals,
                "sdv_home_win_prob": float(pred["home_win_prob"]),
                "sdv_exp_margin": float(pred["exp_margin"]),
                "sdv_exp_total": float(pred["exp_total"]),
                "same_day_source_games": "excluded",
                "as_of_rule": "source_game_date < target_game_date",
            }

            for feature_set, team_games, features in (
                ("shot", shot_team_games, SHOT_FEATURES),
                ("zone", zone_team_games, ZONE_FEATURES),
            ):
                if team_games.empty:
                    for feature in features:
                        row[f"{feature}_diff"] = np.nan
                        row[f"{feature}_sum"] = np.nan
                    continue

                home_means = prior_team_means(
                    team_games, home, target_day, features, min_team_games
                )
                away_means = prior_team_means(
                    team_games, away, target_day, features, min_team_games
                )
                for feature in features:
                    hv = home_means.get(feature, np.nan) if home_means else np.nan
                    av = away_means.get(feature, np.nan) if away_means else np.nan
                    row[f"{feature}_diff"] = hv - av if np.isfinite(hv) and np.isfinite(av) else np.nan
                    row[f"{feature}_sum"] = hv + av if np.isfinite(hv) and np.isfinite(av) else np.nan

            rows.append(row)

    meta = {
        "season_start_year": season,
        "zone_transition_status": zone_status,
        "team_game_rate_rows": game_rates.height,
        "shot_team_game_rows": len(shot_team_games),
        "zone_team_game_rows": len(zone_team_games),
        "prediction_feature_rows": len(rows),
    }
    return pd.DataFrame(rows), meta


def clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 1e-6, 1 - 1e-6)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[Standardizer, np.ndarray]:
    scaler = Standardizer.fit(x)
    xs = scaler.transform(x)
    design = np.column_stack([np.ones(len(xs)), xs])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(design @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = clip_prob(p)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        penalty = 0.5 * l2 * float(np.sum(beta[1:] ** 2)) / len(y)
        grad = design.T @ (p - y) / len(y)
        grad[1:] += l2 * beta[1:] / len(y)
        return loss + penalty, grad

    result = minimize(
        lambda b: objective(b)[0],
        np.zeros(design.shape[1]),
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return scaler, result.x


def predict_logistic(
    scaler: Standardizer,
    beta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    xs = scaler.transform(x)
    design = np.column_stack([np.ones(len(xs)), xs])
    z = np.clip(design @ beta, -35.0, 35.0)
    return clip_prob(1.0 / (1.0 + np.exp(-z)))


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[Standardizer, np.ndarray]:
    scaler = Standardizer.fit(x)
    xs = scaler.transform(x)
    design = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(design.shape[1]) * l2
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return scaler, beta


def predict_linear(
    scaler: Standardizer,
    beta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    xs = scaler.transform(x)
    design = np.column_stack([np.ones(len(xs)), xs])
    return design @ beta


def metrics_from_predictions(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    y = frame["home_win"].to_numpy(float)
    p = clip_prob(frame[f"{prefix}_home_win_prob"].to_numpy(float))
    margin = frame["goal_margin"].to_numpy(float)
    margin_pred = frame[f"{prefix}_margin"].to_numpy(float)
    total = frame["goal_total"].to_numpy(float)
    total_pred = frame[f"{prefix}_total"].to_numpy(float)

    return {
        "ml_brier": float(np.mean((p - y) ** 2)),
        "ml_log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "margin_mae": float(np.mean(np.abs(margin_pred - margin))),
        "margin_rmse": float(np.sqrt(np.mean((margin_pred - margin) ** 2))),
        "total_mae": float(np.mean(np.abs(total_pred - total))),
        "total_rmse": float(np.sqrt(np.mean((total_pred - total) ** 2))),
    }


def season_walkforward_compare(
    matrix: pd.DataFrame,
    *,
    candidate_name: str,
    candidate_features: Sequence[str],
    min_train_games: int,
    l2: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_baseline = list(BASELINE_MODEL_FEATURES)
    features_candidate = features_baseline + list(candidate_features)

    required = [
        "season_start_year",
        "home_win",
        "goal_margin",
        "goal_total",
        *features_candidate,
    ]
    work = matrix.dropna(subset=required).copy()
    if work.empty:
        return pd.DataFrame(), {
            "candidate": candidate_name,
            "status": "not_evaluable",
            "reason": "no complete rows",
        }

    predictions: list[pd.DataFrame] = []
    for test_season in sorted(work["season_start_year"].unique()):
        train = work[work["season_start_year"] < test_season]
        test = work[work["season_start_year"] == test_season]
        if len(train) < min_train_games or test.empty:
            continue

        fold = test[
            [
                "season_start_year",
                "game_date",
                "game_id",
                "home_team",
                "away_team",
                "home_win",
                "goal_margin",
                "goal_total",
            ]
        ].copy()
        fold["candidate"] = candidate_name
        fold["train_rows"] = len(train)

        for label, features in (
            ("baseline", features_baseline),
            ("candidate", features_candidate),
        ):
            x_train = train[features].to_numpy(float)
            x_test = test[features].to_numpy(float)

            log_scaler, log_beta = fit_logistic(
                x_train,
                train["home_win"].to_numpy(float),
                l2=l2,
            )
            margin_scaler, margin_beta = fit_ridge(
                x_train,
                train["goal_margin"].to_numpy(float),
                l2=l2,
            )
            total_scaler, total_beta = fit_ridge(
                x_train,
                train["goal_total"].to_numpy(float),
                l2=l2,
            )

            fold[f"{label}_home_win_prob"] = predict_logistic(
                log_scaler, log_beta, x_test
            )
            fold[f"{label}_margin"] = predict_linear(
                margin_scaler, margin_beta, x_test
            )
            fold[f"{label}_total"] = predict_linear(
                total_scaler, total_beta, x_test
            )

        predictions.append(fold)

    if not predictions:
        return pd.DataFrame(), {
            "candidate": candidate_name,
            "status": "not_evaluable",
            "reason": "insufficient prior-season training rows",
        }

    out = pd.concat(predictions, ignore_index=True)
    baseline = metrics_from_predictions(out, "baseline")
    candidate = metrics_from_predictions(out, "candidate")

    deltas = {
        key: candidate[key] - baseline[key]
        for key in baseline
    }
    supported = (
        candidate["ml_log_loss"] < baseline["ml_log_loss"]
        and candidate["ml_brier"] < baseline["ml_brier"]
        and (
            candidate["margin_rmse"] <= baseline["margin_rmse"]
            or candidate["total_rmse"] <= baseline["total_rmse"]
        )
    )

    summary = {
        "candidate": candidate_name,
        "status": "evaluated",
        "oos_rows": len(out),
        "test_seasons": sorted(map(int, out["season_start_year"].unique())),
        "baseline": baseline,
        "candidate_metrics": candidate,
        "candidate_minus_baseline": deltas,
        "incremental_value": (
            "SUPPORTED_FOR_RESEARCH_REVIEW"
            if supported
            else "NOT_DEMONSTRATED"
        ),
    }
    return out, summary


def metric_rows(candidate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    if candidate_summary.get("status") != "evaluated":
        return [
            {
                "candidate": candidate_summary.get("candidate", ""),
                "model": "",
                "metric": "",
                "value": "",
                "baseline_value": "",
                "delta_candidate_minus_baseline": "",
                "oos_rows": 0,
                "incremental_value": candidate_summary.get(
                    "incremental_value", "NOT_EVALUATED"
                ),
                "status": candidate_summary.get("status", "not_evaluable"),
                "reason": candidate_summary.get("reason", ""),
            }
        ]

    rows: list[dict[str, Any]] = []
    baseline = candidate_summary["baseline"]
    candidate = candidate_summary["candidate_metrics"]
    for metric in baseline:
        rows.append(
            {
                "candidate": candidate_summary["candidate"],
                "model": "candidate",
                "metric": metric,
                "value": candidate[metric],
                "baseline_value": baseline[metric],
                "delta_candidate_minus_baseline": (
                    candidate[metric] - baseline[metric]
                ),
                "oos_rows": candidate_summary["oos_rows"],
                "incremental_value": candidate_summary["incremental_value"],
                "status": "evaluated",
                "reason": "",
            }
        )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    sdv_version = enforce_sdv_pin()
    seasons = requested_seasons(args)

    if OUTPUT_ROOT.exists() and not args.overwrite:
        existing = list(OUTPUT_ROOT.glob("*"))
        if existing:
            raise SystemExit(
                f"{OUTPUT_ROOT} already has outputs. Re-run with --overwrite."
            )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log(f"SDV_EDGE_MICROSTATS_VERSION | {SCRIPT_VERSION}")
    log(f"SPORTSDATAVERSE_VERSION | {sdv_version}")
    log(f"SEASONS | {seasons}")
    log("MODE | research_only")
    log("LEAKAGE_RULE | source_game_date < target_game_date")

    coverage_rows: list[dict[str, Any]] = []
    pbp_by_season: dict[int, pl.DataFrame] = {}
    schedule_by_season: dict[int, pl.DataFrame] = {}

    for season in seasons:
        schedule_raw, pbp_raw = read_season(season)
        schedule = normalize_schedule(schedule_raw, season)
        pbp = pbp_raw.with_columns(canonical_game_id_expr())
        schedule_by_season[season] = schedule
        pbp_by_season[season] = pbp
        coverage_rows.append(coverage_record(season, schedule, pbp))

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = OUTPUT_ROOT / "coverage_by_season.csv"
    coverage_df.to_csv(coverage_path, index=False)
    log(f"COVERAGE | {coverage_path}")

    endpoint_df = pd.DataFrame()
    if args.phase in {"coverage", "all"}:
        endpoint_df = probe_edge_endpoints(
            seasons,
            pbp_by_season,
            args.endpoint_sample_size,
            skip_network=args.skip_network,
        )
        endpoint_path = OUTPUT_ROOT / "edge_endpoint_probe.csv"
        endpoint_df.to_csv(endpoint_path, index=False)
        log(f"EDGE_ENDPOINT_PROBE | {endpoint_path} | rows={len(endpoint_df)}")

    feature_matrix = pd.DataFrame()
    season_meta: list[dict[str, Any]] = []
    oos_summaries: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []

    if args.phase in {"oos", "all"}:
        matrices: list[pd.DataFrame] = []
        for season in seasons:
            log(f"OOS_FEATURE_BUILD | season={season}")
            matrix, meta = build_season_feature_matrix(
                season,
                schedule_by_season[season],
                pbp_by_season[season],
                min_team_games=args.min_team_games,
            )
            season_meta.append(meta)
            if not matrix.empty:
                matrices.append(matrix)

        if matrices:
            feature_matrix = pd.concat(matrices, ignore_index=True)
            feature_matrix = feature_matrix.sort_values(
                ["season_start_year", "game_date", "game_id"]
            )
        feature_path = OUTPUT_ROOT / "game_feature_matrix.csv"
        feature_matrix.to_csv(feature_path, index=False)
        log(f"FEATURE_MATRIX | {feature_path} | rows={len(feature_matrix)}")

        candidates: list[tuple[str, Sequence[str]]] = []
        if not feature_matrix.empty:
            candidates.append(("shot_microstats", SHOT_MODEL_FEATURES))
            zone_available = (
                all(column in feature_matrix.columns for column in ZONE_MODEL_FEATURES)
                and feature_matrix[ZONE_MODEL_FEATURES].notna().any().any()
            )
            if zone_available:
                candidates.append(("zone_transitions", ZONE_MODEL_FEATURES))
                candidates.append(
                    ("combined_safe_microstats", [*SHOT_MODEL_FEATURES, *ZONE_MODEL_FEATURES])
                )

        for candidate_name, features in candidates:
            log(f"OOS_EVALUATE | {candidate_name}")
            predictions, summary = season_walkforward_compare(
                feature_matrix,
                candidate_name=candidate_name,
                candidate_features=features,
                min_train_games=args.min_train_games,
                l2=args.ridge,
            )
            oos_summaries.append(summary)
            if not predictions.empty:
                oos_frames.append(predictions)

        oos_comparison = (
            pd.concat(oos_frames, ignore_index=True)
            if oos_frames
            else pd.DataFrame()
        )
        oos_path = OUTPUT_ROOT / "oos_comparison.csv"
        oos_comparison.to_csv(oos_path, index=False)

        metric_data: list[dict[str, Any]] = []
        for summary in oos_summaries:
            metric_data.extend(metric_rows(summary))
        metrics_df = pd.DataFrame(metric_data)
        metrics_path = OUTPUT_ROOT / "oos_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        log(f"OOS_METRICS | {metrics_path}")

    endpoint_summary: dict[str, Any] = {
        "network_probe_skipped": bool(args.skip_network),
        "rows": len(endpoint_df),
        "success_rows": (
            int((endpoint_df["status"] == "success").sum())
            if not endpoint_df.empty
            else 0
        ),
        "nonempty_rows": (
            int(endpoint_df["nonempty"].fillna(False).astype(bool).sum())
            if not endpoint_df.empty
            else 0
        ),
        "endpoint_stability": (
            "ONE_RUN_PROBE_ONLY_NOT_LONGITUDINALLY_VERIFIED"
            if not args.skip_network
            else "NOT_TESTED"
        ),
        "current_day_availability_timing": (
            "UNRESOLVED_REQUIRES_IN_SEASON_PRESTART_OBSERVATION"
        ),
    }

    safe_oos_support = {
        item["candidate"]: item.get("incremental_value", "NOT_EVALUATED")
        for item in oos_summaries
    }

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "sportsdataverse_version": sdv_version,
        "mode": "research_only",
        "production_change": "none",
        "seasons": seasons,
        "season_convention": "CLI=NHL_START_YEAR | SDV=END_YEAR",
        "leakage_rule": "source_game_date < target_game_date",
        "same_day_postgame_sources": "excluded",
        "historical_score_ground_truth": (
            "PBP-derived team_game_xg_rates goals; schedule score fields not used"
        ),
        "edge_aggregate_safety": EDGE_SAFETY,
        "coverage": coverage_rows,
        "endpoint_probe": endpoint_summary,
        "season_feature_build": season_meta,
        "oos": oos_summaries,
        "safe_microstat_incremental_value": safe_oos_support,
        "promotion_gate": {
            "edge_season_aggregates": (
                "BLOCKED_UNTIL_TIMESTAMPED_HISTORICAL_PREGAME_SNAPSHOTS_EXIST"
            ),
            "pbp_shot_location_characteristics": (
                "RESEARCH_ONLY_PENDING_OOS_RESULT"
                if "shot_microstats" not in safe_oos_support
                else safe_oos_support["shot_microstats"]
            ),
            "pbp_zone_transitions": safe_oos_support.get(
                "zone_transitions",
                "NOT_EVALUATED_OR_SCHEMA_UNAVAILABLE",
            ),
            "automatic_production_promotion": False,
        },
    }

    summary_path = OUTPUT_ROOT / "summary.json"
    write_json(summary_path, summary)

    log(f"SUMMARY | {summary_path}")
    log("DONE")
    log(
        "NEXT | Send summary.json and oos_metrics.csv back for SDV-P9 promotion/rejection review."
    )


if __name__ == "__main__":
    main()
