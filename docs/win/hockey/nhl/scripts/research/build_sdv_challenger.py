#!/usr/bin/env python3
"""
SDV-P6 — SportsDataverse challenger model research runner.

Repository path:
    docs/win/hockey/nhl/scripts/research/build_sdv_challenger.py

Output root:
    docs/win/hockey/nhl/research/sdv_challenger/

Purpose
-------
Build and evaluate SportsDataverse's NHL pregame market model as an
independent challenger to the existing D-Ratings predictions.

The challenger produces the three quantities aligned with the current NHL
pipeline markets:
    - home win probability
    - expected home-minus-away goal margin
    - expected total goals

Leakage rules
-------------
1. Canonical game key is the official 10-digit NHL ``game_id``.
2. SportsDataverse ratings for target date D use only source games whose
   source_game_date < D.
3. Same-day completed games are deliberately excluded from ratings for every
   other game on that date. This is conservative and reproducible.
4. Realized outcomes are read from the repository's validated final-score
   files, not from historical SportsDataverse schedule score columns.
5. Ensemble/meta-model parameters for target date D are trained only on
   completed comparison rows with game_date < D.
6. Research outputs support the approved production role: D-Ratings primary, SDV challenger/disagreement, and calibrated secondary ensemble/meta signals. This research runner is not the production execution step.

SportsDataverse contract
------------------------
Pinned project dependency: sportsdataverse==0.0.75

This script uses the public SportsDataverse functions:
    sportsdataverse.nhl.nhl_team_ratings.adjust_rate_opponent
    sportsdataverse.nhl.nhl_market.nhl_predict_games
    sportsdataverse.nhl.nhl_prediction_constants.get_constants

Rather than re-downloading historical data for each walk-forward date, the
script reconstructs the same team-rating calculation from the local SDV-P5
Parquet store.

Examples
--------
Standalone challenger evaluation for the 2025-26 NHL season::

    python docs/win/hockey/nhl/scripts/research/build_sdv_challenger.py \
        --season 2025

After standalone validation exists, run the explicitly gated ensemble phase::

    python docs/win/hockey/nhl/scripts/research/build_sdv_challenger.py \
        --season 2025 --evaluate-ensemble

Use ``--include-playoffs`` to score playoff target games too. Ratings remain
based on regular-season games, matching SportsDataverse ``nhl_team_ratings``.
"""

from __future__ import annotations

import sys

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

# Set before importing Polars/SportsDataverse. Some SDV Parquet assets contain
# R Arrow extension metadata.
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize


RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

# noinspection PyPep8
from sdv_ratings_common import prior_adjusted_ratings


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# noinspection PyPep8
from model_blend_common import (
    design_matrix,
    fit_blend_weights,
    ridge_linear_coefficients,
)


SCRIPT_VERSION = "SDV-P6-2026-08-30-v2"
PINNED_SDV_VERSION = "0.0.75"

REPO_ROOT = Path(__file__).resolve().parents[6]
NHL_ROOT = REPO_ROOT / "docs" / "win" / "hockey" / "nhl"

SDV_HISTORY_ROOT = NHL_ROOT / "research" / "sdv_history"
OUTPUT_ROOT = NHL_ROOT / "research" / "sdv_challenger"
PREDICTIONS_ROOT = NHL_ROOT / "00_intake" / "predictions"
FINAL_SCORES_ROOT = NHL_ROOT / "05_final_scores" / "final_scores"

UTC = timezone.utc
OFFICIAL_GAME_ID_RE = re.compile(r"^\d{10}$")

# NHL official game-id type codes embedded after the four-digit start year.
REGULAR_GAME_CODE = "02"
PLAYOFF_GAME_CODE = "03"

# Standalone validation must exist before the optional ensemble stage is
# allowed to run. This is a sample sufficiency gate, not a performance-based
# deployment rule.
DEFAULT_MIN_ENSEMBLE_TRAIN_ROWS = 100
DEFAULT_MIN_TEAM_GAMES = 1

EPS = 1e-6


@dataclass(frozen=True)
class StandaloneMetrics:
    rows: int
    brier: float
    log_loss: float
    margin_mae: float
    margin_rmse: float
    total_mae: float
    total_rmse: float


@dataclass(frozen=True)
class LinearModel:
    coefficients: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coefficients


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = np.clip(x @ self.coefficients, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the SportsDataverse NHL challenger model."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NHL season START YEAR; e.g. 2025 means 2025-26.",
    )
    parser.add_argument(
        "--include-playoffs",
        action="store_true",
        help="Also score playoff target games. Ratings still use regular-season history only.",
    )
    parser.add_argument(
        "--evaluate-ensemble",
        action="store_true",
        help=(
            "After standalone validation is built, run past-only calibrated weighted, "
            "meta-model, and disagreement analyses."
        ),
    )
    parser.add_argument(
        "--min-ensemble-train-rows",
        type=int,
        default=DEFAULT_MIN_ENSEMBLE_TRAIN_ROWS,
        help="Minimum prior completed comparison rows required for ensemble/meta fitting.",
    )
    parser.add_argument(
        "--min-team-games",
        type=int,
        default=DEFAULT_MIN_TEAM_GAMES,
        help="Minimum prior regular-season games required for both SDV-rated teams.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing challenger output files for the season.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sdv_season_year(nhl_season_start_year: int) -> int:
    return nhl_season_start_year + 1


def game_type_code(game_id: str) -> str:
    text = str(game_id).strip()
    if not OFFICIAL_GAME_ID_RE.fullmatch(text):
        return ""
    return text[4:6]


def validate_official_game_ids(values: Iterable[Any], *, label: str, season: int | None = None) -> None:
    bad: list[str] = []
    wrong_season: list[str] = []
    for value in values:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        # CSV readers may parse integer-like IDs as floats.
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        if not OFFICIAL_GAME_ID_RE.fullmatch(text):
            bad.append(text)
            continue
        if season is not None and not text.startswith(str(season)):
            wrong_season.append(text)
    if bad:
        sample = ", ".join(bad[:5])
        raise RuntimeError(f"{label}: non-canonical game_id values found; examples: {sample}")
    if wrong_season:
        sample = ", ".join(wrong_season[:5])
        raise RuntimeError(
            f"{label}: game_id values outside requested NHL season start year {season}; examples: {sample}"
        )


def canonical_game_id_series(series: pd.Series) -> pd.Series:
    def one(value: Any) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text

    return series.map(one).astype("string")


def enforce_sdv_pin() -> str:
    try:
        installed = version("sportsdataverse")
    except PackageNotFoundError as exc:
        raise SystemExit("sportsdataverse is not installed") from exc
    if installed != PINNED_SDV_VERSION:
        raise SystemExit(
            f"Refusing to run with sportsdataverse=={installed}; required {PINNED_SDV_VERSION}."
        )
    return installed


def season_paths(season: int) -> tuple[Path, Path]:
    schedule = SDV_HISTORY_ROOT / "schedules" / f"season_{season}.parquet"
    pbp = SDV_HISTORY_ROOT / "play_by_play" / f"season_{season}.parquet"
    return schedule, pbp


def require_p5_inputs(season: int) -> tuple[Path, Path]:
    schedule, pbp = season_paths(season)
    missing = [path for path in (schedule, pbp) if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "SDV-P5 local history is missing required files:\n"
            f"{joined}\n"
            "Run build_sdv_history.py for this season first."
        )
    return schedule, pbp


def first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def normalize_schedule_for_sdv(schedule: pl.DataFrame, season: int) -> pl.DataFrame:
    """Convert P5 schedule to the exact shape used by SDV team_game_xg_rates."""
    required = {"game_id", "source_game_date"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise RuntimeError(f"P5 schedule missing required columns: {', '.join(missing)}")

    home_col = first_column(
        schedule.columns,
        ("home_team_abbr", "home_team_abbrev", "home_abbr"),
    )
    away_col = first_column(
        schedule.columns,
        ("away_team_abbr", "away_team_abbrev", "away_abbr"),
    )
    if not home_col or not away_col:
        raise RuntimeError("P5 schedule is missing home/away team abbreviation columns")

    # Use official game-id type code rather than relying on potentially drifting
    # upstream textual game_type values.
    out = schedule.with_columns(
        pl.col("game_id").cast(pl.Int64, strict=False).cast(pl.String),
        pl.col("source_game_date")
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        .alias("date"),
        pl.col(home_col).cast(pl.String).alias("home_abbr"),
        pl.col(away_col).cast(pl.String).alias("away_abbr"),
        pl.lit(False).alias("neutral_site"),
        pl.lit(sdv_season_year(season), dtype=pl.Int64).alias("season"),
    )

    invalid_dates = out.filter(pl.col("date").is_null()).height
    if invalid_dates:
        raise RuntimeError(f"P5 schedule has {invalid_dates} rows without a parseable source_game_date")

    ids = out.get_column("game_id").to_list()
    validate_official_game_ids(ids, label="P5 schedule", season=season)

    return out.select(
        "game_id", "season", "date", "home_abbr", "away_abbr", "neutral_site"
    ).unique(subset=["game_id"], keep="first")


def normalize_pbp_for_sdv(pbp: pl.DataFrame, season: int) -> pl.DataFrame:
    """Validate the local P5 PBP fields consumed by SDV team_game_xg_rates."""
    required = {
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
    missing = sorted(required - set(pbp.columns))
    if missing:
        raise RuntimeError(
            "P5 play_by_play is missing fields required by SportsDataverse team ratings: "
            + ", ".join(missing)
        )
    out = pbp.with_columns(pl.col("game_id").cast(pl.Int64, strict=False).cast(pl.String))
    validate_official_game_ids(out.get_column("game_id").drop_nulls().unique().to_list(), label="P5 PBP", season=season)
    return out


def load_p5_game_rates(season: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build all regular-season per-team game rates once from local P5 data."""
    from sportsdataverse.nhl.nhl_team_ratings import team_game_xg_rates

    schedule_path, pbp_path = require_p5_inputs(season)
    schedule_raw = pl.read_parquet(schedule_path)
    pbp_raw = pl.read_parquet(pbp_path)

    schedule = normalize_schedule_for_sdv(schedule_raw, season)
    pbp = normalize_pbp_for_sdv(pbp_raw, season)

    regular_schedule = schedule.filter(
        pl.col("game_id").str.slice(4, 2) == REGULAR_GAME_CODE
    )
    if regular_schedule.is_empty():
        raise RuntimeError("No regular-season games found in P5 schedule")

    regular_ids = regular_schedule.get_column("game_id")
    pbp_regular = pbp.filter(pl.col("game_id").is_in(regular_ids))
    game_rates = team_game_xg_rates(pbp_regular, regular_schedule)
    if game_rates.is_empty():
        raise RuntimeError("SportsDataverse team_game_xg_rates returned no rows")

    return schedule, game_rates


def ratings_from_prior_games(
    game_rates: pl.DataFrame,
    target_date: date,
) -> pl.DataFrame:
    out = prior_adjusted_ratings(
        game_rates,
        target_date,
    )
    if out.is_empty():
        return out

    net_mean = out.get_column("adj_xg_net").mean()
    net_std = out.get_column("adj_xg_net").std()

    out = out.with_columns(
        pl.col("adj_xgf")
        .rank(method="ordinal", descending=True)
        .cast(pl.Int64)
        .alias("off_rank"),
        pl.col("adj_xga")
        .rank(method="ordinal", descending=False)
        .cast(pl.Int64)
        .alias("def_rank"),
        pl.col("adj_xg_net")
        .rank(method="ordinal", descending=True)
        .cast(pl.Int64)
        .alias("net_rank"),
        (
            (
                (pl.col("adj_xg_net") - float(net_mean))
                / float(net_std)
            )
            if net_std
            else pl.lit(0.0)
        ).alias("net_z"),
    )

    return out.select(
        "season",
        "team",
        "adj_xgf",
        "adj_xga",
        "adj_xg_net",
        "adj_gf",
        "adj_ga",
        "games",
        "off_rank",
        "def_rank",
        "net_rank",
        "net_z",
    )


def target_schedule(schedule: pl.DataFrame, include_playoffs: bool) -> pl.DataFrame:
    codes = [REGULAR_GAME_CODE]
    if include_playoffs:
        codes.append(PLAYOFF_GAME_CODE)
    return schedule.filter(pl.col("game_id").str.slice(4, 2).is_in(codes)).sort(["date", "game_id"])


def build_sdv_walkforward_predictions(
    schedule: pl.DataFrame,
    game_rates: pl.DataFrame,
    *,
    season: int,
    include_playoffs: bool,
    min_team_games: int,
) -> pd.DataFrame:
    """Predict every eligible target date from strictly prior game information."""
    from sportsdataverse.nhl.nhl_market import nhl_predict_games

    targets = target_schedule(schedule, include_playoffs)
    if targets.is_empty():
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    unique_dates = targets.get_column("date").unique().sort().to_list()

    for target_day in unique_dates:
        day_games = targets.filter(pl.col("date") == pl.lit(target_day))
        ratings = ratings_from_prior_games(game_rates, target_day)
        if ratings.is_empty():
            continue

        if min_team_games > 0:
            ratings = ratings.filter(pl.col("games") >= min_team_games)
        if ratings.is_empty():
            continue

        games = day_games.select(
            "game_id",
            pl.col("home_abbr").alias("home_team"),
            pl.col("away_abbr").alias("away_team"),
            "neutral_site",
        )

        # Keep only games where both sides have leakage-safe ratings.
        rated_teams = set(ratings.get_column("team").to_list())
        games = games.filter(
            pl.col("home_team").is_in(rated_teams) & pl.col("away_team").is_in(rated_teams)
        )
        if games.is_empty():
            continue

        preds = nhl_predict_games(games, ratings)
        if preds.is_empty():
            continue

        day_df = preds.to_pandas()
        day_df.insert(1, "game_date", target_day.isoformat())
        day_df["nhl_season_start_year"] = season
        day_df["sdv_season_year"] = sdv_season_year(season)
        day_df["sdv_as_of_rule"] = "source_game_date < target_game_date"
        rows.append(day_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["game_id"] = canonical_game_id_series(out["game_id"])
    validate_official_game_ids(out["game_id"], label="SDV predictions", season=season)
    return out.sort_values(["game_date", "game_id"]).drop_duplicates("game_id", keep="first")


def prediction_csv_paths() -> list[Path]:
    # Search the active prediction folder plus its archives. Ignore anything
    # outside CSVs; later canonical game_id deduplication prevents duplicates.
    return sorted(path for path in PREDICTIONS_ROOT.rglob("*.csv") if path.is_file())


def load_dratings_predictions(season: int) -> pd.DataFrame:
    required = {
        "game_id",
        "game_date",
        "home_prob_moneyline",
        "away_projected_goals",
        "home_projected_goals",
        "total_projected_goals",
    }
    frames: list[pd.DataFrame] = []

    for path in prediction_csv_paths():
        try:
            frame = pd.read_csv(path, dtype={"game_id": "string"})
        except (OSError, UnicodeError, ValueError):
            continue
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["_source_file"] = repo_relative(path)
        frame["game_id"] = canonical_game_id_series(frame["game_id"])
        frame = frame[frame["game_id"].str.startswith(str(season), na=False)]
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"No D-Ratings prediction rows were found for NHL season start year {season}"
        )

    out = pd.concat(frames, ignore_index=True)
    validate_official_game_ids(out["game_id"], label="D-Ratings predictions", season=season)

    # Normalize game_date to ISO; source files use YYYY_MM_DD.
    out["game_date"] = (
        out["game_date"].astype("string").str.replace("_", "-", regex=False)
    )
    out["_parsed_game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out = out[out["_parsed_game_date"].notna()].copy()

    # Prefer a row whose declared game_date agrees with the official game date.
    # If the same game appears more than once, the lexicographically earliest
    # source file is used only after exact-date rows are prioritized. No values
    # are averaged across snapshots.
    out = out.sort_values(["game_id", "_parsed_game_date", "_source_file"])
    out = out.drop_duplicates(subset=["game_id"], keep="first")

    numeric = [
        "home_prob_moneyline",
        "away_projected_goals",
        "home_projected_goals",
        "total_projected_goals",
    ]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["drat_home_win_prob"] = out["home_prob_moneyline"]
    out["drat_exp_margin"] = out["home_projected_goals"] - out["away_projected_goals"]
    out["drat_exp_total"] = out["total_projected_goals"]

    return out[
        [
            "game_id",
            "game_date",
            "drat_home_win_prob",
            "drat_exp_margin",
            "drat_exp_total",
            "_source_file",
        ]
    ].copy()


def load_final_scores(season: int) -> pd.DataFrame:
    required = {"game_id", "game_date", "away_score", "home_score", "total_score"}
    frames: list[pd.DataFrame] = []
    for path in sorted(FINAL_SCORES_ROOT.glob("*.csv")):
        try:
            frame = pd.read_csv(path, dtype={"game_id": "string"})
        except (OSError, UnicodeError, ValueError):
            continue
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame["game_id"] = canonical_game_id_series(frame["game_id"])
        frame = frame[frame["game_id"].str.startswith(str(season), na=False)]
        if frame.empty:
            continue
        frame["_score_source_file"] = repo_relative(path)
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No final-score rows found for NHL season start year {season}")

    out = pd.concat(frames, ignore_index=True)
    validate_official_game_ids(out["game_id"], label="final scores", season=season)
    out = out.drop_duplicates(subset=["game_id"], keep="last")

    for column in ("home_score", "away_score", "total_score"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["home_score", "away_score", "total_score"])
    out["actual_home_win"] = (out["home_score"] > out["away_score"]).astype(int)
    out["actual_margin"] = out["home_score"] - out["away_score"]
    out["actual_total"] = out["total_score"]
    out["game_date"] = out["game_date"].astype("string").str.replace("_", "-", regex=False)

    return out[
        [
            "game_id",
            "game_date",
            "home_score",
            "away_score",
            "actual_home_win",
            "actual_margin",
            "actual_total",
            "_score_source_file",
        ]
    ].copy()


def build_comparison_frame(sdv: pd.DataFrame, drat: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if sdv.empty:
        return pd.DataFrame()

    renamed = sdv.rename(
        columns={
            "home_win_prob": "sdv_home_win_prob",
            "exp_margin": "sdv_exp_margin",
            "exp_total": "sdv_exp_total",
        }
    )
    out = renamed.merge(drat, on="game_id", how="inner", suffixes=("", "_drat"))
    out = out.merge(scores, on="game_id", how="inner", suffixes=("", "_score"))
    if out.empty:
        return out

    # Canonical target date comes from the P5 schedule / SDV walk-forward row.
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")
    out = out.dropna(
        subset=[
            "sdv_home_win_prob",
            "sdv_exp_margin",
            "sdv_exp_total",
            "drat_home_win_prob",
            "drat_exp_margin",
            "drat_exp_total",
            "actual_home_win",
            "actual_margin",
            "actual_total",
        ]
    ).copy()

    out["prob_disagreement"] = (out["sdv_home_win_prob"] - out["drat_home_win_prob"]).abs()
    out["margin_disagreement"] = (out["sdv_exp_margin"] - out["drat_exp_margin"]).abs()
    out["total_disagreement"] = (out["sdv_exp_total"] - out["drat_exp_total"]).abs()
    return out.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def clipped_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), EPS, 1.0 - EPS)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    p = clipped_prob(p)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = clipped_prob(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def standalone_metrics(frame: pd.DataFrame, prefix: str) -> StandaloneMetrics:
    y = frame["actual_home_win"].to_numpy(float)
    p = frame[f"{prefix}_home_win_prob"].to_numpy(float)
    actual_margin = frame["actual_margin"].to_numpy(float)
    margin = frame[f"{prefix}_exp_margin"].to_numpy(float)
    actual_total = frame["actual_total"].to_numpy(float)
    total = frame[f"{prefix}_exp_total"].to_numpy(float)
    return StandaloneMetrics(
        rows=len(frame),
        brier=brier_score(y, p),
        log_loss=log_loss(y, p),
        margin_mae=mae(actual_margin, margin),
        margin_rmse=rmse(actual_margin, margin),
        total_mae=mae(actual_total, total),
        total_rmse=rmse(actual_total, total),
    )


def metrics_to_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ("drat", "sdv"):
        m = standalone_metrics(frame, name)
        rows.extend(
            [
                {"model": name, "market": "moneyline", "metric": "brier", "value": m.brier, "rows": m.rows},
                {"model": name, "market": "moneyline", "metric": "log_loss", "value": m.log_loss, "rows": m.rows},
                {"model": name, "market": "margin", "metric": "mae", "value": m.margin_mae, "rows": m.rows},
                {"model": name, "market": "margin", "metric": "rmse", "value": m.margin_rmse, "rows": m.rows},
                {"model": name, "market": "total", "metric": "mae", "value": m.total_mae, "rows": m.rows},
                {"model": name, "market": "total", "metric": "rmse", "value": m.total_rmse, "rows": m.rows},
            ]
        )
    return pd.DataFrame(rows)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> LogisticModel:
    """L2-stabilized logistic regression using SciPy; no sklearn dependency."""
    y = np.asarray(y, dtype=float)
    design = design_matrix(x)

    def objective(coefficients: np.ndarray) -> float:
        z = np.clip(design @ coefficients, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        nll = -np.sum(y * np.log(np.clip(p, EPS, 1 - EPS)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1 - EPS)))
        penalty = 1e-4 * float(np.sum(coefficients[1:] ** 2))
        return float(nll + penalty)

    result = minimize(objective, np.zeros(design.shape[1]), method="BFGS")
    beta = result.x if result.success or np.isfinite(result.fun) else np.zeros(design.shape[1])
    return LogisticModel(beta)


def fit_linear(
    x: np.ndarray,
    y: np.ndarray,
) -> LinearModel:
    return LinearModel(
        ridge_linear_coefficients(x, y)
    )


def apply_logistic(model: LogisticModel, x: np.ndarray) -> np.ndarray:
    design = design_matrix(x)
    return model.predict_proba(design)


def apply_linear(model: LinearModel, x: np.ndarray) -> np.ndarray:
    design = design_matrix(x)
    return model.predict(design)


def build_walkforward_ensemble(frame: pd.DataFrame, min_train_rows: int) -> pd.DataFrame:
    """
    Evaluate three challenger-combination families with strict date splits:
      1) calibrated weighted ensemble,
      2) SDV + D-Ratings predictions feeding a meta-model,
      3) disagreement as a past-derived filter/feature.
    """
    if frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    work["_date"] = pd.to_datetime(work["game_date"], errors="coerce").dt.date
    work = work[work["_date"].notna()].sort_values(["_date", "game_id"]).copy()
    results: list[pd.DataFrame] = []

    for target_day in sorted(work["_date"].unique()):
        train = work[work["_date"] < target_day].copy()
        test = work[work["_date"] == target_day].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        if train["actual_home_win"].nunique() < 2:
            continue

        y_train = train["actual_home_win"].to_numpy(float)
        (
            drat_cal,
            sdv_cal,
            prob_weight,
            margin_weight,
            total_weight,
        ) = fit_blend_weights(
            train,
            y_train,
            fit_logistic,
            apply_logistic,
        )

        drat_test_cal = apply_logistic(
            drat_cal,
            test[["drat_home_win_prob"]].to_numpy(float),
        )
        sdv_test_cal = apply_logistic(
            sdv_cal,
            test[["sdv_home_win_prob"]].to_numpy(float),
        )
        weighted_prob = (
            prob_weight * drat_test_cal
            + (1.0 - prob_weight) * sdv_test_cal
        )

        # Meta-model: both models + disagreement enter as features. Fits use
        # only rows earlier than the target date.
        prob_meta_x_train = train[
            ["drat_home_win_prob", "sdv_home_win_prob", "prob_disagreement"]
        ].to_numpy(float)
        prob_meta_x_test = test[
            ["drat_home_win_prob", "sdv_home_win_prob", "prob_disagreement"]
        ].to_numpy(float)
        prob_meta = fit_logistic(prob_meta_x_train, y_train)

        margin_meta = fit_linear(
            train[["drat_exp_margin", "sdv_exp_margin", "margin_disagreement"]].to_numpy(float),
            train["actual_margin"].to_numpy(float),
        )
        total_meta = fit_linear(
            train[["drat_exp_total", "sdv_exp_total", "total_disagreement"]].to_numpy(float),
            train["actual_total"].to_numpy(float),
        )

        # Disagreement thresholds are learned from prior rows only.
        prob_disagreement_threshold = float(
            train["prob_disagreement"].quantile(0.75)
        )
        margin_disagreement_threshold = float(
            train["margin_disagreement"].quantile(0.75)
        )
        total_disagreement_threshold = float(
            train["total_disagreement"].quantile(0.75)
        )

        result = test.copy()
        result["ensemble_train_rows"] = len(train)
        result["weighted_prob_drat_weight"] = prob_weight
        result["weighted_margin_drat_weight"] = margin_weight
        result["weighted_total_drat_weight"] = total_weight
        result["weighted_home_win_prob"] = weighted_prob
        result["weighted_exp_margin"] = (
            margin_weight * result["drat_exp_margin"]
            + (1.0 - margin_weight) * result["sdv_exp_margin"]
        )
        result["weighted_exp_total"] = (
            total_weight * result["drat_exp_total"]
            + (1.0 - total_weight) * result["sdv_exp_total"]
        )
        result["meta_home_win_prob"] = apply_logistic(prob_meta, prob_meta_x_test)
        result["meta_exp_margin"] = apply_linear(
            margin_meta,
            result[["drat_exp_margin", "sdv_exp_margin", "margin_disagreement"]].to_numpy(float),
        )
        result["meta_exp_total"] = apply_linear(
            total_meta,
            result[["drat_exp_total", "sdv_exp_total", "total_disagreement"]].to_numpy(float),
        )
        result["disagreement_threshold_p75_prior"] = prob_disagreement_threshold
        result["high_disagreement_flag"] = (
            result["prob_disagreement"] >= prob_disagreement_threshold
        )
        result["prob_disagreement_threshold_p75_prior"] = prob_disagreement_threshold
        result["margin_disagreement_threshold_p75_prior"] = margin_disagreement_threshold
        result["total_disagreement_threshold_p75_prior"] = total_disagreement_threshold
        result["high_prob_disagreement_flag"] = (
            result["prob_disagreement"] >= prob_disagreement_threshold
        )
        result["high_margin_disagreement_flag"] = (
            result["margin_disagreement"] >= margin_disagreement_threshold
        )
        result["high_total_disagreement_flag"] = (
            result["total_disagreement"] >= total_disagreement_threshold
        )
        results.append(result)

    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    return out.drop(columns=["_date"], errors="ignore").sort_values(["game_date", "game_id"])


def combined_model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    specs = {
        "weighted": ("weighted_home_win_prob", "weighted_exp_margin", "weighted_exp_total"),
        "meta": ("meta_home_win_prob", "meta_exp_margin", "meta_exp_total"),
    }
    rows: list[dict[str, Any]] = []
    for model, (prob_col, margin_col, total_col) in specs.items():
        subset = frame.dropna(subset=[prob_col, margin_col, total_col]).copy()
        if subset.empty:
            continue
        y = subset["actual_home_win"].to_numpy(float)
        rows.extend(
            [
                {"model": model, "market": "moneyline", "metric": "brier", "value": brier_score(y, subset[prob_col].to_numpy(float)), "rows": len(subset)},
                {"model": model, "market": "moneyline", "metric": "log_loss", "value": log_loss(y, subset[prob_col].to_numpy(float)), "rows": len(subset)},
                {"model": model, "market": "margin", "metric": "mae", "value": mae(subset["actual_margin"].to_numpy(float), subset[margin_col].to_numpy(float)), "rows": len(subset)},
                {"model": model, "market": "margin", "metric": "rmse", "value": rmse(subset["actual_margin"].to_numpy(float), subset[margin_col].to_numpy(float)), "rows": len(subset)},
                {"model": model, "market": "total", "metric": "mae", "value": mae(subset["actual_total"].to_numpy(float), subset[total_col].to_numpy(float)), "rows": len(subset)},
                {"model": model, "market": "total", "metric": "rmse", "value": rmse(subset["actual_total"].to_numpy(float), subset[total_col].to_numpy(float)), "rows": len(subset)},
            ]
        )

    # Disagreement filter is an analysis slice, not a third probability model.
    for flag, label in ((False, "low_or_normal_disagreement"), (True, "high_disagreement")):
        subset = frame[frame["high_disagreement_flag"] == flag]
        if subset.empty:
            continue
        for base in ("drat", "sdv"):
            y = subset["actual_home_win"].to_numpy(float)
            rows.append(
                {
                    "model": f"{base}:{label}",
                    "market": "moneyline",
                    "metric": "brier",
                    "value": brier_score(y, subset[f"{base}_home_win_prob"].to_numpy(float)),
                    "rows": len(subset),
                }
            )
            rows.append(
                {
                    "model": f"{base}:{label}",
                    "market": "moneyline",
                    "metric": "log_loss",
                    "value": log_loss(y, subset[f"{base}_home_win_prob"].to_numpy(float)),
                    "rows": len(subset),
                }
            )
    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def output_paths(season: int) -> dict[str, Path]:
    season_dir = OUTPUT_ROOT / f"season_{season}"
    return {
        "dir": season_dir,
        "sdv_predictions": season_dir / "sdv_walkforward_predictions.csv",
        "standalone_predictions": season_dir / "standalone_comparison.csv",
        "standalone_metrics": season_dir / "standalone_metrics.csv",
        "ensemble_predictions": season_dir / "ensemble_walkforward_predictions.csv",
        "ensemble_metrics": season_dir / "ensemble_metrics.csv",
        "summary": season_dir / "summary.json",
    }


def refuse_existing(paths: dict[str, Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for key, path in paths.items() if key != "dir" and path.exists()]
    if existing:
        listing = "\n".join(f"  - {repo_relative(path)}" for path in existing)
        raise SystemExit(
            "Challenger output already exists. Use --overwrite to replace it:\n" + listing
        )


def standalone_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for model in ("drat", "sdv"):
        rows = metrics[metrics["model"] == model]
        payload[model] = {
            f"{row.market}_{row.metric}": float(row.value)
            for row in rows.itertuples(index=False)
        }
    return payload


def main() -> int:
    args = parse_args()
    if args.min_ensemble_train_rows < 20:
        raise SystemExit("--min-ensemble-train-rows must be at least 20")
    if args.min_team_games < 0:
        raise SystemExit("--min-team-games cannot be negative")

    installed = enforce_sdv_pin()
    require_p5_inputs(args.season)
    paths = output_paths(args.season)
    refuse_existing(paths, overwrite=args.overwrite)

    print(f"BUILD_SDV_CHALLENGER_VERSION | {SCRIPT_VERSION}")
    print(f"SPORTSDATAVERSE_VERSION | {installed}")
    print("MODE | RESEARCH_SUPPORTING_PRODUCTION | deployment_role=SECONDARY_CONFIRMATION_AND_CALIBRATED_SIGNAL")
    print(f"SEASON | NHL {args.season}-{str(args.season + 1)[-2:]} | SDV {sdv_season_year(args.season)}")
    print("LEAKAGE_RULE | source_game_date < target_game_date")

    schedule, game_rates = load_p5_game_rates(args.season)
    print(f"P5_GAME_RATES | {game_rates.height} team-game rows")

    sdv_predictions = build_sdv_walkforward_predictions(
        schedule,
        game_rates,
        season=args.season,
        include_playoffs=args.include_playoffs,
        min_team_games=args.min_team_games,
    )
    if sdv_predictions.empty:
        raise SystemExit("No leakage-safe SDV predictions could be generated")
    write_csv_atomic(sdv_predictions, paths["sdv_predictions"])
    print(f"SDV_PREDICTIONS | {len(sdv_predictions)} rows")

    drat = load_dratings_predictions(args.season)
    scores = load_final_scores(args.season)
    comparison = build_comparison_frame(sdv_predictions, drat, scores)
    if comparison.empty:
        raise SystemExit(
            "No official game_id overlap exists among SDV predictions, D-Ratings predictions, and final scores"
        )

    write_csv_atomic(comparison, paths["standalone_predictions"])
    standalone = metrics_to_frame(comparison)
    write_csv_atomic(standalone, paths["standalone_metrics"])
    print(f"STANDALONE_COMPARISON | {len(comparison)} completed matched games")

    ensemble_status = "not_requested"
    ensemble_rows = 0
    ensemble_metrics_payload: list[dict[str, Any]] = []

    if args.evaluate_ensemble:
        if len(comparison) < args.min_ensemble_train_rows:
            ensemble_status = "blocked_insufficient_standalone_rows"
            print(
                "ENSEMBLE_BLOCKED | standalone comparison has "
                f"{len(comparison)} rows; minimum is {args.min_ensemble_train_rows}"
            )
        else:
            ensemble = build_walkforward_ensemble(comparison, args.min_ensemble_train_rows)
            if ensemble.empty:
                ensemble_status = "blocked_no_walkforward_test_rows_after_training_gate"
                print("ENSEMBLE_BLOCKED | no post-training-gate target dates available")
            else:
                write_csv_atomic(ensemble, paths["ensemble_predictions"])
                ensemble_metrics = combined_model_metrics(ensemble)
                write_csv_atomic(ensemble_metrics, paths["ensemble_metrics"])
                ensemble_status = "evaluated_walkforward"
                ensemble_rows = len(ensemble)
                ensemble_metrics_payload = ensemble_metrics.to_dict(orient="records")
                print(f"ENSEMBLE_EVALUATED | {ensemble_rows} strict walk-forward test rows")

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "mode": "research_supporting_production",
        "deployment_role": "secondary_confirmation_and_calibrated_signal",
        "production_architecture": {
            "primary_model": "D-Ratings",
            "sdv_role": "challenger_confirmation_and_disagreement",
            "calibrated_secondary_role": "weighted_ensemble_and_meta_signal",
            "sole_model_replacement": False,
        },
        "deployment_candidates": [
            "secondary confirmation model",
            "disagreement filter",
            "calibrated weighted ensemble signal",
            "calibrated meta-model signal",
        ],
        "sportsdataverse_version": installed,
        "nhl_season_start_year": args.season,
        "sdv_season_year": sdv_season_year(args.season),
        "include_playoffs": bool(args.include_playoffs),
        "leakage_rule": "source_game_date < target_game_date",
        "same_day_source_games": "excluded",
        "canonical_join_key": "official 10-digit NHL game_id",
        "grading_source": repo_relative(FINAL_SCORES_ROOT),
        "sdv_history_source": repo_relative(SDV_HISTORY_ROOT),
        "dratings_source": repo_relative(PREDICTIONS_ROOT),
        "sdv_predictions_rows": len(sdv_predictions),
        "standalone_matched_completed_rows": len(comparison),
        "standalone_metrics": standalone_summary(standalone),
        "ensemble_requested": bool(args.evaluate_ensemble),
        "ensemble_status": ensemble_status,
        "ensemble_min_prior_rows": args.min_ensemble_train_rows,
        "ensemble_walkforward_test_rows": ensemble_rows,
        "ensemble_metrics": ensemble_metrics_payload,
        "ensemble_methods": {
            "calibrated_weighted": (
                "Each model probability is Platt-calibrated on prior rows only; "
                "D-Ratings weight is selected on a 0..1 grid by prior log loss. "
                "Margin and total weights are selected independently by prior RMSE."
            ),
            "sdv_features_feeding_another_model": (
                "Past-only logistic/linear meta-models use D-Ratings prediction, "
                "SDV prediction, and model disagreement as features."
            ),
            "disagreement_filter": (
                "High disagreement is defined independently for probability, margin, "
                "and total by the prior-row 75th percentile of absolute disagreement. "
                "The legacy high_disagreement_flag remains the moneyline probability flag."
            ),
        },
        "production_change": (
            "Approved for production as secondary confirmation/disagreement plus "
            "calibrated ensemble/meta signals; D-Ratings remains primary."
        ),
    }
    write_json_atomic(summary, paths["summary"])

    print(f"DONE | {repo_relative(paths['dir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
