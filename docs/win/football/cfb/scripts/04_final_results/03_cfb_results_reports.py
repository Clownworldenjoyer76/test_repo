#!/usr/bin/env python3
"""
CFB graded-results report builder.

Consumes the one-row-per-bet ledger produced by 02_cfb_results_analyze.py and
writes season, week, market, side, EV, odds, Kelly, probability, spread-line,
and total-range reports modeled on the MLB grading-report layer.

READS:
    docs/win/football/cfb/04_final_results/intermediate/work_cfb.csv

WRITES:
    docs/win/football/cfb/04_final_results/cfb_summary_overall.csv
    docs/win/football/cfb/04_final_results/reports/...
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCRIPT_VERSION = "cfb-results-reports-v3-signed-spread-sort-2026-10-01"

SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
REPORT_ROOT = CFB_ROOT / "errors"
CURRENT_WEEK_CONFIG = CFB_ROOT / "config" / "current_week.yaml"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

DEFAULT_INPUT_FILE = CFB_ROOT / "04_final_results" / "intermediate" / "work_cfb.csv"

FINAL_RESULTS_DIR = CFB_ROOT / "04_final_results"
FINAL_SUMMARY_FILE = FINAL_RESULTS_DIR / "cfb_summary_overall.csv"
FINAL_REPORTS_DIR = FINAL_RESULTS_DIR / "reports"

@dataclass(frozen=True)
class OutputPaths:
    summary_dir: Path
    reports_dir: Path
    overview_dir: Path
    ml_dir: Path
    spread_dir: Path
    total_dir: Path


def output_paths(root: Path) -> OutputPaths:
    reports_dir = root / "reports"

    return OutputPaths(
        summary_dir=root,
        reports_dir=reports_dir,
        overview_dir=reports_dir / "overview",
        ml_dir=reports_dir / "moneyline",
        spread_dir=reports_dir / "spread",
        total_dir=reports_dir / "totals",
    )

LEAGUE = "CFB"

VALID_RESULTS = {
    "Win",
    "Loss",
    "Push",
    "Void",
    "Pending",
    "Invalid Selection",
    "Invalid Line",
    "No Bet",
}
SETTLED_RESULTS = {"Win", "Loss", "Push", "Void"}
MARKET_TYPES = {"moneyline", "spread", "total"}

SPREAD_LINE_BUCKET_ORDER = {
    "negative_28.0_plus": 0,
    "negative_21.0_to_27.9": 1,
    "negative_14.0_to_20.9": 2,
    "negative_10.0_to_13.9": 3,
    "negative_7.0_to_9.9": 4,
    "negative_3.0_to_6.9": 5,
    "negative_0.0_to_2.9": 6,
    "0.0": 7,
    "positive_0.0_to_2.9": 8,
    "positive_3.0_to_6.9": 9,
    "positive_7.0_to_9.9": 10,
    "positive_10.0_to_13.9": 11,
    "positive_14.0_to_20.9": 12,
    "positive_21.0_to_27.9": 13,
    "positive_28.0_plus": 14,
}
SIDE_GROUP_ORDER = {
    "HOME": 0,
    "AWAY": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def to_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def clear_report_outputs(
    paths: OutputPaths,
) -> None:
    if paths.reports_dir.exists():
        shutil.rmtree(paths.reports_dir)

    for directory in [
        paths.overview_dir,
        paths.ml_dir,
        paths.spread_dir,
        paths.total_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )



def normalize_result(value: Any) -> str:
    raw = clean(value).casefold()
    mapping = {
        "win": "Win",
        "loss": "Loss",
        "push": "Push",
        "void": "Void",
        "pending": "Pending",
        "invalid selection": "Invalid Selection",
        "invalid line": "Invalid Line",
        "no bet": "No Bet",
    }

    result = mapping.get(raw)

    if result is None:
        raise RuntimeError(
            f"Unsupported bet_result value: {value!r}"
        )

    return result

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def numeric_sort_value(value: Any) -> tuple[int, float | str]:
    number = to_float(value)
    return (0, number) if number is not None else (1, clean(value))


def required_float(value: Any, label: str) -> float:
    text = clean(value)

    if not text:
        raise RuntimeError(f"{label} is required")

    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} must be numeric; found {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise RuntimeError(
            f"{label} must be finite; found {value!r}"
        )

    return number


def required_int(value: Any, label: str) -> int:
    number = required_float(value, label)

    if not number.is_integer():
        raise RuntimeError(
            f"{label} must be a whole number; found {value!r}"
        )

    return int(number)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing config: {path}")

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{path} must contain a YAML mapping"
        )

    return data


def resolve_target(
    config: dict[str, Any],
    season_override: int | None,
    season_type_override: int | None,
) -> tuple[int, int]:
    season = (
        required_int(config.get("season"), "current_week.season")
        if season_override is None
        else int(season_override)
    )

    season_type = (
        required_int(
            config.get("season_type"),
            "current_week.season_type",
        )
        if season_type_override is None
        else int(season_type_override)
    )

    if season < 1900:
        raise RuntimeError(f"Invalid season: {season}")

    if season_type <= 0:
        raise RuntimeError(
            f"Invalid season_type: {season_type}"
        )

    return season, season_type


def _validate_selected_bet_units(
    *,
    result: str,
    units: str,
    line: int,
) -> None:
    if result in SETTLED_RESULTS:
        required_float(
            units,
            f"work_cfb.csv line {line}: bet_units",
        )
    elif units:
        required_float(
            units,
            f"work_cfb.csv line {line}: bet_units",
        )


def validate_input(
    df: pd.DataFrame,
    season: int,
    season_type: int,
) -> None:
    required = [
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",
        "game_time",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "side_group",
        "line",
        "odds_american",
        "model_prob",
        "implied_prob",
        "edge",
        "ev",
        "full_kelly",
        "kelly",
        "selection_reason",
        "bet_result",
        "bet_units",
        "final_status",
        "final_away_score",
        "final_home_score",
        "final_total",
        "final_home_margin",
        "ev_bucket",
        "odds_bucket",
        "kelly_bucket",
        "win_prob_bucket",
        "spread_line_bucket",
        "spread_role",
        "total_bucket",
        "day_night",
    ]

    require_columns(df, required, "work_cfb.csv")

    seen: set[tuple[int, int, str, str]] = set()

    for line, (_, row) in enumerate(
        df.iterrows(),
        start=2,
    ):
        row_season = required_int(
            row["season"],
            f"work_cfb.csv line {line}: season",
        )
        row_season_type = required_int(
            row["season_type"],
            f"work_cfb.csv line {line}: season_type",
        )
        week = required_int(
            row["week"],
            f"work_cfb.csv line {line}: week",
        )
        game_id = clean(row["game_id"])
        market = clean(row["market_type"]).lower()
        result = normalize_result(row["bet_result"])
        side_group = clean(row["side_group"]).upper()

        if row_season != season:
            raise RuntimeError(
                f"work_cfb.csv line {line}: wrong season"
            )

        if row_season_type != season_type:
            raise RuntimeError(
                f"work_cfb.csv line {line}: wrong season_type"
            )

        if week <= 0:
            raise RuntimeError(
                f"work_cfb.csv line {line}: invalid week"
            )

        if not game_id:
            raise RuntimeError(
                f"work_cfb.csv line {line}: blank game_id"
            )

        if market not in MARKET_TYPES:
            raise RuntimeError(
                f"work_cfb.csv line {line}: invalid market_type"
            )

        if result == "No Bet":
            raise RuntimeError(
                f"work_cfb.csv line {line}: selected ledger contains No Bet"
            )

        allowed_sides = (
            {"OVER", "UNDER"}
            if market == "total"
            else {"HOME", "AWAY"}
        )

        if side_group not in allowed_sides:
            raise RuntimeError(
                f"work_cfb.csv line {line}: invalid side_group"
            )

        odds = required_float(
            row["odds_american"],
            f"work_cfb.csv line {line}: odds_american",
        )
        if odds == 0:
            raise RuntimeError(
                f"work_cfb.csv line {line}: zero odds"
            )

        for column in (
            "ev",
            "edge",
            "kelly",
            "full_kelly",
        ):
            required_float(
                row[column],
                f"work_cfb.csv line {line}: {column}",
            )

        for column in (
            "model_prob",
            "implied_prob",
        ):
            probability = required_float(
                row[column],
                f"work_cfb.csv line {line}: {column}",
            )

            if not 0.0 <= probability <= 1.0:
                raise RuntimeError(
                    f"work_cfb.csv line {line}: "
                    f"{column} outside [0,1]"
                )

        units = clean(row["bet_units"])

        _validate_selected_bet_units(
            result=result,
            units=units,
            line=line,
        )

        key = (
            row_season,
            week,
            game_id,
            market,
        )

        if key in seen:
            raise RuntimeError(
                f"work_cfb.csv line {line}: duplicate selected market"
            )

        seen.add(key)


def read_generated_csv(
    path: Path,
    require_rows: bool,
) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Generated report missing or empty: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if len(frame.columns) == 0:
        raise RuntimeError(
            f"Generated report missing header: {path}"
        )

    if require_rows and frame.empty:
        raise RuntimeError(
            f"Generated report contains no rows: {path}"
        )

    return frame


def validate_generated(
    root: Path,
    work: pd.DataFrame,
) -> None:
    has_bets = not work.empty

    top = read_generated_csv(
        root / "cfb_summary_overall.csv",
        has_bets,
    )

    require_columns(
        top,
        [
            "league",
            "season",
            "market_type",
            "Selected",
            "units",
        ],
        "generated cfb_summary_overall.csv",
    )

    core = [
        "cfb_summary_overall.csv",
        "cfb_summary_by_market.csv",
        "cfb_summary_by_week.csv",
        "cfb_bet_log.csv",
    ]

    for name in core:
        read_generated_csv(
            root / "reports" / "overview" / name,
            has_bets,
        )

    for directory in (
        "moneyline",
        "spread",
        "totals",
    ):
        files = sorted(
            (root / "reports" / directory).glob("*.csv")
        )

        if not files:
            raise RuntimeError(
                f"Generated {directory} directory has no CSV reports"
            )

        for path in files:
            read_generated_csv(path, False)

    if has_bets:
        selected = sum(
            required_int(value, "generated Selected")
            for value in top["Selected"]
        )

        if selected != len(work):
            raise RuntimeError(
                "Generated summary Selected total does not match work_cfb.csv"
            )

        input_units = float(
            pd.to_numeric(
                work["bet_units"],
                errors="coerce",
            )
            .dropna()
            .sum()
        )

        report_units = sum(
            required_float(value, "generated units")
            for value in top["units"]
        )

        if not math.isclose(
            input_units,
            report_units,
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise RuntimeError(
                "Generated summary units do not match work_cfb.csv"
            )

    elif not top.empty:
        raise RuntimeError(
            "Generated summary has rows despite zero selected bets"
        )


def tree_snapshot(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        return {}

    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.csv"))
    }


def publish_generated(stage_root: Path) -> bool:
    stage_summary = stage_root / "cfb_summary_overall.csv"
    stage_reports = stage_root / "reports"

    modified = (
        not FINAL_SUMMARY_FILE.is_file()
        or FINAL_SUMMARY_FILE.read_bytes()
        != stage_summary.read_bytes()
        or tree_snapshot(FINAL_REPORTS_DIR)
        != tree_snapshot(stage_reports)
    )

    if not modified:
        return False

    token = uuid.uuid4().hex
    summary_backup = FINAL_RESULTS_DIR / f".cfb_summary_overall.{token}.bak"
    reports_backup = FINAL_RESULTS_DIR / f".reports.{token}.bak"

    had_summary = FINAL_SUMMARY_FILE.exists()
    had_reports = FINAL_REPORTS_DIR.exists()

    try:
        if had_summary:
            os.replace(
                FINAL_SUMMARY_FILE,
                summary_backup,
            )

        if had_reports:
            os.replace(
                FINAL_REPORTS_DIR,
                reports_backup,
            )

        os.replace(
            stage_summary,
            FINAL_SUMMARY_FILE,
        )

        os.replace(
            stage_reports,
            FINAL_REPORTS_DIR,
        )

    except Exception:
        FINAL_SUMMARY_FILE.unlink(
            missing_ok=True
        )

        if FINAL_REPORTS_DIR.exists():
            shutil.rmtree(
                FINAL_REPORTS_DIR,
                ignore_errors=True,
            )

        if summary_backup.exists():
            os.replace(
                summary_backup,
                FINAL_SUMMARY_FILE,
            )

        if reports_backup.exists():
            os.replace(
                reports_backup,
                FINAL_REPORTS_DIR,
            )

        raise

    finally:
        summary_backup.unlink(
            missing_ok=True
        )

        if reports_backup.exists():
            shutil.rmtree(
                reports_backup,
                ignore_errors=True,
            )

    return True


# ---------------------------------------------------------------------------
# Metric definitions / aggregation
# ---------------------------------------------------------------------------

def write_metric_definitions(
    paths: OutputPaths,
) -> None:
    definitions = pd.DataFrame(
        [
            {
                "metric": "Win_Pct",
                "definition": "wins / (wins + losses)",
                "push_handling": "pushes excluded from denominator",
            },
            {
                "metric": "Win_Pct_All_Bets",
                "definition": "wins / (wins + losses + pushes)",
                "push_handling": "pushes included in denominator",
            },
            {
                "metric": "units",
                "definition": "sum of grader-produced bet_units",
                "push_handling": "pushes and voids contribute 0 units",
            },
            {
                "metric": "ROI_Excluding_Pushes",
                "definition": "units / (wins + losses)",
                "push_handling": "pushes excluded from denominator",
            },
            {
                "metric": "ROI_Including_Pushes",
                "definition": "units / (wins + losses + pushes)",
                "push_handling": "pushes included in denominator",
            },
            {
                "metric": "Total",
                "definition": "wins + losses + pushes",
                "push_handling": "void, pending, and invalid bets excluded",
            },
            {
                "metric": "Selected",
                "definition": "all selected wagers present in work_cfb.csv",
                "push_handling": "includes void, pending, and invalid selected wagers",
            },
        ]
    )
    write_csv(definitions, paths.overview_dir / "cfb_report_metric_definitions.csv")


def build_metric_row(sub: pd.DataFrame) -> dict[str, Any]:
    results = sub["bet_result"]

    wins = int((results == "Win").sum())
    losses = int((results == "Loss").sum())
    pushes = int((results == "Push").sum())
    voids = int((results == "Void").sum())
    pending = int((results == "Pending").sum())
    invalid = int(
        (~results.isin({"Win", "Loss", "Push", "Void", "Pending"})).sum()
    )

    bets_excluding_pushes = wins + losses
    bets_including_pushes = wins + losses + pushes
    selected = len(sub)

    win_pct = (
        round(wins / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )
    win_pct_all_bets = (
        round(wins / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    units = pd.to_numeric(sub["bet_units"], errors="coerce").dropna()
    total_units = round(float(units.sum()), 4) if not units.empty else 0.0

    roi_excluding_pushes = (
        round(total_units / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )
    roi_including_pushes = (
        round(total_units / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    ev_vals = pd.to_numeric(sub["ev"], errors="coerce").dropna()
    odds_vals = pd.to_numeric(sub["odds_american"], errors="coerce").dropna()
    prob_vals = pd.to_numeric(sub["model_prob"], errors="coerce").dropna()
    kelly_vals = pd.to_numeric(sub["kelly"], errors="coerce").dropna()

    return {
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "Void": voids,
        "Pending": pending,
        "Invalid": invalid,
        "Total": bets_including_pushes,
        "Selected": selected,
        "bets_excluding_pushes": bets_excluding_pushes,
        "bets_including_pushes": bets_including_pushes,
        "Win_Pct": win_pct,
        "Win_Pct_All_Bets": win_pct_all_bets,
        "units": total_units,
        "ROI_Excluding_Pushes": roi_excluding_pushes,
        "ROI_Including_Pushes": roi_including_pushes,
        "avg_ev": round(float(ev_vals.mean()), 4) if not ev_vals.empty else None,
        "avg_odds": round(float(odds_vals.mean()), 1) if not odds_vals.empty else None,
        "avg_model_prob": (
            round(float(prob_vals.mean()), 4) if not prob_vals.empty else None
        ),
        "avg_kelly": (
            round(float(kelly_vals.mean()), 4) if not kelly_vals.empty else None
        ),
    }


def metric_columns(prefix: list[str]) -> list[str]:
    return prefix + [
        "Win",
        "Loss",
        "Push",
        "Void",
        "Pending",
        "Invalid",
        "Total",
        "Selected",
        "bets_excluding_pushes",
        "bets_including_pushes",
        "Win_Pct",
        "Win_Pct_All_Bets",
        "units",
        "ROI_Excluding_Pushes",
        "ROI_Including_Pushes",
        "avg_ev",
        "avg_odds",
        "avg_model_prob",
        "avg_kelly",
    ]


def aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    variable_label: str | None = None,
) -> pd.DataFrame:
    prefix_cols = [
        "variable" if variable_label and col == group_cols[-1] else col
        for col in group_cols
    ]

    if df.empty:
        return pd.DataFrame(columns=metric_columns(prefix_cols))

    rows: list[dict[str, Any]] = []

    for keys, sub in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = {}
        for index, column in enumerate(group_cols):
            label = "variable" if variable_label and index == len(group_cols) - 1 else column
            row[label] = keys[index]

        row.update(build_metric_row(sub))
        rows.append(row)

    return pd.DataFrame(rows)


def write_bucket_report(
    src: pd.DataFrame,
    output_dir: Path,
    bucket_col: str,
    filename: str,
    extra_group_cols: list[str] | None = None,
) -> None:
    if src.empty or bucket_col not in src.columns:
        return

    usable = src[src[bucket_col].astype(str).ne("UNBUCKETED")].copy()
    if usable.empty:
        return

    groups = ["league", "season", "market_type"]
    if extra_group_cols:
        groups.extend(extra_group_cols)
    groups.append(bucket_col)

    report = aggregate(usable, groups, variable_label=bucket_col)
    write_csv(report, output_dir / filename)


def write_spread_line_home_away_report(
    src: pd.DataFrame,
    output_dir: Path,
) -> None:
    bucket_col = "spread_line_bucket"

    if src.empty or bucket_col not in src.columns:
        return

    usable = src[src[bucket_col].astype(str).ne("UNBUCKETED")].copy()
    if usable.empty:
        return

    report = aggregate(
        usable,
        [
            "league",
            "season",
            "market_type",
            "side_group",
            bucket_col,
        ],
        variable_label=bucket_col,
    )

    report["_side_sort"] = report["side_group"].map(
        SIDE_GROUP_ORDER
    )
    report["_variable_sort"] = report["variable"].map(
        SPREAD_LINE_BUCKET_ORDER
    )

    if report["_side_sort"].isna().any():
        unknown = sorted(
            report.loc[
                report["_side_sort"].isna(),
                "side_group",
            ].astype(str).unique().tolist()
        )
        raise RuntimeError(
            "Unsupported side_group in spread line report: "
            f"{unknown}"
        )

    if report["_variable_sort"].isna().any():
        unknown = sorted(
            report.loc[
                report["_variable_sort"].isna(),
                "variable",
            ].astype(str).unique().tolist()
        )
        raise RuntimeError(
            "Unsupported spread_line_bucket in spread line report: "
            f"{unknown}"
        )

    report = report.sort_values(
        [
            "_side_sort",
            "_variable_sort",
        ],
        kind="stable",
    ).drop(
        columns=[
            "_side_sort",
            "_variable_sort",
        ]
    )

    write_csv(
        report,
        output_dir / "cfb_spread_by_line_home_away_summary.csv",
    )


# ---------------------------------------------------------------------------
# Enrichment / reports
# ---------------------------------------------------------------------------


def enrich(
    df: pd.DataFrame,
    season: int,
    season_type: int,
) -> pd.DataFrame:
    work = df.copy()

    validate_input(
        work,
        season,
        season_type,
    )

    work["league"] = LEAGUE
    work["market_type"] = (
        work["market_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["bet_side"] = (
        work["bet_side"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    work["side_group"] = (
        work["side_group"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    work["bet_result"] = work["bet_result"].map(
        normalize_result
    )

    return work

def build_probability_validation(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    probability_metric_columns = [
        "league",
        "season",
        "market_type",
        "settled_bets",
        "brier_score",
        "log_loss",
        "avg_model_prob",
        "observed_win_rate",
        "calibration_bias",
        "expected_calibration_error",
    ]
    calibration_columns = [
        "league",
        "season",
        "market_type",
        "probability_bucket",
        "settled_bets",
        "avg_model_prob",
        "observed_win_rate",
        "calibration_gap",
        "abs_calibration_gap",
    ]

    settled = df[df["bet_result"].isin({"Win", "Loss"})].copy()
    settled["_model_prob"] = pd.to_numeric(
        settled["model_prob"],
        errors="coerce",
    )
    settled = settled[
        settled["_model_prob"].between(0.0, 1.0, inclusive="both")
    ].copy()
    settled["_outcome"] = settled["bet_result"].eq("Win").astype(float)

    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    scopes: list[tuple[Any, Any, str, pd.DataFrame]] = []

    for (league, season), season_sub in settled.groupby(
        ["league", "season"],
        dropna=False,
        sort=False,
    ):
        scopes.append((league, season, "all", season_sub))

        for market_type, market_sub in season_sub.groupby(
            "market_type",
            dropna=False,
            sort=False,
        ):
            scopes.append((league, season, str(market_type), market_sub))

    for league, season, market_type, sub in scopes:
        probabilities = sub["_model_prob"].astype(float)
        outcomes = sub["_outcome"].astype(float)
        clipped = probabilities.clip(lower=1e-15, upper=1.0 - 1e-15)

        brier_score = float(((probabilities - outcomes) ** 2).mean())
        log_loss = float(
            -(
                outcomes * clipped.map(math.log)
                + (1.0 - outcomes) * (1.0 - clipped).map(math.log)
            ).mean()
        )
        avg_model_prob = float(probabilities.mean())
        observed_win_rate = float(outcomes.mean())

        expected_calibration_error = 0.0

        for probability_bucket, bucket_sub in sub.groupby(
            "win_prob_bucket",
            dropna=False,
            sort=False,
        ):
            bucket_prob = float(bucket_sub["_model_prob"].mean())
            bucket_win_rate = float(bucket_sub["_outcome"].mean())
            calibration_gap = bucket_win_rate - bucket_prob
            bucket_count = len(bucket_sub)

            expected_calibration_error += (
                bucket_count / len(sub)
            ) * abs(calibration_gap)

            calibration_rows.append(
                {
                    "league": league,
                    "season": season,
                    "market_type": market_type,
                    "probability_bucket": probability_bucket,
                    "settled_bets": bucket_count,
                    "avg_model_prob": round(bucket_prob, 6),
                    "observed_win_rate": round(bucket_win_rate, 6),
                    "calibration_gap": round(calibration_gap, 6),
                    "abs_calibration_gap": round(abs(calibration_gap), 6),
                }
            )

        metric_rows.append(
            {
                "league": league,
                "season": season,
                "market_type": market_type,
                "settled_bets": len(sub),
                "brier_score": round(brier_score, 6),
                "log_loss": round(log_loss, 6),
                "avg_model_prob": round(avg_model_prob, 6),
                "observed_win_rate": round(observed_win_rate, 6),
                "calibration_bias": round(
                    observed_win_rate - avg_model_prob,
                    6,
                ),
                "expected_calibration_error": round(
                    expected_calibration_error,
                    6,
                ),
            }
        )

    write_csv(
        pd.DataFrame(
            metric_rows,
            columns=probability_metric_columns,
        ),
        paths.overview_dir / "cfb_probability_metrics.csv",
    )
    write_csv(
        pd.DataFrame(calibration_rows, columns=calibration_columns),
        paths.overview_dir / "cfb_calibration_by_probability.csv",
    )

def build_top_summary(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    report = aggregate(df, ["league", "season", "market_type"])
    write_csv(report, paths.summary_dir / "cfb_summary_overall.csv")


def build_overview(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    write_metric_definitions(paths)
    build_probability_validation(
        df,
        paths,
    )

    write_csv(
        aggregate(df, ["league", "season"]),
        paths.overview_dir / "cfb_summary_overall.csv",
    )

    write_csv(
        aggregate(
            df,
            ["league", "season", "market_type"],
            variable_label="market_type",
        ),
        paths.overview_dir / "cfb_summary_by_market.csv",
    )

    side = df[df["side_group"].isin(["HOME", "AWAY", "OVER", "UNDER"])].copy()
    write_csv(
        aggregate(
            side,
            ["league", "season", "side_group"],
            variable_label="side_group",
        ),
        paths.overview_dir / "cfb_summary_by_side_group.csv",
    )

    by_week = aggregate(
        df,
        ["league", "season", "week"],
        variable_label="week",
    )

    if not by_week.empty:
        by_week["_season_sort"] = pd.to_numeric(by_week["season"], errors="coerce")
        by_week["_week_sort"] = pd.to_numeric(by_week["variable"], errors="coerce")
        by_week = by_week.sort_values(
            ["_season_sort", "_week_sort"],
            kind="stable",
        ).drop(columns=["_season_sort", "_week_sort"])

    write_csv(by_week, paths.overview_dir / "cfb_summary_by_week.csv")

    cumulative = by_week.copy()
    if not cumulative.empty:
        cumulative["cumulative_units"] = (
            cumulative.groupby("season", dropna=False)["units"].cumsum().round(4)
        )
    write_csv(cumulative, paths.overview_dir / "cfb_cumulative_units_by_week.csv")

    if "day_night" in df.columns:
        timed = df[df["day_night"].astype(str).str.strip().ne("")].copy()
        write_csv(
            aggregate(
                timed,
                ["league", "season", "day_night"],
                variable_label="day_night",
            ),
            paths.overview_dir / "cfb_summary_by_day_night.csv",
        )

    log_columns = [
        "season",
        "season_type",
        "week",
        "game_date",
        "game_time",
        "game_id",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "side_group",
        "line",
        "odds_american",
        "model_prob",
        "implied_prob",
        "edge",
        "ev",
        "kelly",
        "selection_reason",
        "bet_result",
        "bet_units",
        "final_status",
        "final_away_score",
        "final_home_score",
        "final_total",
        "final_home_margin",
    ]
    available = [column for column in log_columns if column in df.columns]
    write_csv(df[available].copy(), paths.overview_dir / "cfb_bet_log.csv")


def build_moneyline(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    ml = df[df["market_type"].eq("moneyline")].copy()
    write_bucket_report(ml, paths.ml_dir, "ev_bucket", "cfb_moneyline_by_ev.csv")
    write_bucket_report(ml, paths.ml_dir, "odds_bucket", "cfb_moneyline_by_odds.csv")
    write_bucket_report(ml, paths.ml_dir, "kelly_bucket", "cfb_moneyline_by_kelly.csv")
    write_bucket_report(ml, paths.ml_dir, "win_prob_bucket", "cfb_moneyline_by_win_prob.csv")

    home_away = ml[ml["side_group"].isin(["HOME", "AWAY"])].copy()
    write_csv(
        aggregate(
            home_away,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        paths.ml_dir / "cfb_moneyline_by_home_away.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_moneyline_by_ev_home_away_summary.csv"),
        ("odds_bucket", "cfb_moneyline_by_odds_home_away_summary.csv"),
        ("kelly_bucket", "cfb_moneyline_by_kelly_home_away_summary.csv"),
        ("win_prob_bucket", "cfb_moneyline_by_win_prob_home_away_summary.csv"),
    ]:
        write_bucket_report(
            home_away,
            paths.ml_dir,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )


def build_spread(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    spread = df[df["market_type"].eq("spread")].copy()
    write_bucket_report(spread, paths.spread_dir, "ev_bucket", "cfb_spread_by_ev.csv")
    write_bucket_report(spread, paths.spread_dir, "odds_bucket", "cfb_spread_by_odds.csv")
    write_bucket_report(spread, paths.spread_dir, "kelly_bucket", "cfb_spread_by_kelly.csv")
    write_bucket_report(spread, paths.spread_dir, "win_prob_bucket", "cfb_spread_by_win_prob.csv")
    write_bucket_report(spread, paths.spread_dir, "spread_line_bucket", "cfb_spread_by_line.csv")

    home_away = spread[spread["side_group"].isin(["HOME", "AWAY"])].copy()
    write_csv(
        aggregate(
            home_away,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        paths.spread_dir / "cfb_spread_by_home_away.csv",
    )

    roles = spread[spread["spread_role"].ne("UNBUCKETED")].copy()
    write_csv(
        aggregate(
            roles,
            ["league", "season", "market_type", "spread_role"],
            variable_label="spread_role",
        ),
        paths.spread_dir / "cfb_spread_by_favorite_underdog.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_spread_by_ev_home_away_summary.csv"),
        ("odds_bucket", "cfb_spread_by_odds_home_away_summary.csv"),
        ("kelly_bucket", "cfb_spread_by_kelly_home_away_summary.csv"),
        ("win_prob_bucket", "cfb_spread_by_win_prob_home_away_summary.csv"),
    ]:
        write_bucket_report(
            home_away,
            paths.spread_dir,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )

    write_spread_line_home_away_report(
        home_away,
        paths.spread_dir,
    )


def build_totals(
    df: pd.DataFrame,
    paths: OutputPaths,
) -> None:
    totals = df[df["market_type"].eq("total")].copy()
    write_bucket_report(totals, paths.total_dir, "ev_bucket", "cfb_total_by_ev.csv")
    write_bucket_report(totals, paths.total_dir, "odds_bucket", "cfb_total_by_odds.csv")
    write_bucket_report(totals, paths.total_dir, "kelly_bucket", "cfb_total_by_kelly.csv")
    write_bucket_report(totals, paths.total_dir, "win_prob_bucket", "cfb_total_by_win_prob.csv")
    write_bucket_report(totals, paths.total_dir, "total_bucket", "cfb_total_by_total_range.csv")

    over_under = totals[totals["side_group"].isin(["OVER", "UNDER"])].copy()
    write_csv(
        aggregate(
            over_under,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        paths.total_dir / "cfb_total_by_over_under.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_total_by_ev_over_under_summary.csv"),
        ("odds_bucket", "cfb_total_by_odds_over_under_summary.csv"),
        ("kelly_bucket", "cfb_total_by_kelly_over_under_summary.csv"),
        ("win_prob_bucket", "cfb_total_by_win_prob_over_under_summary.csv"),
        ("total_bucket", "cfb_total_by_total_range_over_under_summary.csv"),
    ]:
        write_bucket_report(
            over_under,
            paths.total_dir,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validated CFB grading reports."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "Target season. Defaults to "
            "config/current_week.yaml."
        ),
    )

    parser.add_argument(
        "--season-type",
        type=int,
        default=None,
        help=(
            "Target season type. Defaults to "
            "config/current_week.yaml."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="04_final_results",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        season=args.season,
        week=None,
        extra_context={
            "script_version": SCRIPT_VERSION,
            "output_scope": "grading_reports",
        },
    ) as report:
        input_file = args.input.resolve()

        report.add_input(
            CURRENT_WEEK_CONFIG
        )
        report.add_input(
            input_file
        )

        if not input_file.is_file():
            raise FileNotFoundError(
                f"Input file not found: {input_file}"
            )

        config = load_config(
            CURRENT_WEEK_CONFIG
        )

        season, season_type = resolve_target(
            config,
            args.season,
            args.season_type,
        )

        report.season = season

        df = pd.read_csv(
            input_file,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        df = enrich(
            df,
            season,
            season_type,
        )

        stage_root = Path(
            tempfile.mkdtemp(
                prefix=".cfb_reports_",
                dir=FINAL_RESULTS_DIR,
            )
        )

        paths = output_paths(
            stage_root
        )

        try:
            clear_report_outputs(
                paths
            )

            build_top_summary(
                df,
                paths,
            )
            build_overview(
                df,
                paths,
            )
            build_moneyline(
                df,
                paths,
            )
            build_spread(
                df,
                paths,
            )
            build_totals(
                df,
                paths,
            )

            validate_generated(
                stage_root,
                df,
            )

            output_modified = publish_generated(
                stage_root
            )

        finally:
            shutil.rmtree(
                stage_root,
                ignore_errors=True,
            )

        report.add_output(
            FINAL_SUMMARY_FILE
        )

        report_files = sorted(
            FINAL_REPORTS_DIR.rglob(
                "*.csv"
            )
        )

        for path in report_files:
            report.add_output(
                path
            )

        market_counts = (
            df["market_type"]
            .value_counts(dropna=False)
            .to_dict()
            if not df.empty
            else {}
        )

        result_counts = (
            df["bet_result"]
            .value_counts(dropna=False)
            .to_dict()
            if not df.empty
            else {}
        )

        settled_bets = int(
            df["bet_result"]
            .isin(SETTLED_RESULTS)
            .sum()
        )

        probability_settled_bets = int(
            df["bet_result"]
            .isin({"Win", "Loss"})
            .sum()
        )

        units = (
            float(
                pd.to_numeric(
                    df["bet_units"],
                    errors="coerce",
                )
                .dropna()
                .sum()
            )
            if not df.empty
            else 0.0
        )

        report.set_rows(
            rows_in=len(df),
            rows_out=len(df),
        )

        report.update_details(
            {
                "script_version":
                    SCRIPT_VERSION,
                "season_type":
                    season_type,
                "selected_bets":
                    len(df),
                "settled_bets":
                    settled_bets,
                "probability_settled_bets":
                    probability_settled_bets,
                "units":
                    units,
                "market_counts":
                    market_counts,
                "result_counts":
                    result_counts,
                "report_files":
                    len(report_files) + 1,
                "output_modified":
                    output_modified,
            }
        )

        print(
            "CFB reports complete: "
            f"version={SCRIPT_VERSION} "
            f"selected_bets={len(df)} "
            f"report_files={len(report_files) + 1} "
            "output_modified="
            f"{'yes' if output_modified else 'no'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
