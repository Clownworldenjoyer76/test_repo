#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/03_nhl_results_reports.py

from datetime import datetime, UTC
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# noinspection PyPep8
from market_common import normalize_market, normalize_side
from odds_price_common import american_to_decimal, price_metrics


###############################################################
######################## PATH CONFIG ##########################
###############################################################

NHL_ROOT = Path("docs/win/hockey/nhl")
FINAL_ROOT = NHL_ROOT / "05_final_scores"

INPUT_FILE = FINAL_ROOT / "intermediate" / "work_nhl.csv"

REPORT_ROOT = FINAL_ROOT / "reports"
MONEYLINE_DIR = REPORT_ROOT / "moneyline"
PUCKLINE_DIR = REPORT_ROOT / "puckline"
TOTAL_DIR = REPORT_ROOT / "total"
CALIBRATION_DIR = REPORT_ROOT / "calibration"

TALLY_FILE = FINAL_ROOT / "nhl_market_tally.csv"

ERROR_DIR = FINAL_ROOT / "errors"
ERROR_DIR.mkdir(parents=True, exist_ok=True)

REPORT_ROOT.mkdir(parents=True, exist_ok=True)
MONEYLINE_DIR.mkdir(parents=True, exist_ok=True)
PUCKLINE_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = ERROR_DIR / "03_nhl_results_reports_errors.txt"
SUMMARY_LOG = ERROR_DIR / "03_nhl_results_reports_summary.txt"
UNRESOLVED_FILE = ERROR_DIR / "01_nhl_results_grade_unresolved.csv"
PENDING_FILE = FINAL_ROOT / "intermediate" / "01_nhl_results_grade_pending.csv"


###############################################################
######################## REPORT COLUMNS #######################
###############################################################

REPORT_COLUMNS = [
    "league",
    "market_type",
    "side_group",
    "variable",
    "Win",
    "Loss",
    "Push",
    "Total",
    "bets_excluding_pushes",
    "bets_including_pushes",
    "Win_Pct",
    "units",
    "roi",
    "avg_odds",
    "avg_ev",
    "avg_kelly",
    "avg_win_prob",
    "avg_my_model_prob",
]

CALIBRATION_METRICS_COLUMNS = [
    "league",
    "market_type",
    "bets",
    "expected_win_rate",
    "realized_win_rate",
    "calibration_gap",
    "brier_score",
    "log_loss",
    "expected_wins",
    "realized_wins",
    "my_model_bets",
    "my_model_expected_win_rate",
    "my_model_realized_win_rate",
    "my_model_calibration_gap",
    "my_model_brier_score",
    "my_model_log_loss",
    "my_model_expected_wins",
    "my_model_realized_wins",
]

PROBABILITY_CALIBRATION_COLUMNS = [
    "league",
    "market_type",
    "probability_bucket",
    "bets",
    "avg_model_prob",
    "realized_win_rate",
    "calibration_gap",
    "brier_score",
    "log_loss",
    "expected_wins",
    "realized_wins",
    "avg_my_model_prob",
    "my_model_bets",
    "my_model_realized_win_rate",
    "my_model_calibration_gap",
    "my_model_brier_score",
    "my_model_log_loss",
    "my_model_expected_wins",
    "my_model_realized_wins",
]

EXPECTED_VS_REALIZED_COLUMNS = [
    "league",
    "market_type",
    "bets",
    "expected_probability",
    "realized_probability",
    "difference",
    "expected_wins",
    "realized_wins",
    "my_model_bets",
    "my_model_expected_probability",
    "my_model_realized_probability",
    "my_model_difference",
    "my_model_expected_wins",
    "my_model_realized_wins",
]

WALK_FORWARD_COLUMNS = [
    "league",
    "market_type",
    "through_game_date",
    "Win",
    "Loss",
    "Push",
    "bets_excluding_pushes",
    "bets_including_pushes",
    "win_pct",
    "units",
    "roi",
    "avg_odds",
    "avg_ev",
    "avg_kelly",
    "avg_model_prob",
    "avg_my_model_prob",
    "expected_win_rate",
    "realized_win_rate",
    "calibration_gap",
    "brier_score",
    "log_loss",
    "my_model_bets",
    "my_model_expected_win_rate",
    "my_model_realized_win_rate",
    "my_model_calibration_gap",
    "my_model_brier_score",
    "my_model_log_loss",
]


###############################################################
######################## BUCKET CONFIG ########################
###############################################################

EV_BANDS = [
    (0.00, 0.01),
    (0.01, 0.02),
    (0.02, 0.03),
    (0.03, 0.04),
    (0.04, 0.05),
    (0.05, 0.075),
    (0.075, 0.10),
    (0.10, 999),
]

KELLY_BANDS = [
    (0.00, 0.01),
    (0.01, 0.02),
    (0.02, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.50),
    (0.50, 999),
]

ODDS_BANDS = [
    (-999, -200),
    (-199, -150),
    (-149, -125),
    (-124, -110),
    (-109, 100),
    (101, 125),
    (126, 150),
    (151, 200),
    (201, 999),
]

WIN_PROB_BANDS = {
    "moneyline": [
        (0.00, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.00),
    ],
    "puck_line": [
        (0.00, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.00),
    ],
    "total": [
        (0.00, 0.45),
        (0.45, 0.50),
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.00),
    ],
}

TOTAL_RANGE_BANDS = [
    (0.0, 5.5),
    (5.5, 6.0),
    (6.0, 6.5),
    (6.5, 7.0),
    (7.0, 7.5),
    (7.5, 8.0),
    (8.0, 999),
]

CALIBRATION_BANDS = [
    (0.00, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 1.00),
]

LOG_LOSS_EPSILON = 1e-15


###############################################################
######################## LOGGING ##############################
###############################################################

def reset_logs() -> None:
    ERROR_LOG.write_text("", encoding="utf-8")
    SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg: str) -> None:
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def log_summary(msg: str) -> None:
    with SUMMARY_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


###############################################################
######################## HELPERS ##############################
###############################################################

def safe_read(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"MISSING FILE | {path}"
        )

    try:
        return pd.read_csv(path)

    except Exception as e:
        raise RuntimeError(
            f"READ ERROR | {path} | {e}"
        ) from e


def side_group(row) -> str:
    market = normalize_market(row.get("market_type", ""))
    side = normalize_side(row.get("bet_side", ""))

    if market in {"moneyline", "puck_line"}:
        if side == "home":
            return "HOME"
        if side == "away":
            return "AWAY"

    if market == "total":
        if side == "over":
            return "OVER"
        if side == "under":
            return "UNDER"

    return side.upper()


def band_label(low: float, high: float) -> str:
    def fmt(v: float) -> str:
        if abs(v) >= 100:
            return str(int(v))
        if float(v).is_integer():
            return f"{v:.1f}" if abs(v) < 10 else str(int(v))
        return str(v).rstrip("0").rstrip(".")

    return f"{fmt(low)}_to_{fmt(high)}"


def bucket_value(value, bands: list[tuple[float, float]]) -> str:
    if pd.isna(value):
        return "missing"

    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return "missing"

    for low, high in bands:
        if low <= v <= high:
            return band_label(low, high)

    return "out_of_range"


def win_prob_bucket(row) -> str:
    market = normalize_market(row.get("market_type", ""))
    bands = WIN_PROB_BANDS.get(market, [])
    return bucket_value(row.get("model_prob"), bands)


def american_to_profit_per_unit(odds) -> float:
    if pd.isna(odds):
        return np.nan

    try:
        odds = float(odds)
    except (TypeError, ValueError, OverflowError):
        return np.nan

    if odds > 0:
        return odds / 100.0

    if odds < 0:
        return 100.0 / abs(odds)

    return np.nan


def grade_to_units(row) -> float:
    result = str(row.get("bet_result", "")).strip().lower()
    odds = row.get("dk_odds_american", np.nan)
    profit_per_unit = american_to_profit_per_unit(odds)

    if result == "win":
        return profit_per_unit if not pd.isna(profit_per_unit) else np.nan

    if result == "loss":
        return -1.0

    if result == "push":
        return 0.0

    return np.nan


def empty_report_df() -> pd.DataFrame:
    return pd.DataFrame(columns=REPORT_COLUMNS)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["league"] = "nhl"
    df["market_type"] = df["market_type"].map(normalize_market)
    df["bet_side"] = df["bet_side"].map(normalize_side)

    if df.empty:
        df["side_group"] = pd.Series(
            dtype=str
        )
    else:
        df["side_group"] = df.apply(
            side_group,
            axis=1,
        )

    numeric_cols = [
        "line",
        "dk_odds_american",
        "dk_odds_decimal",
        "model_prob",
        "my_model_prob",
        "edge",
        "ev",
        "kelly",
        "away_score",
        "home_score",
        "total_score",
        "away_puck_line_result",
        "home_puck_line_result",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    result_clean = (
        df["bet_result"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["is_win"] = (
        result_clean == "win"
    ).astype(int)

    df["is_loss"] = (
        result_clean == "loss"
    ).astype(int)

    df["is_push"] = (
        result_clean == "push"
    ).astype(int)

    # Recalculate report valuation against the exact same quote used for
    # settlement; the immutable selected/graded CSVs are never edited here.
    canonical = df["dk_odds_american"].map(american_to_decimal)
    original_decimal = df["dk_odds_decimal"].copy()
    original_edge = df["edge"].copy()
    original_ev = df["ev"].copy()
    original_kelly = df["kelly"].copy()

    df["dk_odds_decimal"] = canonical
    df["edge"], df["ev"], df["kelly"] = price_metrics(
        df["model_prob"],
        df["dk_odds_decimal"],
    )
    # Historical comparisons are explicitly revaluations, not a claim that
    # the original selections or an actual bet ticket used these prices.
    changed = (
        original_decimal.notna()
        & canonical.notna()
        & (original_decimal.sub(canonical).abs() > 1e-6)
    )
    audit = df.reindex(columns=[
        "game_date",
        "game_id",
        "market_type",
        "bet_side",
        "selected_provider_name",
        "pulled_at",
        "dk_odds_american",
        "model_prob",
    ]).loc[changed].copy()
    audit["original_decimal"] = original_decimal.loc[changed]
    audit["revalued_decimal"] = canonical.loc[changed]
    audit["original_edge"] = original_edge.loc[changed]
    audit["revalued_edge"] = df.loc[changed, "edge"]
    audit["original_ev"] = original_ev.loc[changed]
    audit["revalued_ev"] = df.loc[changed, "ev"]
    audit["original_kelly"] = original_kelly.loc[changed]
    audit["revalued_kelly"] = df.loc[changed, "kelly"]
    audit["status"] = "counterfactual_report_revaluation_not_execution"
    audit.to_csv(
        FINAL_ROOT / "reports" / "nhl_price_revaluations.csv",
        index=False,
    )

    df["units"] = df.apply(
        grade_to_units,
        axis=1,
    )

    df["ev_bucket"] = df["ev"].apply(
        lambda v: bucket_value(
            v,
            EV_BANDS,
        )
    )

    df["kelly_bucket"] = df["kelly"].apply(
        lambda v: bucket_value(
            v,
            KELLY_BANDS,
        )
    )

    df["odds_bucket"] = (
        df["dk_odds_american"]
        .apply(
            lambda v: bucket_value(
                v,
                ODDS_BANDS,
            )
        )
    )

    df["win_prob_bucket"] = df.apply(
        win_prob_bucket,
        axis=1,
    )

    df["my_model_prob_bucket"] = df.apply(
        lambda row: bucket_value(
            row.get("my_model_prob"),
            WIN_PROB_BANDS.get(
                normalize_market(row.get("market_type", "")),
                [],
            ),
        ),
        axis=1,
    )

    df["side_bucket"] = df["side_group"]

    df["total_range_bucket"] = (
        df["line"]
        .apply(
            lambda v: bucket_value(
                v,
                TOTAL_RANGE_BANDS,
            )
        )
    )

    return df


###############################################################
######################## SUMMARIES ############################
###############################################################

def summarize(
    df: pd.DataFrame,
    market_type: str,
    variable_col: str,
    include_side: bool,
) -> pd.DataFrame:
    if df.empty:
        return empty_report_df()

    group_cols = [
        "league",
        "market_type",
        variable_col,
    ]

    if include_side:
        group_cols.insert(
            2,
            "side_group",
        )

    grouped = (
        df.groupby(
            group_cols,
            dropna=False,
        )
        .agg(
            Win=("is_win", "sum"),
            Loss=("is_loss", "sum"),
            Push=("is_push", "sum"),
            units=("units", "sum"),
            avg_odds=(
                "dk_odds_american",
                "mean",
            ),
            avg_ev=("ev", "mean"),
            avg_kelly=("kelly", "mean"),
            avg_win_prob=(
                "model_prob",
                "mean",
            ),
            avg_my_model_prob=(
                "my_model_prob",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["Total"] = (
        grouped["Win"]
        + grouped["Loss"]
        + grouped["Push"]
    )

    grouped["bets_excluding_pushes"] = (
        grouped["Win"]
        + grouped["Loss"]
    )

    grouped["bets_including_pushes"] = (
        grouped["Total"]
    )

    grouped["Win_Pct"] = np.where(
        grouped["bets_excluding_pushes"] > 0,
        grouped["Win"]
        / grouped["bets_excluding_pushes"],
        np.nan,
    )

    grouped["roi"] = np.where(
        grouped["bets_including_pushes"] > 0,
        grouped["units"]
        / grouped["bets_including_pushes"],
        np.nan,
    )

    grouped = grouped.rename(
        columns={
            variable_col: "variable",
        }
    )

    if not include_side:
        grouped["side_group"] = "ALL"

    grouped["market_type"] = market_type

    grouped = grouped[
        REPORT_COLUMNS
    ]

    grouped = (
        grouped.sort_values(
            [
                "league",
                "market_type",
                "side_group",
                "variable",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return grouped


def write_report(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    log_summary(
        f"WROTE REPORT | {path} | "
        f"rows={len(df)}"
    )


def write_pair(
    market_df: pd.DataFrame,
    market_type: str,
    out_dir: Path,
    prefix: str,
    report_name: str,
    variable_col: str,
) -> None:
    base = summarize(
        market_df,
        market_type,
        variable_col,
        include_side=False,
    )

    side = summarize(
        market_df,
        market_type,
        variable_col,
        include_side=True,
    )

    write_report(
        base,
        out_dir
        / f"{prefix}_by_{report_name}.csv",
    )

    write_report(
        side,
        out_dir
        / (
            f"{prefix}_by_{report_name}"
            "_home_away_summary.csv"
        ),
    )


def write_market_tally(
    df: pd.DataFrame,
) -> None:
    rows = []

    for market_type in [
        "moneyline",
        "puck_line",
        "total",
    ]:
        market_df = df[
            df["market_type"]
            == market_type
        ].copy()

        if market_df.empty:
            row = {
                "league": "nhl",
                "market_type": market_type,
                "side_group": "ALL",
                "variable": "ALL",
                "Win": 0,
                "Loss": 0,
                "Push": 0,
                "Total": 0,
                "bets_excluding_pushes": 0,
                "bets_including_pushes": 0,
                "Win_Pct": np.nan,
                "units": np.nan,
                "roi": np.nan,
                "avg_odds": np.nan,
                "avg_ev": np.nan,
                "avg_kelly": np.nan,
                "avg_win_prob": np.nan,
                "avg_my_model_prob": np.nan,
            }

            rows.append(row)
            continue

        summary = summarize(
            market_df,
            market_type,
            "all_bucket",
            include_side=False,
        )

        rows.extend(
            summary.to_dict(
                "records"
            )
        )

    tally = pd.DataFrame(
        rows,
        columns=REPORT_COLUMNS,
    )

    tally.to_csv(
        TALLY_FILE,
        index=False,
    )

    log_summary(
        f"WROTE MARKET TALLY | "
        f"{TALLY_FILE} | "
        f"rows={len(tally)}"
    )


###############################################################
######################## CALIBRATION ##########################
###############################################################

def prepare_calibration_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    calibration = df[
        df["bet_result"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "win",
                "loss",
            ]
        )
    ].copy()

    if calibration.empty:
        for column, dtype in [
            ("actual_win", float),
            ("brier_component", float),
            ("log_loss_component", float),
            ("calibration_bucket", str),
            ("my_model_actual_win", float),
            ("my_model_brier_component", float),
            ("my_model_log_loss_component", float),
            ("my_model_calibration_bucket", str),
        ]:
            calibration[column] = pd.Series(dtype=dtype)

        return calibration

    invalid_prob = (
        calibration["model_prob"].isna()
        | (calibration["model_prob"] < 0.0)
        | (calibration["model_prob"] > 1.0)
    )

    if invalid_prob.any():
        bad = calibration.loc[
            invalid_prob,
            [
                "game_date",
                "game_id",
                "market_type",
                "bet_side",
                "model_prob",
                "bet_result",
            ],
        ]

        raise RuntimeError(
            "CALIBRATION BLOCKED: "
            "win/loss rows contain missing "
            "or out-of-range model_prob "
            "values | "
            f"count={len(bad)} | "
            f"rows={bad.to_dict('records')}"
        )

    invalid_my_model_prob = (
        calibration["my_model_prob"].notna()
        & (
            (calibration["my_model_prob"] < 0.0)
            | (calibration["my_model_prob"] > 1.0)
        )
    )

    if invalid_my_model_prob.any():
        bad = calibration.loc[
            invalid_my_model_prob,
            [
                "game_date",
                "game_id",
                "market_type",
                "bet_side",
                "my_model_prob",
                "bet_result",
            ],
        ]

        raise RuntimeError(
            "CALIBRATION BLOCKED: "
            "win/loss rows contain out-of-range my_model_prob "
            "values | "
            f"count={len(bad)} | "
            f"rows={bad.to_dict('records')}"
        )

    result_clean = (
        calibration["bet_result"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    calibration["actual_win"] = (
        result_clean == "win"
    ).astype(float)

    calibration["brier_component"] = (
        calibration["model_prob"]
        - calibration["actual_win"]
    ) ** 2

    clipped_prob = (
        calibration["model_prob"]
        .clip(
            LOG_LOSS_EPSILON,
            1.0 - LOG_LOSS_EPSILON,
        )
    )

    calibration["log_loss_component"] = -(
        calibration["actual_win"]
        * np.log(clipped_prob)
        + (1.0 - calibration["actual_win"])
        * np.log(1.0 - clipped_prob)
    )

    calibration["calibration_bucket"] = (
        calibration["model_prob"]
        .apply(
            lambda value: bucket_value(
                value,
                CALIBRATION_BANDS,
            )
        )
    )

    my_model_valid = calibration["my_model_prob"].notna()
    calibration["my_model_actual_win"] = (
        calibration["actual_win"].where(my_model_valid)
    )
    calibration["my_model_brier_component"] = (
        (
            calibration["my_model_prob"]
            - calibration["actual_win"]
        ) ** 2
    ).where(my_model_valid)

    clipped_my_model_prob = (
        calibration["my_model_prob"]
        .clip(
            LOG_LOSS_EPSILON,
            1.0 - LOG_LOSS_EPSILON,
        )
    )

    calibration["my_model_log_loss_component"] = (
        -(
            calibration["actual_win"]
            * np.log(clipped_my_model_prob)
            + (1.0 - calibration["actual_win"])
            * np.log(1.0 - clipped_my_model_prob)
        )
    ).where(my_model_valid)

    calibration["my_model_calibration_bucket"] = (
        calibration["my_model_prob"]
        .apply(
            lambda value: bucket_value(
                value,
                CALIBRATION_BANDS,
            )
        )
    )

    return calibration


def calibration_scope_rows(
    calibration: pd.DataFrame,
) -> list[
    tuple[
        str,
        pd.DataFrame,
    ]
]:
    scopes = [
        (
            "ALL",
            calibration,
        )
    ]

    for market_type in [
        "moneyline",
        "puck_line",
        "total",
    ]:
        scopes.append(
            (
                market_type,
                calibration[
                    calibration["market_type"]
                    == market_type
                ].copy(),
            )
        )

    return scopes


def calibration_metric_row(
    scope: pd.DataFrame,
    market_type: str,
) -> dict:
    bets = len(scope)
    my_model_scope = scope[
        scope["my_model_prob"].notna()
    ].copy()
    my_model_bets = len(my_model_scope)

    if bets == 0:
        return {
            "league": "nhl",
            "market_type": market_type,
            "bets": 0,
            "expected_win_rate": np.nan,
            "realized_win_rate": np.nan,
            "calibration_gap": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "expected_wins": 0.0,
            "realized_wins": 0.0,
            "my_model_bets": 0,
            "my_model_expected_win_rate": np.nan,
            "my_model_realized_win_rate": np.nan,
            "my_model_calibration_gap": np.nan,
            "my_model_brier_score": np.nan,
            "my_model_log_loss": np.nan,
            "my_model_expected_wins": 0.0,
            "my_model_realized_wins": 0.0,
        }

    expected_win_rate = scope["model_prob"].mean()
    realized_win_rate = scope["actual_win"].mean()

    my_model_expected = (
        my_model_scope["my_model_prob"].mean()
        if my_model_bets
        else np.nan
    )
    my_model_realized = (
        my_model_scope["actual_win"].mean()
        if my_model_bets
        else np.nan
    )

    return {
        "league": "nhl",
        "market_type": market_type,
        "bets": bets,
        "expected_win_rate": expected_win_rate,
        "realized_win_rate": realized_win_rate,
        "calibration_gap": (
            realized_win_rate
            - expected_win_rate
        ),
        "brier_score": scope["brier_component"].mean(),
        "log_loss": scope["log_loss_component"].mean(),
        "expected_wins": scope["model_prob"].sum(),
        "realized_wins": scope["actual_win"].sum(),
        "my_model_bets": my_model_bets,
        "my_model_expected_win_rate": my_model_expected,
        "my_model_realized_win_rate": my_model_realized,
        "my_model_calibration_gap": (
            my_model_realized - my_model_expected
            if my_model_bets
            else np.nan
        ),
        "my_model_brier_score": (
            my_model_scope["my_model_brier_component"].mean()
            if my_model_bets
            else np.nan
        ),
        "my_model_log_loss": (
            my_model_scope["my_model_log_loss_component"].mean()
            if my_model_bets
            else np.nan
        ),
        "my_model_expected_wins": (
            my_model_scope["my_model_prob"].sum()
            if my_model_bets
            else 0.0
        ),
        "my_model_realized_wins": (
            my_model_scope["actual_win"].sum()
            if my_model_bets
            else 0.0
        ),
    }


def write_calibration_metrics(
    calibration: pd.DataFrame,
) -> None:
    rows = [
        calibration_metric_row(
            scope,
            market_type,
        )
        for market_type, scope
        in calibration_scope_rows(
            calibration
        )
    ]

    report = pd.DataFrame(
        rows,
        columns=(
            CALIBRATION_METRICS_COLUMNS
        ),
    )

    write_report(
        report,
        CALIBRATION_DIR
        / "nhl_calibration_metrics.csv",
    )


def write_probability_calibration(
    calibration: pd.DataFrame,
) -> None:
    rows = []

    for (
        market_type,
        scope,
    ) in calibration_scope_rows(
        calibration
    ):
        if scope.empty:
            continue

        grouped = (
            scope.groupby(
                "calibration_bucket",
                dropna=False,
            )
            .agg(
                bets=(
                    "actual_win",
                    "size",
                ),
                avg_model_prob=(
                    "model_prob",
                    "mean",
                ),
                realized_win_rate=(
                    "actual_win",
                    "mean",
                ),
                brier_score=(
                    "brier_component",
                    "mean",
                ),
                log_loss=(
                    "log_loss_component",
                    "mean",
                ),
                expected_wins=(
                    "model_prob",
                    "sum",
                ),
                realized_wins=(
                    "actual_win",
                    "sum",
                ),
                avg_my_model_prob=(
                    "my_model_prob",
                    "mean",
                ),
                my_model_bets=(
                    "my_model_prob",
                    "count",
                ),
                my_model_realized_win_rate=(
                    "my_model_actual_win",
                    "mean",
                ),
                my_model_brier_score=(
                    "my_model_brier_component",
                    "mean",
                ),
                my_model_log_loss=(
                    "my_model_log_loss_component",
                    "mean",
                ),
                my_model_expected_wins=(
                    "my_model_prob",
                    "sum",
                ),
                my_model_realized_wins=(
                    "my_model_actual_win",
                    "sum",
                ),
            )
            .reset_index()
        )

        grouped["league"] = "nhl"

        grouped[
            "market_type"
        ] = market_type

        grouped[
            "calibration_gap"
        ] = (
            grouped[
                "realized_win_rate"
            ]
            - grouped[
                "avg_model_prob"
            ]
        )

        grouped[
            "my_model_calibration_gap"
        ] = (
            grouped[
                "my_model_realized_win_rate"
            ]
            - grouped[
                "avg_my_model_prob"
            ]
        )

        grouped = grouped.rename(
            columns={
                "calibration_bucket":
                "probability_bucket",
            }
        )

        rows.extend(
            grouped[
                PROBABILITY_CALIBRATION_COLUMNS
            ].to_dict(
                "records"
            )
        )

    report = pd.DataFrame(
        rows,
        columns=(
            PROBABILITY_CALIBRATION_COLUMNS
        ),
    )

    if not report.empty:
        bucket_order = {
            band_label(
                low,
                high,
            ): index
            for (
                index,
                (
                    low,
                    high,
                ),
            ) in enumerate(
                CALIBRATION_BANDS
            )
        }

        market_order = {
            "ALL": 0,
            "moneyline": 1,
            "puck_line": 2,
            "total": 3,
        }

        report[
            "_market_order"
        ] = (
            report[
                "market_type"
            ]
            .map(
                market_order
            )
            .fillna(999)
        )

        report[
            "_bucket_order"
        ] = (
            report[
                "probability_bucket"
            ]
            .map(
                bucket_order
            )
            .fillna(999)
        )

        report = (
            report.sort_values(
                [
                    "_market_order",
                    "_bucket_order",
                ],
                kind="stable",
            )
            .drop(
                columns=[
                    "_market_order",
                    "_bucket_order",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    write_report(
        report,
        CALIBRATION_DIR
        / (
            "nhl_probability_"
            "calibration.csv"
        ),
    )


def write_expected_vs_realized(
    calibration: pd.DataFrame,
) -> None:
    rows = []

    for (
        market_type,
        scope,
    ) in calibration_scope_rows(
        calibration
    ):
        bets = len(scope)
        my_model_scope = scope[
            scope["my_model_prob"].notna()
        ].copy()
        my_model_bets = len(my_model_scope)

        if bets == 0:
            rows.append(
                {
                    "league": "nhl",
                    "market_type": market_type,
                    "bets": 0,
                    "expected_probability": np.nan,
                    "realized_probability": np.nan,
                    "difference": np.nan,
                    "expected_wins": 0.0,
                    "realized_wins": 0.0,
                    "my_model_bets": 0,
                    "my_model_expected_probability": np.nan,
                    "my_model_realized_probability": np.nan,
                    "my_model_difference": np.nan,
                    "my_model_expected_wins": 0.0,
                    "my_model_realized_wins": 0.0,
                }
            )
            continue

        expected_probability = scope["model_prob"].mean()
        realized_probability = scope["actual_win"].mean()

        my_model_expected_probability = (
            my_model_scope["my_model_prob"].mean()
            if my_model_bets
            else np.nan
        )
        my_model_realized_probability = (
            my_model_scope["actual_win"].mean()
            if my_model_bets
            else np.nan
        )

        rows.append(
            {
                "league": "nhl",
                "market_type": market_type,
                "bets": bets,
                "expected_probability": expected_probability,
                "realized_probability": realized_probability,
                "difference": (
                    realized_probability
                    - expected_probability
                ),
                "expected_wins": scope["model_prob"].sum(),
                "realized_wins": scope["actual_win"].sum(),
                "my_model_bets": my_model_bets,
                "my_model_expected_probability": (
                    my_model_expected_probability
                ),
                "my_model_realized_probability": (
                    my_model_realized_probability
                ),
                "my_model_difference": (
                    my_model_realized_probability
                    - my_model_expected_probability
                    if my_model_bets
                    else np.nan
                ),
                "my_model_expected_wins": (
                    my_model_scope["my_model_prob"].sum()
                    if my_model_bets
                    else 0.0
                ),
                "my_model_realized_wins": (
                    my_model_scope["actual_win"].sum()
                    if my_model_bets
                    else 0.0
                ),
            }
        )

    report = pd.DataFrame(
        rows,
        columns=(
            EXPECTED_VS_REALIZED_COLUMNS
        ),
    )

    write_report(
        report,
        CALIBRATION_DIR
        / (
            "nhl_expected_"
            "vs_realized.csv"
        ),
    )


def walk_forward_scope_row(
    full_scope: pd.DataFrame,
    calibration_scope: pd.DataFrame,
    market_type: str,
    through_game_date: str,
) -> dict:
    wins = int(full_scope["is_win"].sum())
    losses = int(full_scope["is_loss"].sum())
    pushes = int(full_scope["is_push"].sum())

    bets_excluding_pushes = wins + losses
    bets_including_pushes = wins + losses + pushes

    units = full_scope["units"].sum(
        min_count=1
    )

    win_pct = (
        wins / bets_excluding_pushes
        if bets_excluding_pushes > 0
        else np.nan
    )

    roi = (
        units / bets_including_pushes
        if (
            bets_including_pushes > 0
            and not pd.isna(units)
        )
        else np.nan
    )

    if calibration_scope.empty:
        expected_win_rate = np.nan
        realized_win_rate = np.nan
        calibration_gap = np.nan
        brier_score = np.nan
        log_loss = np.nan
    else:
        expected_win_rate = calibration_scope["model_prob"].mean()
        realized_win_rate = calibration_scope["actual_win"].mean()
        calibration_gap = realized_win_rate - expected_win_rate
        brier_score = calibration_scope["brier_component"].mean()
        log_loss = calibration_scope["log_loss_component"].mean()

    my_model_scope = calibration_scope[
        calibration_scope["my_model_prob"].notna()
    ].copy()
    my_model_bets = len(my_model_scope)

    my_model_expected_win_rate = (
        my_model_scope["my_model_prob"].mean()
        if my_model_bets
        else np.nan
    )
    my_model_realized_win_rate = (
        my_model_scope["actual_win"].mean()
        if my_model_bets
        else np.nan
    )

    return {
        "league": "nhl",
        "market_type": market_type,
        "through_game_date": through_game_date,
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "bets_excluding_pushes": bets_excluding_pushes,
        "bets_including_pushes": bets_including_pushes,
        "win_pct": win_pct,
        "units": units,
        "roi": roi,
        "avg_odds": full_scope["dk_odds_american"].mean(),
        "avg_ev": full_scope["ev"].mean(),
        "avg_kelly": full_scope["kelly"].mean(),
        "avg_model_prob": full_scope["model_prob"].mean(),
        "avg_my_model_prob": full_scope["my_model_prob"].mean(),
        "expected_win_rate": expected_win_rate,
        "realized_win_rate": realized_win_rate,
        "calibration_gap": calibration_gap,
        "brier_score": brier_score,
        "log_loss": log_loss,
        "my_model_bets": my_model_bets,
        "my_model_expected_win_rate": my_model_expected_win_rate,
        "my_model_realized_win_rate": my_model_realized_win_rate,
        "my_model_calibration_gap": (
            my_model_realized_win_rate
            - my_model_expected_win_rate
            if my_model_bets
            else np.nan
        ),
        "my_model_brier_score": (
            my_model_scope["my_model_brier_component"].mean()
            if my_model_bets
            else np.nan
        ),
        "my_model_log_loss": (
            my_model_scope["my_model_log_loss_component"].mean()
            if my_model_bets
            else np.nan
        ),
    }


def write_walk_forward_performance(
    df: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    rows = []

    game_dates = sorted(
        value
        for value
        in (
            df["game_date"]
            .dropna()
            .astype(str)
            .unique()
        )
        if value.strip()
    )

    for through_game_date in game_dates:
        cumulative = df[
            df["game_date"]
            .astype(str)
            <= through_game_date
        ].copy()

        cumulative_calibration = (
            calibration[
                calibration[
                    "game_date"
                ]
                .astype(str)
                <= through_game_date
            ].copy()
        )

        for market_type in [
            "ALL",
            "moneyline",
            "puck_line",
            "total",
        ]:
            if market_type == "ALL":
                full_scope = cumulative

                calibration_scope = (
                    cumulative_calibration
                )

            else:
                full_scope = cumulative[
                    cumulative[
                        "market_type"
                    ]
                    == market_type
                ].copy()

                calibration_scope = (
                    cumulative_calibration[
                        cumulative_calibration[
                            "market_type"
                        ]
                        == market_type
                    ].copy()
                )

            rows.append(
                walk_forward_scope_row(
                    full_scope=(
                        full_scope
                    ),
                    calibration_scope=(
                        calibration_scope
                    ),
                    market_type=(
                        market_type
                    ),
                    through_game_date=(
                        through_game_date
                    ),
                )
            )

    report = pd.DataFrame(
        rows,
        columns=(
            WALK_FORWARD_COLUMNS
        ),
    )

    write_report(
        report,
        CALIBRATION_DIR
        / (
            "nhl_walk_forward_"
            "performance.csv"
        ),
    )


def write_calibration_reports(
    df: pd.DataFrame,
) -> None:
    calibration = (
        prepare_calibration_df(
            df
        )
    )

    log_summary(
        "CALIBRATION INPUT | "
        f"win_loss_rows="
        f"{len(calibration)} | "
        f"pushes_excluded="
        f"{int(df['is_push'].sum())}"
    )

    write_calibration_metrics(
        calibration
    )

    write_probability_calibration(
        calibration
    )

    write_expected_vs_realized(
        calibration
    )

    write_walk_forward_performance(
        df,
        calibration,
    )


def fail_if_unresolved_rows_exist() -> None:
    if not UNRESOLVED_FILE.exists():
        return

    try:
        unresolved = pd.read_csv(
            UNRESOLVED_FILE,
            dtype=str,
        )

    except pd.errors.EmptyDataError:
        return

    except Exception as e:
        raise RuntimeError(
            "FAILED TO READ UNRESOLVED "
            "GRADING FILE | "
            f"{UNRESOLVED_FILE} | "
            f"{e}"
        ) from e

    if unresolved.empty:
        return

    raise RuntimeError(
        "FINAL REPORTING BLOCKED: "
        "unresolved completed NHL "
        "grading rows remain | "
        f"count={len(unresolved)} | "
        f"file={UNRESOLVED_FILE}"
    )


###############################################################
######################## MAIN #################################
###############################################################

def main() -> None:
    reset_logs()

    log_summary(
        "START "
        "03_nhl_results_reports.py"
    )

    log_summary(
        f"INPUT_FILE="
        f"{INPUT_FILE}"
    )

    log_summary(
        f"REPORT_ROOT="
        f"{REPORT_ROOT}"
    )

    fail_if_unresolved_rows_exist()

    try:
        df = safe_read(
            INPUT_FILE
        )

    except Exception as e:
        log_error(
            f"REPORT INPUT FAILED | "
            f"{e}"
        )
        raise

    required = [
        "league",
        "game_date",
        "game_id",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "line",
        "dk_odds_american",
        "model_prob",
        "my_model_prob",
        "ev",
        "kelly",
        "bet_result",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        msg = (
            "STOPPING: missing "
            "required columns "
            f"{missing}"
        )

        log_error(msg)
        raise RuntimeError(msg)

    df = prepare_df(df)

    df["all_bucket"] = "ALL"

    moneyline_df = df[
        df["market_type"]
        == "moneyline"
    ].copy()

    puckline_df = df[
        df["market_type"]
        == "puck_line"
    ].copy()

    total_df = df[
        df["market_type"]
        == "total"
    ].copy()

    ###########################################################
    # MONEYLINE
    ###########################################################

    write_pair(
        moneyline_df,
        "moneyline",
        MONEYLINE_DIR,
        "nhl_moneyline",
        "ev",
        "ev_bucket",
    )

    write_pair(
        moneyline_df,
        "moneyline",
        MONEYLINE_DIR,
        "nhl_moneyline",
        "kelly",
        "kelly_bucket",
    )

    write_pair(
        moneyline_df,
        "moneyline",
        MONEYLINE_DIR,
        "nhl_moneyline",
        "odds",
        "odds_bucket",
    )

    write_pair(
        moneyline_df,
        "moneyline",
        MONEYLINE_DIR,
        "nhl_moneyline",
        "win_prob",
        "win_prob_bucket",
    )

    write_pair(
        moneyline_df,
        "moneyline",
        MONEYLINE_DIR,
        "nhl_moneyline",
        "my_model_prob",
        "my_model_prob_bucket",
    )

    ###########################################################
    # PUCK LINE
    ###########################################################

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "ev",
        "ev_bucket",
    )

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "kelly",
        "kelly_bucket",
    )

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "odds",
        "odds_bucket",
    )

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "side",
        "side_bucket",
    )

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "win_prob",
        "win_prob_bucket",
    )

    write_pair(
        puckline_df,
        "puck_line",
        PUCKLINE_DIR,
        "nhl_puck_line",
        "my_model_prob",
        "my_model_prob_bucket",
    )

    ###########################################################
    # TOTAL
    ###########################################################

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "ev",
        "ev_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "kelly",
        "kelly_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "odds",
        "odds_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "side",
        "side_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "total_range",
        "total_range_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "win_prob",
        "win_prob_bucket",
    )

    write_pair(
        total_df,
        "total",
        TOTAL_DIR,
        "nhl_total",
        "my_model_prob",
        "my_model_prob_bucket",
    )

    write_market_tally(
        df
    )

    write_calibration_reports(
        df
    )

    log_summary(
        "END "
        "03_nhl_results_reports.py"
    )

    if df.empty:
        print(
            "NHL reports complete: "
            "no completed graded bets."
        )
    else:
        print(
            "NHL reports complete."
        )


if __name__ == "__main__":
    main()