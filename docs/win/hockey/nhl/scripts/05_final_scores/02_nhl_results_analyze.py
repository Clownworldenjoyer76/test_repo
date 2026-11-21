#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/02_nhl_results_analyze.py

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


###############################################################
######################## PATH CONFIG ##########################
###############################################################

NHL_ROOT = Path("docs/win/hockey/nhl")
FINAL_ROOT = NHL_ROOT / "05_final_scores"

GRADED_DIR = FINAL_ROOT / "graded"
INPUT_FILE = GRADED_DIR / "NHL_final.csv"

INTERMEDIATE_DIR = FINAL_ROOT / "intermediate"
ERROR_DIR = FINAL_ROOT / "errors"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = ERROR_DIR / "02_nhl_results_analyze_errors.txt"
SUMMARY_LOG = ERROR_DIR / "02_nhl_results_analyze_summary.txt"

WORK_FILE = INTERMEDIATE_DIR / "work_nhl.csv"

REQUIRED_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "market_type",
    "bet_side",
    "line",
    "take_bet",
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
    "bet_result",
]


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

def read_input(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise RuntimeError(
            f"MISSING MASTER GRADED FILE | {path}"
        )

    try:
        return pd.read_csv(
            path,
            dtype=str,
        )
    except Exception as e:
        raise RuntimeError(
            f"READ ERROR | {path} | {e}"
        ) from e


def require_columns(
    df: pd.DataFrame,
    required: list[str],
) -> None:
    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"MISSING COLUMNS | {missing}"
        )


def normalize_market(value: str) -> str:
    value = str(value).strip().lower()

    if value in {"moneyline", "ml"}:
        return "moneyline"

    if value in {
        "puck_line",
        "puckline",
        "spread",
    }:
        return "puck_line"

    if value in {"total", "totals"}:
        return "total"

    return value


def normalize_side(value: str) -> str:
    return str(value).strip().lower()


def side_group(row) -> str:
    market = normalize_market(
        row.get("market_type", "")
    )
    side = normalize_side(
        row.get("bet_side", "")
    )

    if market in {
        "moneyline",
        "puck_line",
    }:
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


###############################################################
######################## WORK FILE ############################
###############################################################

def build_work() -> int:
    df = read_input(
        INPUT_FILE
    )

    require_columns(
        df,
        REQUIRED_COLUMNS,
    )

    df = df.copy()

    df["game_date"] = (
        df["game_date"]
        .astype(str)
        .str.strip()
        .str.replace(
            "-",
            "_",
            regex=False,
        )
    )

    df["league"] = "nhl"
    df["sport"] = "hockey"

    df["market_type"] = (
        df["market_type"].map(
            normalize_market
        )
    )

    df["bet_side"] = (
        df["bet_side"].map(
            normalize_side
        )
    )

    if df.empty:
        df["side_group"] = pd.Series(
            dtype=str
        )
    else:
        df["side_group"] = df.apply(
            side_group,
            axis=1,
        )

    df["selected_edge"] = df["edge"]
    df["take_odds"] = (
        df["dk_odds_american"]
    )
    df["win_prob"] = df["model_prob"]

    numeric_cols = [
        "line",
        "dk_odds_american",
        "dk_odds_decimal",
        "model_prob",
        "my_model_prob",
        "win_prob",
        "edge",
        "selected_edge",
        "ev",
        "kelly",
        "take_odds",
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

    df.to_csv(
        WORK_FILE,
        index=False,
    )

    log_summary(
        "NHL WORK FILE CREATED | "
        f"rows={len(df)} | "
        f"out={WORK_FILE}"
    )

    return len(df)


###############################################################
######################## MAIN #################################
###############################################################

def main() -> None:
    reset_logs()

    log_summary(
        "START 02_nhl_results_analyze.py"
    )
    log_summary(
        f"INPUT_FILE={INPUT_FILE}"
    )
    log_summary(
        f"WORK_FILE={WORK_FILE}"
    )

    try:
        row_count = build_work()

    except Exception as e:
        log_error(
            f"ANALYSIS PREP FAILED | {e}"
        )
        print(
            f"NHL analysis prep failed: {e}"
        )
        raise

    log_summary(
        "END 02_nhl_results_analyze.py"
    )

    if row_count == 0:
        print(
            "NHL analysis prep complete: "
            "no completed graded bets."
        )
    else:
        print(
            "NHL analysis prep complete."
        )


if __name__ == "__main__":
    main()
