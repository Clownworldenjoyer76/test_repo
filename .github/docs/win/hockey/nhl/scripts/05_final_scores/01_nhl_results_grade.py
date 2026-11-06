#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/01_nhl_results_grade.py

from datetime import UTC, datetime
from pathlib import Path
import sys
import re

import pandas as pd


SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# noinspection PyPep8
from selection_schema_common import (
    SELECTION_COLUMNS as SELECT_REQUIRED_COLUMNS,
)
# noinspection PyPep8
from market_common import normalize_market, normalize_side


###############################################################
######################## PATH CONFIG ##########################
###############################################################

NHL_ROOT = Path("docs/win/hockey/nhl")
FINAL_ROOT = NHL_ROOT / "05_final_scores"

SELECT_DIR = NHL_ROOT / "04_select"
SCORE_DIR = FINAL_ROOT / "final_scores"
GRADED_DIR = FINAL_ROOT / "graded"
INTERMEDIATE_DIR = FINAL_ROOT / "intermediate"
ERROR_DIR = FINAL_ROOT / "errors"

GRADED_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ERROR_LOG = ERROR_DIR / "01_nhl_results_grade_errors.txt"
GRADE_SUMMARY_LOG = ERROR_DIR / "01_nhl_results_grade_summary.txt"

SELECT_PATTERN = "*_NHL.csv"
SCORE_PATTERN = "*_NHL_final_scores.csv"

MASTER_FILE = GRADED_DIR / "NHL_final.csv"
STATUS_FILE = INTERMEDIATE_DIR / "nhl_game_status.csv"
PENDING_FILE = INTERMEDIATE_DIR / "01_nhl_results_grade_pending.csv"
UNRESOLVED_FILE = ERROR_DIR / "01_nhl_results_grade_unresolved.csv"

FINAL_GAME_STATES = {"FINAL", "OFF"}
GAME_ID_RE = re.compile(r"^\d{10}$")

SCORE_REQUIRED_COLUMNS = [
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

STATUS_REQUIRED_COLUMNS = [
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


GRADED_OUTPUT_COLUMNS = (
    SELECT_REQUIRED_COLUMNS
    + ["source_select_file"]
    + [
        "game_state",
        "game_schedule_state",
        "is_final",
        "status_observed_at",
        "source_status_file",
        "away_score",
        "home_score",
        "total_score",
        "away_puck_line_result",
        "home_puck_line_result",
        "source_score_file",
        "bet_result",
    ]
)


###############################################################
######################## LOGGING ##############################
###############################################################

def reset_logs() -> None:
    GRADE_ERROR_LOG.write_text("", encoding="utf-8")
    GRADE_SUMMARY_LOG.write_text("", encoding="utf-8")


def log_error(msg: str) -> None:
    with GRADE_ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


def log_summary(msg: str) -> None:
    with GRADE_SUMMARY_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(UTC).isoformat()}] {msg}\n")


###############################################################
######################## HELPERS ##############################
###############################################################

def read_csv(
    path: Path,
    *,
    allow_empty: bool,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"MISSING FILE | {path}")

    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"READ ERROR | {path} | {e}") from e

    if df is None:
        raise RuntimeError(f"READ RETURNED NO DATAFRAME | {path}")

    if df.empty and not allow_empty:
        raise RuntimeError(f"EMPTY FILE | {path}")

    return df


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"MISSING COLUMNS | {label} | {missing}"
        )


def normalize_date(value) -> str:
    return str(value).strip().replace("-", "_")


def normalize_team(value) -> str:
    return str(value).strip()


def normalize_identity_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df["game_date"] = (
        df["game_date"].map(normalize_date)
    )
    df["away_team"] = (
        df["away_team"].map(normalize_team)
    )
    df["home_team"] = (
        df["home_team"].map(normalize_team)
    )
    df["game_id"] = (
        df["game_id"]
        .astype(str)
        .str.strip()
    )
    return df


def to_float(value):
    try:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        return float(text)

    except (TypeError, ValueError, OverflowError):
        return None


def validate_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    invalid = []

    for row_number, value in enumerate(
        df["game_id"],
        start=2,
    ):
        game_id = str(value).strip()

        if not GAME_ID_RE.fullmatch(game_id):
            invalid.append(
                (row_number, game_id)
            )

    if invalid:
        for row_number, game_id in invalid:
            log_error(
                "INVALID CANONICAL game_id | "
                f"{label} | "
                f"row={row_number} | "
                f"game_id={game_id}"
            )

        raise RuntimeError(
            "INVALID CANONICAL game_id VALUES | "
            f"{label} | count={len(invalid)}"
        )


def clean_old_outputs() -> None:
    for path in GRADED_DIR.glob(
        "*_results_NHL.csv"
    ):
        path.unlink(missing_ok=True)

    MASTER_FILE.unlink(missing_ok=True)
    PENDING_FILE.unlink(missing_ok=True)
    UNRESOLVED_FILE.unlink(missing_ok=True)

    log_summary(
        "CLEARED OLD NHL GRADED OUTPUTS"
    )


###############################################################
####################### LOAD INPUTS ###########################
###############################################################

def load_select_rows() -> pd.DataFrame:
    select_files = sorted(
        SELECT_DIR.glob(SELECT_PATTERN)
    )

    if not select_files:
        raise RuntimeError(
            "NO SELECT FILES FOUND | "
            f"{SELECT_DIR} | "
            f"pattern={SELECT_PATTERN}"
        )

    parts = []

    for path in select_files:
        df = read_csv(
            path,
            allow_empty=True,
        )

        require_columns(
            df,
            SELECT_REQUIRED_COLUMNS,
            str(path),
        )

        # Header-only Stage 04 files are valid:
        # they mean zero bets were selected for that slate.
        if df.empty:
            log_summary(
                "EMPTY SELECT FILE | "
                f"{path} | rows=0 | skipped"
            )
            continue

        df = df.copy()
        df["source_select_file"] = path.name
        df = normalize_identity_columns(df)
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

        validate_game_ids(
            df,
            str(path),
        )

        parts.append(df)

    if not parts:
        out = pd.DataFrame(
            columns=(
                SELECT_REQUIRED_COLUMNS
                + ["source_select_file"]
            )
        )

        log_summary(
            "SELECT ROWS LOADED | "
            f"files={len(select_files)} | "
            "rows=0"
        )

        return out

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    log_summary(
        "SELECT ROWS LOADED | "
        f"files={len(select_files)} | "
        f"rows={len(out)}"
    )

    return out


def load_score_rows() -> pd.DataFrame:
    score_files = sorted(
        SCORE_DIR.glob(SCORE_PATTERN)
    )

    if not score_files:
        log_summary(
            "NO FINAL SCORE FILES FOUND | "
            f"{SCORE_DIR} | "
            f"pattern={SCORE_PATTERN} | "
            "using empty final-score set"
        )

        return pd.DataFrame(
            columns=(
                SCORE_REQUIRED_COLUMNS
                + ["source_score_file"]
            )
        )

    parts = []

    for path in score_files:
        df = read_csv(
            path,
            allow_empty=True,
        )

        require_columns(
            df,
            SCORE_REQUIRED_COLUMNS,
            str(path),
        )

        if df.empty:
            log_summary(
                "EMPTY FINAL SCORE FILE | "
                f"{path} | rows=0 | skipped"
            )
            continue

        df = df.copy()
        df["source_score_file"] = path.name
        df = normalize_identity_columns(df)

        validate_game_ids(
            df,
            str(path),
        )

        parts.append(df)

    if not parts:
        return pd.DataFrame(
            columns=(
                SCORE_REQUIRED_COLUMNS
                + ["source_score_file"]
            )
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    log_summary(
        "FINAL SCORE ROWS LOADED | "
        f"files={len(score_files)} | "
        f"rows={len(out)}"
    )

    return out


def load_status_rows() -> pd.DataFrame:
    df = read_csv(
        STATUS_FILE,
        allow_empty=False,
    )

    require_columns(
        df,
        STATUS_REQUIRED_COLUMNS,
        str(STATUS_FILE),
    )

    df = df.copy()
    df["source_status_file"] = (
        STATUS_FILE.name
    )
    df = normalize_identity_columns(df)
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

    validate_game_ids(
        df,
        str(STATUS_FILE),
    )

    if df["game_state"].eq("").any():
        raise RuntimeError(
            "BLANK official game_state VALUES | "
            f"{STATUS_FILE}"
        )

    log_summary(
        "OFFICIAL STATUS ROWS LOADED | "
        f"rows={len(df)}"
    )

    return df


###############################################################
###################### SCORE LOOKUPS ##########################
###############################################################

def build_index(
    df: pd.DataFrame,
    label: str,
) -> dict[str, dict]:
    by_game_id: dict[str, dict] = {}

    for _, row in df.iterrows():
        rec = row.to_dict()
        game_id = str(
            rec.get("game_id", "")
        ).strip()

        if game_id in by_game_id:
            raise RuntimeError(
                f"DUPLICATE {label} game_id | "
                f"{game_id}"
            )

        by_game_id[game_id] = rec

    return by_game_id


###############################################################
######################## OUTCOME LOGIC ########################
###############################################################

def determine_outcome(row: dict) -> str:
    try:
        market = normalize_market(
            row.get("market_type", "")
        )
        side = normalize_side(
            row.get("bet_side", "")
        )

        away_score = to_float(
            row.get("away_score")
        )
        home_score = to_float(
            row.get("home_score")
        )
        line = to_float(
            row.get("line")
        )

        if (
            away_score is None
            or home_score is None
        ):
            return "Unknown"

        if market == "moneyline":
            if away_score == home_score:
                return "Push"

            if side == "home":
                return (
                    "Win"
                    if home_score > away_score
                    else "Loss"
                )

            if side == "away":
                return (
                    "Win"
                    if away_score > home_score
                    else "Loss"
                )

            return "Unknown"

        if market == "puck_line":
            if line is None:
                return "Unknown"

            if side == "home":
                result = to_float(
                    row.get(
                        "home_puck_line_result"
                    )
                )

                if result is None:
                    result = (
                        home_score - away_score
                    )

                diff = result + line

            elif side == "away":
                result = to_float(
                    row.get(
                        "away_puck_line_result"
                    )
                )

                if result is None:
                    result = (
                        away_score - home_score
                    )

                diff = result + line

            else:
                return "Unknown"

            if abs(diff) < 1e-9:
                return "Push"

            return (
                "Win"
                if diff > 0
                else "Loss"
            )

        if market == "total":
            if line is None:
                return "Unknown"

            total_score = to_float(
                row.get("total_score")
            )

            if total_score is None:
                total_score = (
                    away_score + home_score
                )

            diff = total_score - line

            if abs(diff) < 1e-9:
                return "Push"

            if side == "over":
                return (
                    "Win"
                    if diff > 0
                    else "Loss"
                )

            if side == "under":
                return (
                    "Win"
                    if diff < 0
                    else "Loss"
                )

            return "Unknown"

    except Exception as e:
        log_error(
            "DETERMINE OUTCOME ERROR | "
            f"{e} | row={row}"
        )

    return "Unknown"


###############################################################
######################## GRADING ##############################
###############################################################

def grade_rows(
    bets: pd.DataFrame,
    scores: pd.DataFrame,
    statuses: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    score_by_game_id = build_index(
        scores,
        "FINAL SCORE",
    )
    status_by_game_id = build_index(
        statuses,
        "OFFICIAL STATUS",
    )

    graded_rows = []
    pending_rows = []
    unresolved_rows = []

    score_cols = [
        "away_score",
        "home_score",
        "total_score",
        "away_puck_line_result",
        "home_puck_line_result",
        "source_score_file",
    ]

    status_cols = [
        "game_state",
        "game_schedule_state",
        "is_final",
        "status_observed_at",
        "source_status_file",
    ]

    for _, bet_row in bets.iterrows():
        bet = bet_row.to_dict()
        game_id = str(
            bet.get("game_id", "")
        ).strip()

        status = status_by_game_id.get(
            game_id
        )
        score = score_by_game_id.get(
            game_id
        )

        if status is None:
            unresolved = dict(bet)
            unresolved[
                "unresolved_reason"
            ] = "official_game_status_missing"

            unresolved_rows.append(
                unresolved
            )

            log_error(
                "OFFICIAL GAME STATUS MISSING | "
                f"game_date={bet.get('game_date', '')} | "
                f"game_id={game_id}"
            )

            continue

        combined = dict(bet)

        for col in status_cols:
            combined[col] = status.get(
                col,
                "",
            )

        game_state = str(
            status.get("game_state", "")
        ).strip().upper()

        if (
            game_state
            not in FINAL_GAME_STATES
        ):
            if score is not None:
                combined[
                    "unresolved_reason"
                ] = (
                    "score_present_for_nonfinal_game"
                )

                unresolved_rows.append(
                    combined
                )

                log_error(
                    "FINAL SCORE PRESENT FOR "
                    "NONFINAL GAME | "
                    f"game_id={game_id} | "
                    f"game_state={game_state}"
                )

                continue

            combined[
                "pending_reason"
            ] = "official_game_not_final"

            pending_rows.append(
                combined
            )

            log_summary(
                "PENDING GAME | "
                f"game_id={game_id} | "
                f"game_state={game_state}"
            )

            continue

        if score is None:
            combined[
                "unresolved_reason"
            ] = "final_game_missing_final_score"

            unresolved_rows.append(
                combined
            )

            log_error(
                "FINAL GAME MISSING FINAL SCORE | "
                f"game_id={game_id} | "
                f"game_state={game_state}"
            )

            continue

        score_game_id = str(
            score.get("game_id", "")
        ).strip()

        if game_id != score_game_id:
            raise RuntimeError(
                "CANONICAL game_id MISMATCH | "
                f"bet_game_id={game_id} | "
                f"score_game_id={score_game_id}"
            )

        for col in score_cols:
            combined[col] = score.get(
                col,
                "",
            )

        result = determine_outcome(
            combined
        )

        if result == "Unknown":
            combined["bet_result"] = result
            combined[
                "unresolved_reason"
            ] = "unknown_outcome"

            unresolved_rows.append(
                combined
            )

            log_error(
                "UNRESOLVED OUTCOME | "
                f"game_id={game_id} | "
                f"market={combined.get('market_type', '')} | "
                f"side={combined.get('bet_side', '')}"
            )

            continue

        combined["bet_result"] = result
        graded_rows.append(combined)

    graded_df = pd.DataFrame(
        graded_rows
    )
    pending_df = pd.DataFrame(
        pending_rows
    )
    unresolved_df = pd.DataFrame(
        unresolved_rows
    )

    selected_count = len(bets)
    graded_count = len(graded_df)
    pending_count = len(pending_df)
    unresolved_count = len(unresolved_df)

    if (
        selected_count
        != graded_count
        + pending_count
        + unresolved_count
    ):
        raise RuntimeError(
            "LOSSLESS RECONCILIATION FAILED | "
            f"selected={selected_count} | "
            f"graded={graded_count} | "
            f"pending={pending_count} | "
            f"unresolved={unresolved_count}"
        )

    log_summary(
        "GRADING RECONCILIATION | "
        f"selected={selected_count} | "
        f"graded={graded_count} | "
        f"pending={pending_count} | "
        f"unresolved={unresolved_count}"
    )

    return (
        graded_df,
        pending_df,
        unresolved_df,
    )


###############################################################
######################## OUTPUT ###############################
###############################################################

def write_partition(
    df: pd.DataFrame,
    path: Path,
    empty_message: str,
    written_message: str,
) -> None:
    if df.empty:
        path.unlink(missing_ok=True)
        log_summary(empty_message)
        return

    df = df.copy()

    if "game_date" in df.columns:
        df["game_date"] = (
            df["game_date"].map(
                normalize_date
            )
        )

    sort_cols = [
        c
        for c in [
            "game_date",
            "game_time",
            "game_id",
            "away_team",
            "home_team",
            "market_type",
            "bet_side",
            "line",
        ]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(
            sort_cols,
            kind="mergesort",
        )

    df.to_csv(
        path,
        index=False,
    )

    log_summary(
        f"{written_message} | "
        f"{path} | rows={len(df)}"
    )


def write_graded_outputs(
    df: pd.DataFrame,
) -> None:
    if df.empty:
        empty_master = pd.DataFrame(
            columns=GRADED_OUTPUT_COLUMNS
        )

        empty_master.to_csv(
            MASTER_FILE,
            index=False,
        )

        log_summary(
            "WROTE HEADER-ONLY MASTER GRADED | "
            f"{MASTER_FILE} | rows=0"
        )

        return

    df = df.copy()
    df["game_date"] = (
        df["game_date"].map(
            normalize_date
        )
    )

    sort_cols = [
        c
        for c in [
            "game_date",
            "game_time",
            "game_id",
            "away_team",
            "home_team",
            "market_type",
            "bet_side",
            "line",
        ]
        if c in df.columns
    ]

    if sort_cols:
        df = df.sort_values(
            sort_cols,
            kind="mergesort",
        )

    for game_date, date_df in df.groupby(
        "game_date",
        dropna=False,
    ):
        out_path = (
            GRADED_DIR
            / f"{game_date}_results_NHL.csv"
        )

        date_df.to_csv(
            out_path,
            index=False,
        )

        log_summary(
            "WROTE DAILY GRADED | "
            f"{out_path} | "
            f"rows={len(date_df)}"
        )

    df.to_csv(
        MASTER_FILE,
        index=False,
    )

    log_summary(
        "WROTE MASTER GRADED | "
        f"{MASTER_FILE} | rows={len(df)}"
    )


###############################################################
######################## MAIN #################################
###############################################################

def main() -> None:
    reset_logs()

    log_summary(
        "START 01_nhl_results_grade.py"
    )

    clean_old_outputs()

    try:
        bets = load_select_rows()

        # Zero selections are a valid outcome.
        # Nothing else in Stage 05 needs to be loaded
        # when there are no selected bets.
        if bets.empty:
            log_summary(
                "NO SELECTED NHL BETS TO GRADE | "
                "zero-selection slate"
            )

            write_graded_outputs(
                pd.DataFrame(
                    columns=GRADED_OUTPUT_COLUMNS
                )
            )

            print(
                "NHL grading complete: "
                "no selected bets."
            )

            log_summary(
                "END 01_nhl_results_grade.py"
            )

            return

        scores = load_score_rows()
        statuses = load_status_rows()

        (
            graded,
            pending,
            unresolved,
        ) = grade_rows(
            bets,
            scores,
            statuses,
        )

        write_partition(
            pending,
            PENDING_FILE,
            "NO PENDING GRADING ROWS",
            "WROTE PENDING GRADING ROWS",
        )

        write_partition(
            unresolved,
            UNRESOLVED_FILE,
            "NO UNRESOLVED GRADING ROWS",
            "WROTE UNRESOLVED GRADING ROWS",
        )

        write_graded_outputs(
            graded
        )

        if not unresolved.empty:
            raise RuntimeError(
                "UNRESOLVED COMPLETED NHL "
                "GRADING ROWS REMAIN | "
                f"count={len(unresolved)} | "
                f"file={UNRESOLVED_FILE}"
            )

    except Exception as e:
        log_error(
            f"GRADING FAILED | {e}"
        )

        print(
            f"NHL grading failed: {e}"
        )

        raise

    log_summary(
        "END 01_nhl_results_grade.py"
    )

    print(
        "NHL grading complete."
    )


if __name__ == "__main__":
    main()
