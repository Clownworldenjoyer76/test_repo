#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/01_merge/build_juice_files.py

import math
import sys
import traceback
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
from scipy.stats import poisson, skellam


SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# noinspection PyPep8
from feature_columns_common import (
    FATIGUE_FEATURE_COLUMNS,
    GOALIE_FEATURE_COLUMNS,
    GOALIE_NUMERIC_FEATURE_COLUMNS,
    LINEUP_FEATURE_COLUMNS,
    LINEUP_NUMERIC_FEATURE_COLUMNS,
    SDV_PREDICTION_COLUMNS,
    TEAM_STRENGTH_FEATURE_COLUMNS,
)

BASE_DIR = Path("docs/win/hockey/nhl")
INPUT_DIR = BASE_DIR / "01_merge"
OUTPUT_DIR = INPUT_DIR / "01_merguiced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR = BASE_DIR / "errors" / "01_merge"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_juice_files.txt"

BASE_COLUMNS = [
    "sport","league","game_date","game_time","game_id","away_team","home_team",
    *FATIGUE_FEATURE_COLUMNS,*TEAM_STRENGTH_FEATURE_COLUMNS,*GOALIE_FEATURE_COLUMNS,*LINEUP_FEATURE_COLUMNS,
    *SDV_PREDICTION_COLUMNS,
]

ODDS_PROVENANCE_COLUMNS = [
    "odds_source",
    "pulled_at",
]

MONEYLINE_PROVENANCE_COLUMNS = [
    *ODDS_PROVENANCE_COLUMNS,
    "moneyline_provider_id",
    "moneyline_provider_name",
]

PUCK_LINE_PROVENANCE_COLUMNS = [
    *ODDS_PROVENANCE_COLUMNS,
    "puck_line_provider_id",
    "puck_line_provider_name",
]

TOTAL_PROVENANCE_COLUMNS = [
    *ODDS_PROVENANCE_COLUMNS,
    "total_provider_id",
    "total_provider_name",
]

ALL_ODDS_PROVENANCE_COLUMNS = [
    "odds_source",
    "moneyline_provider_id",
    "moneyline_provider_name",
    "puck_line_provider_id",
    "puck_line_provider_name",
    "total_provider_id",
    "total_provider_name",
    "pulled_at",
]

MERGED_REQUIRED_COLUMNS = BASE_COLUMNS + [
    "away_prob_moneyline","home_prob_moneyline","away_projected_goals","home_projected_goals",
    "total_projected_goals","away_puck_line","home_puck_line","total",
    "away_dk_moneyline_american","home_dk_moneyline_american",
    "away_dk_moneyline_decimal","home_dk_moneyline_decimal",
    "away_dk_puck_line_american","home_dk_puck_line_american",
    "away_dk_puck_line_decimal","home_dk_puck_line_decimal",
    "dk_total_over_american","dk_total_under_american",
    "dk_total_over_decimal","dk_total_under_decimal",
    *ALL_ODDS_PROVENANCE_COLUMNS,
]

MONEYLINE_COLUMNS = BASE_COLUMNS + [
    "away_prob_moneyline","home_prob_moneyline",
    "away_fair_decimal_moneyline","home_fair_decimal_moneyline",
    "away_dk_moneyline_american","home_dk_moneyline_american",
    "away_dk_moneyline_decimal","home_dk_moneyline_decimal",
    *MONEYLINE_PROVENANCE_COLUMNS,
]

PUCK_LINE_COLUMNS = BASE_COLUMNS + [
    "away_puck_line","home_puck_line","away_prob_puck_line","home_prob_puck_line",
    "away_fair_decimal_puck_line","home_fair_decimal_puck_line",
    "away_dk_puck_line_american","home_dk_puck_line_american",
    "away_dk_puck_line_decimal","home_dk_puck_line_decimal",
    *PUCK_LINE_PROVENANCE_COLUMNS,
]

TOTAL_COLUMNS = BASE_COLUMNS + [
    "total","total_projected_goals","over_prob_total","under_prob_total",
    "over_fair_decimal_total","under_fair_decimal_total",
    "dk_total_over_american","dk_total_under_american",
    "dk_total_over_decimal","dk_total_under_decimal",
    *TOTAL_PROVENANCE_COLUMNS,
]

with open(LOG_FILE, "w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== build_juice_files RUN {datetime.now(UTC).isoformat()} ===\n")

def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()} | {msg}\n")

def wipe_output_dir() -> None:
    removed = 0
    for path in OUTPUT_DIR.glob("*.csv"):
        path.unlink()
        removed += 1
    log(f"Wiped pre-juice CSV outputs: {removed}")

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def fair_decimal(prob):
    if pd.isna(prob) or prob <= 0:
        return None
    return 1 / prob

def _calculate_puck_probability(
    line,
    projected_goals,
    opponent_projected_goals,
):
    if any(
        pd.isna(value)
        for value in (
            line,
            projected_goals,
            opponent_projected_goals,
        )
    ):
        return None
    if projected_goals <= 0 or opponent_projected_goals <= 0:
        return None
    threshold = math.floor(-line)
    probability = 1 - skellam.cdf(
        threshold,
        projected_goals,
        opponent_projected_goals,
    )
    if pd.isna(probability):
        return None
    return min(max(probability, 0.01), 0.99)


def calculate_home_puck_probability(
    home_line,
    home_projected_goals,
    away_projected_goals,
):
    return _calculate_puck_probability(
        home_line,
        home_projected_goals,
        away_projected_goals,
    )

def calculate_away_puck_probability(
    away_line,
    away_projected_goals,
    home_projected_goals,
):
    return _calculate_puck_probability(
        away_line,
        away_projected_goals,
        home_projected_goals,
    )

def calculate_total_probabilities(total_line, total_projected_goals):
    if pd.isna(total_line) or pd.isna(total_projected_goals) or total_projected_goals <= 0:
        return None, None
    total_line = float(total_line)
    if total_line.is_integer():
        push_total = int(total_line)
        under_prob = poisson.cdf(push_total - 1, total_projected_goals)
        over_prob = 1 - poisson.cdf(push_total, total_projected_goals)
        no_push_prob = under_prob + over_prob
        if pd.isna(no_push_prob) or no_push_prob <= 0:
            return None, None
        under_prob /= no_push_prob
        over_prob /= no_push_prob
    else:
        cutoff = math.floor(total_line)
        under_prob = poisson.cdf(cutoff, total_projected_goals)
        over_prob = 1 - under_prob
    if pd.isna(over_prob) or pd.isna(under_prob):
        return None, None
    return min(max(over_prob, 0.01), 0.99), min(max(under_prob, 0.01), 0.99)

def validate_schema(df: pd.DataFrame) -> list[str]:
    return [col for col in MERGED_REQUIRED_COLUMNS if col not in df.columns]

def build_moneyline(df: pd.DataFrame, output_path: Path) -> int:
    moneyline = df.copy()
    moneyline["away_fair_decimal_moneyline"] = moneyline["away_prob_moneyline"].apply(fair_decimal)
    moneyline["home_fair_decimal_moneyline"] = moneyline["home_prob_moneyline"].apply(fair_decimal)
    moneyline = moneyline[MONEYLINE_COLUMNS]
    moneyline.to_csv(output_path, index=False)
    log(f"WROTE {output_path} ({len(moneyline)} rows)")
    return len(moneyline)

def _append_probability_pair(
    first_probability,
    second_probability,
    first_probabilities,
    second_probabilities,
    first_fair_prices,
    second_fair_prices,
    issue_label,
    row_number,
    game_id,
) -> None:
    if first_probability is None or second_probability is None:
        log(
            f"ROW ISSUE: {issue_label} probability unavailable "
            f"row_number={row_number} game_id={game_id}"
        )
    first_probabilities.append(first_probability)
    second_probabilities.append(second_probability)
    first_fair_prices.append(
        fair_decimal(first_probability)
        if first_probability is not None
        else None
    )
    second_fair_prices.append(
        fair_decimal(second_probability)
        if second_probability is not None
        else None
    )


def build_puck_line(df: pd.DataFrame, output_path: Path) -> int:
    puck_line = df.copy()
    away_probs, home_probs, away_fair, home_fair = [], [], [], []

    for row_number, (_, row) in enumerate(puck_line.iterrows()):
        hp = calculate_home_puck_probability(
            row["home_puck_line"],
            row["home_projected_goals"],
            row["away_projected_goals"],
        )
        ap = calculate_away_puck_probability(
            row["away_puck_line"],
            row["away_projected_goals"],
            row["home_projected_goals"],
        )
        _append_probability_pair(
            hp, ap, home_probs, away_probs, home_fair, away_fair,
            "puck-line", row_number, row.get("game_id", ""),
        )

    puck_line["away_prob_puck_line"] = away_probs
    puck_line["home_prob_puck_line"] = home_probs
    puck_line["away_fair_decimal_puck_line"] = away_fair
    puck_line["home_fair_decimal_puck_line"] = home_fair
    puck_line = puck_line[PUCK_LINE_COLUMNS]
    puck_line.to_csv(output_path, index=False)
    log(f"WROTE {output_path} ({len(puck_line)} rows)")
    return len(puck_line)

def build_total(df: pd.DataFrame, output_path: Path) -> int:
    total = df.copy()
    over_probs, under_probs, over_fair, under_fair = [], [], [], []

    for row_number, (_, row) in enumerate(total.iterrows()):
        op, up = calculate_total_probabilities(
            row["total"],
            row["total_projected_goals"],
        )
        _append_probability_pair(
            op, up, over_probs, under_probs, over_fair, under_fair,
            "total", row_number, row.get("game_id", ""),
        )

    total["over_prob_total"] = over_probs
    total["under_prob_total"] = under_probs
    total["over_fair_decimal_total"] = over_fair
    total["under_fair_decimal_total"] = under_fair
    total = total[TOTAL_COLUMNS]
    total.to_csv(output_path, index=False)
    log(f"WROTE {output_path} ({len(total)} rows)")
    return len(total)

def process_file(path: Path) -> list[tuple[str, int]]:
    df = pd.read_csv(path)
    if df.empty:
        log(f"EMPTY: {path} — skipping")
        return []
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    numeric_columns = [
        *FATIGUE_FEATURE_COLUMNS,*TEAM_STRENGTH_FEATURE_COLUMNS,*GOALIE_NUMERIC_FEATURE_COLUMNS,*LINEUP_NUMERIC_FEATURE_COLUMNS,
        *SDV_PREDICTION_COLUMNS,
        "away_prob_moneyline","home_prob_moneyline","away_projected_goals","home_projected_goals",
        "total_projected_goals","away_puck_line","home_puck_line","total",
        "away_dk_moneyline_american","home_dk_moneyline_american",
        "away_dk_moneyline_decimal","home_dk_moneyline_decimal",
        "away_dk_puck_line_american","home_dk_puck_line_american",
        "away_dk_puck_line_decimal","home_dk_puck_line_decimal",
        "dk_total_over_american","dk_total_under_american",
        "dk_total_over_decimal","dk_total_under_decimal",
    ]
    for col in numeric_columns:
        df[col] = to_numeric(df[col])
    slate_date = path.name.replace("_NHL_merged.csv", "")
    outputs = [
        (OUTPUT_DIR / f"{slate_date}_NHL_moneyline.csv", build_moneyline),
        (OUTPUT_DIR / f"{slate_date}_NHL_puck_line.csv", build_puck_line),
        (OUTPUT_DIR / f"{slate_date}_NHL_total.csv", build_total),
    ]
    written = []
    for output_path, fn in outputs:
        written.append((str(output_path), fn(df, output_path)))
    return written

def main() -> None:
    try:
        wipe_output_dir()
        input_files = sorted(INPUT_DIR.glob("*_NHL_merged.csv"))
        log(f"Input files found: {len(input_files)}")
        if not input_files:
            raise FileNotFoundError(f"No merged input files found in {INPUT_DIR}")
        files_written = []
        for path in input_files:
            files_written.extend(process_file(path))
        log(f"Files written: {len(files_written)}")
        log("STATUS: SUCCESS")
    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
