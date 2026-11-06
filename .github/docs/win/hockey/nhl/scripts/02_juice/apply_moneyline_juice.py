#!/usr/bin/env python3
# hockey/nhl/scripts/02_juice/apply_moneyline_juice.py

import math
import sys
import traceback
from pathlib import Path

import pandas as pd

SCRIPTS_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)
from odds_price_common import reconcile_dataframe_prices

JUICE_HELPER_DIR = str(Path(__file__).resolve().parent)
if JUICE_HELPER_DIR not in sys.path:
    sys.path.insert(0, JUICE_HELPER_DIR)

from juice_common import (
    FATIGUE_FEATURE_COLUMNS,
    GOALIE_FEATURE_COLUMNS,
    GOALIE_NUMERIC_FEATURE_COLUMNS,
    LINEUP_FEATURE_COLUMNS,
    LINEUP_NUMERIC_FEATURE_COLUMNS,
    TEAM_STRENGTH_FEATURE_COLUMNS,
    calculate_juiced_probabilities_or_quarantine,
    apply_adjustments_or_quarantine,
    finalize_processed_file,
    load_juice_config,
    make_logger,
    quarantine_row,
    run_input_files,
    validate_columns,
    wipe_market_outputs,
)


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "01_merge" / "01_merguiced"
OUTPUT_DIR = BASE_DIR / "02_juice"
JUICE_FILE = BASE_DIR / "config" / "juice" / "nhl_moneyline_juice.csv"

ERROR_DIR = BASE_DIR / "errors" / "02_juice"
LOG_FILE = ERROR_DIR / "apply_moneyline_juice.txt"

reset_log, log = make_logger(
    LOG_FILE,
    "apply_moneyline_juice",
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_INPUT_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "game_id",
    "away_team",
    "home_team",
    *FATIGUE_FEATURE_COLUMNS,
    *TEAM_STRENGTH_FEATURE_COLUMNS,
    *GOALIE_FEATURE_COLUMNS,
    *LINEUP_FEATURE_COLUMNS,
    "away_prob_moneyline",
    "home_prob_moneyline",
    "away_fair_decimal_moneyline",
    "home_fair_decimal_moneyline",
    "away_dk_moneyline_american",
    "home_dk_moneyline_american",
    "away_dk_moneyline_decimal",
    "home_dk_moneyline_decimal",
    "odds_source",
    "moneyline_provider_id",
    "moneyline_provider_name",
    "pulled_at",
]

REQUIRED_CONFIG_COLUMNS = [
    "band",
    "band_min",
    "band_max",
    "fav_ud",
    "venue",
    "model_calibration_adjustment",
]

OUTPUT_COLUMNS = REQUIRED_INPUT_COLUMNS + [
    "away_juiced_decimal_moneyline",
    "home_juiced_decimal_moneyline",
    "away_juiced_prob_moneyline",
    "home_juiced_prob_moneyline",
    "away_normalized_prob_moneyline",
    "home_normalized_prob_moneyline",
]


def find_model_calibration_adjustment(
    juice_df: pd.DataFrame,
    american: float,
    fav_ud: str,
    venue: str,
):
    band = juice_df[
        (juice_df["band_min"] <= american)
        & (american <= juice_df["band_max"])
        & (juice_df["fav_ud"] == fav_ud)
        & (juice_df["venue"] == venue)
    ]

    if len(band) != 1:
        return None

    return float(
        band.iloc[0]["model_calibration_adjustment"]
    )


def process_file(
    path: Path,
    juice_df: pd.DataFrame,
) -> tuple[int, int, int]:
    original_df = pd.read_csv(path)
    validate_columns(
        path,
        original_df,
        REQUIRED_INPUT_COLUMNS,
    )

    df = original_df.copy()
    reconcile_dataframe_prices(
        df,
        [["away_dk_moneyline_american","away_dk_moneyline_decimal"],["home_dk_moneyline_american","home_dk_moneyline_decimal"]],
        log=lambda message, level: log(f"{level} {message}"),
    )

    for col in [
        *FATIGUE_FEATURE_COLUMNS,
        *TEAM_STRENGTH_FEATURE_COLUMNS,
        *GOALIE_NUMERIC_FEATURE_COLUMNS,
        *LINEUP_NUMERIC_FEATURE_COLUMNS,
        "away_prob_moneyline",
        "home_prob_moneyline",
        "away_fair_decimal_moneyline",
        "home_fair_decimal_moneyline",
        "away_dk_moneyline_american",
        "home_dk_moneyline_american",
        "away_dk_moneyline_decimal",
        "home_dk_moneyline_decimal",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for col in [
        "away_juiced_decimal_moneyline",
        "home_juiced_decimal_moneyline",
        "away_juiced_prob_moneyline",
        "home_juiced_prob_moneyline",
        "away_normalized_prob_moneyline",
        "home_normalized_prob_moneyline",
    ]:
        df[col] = pd.NA

    accepted_indices = []
    quarantine_rows = []

    applied = 0
    skipped_bad = 0
    skipped_noband = 0

    for row_number, (idx, row) in enumerate(df.iterrows()):
        try:
            away_american = float(
                row[
                    "away_dk_moneyline_american"
                ]
            )
            home_american = float(
                row[
                    "home_dk_moneyline_american"
                ]
            )
            away_fair = float(
                row[
                    "away_fair_decimal_moneyline"
                ]
            )
            home_fair = float(
                row[
                    "home_fair_decimal_moneyline"
                ]
            )
        except (TypeError, ValueError, OverflowError):
            reason = "bad_numeric_parse"
            skipped_bad += 1
            quarantine_row(
                original_df,
                idx,
                reason,
                quarantine_rows,
            )
            log(
                f"ROW QUARANTINE: "
                f"{path.name} row_number={row_number} "
                f"reason={reason}"
            )
            continue

        if (
            not math.isfinite(away_american)
            or not math.isfinite(home_american)
            or not math.isfinite(away_fair)
            or not math.isfinite(home_fair)
            or away_fair <= 1
            or home_fair <= 1
        ):
            reason = "bad_moneyline_values"
            skipped_bad += 1
            quarantine_row(
                original_df,
                idx,
                reason,
                quarantine_rows,
            )
            log(
                f"ROW QUARANTINE: "
                f"{path.name} row_number={row_number} "
                f"reason={reason}"
            )
            continue

        away_fav_ud = (
            "favorite"
            if away_american < 0
            else "underdog"
        )
        home_fav_ud = (
            "favorite"
            if home_american < 0
            else "underdog"
        )

        away_adjustment = find_model_calibration_adjustment(
            juice_df,
            away_american,
            away_fav_ud,
            "away",
        )
        home_adjustment = find_model_calibration_adjustment(
            juice_df,
            home_american,
            home_fav_ud,
            "home",
        )

        adjusted_decimals = apply_adjustments_or_quarantine(
            original_df=original_df,
            idx=idx,
            path=path,
            row_number=row_number,
            quarantine_rows=quarantine_rows,
            log=log,
            away_fair=away_fair,
            home_fair=home_fair,
            away_adjustment=away_adjustment,
            home_adjustment=home_adjustment,
            diagnostic_fields={
                "away_american": away_american,
                "home_american": home_american,
            },
        )

        if adjusted_decimals is None:
            skipped_noband += 1
            continue

        (
            away_juiced_decimal,
            home_juiced_decimal,
        ) = adjusted_decimals

        probabilities = (
            calculate_juiced_probabilities_or_quarantine(
                original_df=original_df,
                idx=idx,
                path=path,
                row_number=row_number,
                quarantine_rows=quarantine_rows,
                log=log,
                first_decimal=away_juiced_decimal,
                second_decimal=home_juiced_decimal,
            )
        )

        if probabilities is None:
            skipped_bad += 1
            continue

        (
            away_juiced_prob,
            home_juiced_prob,
            prob_total,
        ) = probabilities

        df.at[
            idx,
            "away_juiced_decimal_moneyline",
        ] = away_juiced_decimal

        df.at[
            idx,
            "home_juiced_decimal_moneyline",
        ] = home_juiced_decimal

        df.at[
            idx,
            "away_juiced_prob_moneyline",
        ] = away_juiced_prob

        df.at[
            idx,
            "home_juiced_prob_moneyline",
        ] = home_juiced_prob

        df.at[
            idx,
            "away_normalized_prob_moneyline",
        ] = (
            away_juiced_prob
            / prob_total
        )

        df.at[
            idx,
            "home_normalized_prob_moneyline",
        ] = (
            home_juiced_prob
            / prob_total
        )

        accepted_indices.append(idx)
        applied += 1

    return finalize_processed_file(
        path=path,
        df=df,
        original_df=original_df,
        accepted_indices=accepted_indices,
        output_columns=OUTPUT_COLUMNS,
        output_dir=OUTPUT_DIR,
        error_dir=ERROR_DIR,
        quarantine_rows=quarantine_rows,
        applied=applied,
        skipped_bad=skipped_bad,
        skipped_noband=skipped_noband,
        log=log,
    )


def main() -> None:
    reset_log()

    try:
        wipe_market_outputs(
            OUTPUT_DIR,
            ERROR_DIR,
            output_glob="*moneyline*.csv",
            quarantine_glob="*moneyline*_quarantine.csv",
            label="moneyline",
            log=log,
        )

        log(f"INPUT_DIR: {INPUT_DIR}")
        log(f"OUTPUT_DIR: {OUTPUT_DIR}")
        log(f"JUICE_FILE: {JUICE_FILE}")
        log(f"QUARANTINE_DIR: {ERROR_DIR}")

        juice_df = load_juice_config(
            JUICE_FILE,
            REQUIRED_CONFIG_COLUMNS,
            text_columns=["fav_ud", "venue"],
        )

        input_files = sorted(
            INPUT_DIR.glob(
                "*_NHL_moneyline.csv"
            )
        )

        log(
            f"Input files found: "
            f"{len(input_files)}"
        )

        if not input_files:
            raise FileNotFoundError(
                "No moneyline input files "
                f"found in {INPUT_DIR}"
            )

        run_input_files(
            input_files=input_files,
            juice_df=juice_df,
            process_file=process_file,
            log=log,
        )

        print(
            "apply_moneyline_juice complete."
        )

    except Exception as e:
        log(
            f"FATAL ERROR: {e}\n"
            f"{traceback.format_exc()}"
        )
        log("STATUS: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()