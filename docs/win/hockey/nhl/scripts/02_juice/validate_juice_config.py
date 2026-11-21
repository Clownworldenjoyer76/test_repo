#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/02_juice/validate_juice_config.py

import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "config" / "juice"
ERROR_DIR = BASE_DIR / "errors" / "02_juice"
LOG_FILE = ERROR_DIR / "validate_juice_config.txt"

ERROR_DIR.mkdir(parents=True, exist_ok=True)

MONEYLINE_FILE = CONFIG_DIR / "nhl_moneyline_juice.csv"
PUCK_LINE_FILE = CONFIG_DIR / "nhl_puck_line_juice.csv"
TOTAL_FILE = CONFIG_DIR / "nhl_total_juice.csv"

ADJUSTMENT_COLUMN = "model_calibration_adjustment"

MONEYLINE_REQUIRED = [
    "band",
    "band_min",
    "band_max",
    "fav_ud",
    "venue",
    ADJUSTMENT_COLUMN,
]

PUCK_LINE_REQUIRED = [
    "band",
    "band_min",
    "band_max",
    "venue",
    "fav_ud",
    ADJUSTMENT_COLUMN,
]

TOTAL_REQUIRED = [
    "band",
    "band_min",
    "band_max",
    "side",
    ADJUSTMENT_COLUMN,
]

# Supported Stage 02 lookup values.
MONEYLINE_FAVORITE_RANGE = range(-999, -99)
MONEYLINE_UNDERDOG_RANGE = range(100, 600)

PUCK_LINES = (
    -1.5,
    1.5,
)

TOTAL_LINES = (
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
)

VALID_VENUES = {
    "away",
    "home",
}

VALID_FAV_UD = {
    "favorite",
    "underdog",
}

VALID_TOTAL_SIDES = {
    "over",
    "under",
}

# build_juice_files.py clips puck-line and total probabilities
# to 0.01..0.99. Their smallest possible fair decimal is 1 / 0.99.
CLIPPED_MIN_FAIR_DECIMAL = 1 / 0.99


###############################################################
######################## LOGGING ##############################
###############################################################

def now() -> str:
    return datetime.now(UTC).isoformat()


def reset_log() -> None:
    LOG_FILE.write_text(
        f"=== validate_juice_config RUN {now()} ===\n",
        encoding="utf-8",
    )


def log(msg: str) -> None:
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{now()} | {msg}\n"
        )


def fail(
    errors: list[str],
    msg: str,
) -> None:
    errors.append(msg)
    log(
        f"ERROR | {msg}"
    )


def warn(
    warnings: list[str],
    msg: str,
) -> None:
    warnings.append(msg)
    log(
        f"WARNING | {msg}"
    )


###############################################################
######################## LOAD CONFIG ##########################
###############################################################

def load_config(
    path: Path,
    required: list[str],
    errors: list[str],
) -> pd.DataFrame | None:

    if not path.exists():
        fail(
            errors,
            f"MISSING CONFIG | {path}",
        )
        return None

    try:
        df = pd.read_csv(
            path,
            dtype=str,
        )

    except Exception as e:
        fail(
            errors,
            f"READ ERROR | {path} | {e}",
        )
        return None

    if df.empty:
        fail(
            errors,
            f"EMPTY CONFIG | {path}",
        )
        return None

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        fail(
            errors,
            f"MISSING COLUMNS | {path} | {missing}",
        )
        return None

    df = df.copy()

    for col in [
        "band_min",
        "band_max",
        ADJUSTMENT_COLUMN,
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for col in required:
        if col not in {
            "band_min",
            "band_max",
            ADJUSTMENT_COLUMN,
        }:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    bad_numeric = []

    for idx, row in df.iterrows():
        for col in [
            "band_min",
            "band_max",
            ADJUSTMENT_COLUMN,
        ]:
            value = row[col]

            if (
                pd.isna(value)
                or not math.isfinite(float(value))
            ):
                bad_numeric.append(
                    (
                        int(idx) + 2,
                        col,
                    )
                )

    if bad_numeric:
        fail(
            errors,
            "INVALID NUMERIC VALUES | "
            f"{path.name} | "
            f"count={len(bad_numeric)} | "
            f"sample={bad_numeric[:10]}",
        )

    bad_ranges = df[
        df["band_min"].notna()
        & df["band_max"].notna()
        & (
            df["band_min"]
            > df["band_max"]
        )
    ]

    if not bad_ranges.empty:
        fail(
            errors,
            "BAND_MIN EXCEEDS BAND_MAX | "
            f"{path.name} | "
            f"rows={[(int(i) + 2) for i in bad_ranges.index]}",
        )

    blank_band = (
        df["band"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_band.any():
        fail(
            errors,
            "BLANK BAND LABEL | "
            f"{path.name} | "
            f"rows={[(int(i) + 2) for i in df.index[blank_band]]}",
        )

    return df


###############################################################
##################### CONFIG VALIDATION #######################
###############################################################

def validate_categories(
    df: pd.DataFrame,
    path: Path,
    column: str,
    allowed: set[str],
    errors: list[str],
) -> None:

    invalid = sorted(
        set(df[column]) - allowed
    )

    if invalid:
        fail(
            errors,
            f"INVALID {column.upper()} VALUES | "
            f"{path.name} | {invalid}",
        )


def validate_expected_combinations(
    df: pd.DataFrame,
    path: Path,
    columns: list[str],
    expected: set[tuple[str, ...]],
    errors: list[str],
) -> None:

    actual = {
        tuple(
            str(row[column]).strip()
            for column in columns
        )
        for _, row in df.iterrows()
    }

    missing = sorted(
        expected - actual
    )

    if missing:
        fail(
            errors,
            "MISSING EXPECTED COMBINATIONS | "
            f"{path.name} | "
            f"columns={columns} | "
            f"missing={missing}",
        )


def validate_duplicates_and_overlaps(
    df: pd.DataFrame,
    path: Path,
    group_columns: list[str],
    errors: list[str],
) -> None:

    key_columns = (
        group_columns
        + [
            "band_min",
            "band_max",
        ]
    )

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_mask.any():
        fail(
            errors,
            "DUPLICATE CONFIG KEYS | "
            f"{path.name} | "
            f"rows={[(int(i) + 2) for i in df.index[duplicate_mask]]}",
        )

    usable = df.dropna(
        subset=[
            "band_min",
            "band_max",
        ]
    )

    group_arg = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    overlaps = []

    for group_key, group in usable.groupby(
        group_arg,
        dropna=False,
    ):

        ordered = group.sort_values(
            [
                "band_min",
                "band_max",
            ],
            kind="stable",
        )

        active_idx = None
        active_max = None

        for idx, row in ordered.iterrows():

            band_min = float(
                row["band_min"]
            )

            band_max = float(
                row["band_max"]
            )

            if (
                active_max is not None
                and band_min <= active_max
            ):
                overlaps.append(
                    (
                        group_key,
                        int(active_idx) + 2,
                        int(idx) + 2,
                    )
                )

            if (
                active_max is None
                or band_max > active_max
            ):
                active_idx = idx
                active_max = band_max

    if overlaps:
        fail(
            errors,
            "OVERLAPPING CONFIG KEYS | "
            f"{path.name} | "
            f"count={len(overlaps)} | "
            f"sample={overlaps[:10]}",
        )


def matches(
    df: pd.DataFrame,
    value: float,
    filters: dict[str, str],
) -> pd.DataFrame:

    mask = (
        df["band_min"].notna()
        & df["band_max"].notna()
        & (
            df["band_min"]
            <= value
        )
        & (
            value
            <= df["band_max"]
        )
    )

    for column, expected in filters.items():
        mask &= (
            df[column]
            == expected
        )

    return df.loc[
        mask
    ]


def validate_coverage_cases(
    df: pd.DataFrame,
    label: str,
    cases: list[
        tuple[
            float,
            dict[str, str],
        ]
    ],
    errors: list[str],
) -> None:

    bad = []

    for value, filters in cases:

        count = len(
            matches(
                df,
                value,
                filters,
            )
        )

        if count != 1:
            bad.append(
                (
                    value,
                    filters,
                    count,
                )
            )

    if bad:
        fail(
            errors,
            f"{label} COVERAGE ERRORS | "
            f"count={len(bad)} | "
            f"sample={bad[:10]}",
        )


###############################################################
############### ADJUSTED DECIMAL SAFETY #######################
###############################################################

def validate_adjustment_value(
    path: Path,
    row_number: int,
    band: str,
    adjustment: float,
    errors: list[str],
) -> bool:

    if not math.isfinite(
        adjustment
    ):
        return False

    if adjustment >= 1:
        fail(
            errors,
            "INVALID CALIBRATION ADJUSTMENT | "
            f"{path.name} | "
            f"row={row_number} | "
            f"band={band} | "
            f"adjustment={adjustment} | "
            "adjustment must be < 1",
        )
        return False

    return True


def iter_valid_adjustments(
    df: pd.DataFrame,
    path: Path,
    errors: list[str],
):
    for idx, row in df.iterrows():
        value = row[
            ADJUSTMENT_COLUMN
        ]

        if pd.isna(
            value
        ):
            continue

        adjustment = float(
            value
        )
        row_number = (
            int(idx)
            + 2
        )
        band = str(
            row["band"]
        )

        if not validate_adjustment_value(
            path,
            row_number,
            band,
            adjustment,
            errors,
        ):
            continue

        yield (
            row_number,
            band,
            adjustment,
        )


def validate_moneyline_adjusted_decimal(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    for (
        row_number,
        band,
        adjustment,
    ) in iter_valid_adjustments(
        df,
        MONEYLINE_FILE,
        errors,
    ):
        if adjustment > 0:
            threshold = (
                1
                / (
                    1
                    - adjustment
                )
            )

            warn(
                warnings,
                "UNSAFE ADJUSTED DECIMAL POSSIBLE | "
                f"{MONEYLINE_FILE.name} | "
                f"row={row_number} | "
                f"band={band} | "
                f"adjustment={adjustment} | "
                "fair_decimal at or below "
                f"{threshold:.12g} can produce "
                "adjusted_decimal <= 1",
            )


def validate_clipped_adjusted_decimal(
    df: pd.DataFrame,
    path: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    for (
        row_number,
        band,
        adjustment,
    ) in iter_valid_adjustments(
        df,
        path,
        errors,
    ):
        adjusted_decimal = (
            CLIPPED_MIN_FAIR_DECIMAL
            * (
                1
                - adjustment
            )
        )

        if (
            not math.isfinite(
                adjusted_decimal
            )
            or adjusted_decimal <= 1
        ):
            warn(
                warnings,
                "UNSAFE ADJUSTED DECIMAL POSSIBLE | "
                f"{path.name} | "
                f"row={row_number} | "
                f"band={band} | "
                f"adjustment={adjustment} | "
                f"minimum_supported_fair_decimal="
                f"{CLIPPED_MIN_FAIR_DECIMAL:.12g} | "
                f"adjusted_decimal={adjusted_decimal:.12g}",
            )


###############################################################
###################### MONEYLINE CONFIG #######################
###############################################################

def validate_moneyline_static(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:

    validate_categories(
        df,
        MONEYLINE_FILE,
        "venue",
        VALID_VENUES,
        errors,
    )

    validate_categories(
        df,
        MONEYLINE_FILE,
        "fav_ud",
        VALID_FAV_UD,
        errors,
    )

    validate_expected_combinations(
        df,
        MONEYLINE_FILE,
        [
            "fav_ud",
            "venue",
        ],
        {
            (
                "favorite",
                "away",
            ),
            (
                "favorite",
                "home",
            ),
            (
                "underdog",
                "away",
            ),
            (
                "underdog",
                "home",
            ),
        },
        errors,
    )

    validate_duplicates_and_overlaps(
        df,
        MONEYLINE_FILE,
        [
            "fav_ud",
            "venue",
        ],
        errors,
    )

    sign_mismatches = []

    for idx, row in df.dropna(
        subset=[
            "band_min",
            "band_max",
        ]
    ).iterrows():

        band_min = float(
            row["band_min"]
        )

        band_max = float(
            row["band_max"]
        )

        fav_ud = row[
            "fav_ud"
        ]

        if (
            band_max < 0
            and fav_ud != "favorite"
        ):
            sign_mismatches.append(
                int(idx)
                + 2
            )

        elif (
            band_min > 0
            and fav_ud != "underdog"
        ):
            sign_mismatches.append(
                int(idx)
                + 2
            )

        elif (
            band_min
            <= 0
            <= band_max
        ):
            sign_mismatches.append(
                int(idx)
                + 2
            )

    if sign_mismatches:
        fail(
            errors,
            "MONEYLINE FAVORITE/UNDERDOG "
            "BAND MISMATCH | "
            f"rows={sign_mismatches}",
        )

    cases = []

    for odds in MONEYLINE_FAVORITE_RANGE:

        for venue in VALID_VENUES:

            cases.append(
                (
                    float(
                        odds
                    ),
                    {
                        "fav_ud": "favorite",
                        "venue": venue,
                    },
                )
            )

    for odds in MONEYLINE_UNDERDOG_RANGE:

        for venue in VALID_VENUES:

            cases.append(
                (
                    float(
                        odds
                    ),
                    {
                        "fav_ud": "underdog",
                        "venue": venue,
                    },
                )
            )

    validate_coverage_cases(
        df,
        "MONEYLINE",
        cases,
        errors,
    )

    validate_moneyline_adjusted_decimal(
        df,
        errors,
        warnings,
    )


###############################################################
###################### PUCK-LINE CONFIG #######################
###############################################################

def validate_puck_line_static(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:

    validate_categories(
        df,
        PUCK_LINE_FILE,
        "venue",
        VALID_VENUES,
        errors,
    )

    validate_categories(
        df,
        PUCK_LINE_FILE,
        "fav_ud",
        VALID_FAV_UD,
        errors,
    )

    validate_duplicates_and_overlaps(
        df,
        PUCK_LINE_FILE,
        [
            "fav_ud",
            "venue",
        ],
        errors,
    )

    expected = {
        (
            -1.5,
            -1.5,
            venue,
            "favorite",
        )
        for venue in VALID_VENUES
    } | {
        (
            1.5,
            1.5,
            venue,
            "underdog",
        )
        for venue in VALID_VENUES
    }

    actual = {
        (
            float(
                row["band_min"]
            ),
            float(
                row["band_max"]
            ),
            str(
                row["venue"]
            ),
            str(
                row["fav_ud"]
            ),
        )
        for _, row in df.dropna(
            subset=[
                "band_min",
                "band_max",
            ]
        ).iterrows()
    }

    if actual != expected:
        fail(
            errors,
            "PUCK-LINE EXPECTED COMBINATIONS "
            "MISMATCH | "
            f"missing={sorted(expected - actual)} | "
            f"extra={sorted(actual - expected)}",
        )

    cases = []

    for line in PUCK_LINES:

        fav_ud = (
            "favorite"
            if line < 0
            else "underdog"
        )

        for venue in VALID_VENUES:

            cases.append(
                (
                    line,
                    {
                        "fav_ud": fav_ud,
                        "venue": venue,
                    },
                )
            )

    validate_coverage_cases(
        df,
        "PUCK-LINE",
        cases,
        errors,
    )

    validate_clipped_adjusted_decimal(
        df,
        PUCK_LINE_FILE,
        errors,
        warnings,
    )


###############################################################
######################## TOTAL CONFIG #########################
###############################################################

def validate_total_static(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:

    validate_categories(
        df,
        TOTAL_FILE,
        "side",
        VALID_TOTAL_SIDES,
        errors,
    )

    validate_duplicates_and_overlaps(
        df,
        TOTAL_FILE,
        [
            "side",
        ],
        errors,
    )

    expected = {
        (
            line,
            line,
            side,
        )
        for line in TOTAL_LINES
        for side in VALID_TOTAL_SIDES
    }

    actual = {
        (
            float(
                row["band_min"]
            ),
            float(
                row["band_max"]
            ),
            str(
                row["side"]
            ),
        )
        for _, row in df.dropna(
            subset=[
                "band_min",
                "band_max",
            ]
        ).iterrows()
    }

    if actual != expected:
        fail(
            errors,
            "TOTAL EXPECTED COMBINATIONS "
            "MISMATCH | "
            f"missing={sorted(expected - actual)} | "
            f"extra={sorted(actual - expected)}",
        )

    cases = [
        (
            line,
            {
                "side": side,
            },
        )
        for line in TOTAL_LINES
        for side in VALID_TOTAL_SIDES
    ]

    validate_coverage_cases(
        df,
        "TOTAL",
        cases,
        errors,
    )

    validate_clipped_adjusted_decimal(
        df,
        TOTAL_FILE,
        errors,
        warnings,
    )


###############################################################
######################## MAIN #################################
###############################################################

def main() -> None:

    reset_log()

    errors: list[str] = []
    warnings: list[str] = []

    log(
        f"CONFIG_DIR={CONFIG_DIR}"
    )

    log(
        "MONEYLINE SUPPORTED ODDS="
        "-999..-100 and +100..+599"
    )

    log(
        "PUCK-LINE SUPPORTED LINES="
        "-1.5,+1.5"
    )

    log(
        "TOTAL SUPPORTED LINES="
        "5.5,6.0,6.5,7.0,7.5"
    )

    log(
        "MONEYLINE FAIR DECIMAL RULE="
        ">1 with no upstream probability clip"
    )

    log(
        "PUCK-LINE/TOTAL MIN FAIR DECIMAL="
        f"{CLIPPED_MIN_FAIR_DECIMAL:.12g}"
    )

    moneyline = load_config(
        MONEYLINE_FILE,
        MONEYLINE_REQUIRED,
        errors,
    )

    puck_line = load_config(
        PUCK_LINE_FILE,
        PUCK_LINE_REQUIRED,
        errors,
    )

    total = load_config(
        TOTAL_FILE,
        TOTAL_REQUIRED,
        errors,
    )

    if moneyline is not None:
        validate_moneyline_static(
            moneyline,
            errors,
            warnings,
        )

    if puck_line is not None:
        validate_puck_line_static(
            puck_line,
            errors,
            warnings,
        )

    if total is not None:
        validate_total_static(
            total,
            errors,
            warnings,
        )

    log(
        f"VALIDATION ERRORS | "
        f"{len(errors)}"
    )

    log(
        f"VALIDATION WARNINGS | "
        f"{len(warnings)}"
    )

    if errors:

        log(
            "STATUS: FAILED"
        )

        print(
            "NHL Stage 02 calibration "
            "validation FAILED: "
            f"{len(errors)} error(s), "
            f"{len(warnings)} warning(s). "
            f"See {LOG_FILE}"
        )

        sys.exit(1)

    log(
        "STATUS: SUCCESS"
    )

    print(
        "NHL Stage 02 calibration "
        "validation complete: "
        f"0 errors, "
        f"{len(warnings)} warning(s)."
    )


if __name__ == "__main__":
    main()