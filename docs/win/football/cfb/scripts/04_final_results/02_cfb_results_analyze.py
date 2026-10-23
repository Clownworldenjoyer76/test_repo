#!/usr/bin/env python3
"""
Build the CFB one-row-per-bet grading ledger from grade_picks.py outputs.

Reads:
    config/current_week.yaml
    04_final_results/graded/week_*_CFB_graded.csv

Writes:
    04_final_results/intermediate/work_cfb.csv

The grader remains the source of truth for grades and profit units. This script
validates the graded files, verifies its extracted ledger against the grader's
per-game totals, then publishes the reporting ledger.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd
import yaml


SCRIPT_VERSION = "cfb-results-analyze-v4-signed-spread-buckets-2026-10-01"

SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = SCRIPT_DIR.parent
REPORT_ROOT = CFB_ROOT / "errors"
CURRENT_WEEK_CONFIG = CFB_ROOT / "config" / "current_week.yaml"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from pipeline_shared import stage_dataframe_csv


DEFAULT_GRADED_DIR = CFB_ROOT / "04_final_results" / "graded"
DEFAULT_OUTPUT_DIR = CFB_ROOT / "04_final_results" / "intermediate"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "work_cfb.csv"

GRADED_FILE_RE = re.compile(r"^week_(\d+)_CFB_graded\.csv$")

MARKETS = {
    "moneyline": {
        "prefix": "ml",
        "line_column": None,
        "allowed_sides": {"HOME", "AWAY"},
    },
    "spread": {
        "prefix": "spread",
        "line_column": "spread_line",
        "allowed_sides": {"HOME", "AWAY"},
    },
    "total": {
        "prefix": "total",
        "line_column": "total_line",
        "allowed_sides": {"OVER", "UNDER"},
    },
}

VALID_GRADES = {
    "Win",
    "Loss",
    "Push",
    "Void",
    "Pending",
    "No Bet",
    "Invalid Selection",
    "Invalid Line",
}

SETTLED_GRADES = {"Win", "Loss", "Push", "Void"}
PENDING_GRADES = {"Pending", "Invalid Selection", "Invalid Line"}

BASE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "final_status",
    "final_completed",
    "final_away_score",
    "final_home_score",
    "final_total",
    "final_home_margin",
    "selected_bets",
    "graded_bets",
    "wins",
    "losses",
    "pushes",
    "voids",
    "pending_bets",
    "net_units",
]

MARKET_REQUIRED_SUFFIXES = [
    "selected",
    "selection",
    "selection_reason",
    "odds_american",
    "model_probability",
    "implied_probability",
    "edge",
    "ev",
    "full_kelly",
    "kelly",
    "grade",
    "profit_units",
]

WORK_COLUMNS = [
    "season",
    "season_type",
    "week",
    "week_label",
    "game_id",
    "game_date",
    "game_time",
    "day_night",
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
    "final_completed",
    "final_away_score",
    "final_home_score",
    "final_total",
    "final_home_margin",
    "ev_bucket",
    "odds_bucket",
    "kelly_bucket",
    "model_prob_bucket",
    "win_prob_bucket",
    "spread_line_bucket",
    "spread_role",
    "total_bucket",
]


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""

    return text


def normalize_game_id(value: Any) -> str:
    text = clean(value)

    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]

    return text


def optional_float(value: Any) -> float | None:
    text = clean(value)

    if not text:
        return None

    try:
        result = float(text)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def required_float(value: Any, label: str) -> float:
    text = clean(value)

    if not text:
        fail(f"{label} is required")

    try:
        result = float(text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} must be numeric; found {value!r}"
        ) from exc

    if not math.isfinite(result):
        fail(f"{label} must be finite; found {value!r}")

    return result


def required_int(value: Any, label: str) -> int:
    number = required_float(value, label)

    if not number.is_integer():
        fail(
            f"{label} must be a whole number; "
            f"found {value!r}"
        )

    return int(number)


def strict_flag(value: Any, label: str) -> bool:
    text = clean(value).casefold()

    if text in {"1", "1.0", "true", "yes", "y"}:
        return True

    if text in {"0", "0.0", "false", "no", "n"}:
        return False

    fail(
        f"{label} must be a valid boolean/0/1 flag; "
        f"found {value!r}"
    )


def normalize_grade(value: Any, label: str) -> str:
    raw = clean(value).upper()

    mapping = {
        "WIN": "Win",
        "LOSS": "Loss",
        "PUSH": "Push",
        "VOID": "Void",
        "PENDING": "Pending",
        "NO_BET": "No Bet",
        "INVALID_SELECTION": "Invalid Selection",
        "INVALID_LINE": "Invalid Line",
    }

    result = mapping.get(
        raw,
        raw.title() if raw else "",
    )

    if result not in VALID_GRADES:
        fail(
            f"{label} has unsupported grade "
            f"{value!r}"
        )

    return result


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label} missing required columns: "
            f"{missing}"
        )


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing current-week config: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        fail(
            f"{path} must contain a YAML mapping"
        )

    return data


def resolve_target(
    config: dict[str, Any],
    season_override: int | None,
    season_type_override: int | None,
) -> tuple[int, int]:
    configured_season = required_int(
        config.get("season"),
        "current_week.season",
    )

    configured_season_type = required_int(
        config.get("season_type"),
        "current_week.season_type",
    )

    season = (
        configured_season
        if season_override is None
        else int(season_override)
    )

    season_type = (
        configured_season_type
        if season_type_override is None
        else int(season_type_override)
    )

    if season < 1900:
        fail(
            f"Invalid season: {season}"
        )

    if season_type <= 0:
        fail(
            f"Invalid season_type: {season_type}"
        )

    return season, season_type


def required_columns_for_file() -> list[str]:
    required = list(BASE_REQUIRED_COLUMNS)

    for spec in MARKETS.values():
        prefix = spec["prefix"]

        for suffix in MARKET_REQUIRED_SUFFIXES:
            required.append(
                f"{prefix}_{suffix}"
            )

        if spec["line_column"]:
            required.append(
                spec["line_column"]
            )

    return required


def discover_files(
    graded_dir: Path,
) -> list[tuple[int, Path]]:
    if not graded_dir.is_dir():
        fail(
            f"Graded directory not found: "
            f"{graded_dir}"
        )

    discovered: list[
        tuple[int, Path]
    ] = []

    for path in graded_dir.glob(
        "week_*_CFB_graded.csv"
    ):
        match = GRADED_FILE_RE.fullmatch(
            path.name
        )

        if match:
            discovered.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    return sorted(
        discovered,
        key=lambda item: (
            item[0],
            item[1].name,
        ),
    )


def read_graded_file(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"Missing graded file: {path}"
        )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if df.empty:
        fail(
            f"{path} contains no data rows"
        )

    require_columns(
        df,
        required_columns_for_file(),
        str(path),
    )

    return df


def build_side_group(
    market_type: str,
    bet_side: str,
) -> str:
    side = clean(
        bet_side
    ).upper()

    if (
        market_type in {"moneyline", "spread"}
        and side in {"HOME", "AWAY"}
    ):
        return side

    if (
        market_type == "total"
        and side in {"OVER", "UNDER"}
    ):
        return side

    return ""


def build_day_night(
    value: Any,
    label: str,
) -> str:
    raw = clean(value)

    if not raw:
        fail(
            f"{label} is required"
        )

    formats = (
        "%I:%M %p",
        "%H:%M",
        "%I:%M%p",
        "%H:%M:%S",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(
                raw,
                fmt,
            )

            return (
                "Day"
                if parsed.hour < 17
                else "Night"
            )

        except ValueError:
            continue

    fail(
        f"{label} has unsupported time value "
        f"{value!r}"
    )


def ev_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"
    if value < 0:
        return "<0"
    if value < 0.01:
        return "0.00_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.04:
        return "0.03_to_0.0399"
    if value < 0.05:
        return "0.04_to_0.0499"
    if value < 0.075:
        return "0.05_to_0.0749"
    if value < 0.10:
        return "0.075_to_0.0999"

    return "0.10_plus"


def odds_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"
    if value <= -200:
        return "minus_200_or_lower"
    if value <= -150:
        return "minus_199_to_minus_150"
    if value <= -125:
        return "minus_149_to_minus_125"
    if value <= -110:
        return "minus_124_to_minus_110"
    if value <= -101:
        return "minus_109_to_minus_101"
    if value <= 100:
        return "minus_100_to_plus_100"
    if value <= 125:
        return "plus_101_to_plus_125"
    if value <= 150:
        return "plus_126_to_plus_150"
    if value <= 200:
        return "plus_151_to_plus_200"

    return "plus_201_or_higher"


def kelly_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"
    if value <= 0:
        return "zero_or_below"
    if value < 0.01:
        return "0.001_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.05:
        return "0.03_to_0.0499"
    if value < 0.10:
        return "0.05_to_0.0999"
    if value < 0.15:
        return "0.10_to_0.1499"
    if value < 0.20:
        return "0.15_to_0.1999"

    return "0.20_plus"


def model_prob_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"

    pct = value * 100.0

    if pct < 50:
        return "<50"
    if pct < 55:
        return "50_to_54.9"
    if pct < 60:
        return "55_to_59.9"
    if pct < 65:
        return "60_to_64.9"
    if pct < 70:
        return "65_to_69.9"
    if pct < 75:
        return "70_to_74.9"
    if pct < 80:
        return "75_to_79.9"

    return "80_plus"


def spread_line_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"
    if value == 0:
        return "0.0"

    prefix = (
        "negative"
        if value < 0
        else "positive"
    )
    absolute = abs(value)

    if absolute < 3:
        return f"{prefix}_0.0_to_2.9"
    if absolute < 7:
        return f"{prefix}_3.0_to_6.9"
    if absolute < 10:
        return f"{prefix}_7.0_to_9.9"
    if absolute < 14:
        return f"{prefix}_10.0_to_13.9"
    if absolute < 21:
        return f"{prefix}_14.0_to_20.9"
    if absolute < 28:
        return f"{prefix}_21.0_to_27.9"

    return f"{prefix}_28.0_plus"


def spread_role(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"
    if value < 0:
        return "FAVORITE"
    if value > 0:
        return "UNDERDOG"

    return "PICKEM"


def total_bucket(value: Any) -> str:
    value = optional_float(value)

    if value is None:
        return "UNBUCKETED"

    start = int(
        math.floor(
            value / 5.0
        )
        * 5
    )

    return (
        f"{start}_to_"
        f"{start + 4.9:.1f}"
    )


def validate_probability(
    value: Any,
    label: str,
) -> float:
    number = required_float(
        value,
        label,
    )

    if not 0.0 <= number <= 1.0:
        fail(
            f"{label} must be in [0,1]; "
            f"found {value!r}"
        )

    return number


def validate_nonnegative_int(
    value: Any,
    label: str,
) -> int:
    number = required_int(
        value,
        label,
    )

    if number < 0:
        fail(
            f"{label} cannot be negative; "
            f"found {value!r}"
        )

    return number


def validate_market(
    row: pd.Series,
    market_type: str,
    game_id: str,
) -> tuple[
    bool,
    str,
    float | None,
]:
    spec = MARKETS[
        market_type
    ]

    prefix = spec[
        "prefix"
    ]

    selected = strict_flag(
        row.get(
            f"{prefix}_selected",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_selected"
        ),
    )

    grade = normalize_grade(
        row.get(
            f"{prefix}_grade",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_grade"
        ),
    )

    if (
        selected
        and grade == "No Bet"
    ):
        fail(
            f"game_id={game_id}: selected "
            f"{market_type} bet has grade NO_BET"
        )

    if (
        not selected
        and grade != "No Bet"
    ):
        fail(
            f"game_id={game_id}: unselected "
            f"{market_type} bet has grade "
            f"{grade!r}"
        )

    line: float | None = None

    if not selected:
        return (
            selected,
            grade,
            line,
        )

    selection = clean(
        row.get(
            f"{prefix}_selection",
            "",
        )
    ).upper()

    if (
        grade != "Invalid Selection"
        and selection not in spec[
            "allowed_sides"
        ]
    ):
        fail(
            f"game_id={game_id}: selected "
            f"{market_type} side must be one of "
            f"{sorted(spec['allowed_sides'])}; "
            f"found {selection!r}"
        )

    odds = required_float(
        row.get(
            f"{prefix}_odds_american",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_odds_american"
        ),
    )

    if odds == 0:
        fail(
            f"game_id={game_id}: selected "
            f"{market_type} odds cannot be zero"
        )

    validate_probability(
        row.get(
            f"{prefix}_model_probability",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_model_probability"
        ),
    )

    validate_probability(
        row.get(
            f"{prefix}_implied_probability",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_implied_probability"
        ),
    )

    for suffix in (
        "edge",
        "ev",
        "full_kelly",
        "kelly",
    ):
        required_float(
            row.get(
                f"{prefix}_{suffix}",
                "",
            ),
            (
                f"game_id={game_id}: "
                f"{prefix}_{suffix}"
            ),
        )

    if spec["line_column"] is not None:
        line_raw = row.get(
            spec["line_column"],
            "",
        )

        if grade == "Invalid Line":
            line = optional_float(
                line_raw
            )

        else:
            line = required_float(
                line_raw,
                (
                    f"game_id={game_id}: "
                    f"{spec['line_column']}"
                ),
            )

    units_raw = row.get(
        f"{prefix}_profit_units",
        "",
    )

    if grade in SETTLED_GRADES:
        required_float(
            units_raw,
            (
                f"game_id={game_id}: "
                f"{prefix}_profit_units"
            ),
        )

    elif clean(units_raw):
        required_float(
            units_raw,
            (
                f"game_id={game_id}: "
                f"{prefix}_profit_units"
            ),
        )

    return (
        selected,
        grade,
        line,
    )


def _validate_graded_row(
    row: pd.Series,
    *,
    row_number: int,
    path: Path,
    filename_week: int,
    season: int,
    season_type: int,
    totals: dict[str, float | int],
) -> None:
    game_id = row[
        "game_id"
    ]

    row_label = (
        f"{path}: line {row_number} "
        f"game_id={game_id}"
    )

    row_season = required_int(
        row.get("season"),
        f"{row_label}: season",
    )

    row_season_type = required_int(
        row.get("season_type"),
        f"{row_label}: season_type",
    )

    row_week = required_int(
        row.get("week"),
        f"{row_label}: week",
    )

    if row_season != season:
        fail(
            f"{row_label}: expected "
            f"season={season}; found "
            f"{row_season}"
        )

    if row_season_type != season_type:
        fail(
            f"{row_label}: expected "
            f"season_type={season_type}; "
            f"found {row_season_type}"
        )

    if row_week != filename_week:
        fail(
            f"{row_label}: filename "
            f"week={filename_week} does not "
            f"match row week={row_week}"
        )

    if (
        not clean(
            row.get("away_team")
        )
        or not clean(
            row.get("home_team")
        )
    ):
        fail(
            f"{row_label}: away_team and "
            "home_team are required"
        )

    strict_flag(
        row.get("final_completed"),
        f"{row_label}: final_completed",
    )

    grades: list[str] = []
    selected_count = 0
    selected_units = 0.0

    for market_type in MARKETS:
        (
            selected,
            grade,
            _,
        ) = validate_market(
            row,
            market_type,
            game_id,
        )

        grades.append(
            grade
        )

        if selected:
            selected_count += 1

            prefix = MARKETS[
                market_type
            ][
                "prefix"
            ]

            units = optional_float(
                row.get(
                    f"{prefix}_profit_units",
                    "",
                )
            )

            if units is not None:
                selected_units += units

    derived = {
        "selected_bets":
            selected_count,
        "graded_bets":
            sum(
                grade in SETTLED_GRADES
                for grade in grades
            ),
        "wins":
            grades.count("Win"),
        "losses":
            grades.count("Loss"),
        "pushes":
            grades.count("Push"),
        "voids":
            grades.count("Void"),
        "pending_bets":
            sum(
                grade in PENDING_GRADES
                for grade in grades
            ),
    }

    for column, expected in derived.items():
        actual = validate_nonnegative_int(
            row.get(column),
            f"{row_label}: {column}",
        )

        if actual != expected:
            fail(
                f"{row_label}: grader "
                f"{column}={actual} does not "
                "match market grades="
                f"{expected}"
            )

        totals[
            column
        ] = (
            int(
                totals[
                    column
                ]
            )
            + actual
        )

    grader_units = required_float(
        row.get("net_units"),
        f"{row_label}: net_units",
    )

    # grade_picks.py publishes each game's net_units
    # rounded to six decimal places. Validate against
    # that published precision, while retaining the
    # unrounded market-level units for season totals.
    expected_game_units = round(
        selected_units,
        6,
    )

    if not math.isclose(
        grader_units,
        expected_game_units,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(
            f"{row_label}: grader "
            f"net_units={grader_units} "
            "does not match selected market units "
            "rounded to 6 decimals="
            f"{expected_game_units} "
            f"(raw={selected_units})"
        )

    totals[
        "net_units"
    ] = (
        float(
            totals[
                "net_units"
            ]
        )
        + selected_units
    )



def _validate_graded_rows(
    df: pd.DataFrame,
    *,
    path: Path,
    filename_week: int,
    season: int,
    season_type: int,
    totals: dict[str, float | int],
) -> None:
    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=2,
    ):
        _validate_graded_row(
            row,
            row_number=row_number,
            path=path,
            filename_week=filename_week,
            season=season,
            season_type=season_type,
            totals=totals,
        )


def validate_graded_file(
    df: pd.DataFrame,
    path: Path,
    filename_week: int,
    season: int,
    season_type: int,
) -> dict[str, float | int]:
    df["game_id"] = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    if df[
        "game_id"
    ].eq(
        ""
    ).any():
        fail(
            f"{path} contains blank game_id values"
        )

    if df[
        "game_id"
    ].duplicated().any():
        examples = (
            df.loc[
                df[
                    "game_id"
                ].duplicated(
                    keep=False
                ),
                "game_id",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        fail(
            f"{path} contains duplicate "
            f"game_id values: {examples}"
        )

    totals: dict[
        str,
        float | int,
    ] = {
        "selected_bets": 0,
        "graded_bets": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "voids": 0,
        "pending_bets": 0,
        "net_units": 0.0,
    }

    _validate_graded_rows(
        df,
        path=path,
        filename_week=filename_week,
        season=season,
        season_type=season_type,
        totals=totals,
    )

    return totals


def bet_row(
    row: pd.Series,
    market_type: str,
) -> dict[str, Any] | None:
    spec = MARKETS[
        market_type
    ]

    prefix = spec[
        "prefix"
    ]

    game_id = normalize_game_id(
        row.get(
            "game_id"
        )
    )

    selected = strict_flag(
        row.get(
            f"{prefix}_selected",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_selected"
        ),
    )

    if not selected:
        return None

    selection = clean(
        row.get(
            f"{prefix}_selection",
            "",
        )
    )

    grade = normalize_grade(
        row.get(
            f"{prefix}_grade",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_grade"
        ),
    )

    line: float | None = None

    if spec["line_column"] is not None:
        line = optional_float(
            row.get(
                spec["line_column"],
                "",
            )
        )

    model_prob = required_float(
        row.get(
            f"{prefix}_model_probability",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_model_probability"
        ),
    )

    implied_prob = required_float(
        row.get(
            f"{prefix}_implied_probability",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_implied_probability"
        ),
    )

    odds = required_float(
        row.get(
            f"{prefix}_odds_american",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_odds_american"
        ),
    )

    edge = required_float(
        row.get(
            f"{prefix}_edge",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_edge"
        ),
    )

    ev = required_float(
        row.get(
            f"{prefix}_ev",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_ev"
        ),
    )

    full_kelly = required_float(
        row.get(
            f"{prefix}_full_kelly",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_full_kelly"
        ),
    )

    kelly = required_float(
        row.get(
            f"{prefix}_kelly",
            "",
        ),
        (
            f"game_id={game_id}: "
            f"{prefix}_kelly"
        ),
    )

    output = {
        "season":
            required_int(
                row.get("season"),
                f"game_id={game_id}: season",
            ),
        "season_type":
            required_int(
                row.get("season_type"),
                (
                    f"game_id={game_id}: "
                    "season_type"
                ),
            ),
        "week":
            required_int(
                row.get("week"),
                f"game_id={game_id}: week",
            ),
        "game_id":
            game_id,
        "game_date":
            clean(
                row.get("game_date")
            ),
        "game_time":
            clean(
                row.get("game_time")
            ),
        "away_team":
            clean(
                row.get("away_team")
            ),
        "home_team":
            clean(
                row.get("home_team")
            ),
        "market_type":
            market_type,
        "bet_side":
            selection.lower(),
        "side_group":
            build_side_group(
                market_type,
                selection,
            ),
        "line":
            line,
        "odds_american":
            odds,
        "model_prob":
            model_prob,
        "implied_prob":
            implied_prob,
        "edge":
            edge,
        "ev":
            ev,
        "full_kelly":
            full_kelly,
        "kelly":
            kelly,
        "selection_reason":
            clean(
                row.get(
                    f"{prefix}_selection_reason",
                    "",
                )
            ),
        "bet_result":
            grade,
        "bet_units":
            optional_float(
                row.get(
                    f"{prefix}_profit_units",
                    "",
                )
            ),
        "final_status":
            clean(
                row.get("final_status")
            ),
        "final_completed":
            int(
                strict_flag(
                    row.get("final_completed"),
                    (
                        f"game_id={game_id}: "
                        "final_completed"
                    ),
                )
            ),
        "final_away_score":
            optional_float(
                row.get("final_away_score")
            ),
        "final_home_score":
            optional_float(
                row.get("final_home_score")
            ),
        "final_total":
            optional_float(
                row.get("final_total")
            ),
        "final_home_margin":
            optional_float(
                row.get("final_home_margin")
            ),
        "week_label":
            (
                "Week "
                + str(
                    required_int(
                        row.get("week"),
                        f"game_id={game_id}: week",
                    )
                )
            ),
        "day_night":
            build_day_night(
                clean(
                    row.get("game_time")
                ),
                (
                    f"game_id={game_id}: "
                    "game_time"
                ),
            ),
        "ev_bucket":
            ev_bucket(ev),
        "odds_bucket":
            odds_bucket(odds),
        "kelly_bucket":
            kelly_bucket(kelly),
        "model_prob_bucket":
            model_prob_bucket(model_prob),
        "win_prob_bucket":
            model_prob_bucket(model_prob),
        "spread_line_bucket":
            (
                spread_line_bucket(line)
                if market_type == "spread"
                else "UNBUCKETED"
            ),
        "spread_role":
            (
                spread_role(line)
                if market_type == "spread"
                else "UNBUCKETED"
            ),
        "total_bucket":
            (
                total_bucket(line)
                if market_type == "total"
                else "UNBUCKETED"
            ),
    }

    return output


def ledger_totals(
    work: pd.DataFrame,
) -> dict[str, float | int]:
    if work.empty:
        return {
            "selected_bets": 0,
            "graded_bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "voids": 0,
            "pending_bets": 0,
            "net_units": 0.0,
        }

    results = work[
        "bet_result"
    ]

    units = pd.to_numeric(
        work[
            "bet_units"
        ],
        errors="coerce",
    ).dropna()

    return {
        "selected_bets":
            len(work),
        "graded_bets":
            int(
                results.isin(
                    SETTLED_GRADES
                ).sum()
            ),
        "wins":
            int(
                results.eq(
                    "Win"
                ).sum()
            ),
        "losses":
            int(
                results.eq(
                    "Loss"
                ).sum()
            ),
        "pushes":
            int(
                results.eq(
                    "Push"
                ).sum()
            ),
        "voids":
            int(
                results.eq(
                    "Void"
                ).sum()
            ),
        "pending_bets":
            int(
                results.isin(
                    PENDING_GRADES
                ).sum()
            ),
        "net_units":
            (
                float(
                    units.sum()
                )
                if not units.empty
                else 0.0
            ),
    }


def validate_ledger(
    work: pd.DataFrame,
    season: int,
    season_type: int,
    expected_totals: dict[str, float | int],
) -> None:
    if list(
        work.columns
    ) != WORK_COLUMNS:
        fail(
            "Generated work_cfb.csv "
            "columns changed"
        )

    if not work.empty:
        for column in (
            "season",
            "season_type",
            "week",
        ):
            if work[
                column
            ].isna().any():
                fail(
                    "Generated ledger contains "
                    f"blank {column}"
                )

        if set(
            work[
                "season"
            ].astype(
                int
            ).tolist()
        ) != {
            season
        }:
            fail(
                "Generated ledger contains "
                "season values other than "
                f"{season}"
            )

        if set(
            work[
                "season_type"
            ].astype(
                int
            ).tolist()
        ) != {
            season_type
        }:
            fail(
                "Generated ledger contains "
                "season_type values other than "
                f"{season_type}"
            )

        if work[
            "game_id"
        ].map(
            clean
        ).eq(
            ""
        ).any():
            fail(
                "Generated ledger contains "
                "blank game_id"
            )

        if not set(
            work[
                "market_type"
            ]
        ).issubset(
            set(
                MARKETS
            )
        ):
            fail(
                "Generated ledger contains "
                "unsupported market_type"
            )

        duplicate_key = [
            "season",
            "week",
            "game_id",
            "market_type",
        ]

        if work.duplicated(
            duplicate_key
        ).any():
            examples = work.loc[
                work.duplicated(
                    duplicate_key,
                    keep=False,
                ),
                duplicate_key,
            ].head(
                10
            )

            fail(
                "Duplicate selected market "
                "rows detected:\n"
                + examples.to_string(
                    index=False
                )
            )

    actual_totals = ledger_totals(
        work
    )

    for column in (
        "selected_bets",
        "graded_bets",
        "wins",
        "losses",
        "pushes",
        "voids",
        "pending_bets",
    ):
        if (
            int(
                actual_totals[
                    column
                ]
            )
            != int(
                expected_totals[
                    column
                ]
            )
        ):
            fail(
                f"Ledger {column}="
                f"{actual_totals[column]} "
                "does not match grader "
                f"total={expected_totals[column]}"
            )

    if not math.isclose(
        float(
            actual_totals[
                "net_units"
            ]
        ),
        float(
            expected_totals[
                "net_units"
            ]
        ),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        fail(
            "Ledger net_units="
            f"{actual_totals['net_units']} "
            "does not match grader total="
            f"{expected_totals['net_units']}"
        )


def publish_csv(
    work: pd.DataFrame,
    output_file: Path,
    season: int,
    season_type: int,
    expected_totals: dict[str, float | int],
) -> bool:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_file.with_name(
        f".{output_file.name}."
        f"{uuid.uuid4().hex}.tmp"
    )

    try:
        serialized = stage_dataframe_csv(temporary, work)

        require_columns(
            serialized,
            WORK_COLUMNS,
            str(temporary),
        )

        typed = serialized.copy()

        for column in (
            "season",
            "season_type",
            "week",
        ):
            if not typed.empty:
                typed[
                    column
                ] = typed[
                    column
                ].map(
                    lambda value, c=column: (
                        required_int(
                            value,
                            f"serialized {c}",
                        )
                    )
                )

        if not typed.empty:
            typed[
                "bet_units"
            ] = typed[
                "bet_units"
            ].map(
                optional_float
            )

        validate_ledger(
            typed,
            season,
            season_type,
            expected_totals,
        )

        modified = (
            not output_file.is_file()
            or output_file.read_bytes()
            != temporary.read_bytes()
        )

        if modified:
            os.replace(
                temporary,
                output_file,
            )

        return modified

    finally:
        temporary.unlink(
            missing_ok=True
        )


def build_work_file(
    graded_dir: Path,
    output_file: Path,
    season: int,
    season_type: int,
    report: PipelineReporter,
) -> pd.DataFrame:
    files = discover_files(
        graded_dir
    )

    if not files:
        fail(
            "No week_*_CFB_graded.csv "
            f"files found in {graded_dir}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    source_games = 0

    expected_totals: dict[
        str,
        float | int,
    ] = {
        "selected_bets": 0,
        "graded_bets": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "voids": 0,
        "pending_bets": 0,
        "net_units": 0.0,
    }

    weeks: list[int] = []

    for filename_week, path in files:
        report.add_input(
            path
        )

        df = read_graded_file(
            path
        )

        file_totals = validate_graded_file(
            df,
            path,
            filename_week,
            season,
            season_type,
        )

        weeks.append(
            filename_week
        )

        source_games += len(
            df
        )

        for key in expected_totals:
            if key == "net_units":
                expected_totals[
                    key
                ] = (
                    float(
                        expected_totals[
                            key
                        ]
                    )
                    + float(
                        file_totals[
                            key
                        ]
                    )
                )

            else:
                expected_totals[
                    key
                ] = (
                    int(
                        expected_totals[
                            key
                        ]
                    )
                    + int(
                        file_totals[
                            key
                        ]
                    )
                )

        for _, game in df.iterrows():
            for market_type in MARKETS:
                item = bet_row(
                    game,
                    market_type,
                )

                if item is not None:
                    rows.append(
                        item
                    )

    work = pd.DataFrame(
        rows,
        columns=WORK_COLUMNS,
    )

    if not work.empty:
        work = work.sort_values(
            [
                "season",
                "week",
                "game_date",
                "game_time",
                "game_id",
                "market_type",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    validate_ledger(
        work,
        season,
        season_type,
        expected_totals,
    )

    output_modified = publish_csv(
        work,
        output_file,
        season,
        season_type,
        expected_totals,
    )

    result_counts = (
        work[
            "bet_result"
        ].value_counts(
            dropna=False
        ).to_dict()
        if not work.empty
        else {}
    )

    market_counts = (
        work[
            "market_type"
        ].value_counts(
            dropna=False
        ).to_dict()
        if not work.empty
        else {}
    )

    report.set_rows(
        rows_in=source_games,
        rows_out=len(work),
    )

    report.update_details(
        {
            "script_version":
                SCRIPT_VERSION,
            "graded_dir":
                str(
                    graded_dir
                ),
            "output_file":
                str(
                    output_file
                ),
            "season_type":
                season_type,
            "source_files":
                len(
                    files
                ),
            "source_games":
                source_games,
            "weeks":
                weeks,
            "selected_bets":
                len(
                    work
                ),
            "market_counts":
                market_counts,
            "result_counts":
                result_counts,
            "grader_totals":
                expected_totals,
            "output_modified":
                output_modified,
        }
    )

    print(
        "CFB analyze complete: "
        f"version={SCRIPT_VERSION} "
        f"files={len(files)} "
        f"games={source_games} "
        f"selected_bets={len(work)} "
        f"results={result_counts} "
        "output_modified="
        f"{'yes' if output_modified else 'no'} "
        f"output={output_file}"
    )

    return work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert graded CFB game rows "
            "into a validated one-row-per-bet "
            "ledger."
        )
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

    parser.add_argument(
        "--graded-dir",
        type=Path,
        default=DEFAULT_GRADED_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
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
            "script_version":
                SCRIPT_VERSION,
            "output_scope":
                "grading_analysis_ledger",
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG
        )

        report.add_output(
            args.output.resolve()
        )

        config = load_config(
            CURRENT_WEEK_CONFIG
        )

        (
            season,
            season_type,
        ) = resolve_target(
            config,
            args.season,
            args.season_type,
        )

        report.season = season

        report.update_details(
            {
                "season_type":
                    season_type,
            }
        )

        build_work_file(
            graded_dir=(
                args.grader_dir.resolve()
                if hasattr(args, "grader_dir")
                else args.graded_dir.resolve()
            ),
            output_file=args.output.resolve(),
            season=season,
            season_type=season_type,
            report=report,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )