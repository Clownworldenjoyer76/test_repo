#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


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
    TEAM_STRENGTH_FEATURE_COLUMNS,
)


def make_logger(
    log_file: Path,
    run_name: str,
) -> tuple[Callable[[], None], Callable[[str], None]]:
    def now() -> str:
        return datetime.now(UTC).isoformat()

    def reset_log() -> None:
        with log_file.open("w", encoding="utf-8") as handle:
            handle.write(f"=== {run_name} RUN {now()} ===\n")

    def log(message: str) -> None:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"{now()} | {message}\n")

    return reset_log, log


def validate_columns(
    path: Path,
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}"
        )


def load_juice_config(
    juice_file: Path,
    required_columns: list[str],
    *,
    text_columns: list[str],
) -> pd.DataFrame:
    if not juice_file.exists():
        raise FileNotFoundError(
            f"Missing config file: {juice_file}"
        )

    juice_df = pd.read_csv(juice_file)
    validate_columns(
        juice_file,
        juice_df,
        required_columns,
    )

    for column in (
        "band_min",
        "band_max",
        "model_calibration_adjustment",
    ):
        juice_df[column] = pd.to_numeric(
            juice_df[column],
            errors="coerce",
        )

    for column in text_columns:
        juice_df[column] = (
            juice_df[column]
            .astype(str)
            .str.strip()
        )

    if (
        juice_df[
            [
                "band_min",
                "band_max",
                "model_calibration_adjustment",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            f"{juice_file} has non-numeric "
            "band_min, band_max, or model_calibration_adjustment values"
        )

    return juice_df


def wipe_market_outputs(
    output_dir: Path,
    error_dir: Path,
    *,
    output_glob: str,
    quarantine_glob: str,
    label: str,
    log: Callable[[str], None],
) -> int:
    removed = 0

    for path in output_dir.glob(output_glob):
        path.unlink()
        removed += 1

    for path in error_dir.glob(quarantine_glob):
        path.unlink()
        removed += 1

    log(
        f"Wiped {label} output/quarantine CSVs: {removed}"
    )
    return removed


def quarantine_row(
    original_df: pd.DataFrame,
    idx: Any,
    reason: str,
    quarantine_rows: list[dict],
) -> None:
    rejected = original_df.loc[idx].to_dict()
    rejected["rejection_reason"] = reason
    quarantine_rows.append(rejected)


def apply_adjustments_or_quarantine(
    *,
    original_df: pd.DataFrame,
    idx: Any,
    path: Path,
    row_number: int,
    quarantine_rows: list[dict],
    log: Callable[[str], None],
    away_fair: float,
    home_fair: float,
    away_adjustment: float | None,
    home_adjustment: float | None,
    diagnostic_fields: dict[str, Any],
) -> tuple[float, float] | None:
    if (
        away_adjustment is None
        or home_adjustment is None
    ):
        reason = "no_config_band"
        quarantine_row(
            original_df,
            idx,
            reason,
            quarantine_rows,
        )

        details = " ".join(
            f"{name}={value}"
            for name, value in diagnostic_fields.items()
        )
        detail_suffix = (
            f" {details}"
            if details
            else ""
        )

        log(
            f"ROW QUARANTINE: "
            f"{path.name} row_number={row_number} "
            f"reason={reason}"
            f"{detail_suffix}"
        )
        return None

    return (
        away_fair
        * (1 - away_adjustment),
        home_fair
        * (1 - home_adjustment),
    )



def calculate_juiced_probabilities_or_quarantine(
    *,
    original_df: pd.DataFrame,
    idx: Any,
    path: Path,
    row_number: int,
    quarantine_rows: list[dict],
    log: Callable[[str], None],
    first_decimal: float,
    second_decimal: float,
    diagnostic_fields: dict[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    if (
        not math.isfinite(first_decimal)
        or not math.isfinite(second_decimal)
        or first_decimal <= 1
        or second_decimal <= 1
    ):
        reason = "bad_juiced_decimal"
        quarantine_row(
            original_df,
            idx,
            reason,
            quarantine_rows,
        )

        details = diagnostic_fields or {}
        detail_text = " ".join(
            f"{name}={value}"
            for name, value
            in details.items()
        )
        detail_suffix = (
            f" {detail_text}"
            if detail_text
            else ""
        )

        log(
            f"ROW QUARANTINE: "
            f"{path.name} row_number={row_number} "
            f"reason={reason}"
            f"{detail_suffix}"
        )
        return None

    first_prob = (
        1 / first_decimal
    )
    second_prob = (
        1 / second_decimal
    )
    prob_total = (
        first_prob
        + second_prob
    )

    if (
        not math.isfinite(prob_total)
        or prob_total <= 0
    ):
        reason = "bad_probability_total"
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
        return None

    return (
        first_prob,
        second_prob,
        prob_total,
    )

def write_quarantine(
    error_dir: Path,
    path: Path,
    original_columns: list[str],
    quarantine_rows: list[dict],
) -> Path | None:
    quarantine_path = (
        error_dir
        / f"{path.stem}_quarantine.csv"
    )

    if not quarantine_rows:
        if quarantine_path.exists():
            quarantine_path.unlink()
        return None

    quarantine_df = pd.DataFrame(
        quarantine_rows
    ).reindex(
        columns=(
            original_columns
            + ["rejection_reason"]
        )
    )

    quarantine_df.to_csv(
        quarantine_path,
        index=False,
    )
    return quarantine_path


def finalize_processed_file(
    *,
    path: Path,
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    accepted_indices: list[Any],
    output_columns: list[str],
    output_dir: Path,
    error_dir: Path,
    quarantine_rows: list[dict],
    applied: int,
    skipped_bad: int,
    skipped_noband: int,
    log: Callable[[str], None],
) -> tuple[int, int, int]:
    out_path = output_dir / path.name

    accepted_df = df.loc[
        accepted_indices,
        output_columns,
    ].copy()

    accepted_df.to_csv(
        out_path,
        index=False,
    )

    quarantine_path = write_quarantine(
        error_dir,
        path,
        list(original_df.columns),
        quarantine_rows,
    )

    log(
        f"WROTE {out_path} "
        f"rows={len(accepted_df)} "
        f"applied={applied}"
    )

    if quarantine_path is not None:
        log(
            f"WROTE {quarantine_path} "
            f"rows={len(quarantine_rows)}"
        )

    log(
        f"FILE SUMMARY: {path.name} "
        f"input={len(original_df)} "
        f"accepted={len(accepted_df)} "
        f"quarantined={len(quarantine_rows)} "
        f"bad={skipped_bad} "
        f"no_band={skipped_noband}"
    )

    return (
        applied,
        skipped_bad,
        skipped_noband,
    )


def run_input_files(
    *,
    input_files: list[Path],
    juice_df: pd.DataFrame,
    process_file: Callable[
        [Path, pd.DataFrame],
        tuple[int, int, int],
    ],
    log: Callable[[str], None],
) -> None:
    files_written = 0
    total_applied = 0
    total_skipped_bad = 0
    total_skipped_noband = 0

    for path in input_files:
        log(f"Processing input: {path}")

        (
            applied,
            skipped_bad,
            skipped_noband,
        ) = process_file(
            path,
            juice_df,
        )

        files_written += 1
        total_applied += applied
        total_skipped_bad += skipped_bad
        total_skipped_noband += skipped_noband

    total_quarantined = (
        total_skipped_bad
        + total_skipped_noband
    )

    log("--- SUMMARY ---")
    log(f"Files processed: {len(input_files)}")
    log(f"Files written: {files_written}")
    log(f"Rows applied: {total_applied}")
    log(
        f"Rows quarantined bad: "
        f"{total_skipped_bad}"
    )
    log(
        f"Rows quarantined no band: "
        f"{total_skipped_noband}"
    )
    log(
        f"Rows quarantined total: "
        f"{total_quarantined}"
    )
    log("STATUS: SUCCESS")
