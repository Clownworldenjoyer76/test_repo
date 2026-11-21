#!/usr/bin/env python3
from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def prepare_probability_decimal(
    model_prob,
    book_decimal,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    probability = pd.to_numeric(
        model_prob,
        errors="coerce",
    )
    decimal = pd.to_numeric(
        book_decimal,
        errors="coerce",
    )
    result = pd.Series(
        np.nan,
        index=probability.index,
        dtype="float64",
    )
    valid = (
        decimal.notna()
        & probability.notna()
        & np.isfinite(decimal)
        & np.isfinite(probability)
        & (decimal > 1)
        & (probability > 0)
        & (probability < 1)
    )
    return probability, decimal, result, valid


def read_nonempty_csv(
    input_path: Path,
    per_file_record: dict[str, Any],
    summary: dict[str, int],
    per_file: list[dict[str, Any]],
    log: Callable[..., None],
) -> pd.DataFrame | None:
    frame = pd.read_csv(input_path)

    if frame.empty:
        log(f"{input_path.name} empty â€” skipping")
        per_file_record["status"] = "empty"
        summary["skipped"] += 1
        per_file.append(per_file_record)
        return None

    per_file_record["rows"] = len(frame)
    summary["rows_processed"] += len(frame)
    return frame


def run_market_patterns(
    patterns: Iterable[tuple[str, Callable[..., Any], str]],
    summary: dict[str, int],
    per_file: list[dict[str, Any]],
    process_pattern: Callable[..., None],
    log: Callable[..., None],
    write_summary: Callable[..., None],
) -> None:
    # noinspection PyBroadException
    try:
        for pattern, process_fn, market_label in patterns:
            process_pattern(
                pattern,
                process_fn,
                market_label,
                summary,
                per_file,
            )
    except Exception as exc:
        summary["errors"] += 1
        log(
            f"FATAL: {exc}\n"
            f"{traceback.format_exc()}",
            "ERROR",
        )
        write_summary(
            summary,
            per_file,
        )
        raise

    write_summary(
        summary,
        per_file,
    )
