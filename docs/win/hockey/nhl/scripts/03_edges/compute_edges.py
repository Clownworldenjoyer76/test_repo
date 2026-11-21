#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/03_edges/compute_edges.py

import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd


SCRIPTS_ROOT = str(Path(__file__).resolve().parents[1])
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)
from odds_price_common import reconcile_dataframe_prices

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from edge_pipeline_common import (
    prepare_probability_decimal,
    read_nonempty_csv,
    run_market_patterns,
)


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "02_juice"
OUTPUT_DIR = BASE_DIR / "03_edges"
ERROR_DIR = BASE_DIR / "errors" / "03_edges"
LOG_FILE = ERROR_DIR / "compute_edges.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOGGING
# =========================

def _now():
    return datetime.now(UTC).isoformat()


def _log(msg: str, level: str = "INFO"):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{_now()} | {level:<5} | {msg.rstrip()}\n")


def _write_summary(summary: dict, per_file: list) -> None:
    lines = [
        "",
        "=" * 72,
        f"SUMMARY  {_now()}",
        "=" * 72,
        f"  files_processed  : {summary['files_processed']}",
        f"  rows_processed   : {summary['rows_processed']}",
        f"  rows_accepted    : {summary['rows_accepted']}",
        f"  rows_quarantined : {summary['rows_quarantined']}",
        f"  moneyline_files  : {summary['moneyline_files']}",
        f"  puck_line_files  : {summary['puck_line_files']}",
        f"  total_files      : {summary['total_files']}",
        f"  skipped          : {summary['skipped']}",
        f"  null_edges       : {summary['null_edges']}",
        f"  schema_errors    : {summary['schema_errors']}",
        f"  errors           : {summary['errors']}",
        "",
        (
            f"  {'file':<42} {'market':<12} {'rows':>5} "
            f"{'accepted':>8} {'quarantine':>10} {'null_edges':>10} {'status':>12}"
        ),
    ]

    for pf in per_file:
        lines.append(
            f"  {pf['name']:<42} "
            f"{pf['market']:<12} "
            f"{pf['rows']:>5} "
            f"{pf['accepted']:>8} "
            f"{pf['quarantined']:>10} "
            f"{pf['null_edges']:>10} "
            f"{pf['status']:>12}"
        )

    status = (
        "SUCCESS"
        if summary["errors"] == 0 and summary["schema_errors"] == 0
        else "FAILED"
    )

    lines += [
        "",
        f"STATUS: {status}",
        "=" * 72,
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================
# HELPERS
# =========================

def validate_columns(df, required_cols, file_path):
    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_path.name} missing required columns: {missing}"
        )


def atomic_write_csv(df, output_path):
    tmp = output_path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output_path)


def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def safe_edge_pct(book_decimal, model_prob):
    p, d, out, valid = prepare_probability_decimal(
        model_prob,
        book_decimal,
    )
    out.loc[valid] = (
        p.loc[valid]
        - (1 / d.loc[valid])
    )
    return out


def split_null_edge_rows(
    df: pd.DataFrame,
    edge_columns: list[str],
    input_path: Path,
):
    null_mask = df[edge_columns].isna().any(axis=1)

    null_edge_values = int(
        df[edge_columns]
        .isna()
        .sum()
        .sum()
    )

    accepted_df = df.loc[
        ~null_mask
    ].copy()

    quarantine_df = df.loc[
        null_mask
    ].copy()

    quarantine_path = (
        ERROR_DIR
        / f"{input_path.stem}_null_edges_quarantine.csv"
    )

    if quarantine_df.empty:
        if quarantine_path.exists():
            quarantine_path.unlink()

        return (
            accepted_df,
            quarantine_df,
            null_edge_values,
            None,
        )

    def rejection_reason(row):
        missing = [
            col
            for col in edge_columns
            if pd.isna(row[col])
        ]

        return (
            "null_edge:"
            + "|".join(missing)
        )

    quarantine_df["rejection_reason"] = quarantine_df.apply(
        rejection_reason,
        axis=1,
    )

    atomic_write_csv(
        quarantine_df,
        quarantine_path,
    )

    return (
        accepted_df,
        quarantine_df,
        null_edge_values,
        quarantine_path,
    )


# =========================
# MONEYLINE
# =========================

def compute_moneyline_edges(df, file_path):
    required_cols = [
        "game_id",
        "away_normalized_prob_moneyline",
        "home_normalized_prob_moneyline",
        "away_dk_moneyline_decimal",
        "home_dk_moneyline_decimal",
    ]

    validate_columns(
        df,
        required_cols,
        file_path,
    )

    reconcile_dataframe_prices(
        df,
        [["away_dk_moneyline_american","away_dk_moneyline_decimal"],["home_dk_moneyline_american","home_dk_moneyline_decimal"]],
        log=_log,
    )

    df = to_numeric(
        df,
        required_cols[1:],
    )

    df["away_model_prob_moneyline"] = (
        df["away_normalized_prob_moneyline"]
    )
    df["home_model_prob_moneyline"] = (
        df["home_normalized_prob_moneyline"]
    )

    df["away_edge_pct_moneyline"] = safe_edge_pct(
        df["away_dk_moneyline_decimal"],
        df["away_model_prob_moneyline"],
    )
    df["home_edge_pct_moneyline"] = safe_edge_pct(
        df["home_dk_moneyline_decimal"],
        df["home_model_prob_moneyline"],
    )

    edge_columns = [
        "away_edge_pct_moneyline",
        "home_edge_pct_moneyline",
    ]

    return df, edge_columns


# =========================
# PUCK LINE
# =========================

def compute_puck_line_edges(df, file_path):
    required_cols = [
        "game_id",
        "away_normalized_prob_puck_line",
        "home_normalized_prob_puck_line",
        "away_dk_puck_line_decimal",
        "home_dk_puck_line_decimal",
    ]

    validate_columns(
        df,
        required_cols,
        file_path,
    )

    reconcile_dataframe_prices(
        df,
        [["away_dk_puck_line_american","away_dk_puck_line_decimal"],["home_dk_puck_line_american","home_dk_puck_line_decimal"]],
        log=_log,
    )

    df = to_numeric(
        df,
        required_cols[1:],
    )

    df["away_model_prob_puck_line"] = (
        df["away_normalized_prob_puck_line"]
    )
    df["home_model_prob_puck_line"] = (
        df["home_normalized_prob_puck_line"]
    )

    df["away_edge_pct_puck_line"] = safe_edge_pct(
        df["away_dk_puck_line_decimal"],
        df["away_model_prob_puck_line"],
    )
    df["home_edge_pct_puck_line"] = safe_edge_pct(
        df["home_dk_puck_line_decimal"],
        df["home_model_prob_puck_line"],
    )

    edge_columns = [
        "away_edge_pct_puck_line",
        "home_edge_pct_puck_line",
    ]

    return df, edge_columns


# =========================
# TOTAL
# =========================

def compute_total_edges(df, file_path):
    required_cols = [
        "game_id",
        "over_normalized_prob_total",
        "under_normalized_prob_total",
        "dk_total_over_decimal",
        "dk_total_under_decimal",
    ]

    validate_columns(
        df,
        required_cols,
        file_path,
    )

    reconcile_dataframe_prices(
        df,
        [["dk_total_over_american","dk_total_over_decimal"],["dk_total_under_american","dk_total_under_decimal"]],
        log=_log,
    )

    df = to_numeric(
        df,
        required_cols[1:],
    )

    df["over_model_prob_total"] = (
        df["over_normalized_prob_total"]
    )
    df["under_model_prob_total"] = (
        df["under_normalized_prob_total"]
    )

    df["over_edge_pct_total"] = safe_edge_pct(
        df["dk_total_over_decimal"],
        df["over_model_prob_total"],
    )
    df["under_edge_pct_total"] = safe_edge_pct(
        df["dk_total_under_decimal"],
        df["under_model_prob_total"],
    )

    edge_columns = [
        "over_edge_pct_total",
        "under_edge_pct_total",
    ]

    return df, edge_columns


# =========================
# DRIVER
# =========================

def process_pattern(
    pattern,
    compute_fn,
    market_label,
    summary,
    per_file,
):
    input_files = sorted(
        INPUT_DIR.glob(pattern)
    )

    if not input_files:
        _log(
            f"No input files found for pattern: {pattern}",
            "WARN",
        )
        return

    for input_path in input_files:
        pf = {
            "name": input_path.name,
            "market": market_label,
            "rows": 0,
            "accepted": 0,
            "quarantined": 0,
            "null_edges": 0,
            "status": "ok",
        }

        _log(
            f"--- FILE: {input_path.name} "
            f"market={market_label}"
        )

        try:
            df = read_nonempty_csv(
                input_path,
                pf,
                summary,
                per_file,
                _log,
            )
            if df is None:
                continue

            out_df, edge_columns = compute_fn(
                df,
                input_path,
            )

            (
                accepted_df,
                quarantine_df,
                null_edges,
                quarantine_path,
            ) = split_null_edge_rows(
                out_df,
                edge_columns,
                input_path,
            )

            pf["accepted"] = len(
                accepted_df
            )
            pf["quarantined"] = len(
                quarantine_df
            )
            pf["null_edges"] = (
                null_edges
            )

            summary["rows_accepted"] += len(
                accepted_df
            )
            summary["rows_quarantined"] += len(
                quarantine_df
            )
            summary["null_edges"] += (
                null_edges
            )

            if not quarantine_df.empty:
                pf["status"] = "quarantined"

                _log(
                    f"{input_path.name} | "
                    f"quarantined_rows={len(quarantine_df)} "
                    f"null_edge_values={null_edges}",
                    "WARN",
                )

                _log(
                    f"WROTE QUARANTINE: "
                    f"{quarantine_path}"
                )

            output_path = (
                OUTPUT_DIR
                / input_path.name
            )

            atomic_write_csv(
                accepted_df,
                output_path,
            )

            summary["files_processed"] += 1
            summary[
                f"{market_label}_files"
            ] += 1

            _log(
                f"WROTE: {output_path} "
                f"({len(accepted_df)} accepted rows)"
            )

        except ValueError as e:
            _log(
                f"{input_path.name} schema error: {e}",
                "ERROR",
            )
            pf["status"] = "schema_error"
            summary["schema_errors"] += 1

        except Exception as e:
            _log(
                f"{input_path.name} FAILED: {e}\n"
                f"{traceback.format_exc()}",
                "ERROR",
            )
            pf["status"] = "error"
            summary["errors"] += 1

        per_file.append(pf)


# =========================
# MAIN
# =========================

def main():
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== compute_edges RUN {_now()} ===\n"
        )

    summary = {
        "files_processed": 0,
        "rows_processed": 0,
        "rows_accepted": 0,
        "rows_quarantined": 0,
        "moneyline_files": 0,
        "puck_line_files": 0,
        "total_files": 0,
        "skipped": 0,
        "null_edges": 0,
        "schema_errors": 0,
        "errors": 0,
    }

    per_file = []

    _log(
        f"INPUT_DIR : {INPUT_DIR}"
    )
    _log(
        f"OUTPUT_DIR: {OUTPUT_DIR}"
    )
    _log(
        f"QUARANTINE_DIR: {ERROR_DIR}"
    )

    for output_file in OUTPUT_DIR.glob(
        "*.csv"
    ):
        output_file.unlink()

    for quarantine_file in ERROR_DIR.glob(
        "*_null_edges_quarantine.csv"
    ):
        quarantine_file.unlink()

    run_market_patterns(
        [
            (
                "*_NHL_moneyline.csv",
                compute_moneyline_edges,
                "moneyline",
            ),
            (
                "*_NHL_puck_line.csv",
                compute_puck_line_edges,
                "puck_line",
            ),
            (
                "*_NHL_total.csv",
                compute_total_edges,
                "total",
            ),
        ],
        summary,
        per_file,
        process_pattern,
        _log,
        _write_summary,
    )

    _write_summary(
        summary,
        per_file,
    )

    if (
        summary["schema_errors"] > 0
        or summary["errors"] > 0
    ):
        print(
            "compute_edges failed."
        )
        sys.exit(1)

    print(
        "compute_edges complete."
    )


if __name__ == "__main__":
    main()