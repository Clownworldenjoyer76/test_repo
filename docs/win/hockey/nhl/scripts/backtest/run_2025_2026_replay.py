#!/usr/bin/env python3
"""
Full-season 2025-26 NHL markets replay.

Uses:
  docs/win/hockey/nhl/season_master/2025_2026/games.csv
  docs/win/hockey/nhl/season_master/2025_2026/predictions.csv
  docs/win/hockey/nhl/season_master/2025_2026/sportsbook.csv
  docs/win/hockey/nhl/archive/2025_26/05_final_scores/final_scores/

Runs the CURRENT Stage 01 juice builder, Stage 02 juice scripts,
Stage 03 edges + EV/Kelly, Stage 04 selector, and Stage 05
grading/analysis/reports in an isolated temporary workspace.

Live production Stage 01-05 folders are never modified.

Run from repository root:
  python docs/win/hockey/nhl/scripts/backtest/run_2025_2026_replay.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# noinspection PyPep8
from feature_columns_common import (
    FATIGUE_FEATURE_COLUMNS,
    GOALIE_FEATURE_COLUMNS,
    LINEUP_FEATURE_COLUMNS,
    SDV_PREDICTION_COLUMNS,
    TEAM_STRENGTH_FEATURE_COLUMNS,
)


NHL_REL = Path("docs/win/hockey/nhl")
SEASON = "2025_2026"

SECONDARY_SIGNAL_COLUMNS = [
    "drat_home_win_prob",
    "drat_exp_margin",
    "drat_exp_total",
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
    "prob_disagreement",
    "margin_disagreement",
    "total_disagreement",
    "prob_disagreement_threshold_p75_prior",
    "margin_disagreement_threshold_p75_prior",
    "total_disagreement_threshold_p75_prior",
    "high_prob_disagreement_flag",
    "high_margin_disagreement_flag",
    "high_total_disagreement_flag",
    "ensemble_train_rows",
    "weighted_prob_drat_weight",
    "weighted_margin_drat_weight",
    "weighted_total_drat_weight",
    "weighted_home_win_prob",
    "weighted_exp_margin",
    "weighted_exp_total",
    "meta_home_win_prob",
    "meta_exp_margin",
    "meta_exp_total",
    "secondary_history_max_game_date",
    "secondary_model_status",
    "secondary_signal_version",
]

SPORTSBOOK_FIELDS = [
    "home_dk_moneyline_american",
    "away_dk_moneyline_american",
    "home_puck_line",
    "away_puck_line",
    "total",
    "home_dk_puck_line_american",
    "away_dk_puck_line_american",
    "dk_total_over_american",
    "dk_total_under_american",
    "home_dk_moneyline_decimal",
    "away_dk_moneyline_decimal",
    "home_dk_puck_line_decimal",
    "away_dk_puck_line_decimal",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
]

PROVENANCE_FIELDS = [
    "odds_source",
    "moneyline_provider_id",
    "moneyline_provider_name",
    "puck_line_provider_id",
    "puck_line_provider_name",
    "total_provider_id",
    "total_provider_name",
    "pulled_at",
]


def find_repo_root() -> Path:
    starts = [Path.cwd().resolve(), Path(__file__).resolve()]
    seen = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / NHL_REL).is_dir():
                return candidate
    raise RuntimeError(
        f"Could not find repository root containing {NHL_REL.as_posix()}"
    )


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")


def copy_file(src: Path, dst: Path) -> None:
    require(src, "required file")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    require(src, "required directory")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_script(fake_repo_root: Path, rel: Path) -> None:
    print(f"RUN  {rel.as_posix()}")
    script = fake_repo_root / NHL_REL / rel
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=fake_repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Script failed ({result.returncode}): {rel.as_posix()}"
        )


def american_to_decimal(value):
    try:
        a = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    if not np.isfinite(a) or a == 0:
        return np.nan
    if a > 0:
        return 1.0 + (a / 100.0)
    return 1.0 + (100.0 / abs(a))


def validate_unique_game_ids(df: pd.DataFrame, label: str) -> None:
    if "game_id" not in df.columns:
        raise RuntimeError(f"{label} missing game_id")
    ids = df["game_id"].astype(str).str.strip()
    if ids.eq("").any():
        raise RuntimeError(f"{label} contains blank game_id")
    dupes = ids[ids.duplicated(keep=False)]
    if not dupes.empty:
        raise RuntimeError(
            f"{label} contains duplicate game_id values: "
            + ", ".join(sorted(dupes.unique())[:20])
        )


def read_master(path: Path, label: str) -> pd.DataFrame:
    require(path, label)
    df = pd.read_csv(path, dtype={"game_id": str})
    validate_unique_game_ids(df, label)
    df["game_id"] = df["game_id"].astype(str).str.strip()
    return df


def build_full_season_merged_inputs(
    season_master: Path,
    fake_merge_dir: Path,
) -> dict:
    games = read_master(season_master / "games.csv", "season games")
    preds = read_master(season_master / "predictions.csv", "season predictions")
    book = read_master(season_master / "sportsbook.csv", "season sportsbook")

    required_games = {
        "game_id", "sport", "league", "game_date",
        "game_time", "home_team", "away_team",
    }
    required_preds = {
        "game_id", "home_prob_moneyline", "away_prob_moneyline",
        "away_projected_goals", "home_projected_goals",
        "total_projected_goals",
    }
    required_book = {"game_id", *SPORTSBOOK_FIELDS}

    for df, required_cols, label in [
        (games, required_games, "games.csv"),
        (preds, required_preds, "predictions.csv"),
        (book, required_book, "sportsbook.csv"),
    ]:
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise RuntimeError(f"{label} missing required columns: {missing}")

    base = games[
        [
            "game_id", "sport", "league", "game_date", "game_time",
            "away_team", "home_team",
        ]
    ].copy()

    pred_cols = [
        "game_id", "home_prob_moneyline", "away_prob_moneyline",
        "away_projected_goals", "home_projected_goals",
        "total_projected_goals",
    ]
    book_cols = ["game_id", *SPORTSBOOK_FIELDS]

    merged = base.merge(
        preds[pred_cols],
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        book[book_cols],
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    # Reconstruct decimal prices only where the historical master has
    # American odds but a blank decimal price.
    decimal_pairs = [
        ("home_dk_moneyline_american", "home_dk_moneyline_decimal"),
        ("away_dk_moneyline_american", "away_dk_moneyline_decimal"),
        ("home_dk_puck_line_american", "home_dk_puck_line_decimal"),
        ("away_dk_puck_line_american", "away_dk_puck_line_decimal"),
        ("dk_total_over_american", "dk_total_over_decimal"),
        ("dk_total_under_american", "dk_total_under_decimal"),
    ]
    for american_col, decimal_col in decimal_pairs:
        existing = pd.to_numeric(merged[decimal_col], errors="coerce")
        derived = merged[american_col].map(american_to_decimal)
        merged[decimal_col] = existing.where(existing.notna(), derived)

    # The 2025-26 season master predates these current production
    # features. Keep them explicitly blank; do not fabricate history.
    for col in (
        FATIGUE_FEATURE_COLUMNS
        + TEAM_STRENGTH_FEATURE_COLUMNS
        + GOALIE_FEATURE_COLUMNS
        + LINEUP_FEATURE_COLUMNS
        + SDV_PREDICTION_COLUMNS
    ):
        if col not in merged.columns:
            merged[col] = pd.NA

    merged["odds_source"] = "legacy_2025_26_season_master"
    merged["moneyline_provider_id"] = ""
    merged["moneyline_provider_name"] = "legacy_archive"
    merged["puck_line_provider_id"] = ""
    merged["puck_line_provider_name"] = "legacy_archive"
    merged["total_provider_id"] = ""
    merged["total_provider_name"] = "legacy_archive"
    merged["pulled_at"] = ""

    # Current build_juice_files.py expects this exact logical schema.
    columns = [
        "sport", "league", "game_date", "game_time", "game_id",
        "away_team", "home_team",
        *FATIGUE_FEATURE_COLUMNS,
        *TEAM_STRENGTH_FEATURE_COLUMNS,
        *GOALIE_FEATURE_COLUMNS,
        *LINEUP_FEATURE_COLUMNS,
        *SDV_PREDICTION_COLUMNS,
        "away_prob_moneyline", "home_prob_moneyline",
        "away_projected_goals", "home_projected_goals",
        "total_projected_goals",
        "away_puck_line", "home_puck_line", "total",
        "away_dk_moneyline_american", "home_dk_moneyline_american",
        "away_dk_moneyline_decimal", "home_dk_moneyline_decimal",
        "away_dk_puck_line_american", "home_dk_puck_line_american",
        "away_dk_puck_line_decimal", "home_dk_puck_line_decimal",
        "dk_total_over_american", "dk_total_under_american",
        "dk_total_over_decimal", "dk_total_under_decimal",
        *PROVENANCE_FIELDS,
    ]
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[columns]

    dates = (
        merged["game_date"]
        .astype(str)
        .str.strip()
    )
    valid_dates = dates.str.fullmatch(r"\d{4}_\d{2}_\d{2}")
    if not valid_dates.all():
        bad = sorted(dates[~valid_dates].unique().tolist())
        raise RuntimeError(f"Invalid game_date values: {bad[:20]}")

    min_date = dates.min()
    max_date = dates.max()

    if min_date != "2025_10_07":
        raise RuntimeError(
            f"Full-season replay expected first date 2025_10_07, got {min_date}"
        )

    fake_merge_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for game_date, part in merged.groupby("game_date", sort=True):
        out = fake_merge_dir / f"{game_date}_NHL_merged.csv"
        part.sort_values(["game_time", "game_id"]).to_csv(out, index=False)
        files_written += 1

    return {
        "games": len(games),
        "prediction_rows": len(preds),
        "sportsbook_rows": len(book),
        "merged_rows": len(merged),
        "merged_files": files_written,
        "first_date": min_date,
        "last_date": max_date,
        "games_with_prediction": int(
            merged["home_prob_moneyline"].notna().sum()
        ),
        "games_with_moneyline": int(
            pd.to_numeric(
                merged["home_dk_moneyline_american"],
                errors="coerce",
            ).notna().sum()
        ),
        "games_with_total": int(
            pd.to_numeric(
                merged["total"],
                errors="coerce",
            ).notna().sum()
        ),
        "games_with_puck_line": int(
            pd.to_numeric(
                merged["home_puck_line"],
                errors="coerce",
            ).notna().sum()
        ),
    }


def prepare_selector_inputs(ev_dir: Path, secondary_dir: Path) -> int:
    secondary_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ev_dir.glob("*_NHL_*.csv"))
    if not files:
        raise RuntimeError(f"No EV/Kelly files produced in {ev_dir}")

    count = 0
    for src in files:
        df = pd.read_csv(src, dtype={"game_id": str})
        for col in SECONDARY_SIGNAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        df["secondary_model_status"] = "backtest_disabled"
        df["secondary_signal_version"] = "2025_2026_full_season_backtest"

        # Selector requires market-specific provenance columns.
        if "odds_source" not in df.columns:
            df["odds_source"] = "legacy_2025_26_season_master"
        if "pulled_at" not in df.columns:
            df["pulled_at"] = ""

        if src.name.endswith("_NHL_moneyline.csv"):
            if "moneyline_provider_id" not in df.columns:
                df["moneyline_provider_id"] = ""
            if "moneyline_provider_name" not in df.columns:
                df["moneyline_provider_name"] = "legacy_archive"
        elif src.name.endswith("_NHL_puck_line.csv"):
            if "puck_line_provider_id" not in df.columns:
                df["puck_line_provider_id"] = ""
            if "puck_line_provider_name" not in df.columns:
                df["puck_line_provider_name"] = "legacy_archive"
        elif src.name.endswith("_NHL_total.csv"):
            if "total_provider_id" not in df.columns:
                df["total_provider_id"] = ""
            if "total_provider_name" not in df.columns:
                df["total_provider_name"] = "legacy_archive"
        else:
            continue

        df.to_csv(secondary_dir / src.name, index=False)
        count += 1

    return count


def copy_final_scores_and_build_status(
    archived_scores: Path,
    fake_final_root: Path,
) -> dict:
    score_dst = fake_final_root / "final_scores"
    score_dst.mkdir(parents=True, exist_ok=True)

    files = sorted(archived_scores.glob("*_NHL_final_scores.csv"))
    if not files:
        raise RuntimeError(f"No archived final scores in {archived_scores}")

    status_parts = []
    score_rows = 0

    for src in files:
        df = pd.read_csv(src, dtype={"game_id": str})
        if df.empty:
            continue

        required_cols = [
            "sport", "league", "game_date", "game_id",
            "away_team", "home_team",
            "away_score", "home_score",
            "total_score", "away_puck_line_result",
            "home_puck_line_result",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"{src} missing columns: {missing}")

        shutil.copy2(src, score_dst / src.name)
        score_rows += len(df)

        status = df[
            [
                "sport", "league", "game_date", "game_id",
                "away_team", "home_team",
            ]
        ].copy()
        status["game_state"] = "FINAL"
        status["game_schedule_state"] = "FINAL"
        status["is_final"] = "true"
        status["status_observed_at"] = "historical_2025_26_archive"
        status_parts.append(status)

    if not status_parts:
        raise RuntimeError("Archived final-score files contained no rows.")

    status_df = pd.concat(status_parts, ignore_index=True)
    status_df["game_id"] = status_df["game_id"].astype(str).str.strip()

    duplicates = status_df[
        status_df.duplicated(subset=["game_id"], keep=False)
    ]
    if not duplicates.empty:
        raise RuntimeError(
            "Duplicate final-score game_id values: "
            + ", ".join(
                sorted(duplicates["game_id"].unique().tolist())[:20]
            )
        )

    intermediate = fake_final_root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(
        intermediate / "nhl_game_status.csv",
        index=False,
    )

    return {
        "final_score_files": len(files),
        "final_score_rows": score_rows,
        "final_score_first_date": status_df["game_date"].astype(str).min(),
        "final_score_last_date": status_df["game_date"].astype(str).max(),
    }


def count_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += len(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            pass
    return total


def main() -> None:
    repo_root = find_repo_root()
    nhl = repo_root / NHL_REL

    season_master = nhl / "season_master/2025_2026"
    archived_scores = nhl / "archive/2025_26/05_final_scores/final_scores"
    backtest_config = nhl / "config/markets_backtest.yaml"
    output_root = nhl / f"backtest/{SEASON}"

    scripts = [
        Path("scripts/01_merge/build_juice_files.py"),
        Path("scripts/02_juice/apply_moneyline_juice.py"),
        Path("scripts/02_juice/apply_puck_line_juice.py"),
        Path("scripts/02_juice/apply_total_juice.py"),
        Path("scripts/03_edges/compute_edges.py"),
        Path("scripts/03_edges/compute_ev_kelly.py"),
        Path("scripts/04_select/hockey_select_bets.py"),
        Path("scripts/05_final_scores/01_nhl_results_grade.py"),
        Path("scripts/05_final_scores/02_nhl_results_analyze.py"),
        Path("scripts/05_final_scores/03_nhl_results_reports.py"),
    ]

    juice_configs = [
        Path("config/juice/nhl_moneyline_juice.csv"),
        Path("config/juice/nhl_puck_line_juice.csv"),
        Path("config/juice/nhl_total_juice.csv"),
    ]

    require(season_master, "season master")
    require(archived_scores, "archived final scores")
    require(backtest_config, "markets_backtest.yaml")

    for rel in scripts + juice_configs:
        require(nhl / rel, rel.as_posix())

    temp_root = Path(
        tempfile.mkdtemp(prefix="nhl_full_2025_2026_backtest_")
    )
    fake_nhl = temp_root / NHL_REL

    print(f"Repository: {repo_root}")
    print(f"Isolated replay: {temp_root}")

    try:
        for rel in scripts:
            copy_file(nhl / rel, fake_nhl / rel)

        for rel in juice_configs:
            copy_file(nhl / rel, fake_nhl / rel)

        copy_file(
            backtest_config,
            fake_nhl / "config/markets.yaml",
        )

        coverage = build_full_season_merged_inputs(
            season_master,
            fake_nhl / "01_merge",
        )

        print(
            "Season master coverage: "
            f"{coverage['first_date']} -> {coverage['last_date']} | "
            f"games={coverage['games']} | "
            f"daily_files={coverage['merged_files']}"
        )

        # CURRENT Stage 01/02/03 calculations.
        run_script(temp_root, Path("scripts/01_merge/build_juice_files.py"))
        run_script(temp_root, Path("scripts/02_juice/apply_moneyline_juice.py"))
        run_script(temp_root, Path("scripts/02_juice/apply_puck_line_juice.py"))
        run_script(temp_root, Path("scripts/02_juice/apply_total_juice.py"))
        run_script(temp_root, Path("scripts/03_edges/compute_edges.py"))
        run_script(temp_root, Path("scripts/03_edges/compute_ev_kelly.py"))

        selector_inputs = prepare_selector_inputs(
            fake_nhl / "03_edges/ev_kelly",
            fake_nhl / "03_edges/secondary_signals",
        )

        score_coverage = copy_final_scores_and_build_status(
            archived_scores,
            fake_nhl / "05_final_scores",
        )

        # CURRENT Stage 04/05 using wide-open markets_backtest.yaml.
        run_script(temp_root, Path("scripts/04_select/hockey_select_bets.py"))
        run_script(temp_root, Path("scripts/05_final_scores/01_nhl_results_grade.py"))
        run_script(temp_root, Path("scripts/05_final_scores/02_nhl_results_analyze.py"))
        run_script(temp_root, Path("scripts/05_final_scores/03_nhl_results_reports.py"))

        # Replace only dedicated backtest output.
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for src_rel, dst_name in [
            ("04_select", "selected"),
            ("05_final_scores/graded", "graded"),
            ("05_final_scores/intermediate", "intermediate"),
            ("05_final_scores/reports", "reports"),
            ("errors/01_merge", "errors/01_merge"),
            ("errors/02_juice", "errors/02_juice"),
            ("errors/03_edges", "errors/03_edges"),
            ("errors/04_select", "errors/04_select"),
            ("05_final_scores/errors", "errors/05_final_scores"),
        ]:
            src = fake_nhl / src_rel
            if src.exists():
                copy_tree(src, output_root / dst_name)

        tally = fake_nhl / "05_final_scores/nhl_market_tally.csv"
        if tally.exists():
            shutil.copy2(tally, output_root / "nhl_market_tally.csv")

        selected_files = sorted(
            (output_root / "selected").glob("*_NHL.csv")
        )
        selected_rows = count_rows(selected_files)

        graded_master = output_root / "graded/NHL_final.csv"
        if not graded_master.exists():
            raise RuntimeError(
                f"Backtest did not produce {graded_master}"
            )

        graded = pd.read_csv(graded_master, dtype={"game_id": str})
        graded_dates = (
            graded["game_date"].astype(str)
            if not graded.empty and "game_date" in graded.columns
            else pd.Series(dtype=str)
        )

        summary_lines = [
            "NHL 2025-26 FULL-SEASON MARKETS BACKTEST",
            "========================================",
            f"season_master_first_date={coverage['first_date']}",
            f"season_master_last_date={coverage['last_date']}",
            f"season_master_games={coverage['games']}",
            f"season_master_prediction_rows={coverage['prediction_rows']}",
            f"season_master_sportsbook_rows={coverage['sportsbook_rows']}",
            f"daily_merged_files={coverage['merged_files']}",
            f"games_with_prediction={coverage['games_with_prediction']}",
            f"games_with_moneyline={coverage['games_with_moneyline']}",
            f"games_with_puck_line={coverage['games_with_puck_line']}",
            f"games_with_total={coverage['games_with_total']}",
            f"selector_input_files={selector_inputs}",
            f"final_score_first_date={score_coverage['final_score_first_date']}",
            f"final_score_last_date={score_coverage['final_score_last_date']}",
            f"final_score_rows={score_coverage['final_score_rows']}",
            f"selected_files={len(selected_files)}",
            f"selected_rows={selected_rows}",
            f"graded_rows={len(graded)}",
            (
                f"graded_first_date={graded_dates.min()}"
                if not graded_dates.empty
                else "graded_first_date="
            ),
            (
                f"graded_last_date={graded_dates.max()}"
                if not graded_dates.empty
                else "graded_last_date="
            ),
            "secondary_model_enabled=false",
            "pick_preference=all",
            f"output={output_root}",
            "",
            "Primary file:",
            str(graded_master),
            "",
            "Reports:",
            str(output_root / "reports"),
            "",
        ]

        (output_root / "replay_summary.txt").write_text(
            "\n".join(summary_lines),
            encoding="utf-8",
        )

        print()
        print("FULL-SEASON BACKTEST COMPLETE")
        print(f"Output: {output_root}")
        print(f"Summary: {output_root / 'replay_summary.txt'}")
        print(f"Graded master: {graded_master}")
        print(f"Reports: {output_root / 'reports'}")

    except Exception:
        print()
        print("FULL-SEASON BACKTEST FAILED")
        print(f"Temporary replay retained: {temp_root}")
        raise
    else:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
