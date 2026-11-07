#!/usr/bin/env python3
"""Standalone, manually triggered NHL historical player-strength snapshot builder.

Repository destination: docs/win/hockey/nhl/scripts/00_intake/pull_nhl_player_strength.py

This job is intended to be triggered externally (e.g. cron-job.org dispatching
its GitHub Actions workflow). It does not retrieve confirmed pregame lineups.
It computes the same six historical team/player metrics as the original
pull_sdv.py, using ONLY games before the Eastern date on which this job starts.
No scheduling logic is embedded in this script.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import sportsdataverse.nhl as nhl

# Reuse the original, reviewed aggregation logic. The daily pull_sdv.py now
# reads the snapshots instead of calling this expensive function itself.
import pull_sdv as sdv

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "sdv" / "lineup-strength" / "player_strength_snapshots"
LATEST_FILE = ROOT / "sdv" / "lineup-strength" / "latest_team_player_strength.csv"
VERSION = "1"
COLUMNS = [
    "schema_version", "season", "history_exclusive_date",
    "snapshot_as_of_utc", "source_prior_pbp_rows", "source_prior_shift_rows",
    "team", *sdv.LINEUP_TEAM_METRICS,
]


def write_atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", prefix=".nhl_player_",
            suffix=".tmp", dir=path.parent, delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def normalized_metric(value: object, team: str, name: str) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Bad player metric {team}/{name}: {value!r}") from exc
    if not math.isfinite(number):
        # Invalid statistics remain unavailable; never persist NaN/Infinity.
        return ""
    return repr(number)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Reload full-season play-by-play and shifts instead of using SDV's freshness cache",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    started_at = datetime.now(sdv.NY)
    target_day = started_at.date()
    season = sdv.season_start_for_day(target_day)
    # The local pipeline labels NHL seasons by START year (2026 = 2026-27).
    # SportsDataverse's season-loader arguments use END years (2027 = 2026-27).
    sdv_end_season = season + 1
    sdv.OPTIONS.no_cache = args.no_cache
    print(
        f"PLAYER-STRENGTH START | day={target_day} | "
        f"pipeline_season={season} | SDV_end_season={sdv_end_season}",
        flush=True,
    )

    # Filter each SportsDataverse source by its own historical game_date.
    # Do not use schedule-derived game IDs, which may be on a different scale.
    # The loader accepts the ending year (2027), while each row stores the
    # combined season token (20262027). The snapshot consumer uses 2026.
    expected_sdv_season_token = int(f"{season}{sdv_end_season}")
    failures: list[str] = []
    pbp = sdv.cached_season_pull(
        failures, "pbp", nhl.load_nhl_pbp_full, sdv_end_season, current=True,
    )
    shifts = sdv.cached_season_pull(
        failures, "shifts", nhl.load_nhl_shifts, sdv_end_season, current=True,
    )
    if failures or sdv.frame_empty(pbp) or sdv.frame_empty(shifts):
        raise RuntimeError(f"Missing season data: {failures}")

    for label, source in (("pbp", pbp), ("shifts", shifts)):
        if "game_date" not in source.columns or "game_id" not in source.columns:
            raise RuntimeError(
                f"{label} missing game_date or game_id; "
                f"columns={source.columns}; snapshot NOT published"
            )
        # NHL source rows identify a season by its combined years, e.g.
        # 20262027 for the 2026-27 season. The loader argument is 2027.
        # Reject mixed/mismatched season values rather than publishing data
        # attributed to the wrong season.
        if "season" in source.columns:
            years = set(
                source.get_column("season")
                .cast(pl.Int64, strict=False)
                .drop_nulls()
                .unique()
                .to_list()
            )
            if years != {expected_sdv_season_token}:
                raise RuntimeError(
                    f"{label} expected row season {expected_sdv_season_token} "
                    f"(loader argument {sdv_end_season}), "
                    f"got {sorted(years)}; snapshot NOT published"
                )

    # An empty ID set explicitly selects the helper's date-based filtering.
    # It rejects today's and all future games, preserving historical cutoffs.
    prior_pbp = sdv.frame_strictly_before_game(
        pbp, target_day=target_day, prior_game_ids=set(),
    )
    prior_shifts = sdv.frame_strictly_before_game(
        shifts, target_day=target_day, prior_game_ids=set(),
    )
    print(
        f"PLAYER-STRENGTH FILTER | PBP={pbp.height}->{prior_pbp.height} "
        f"| shifts={shifts.height}->{prior_shifts.height} "
        f"| before={target_day}", flush=True,
    )
    if prior_pbp.is_empty() or prior_shifts.is_empty():
        raise RuntimeError(
            "No historical PBP/shifts rows before target date; "
            f"PBP dates={pbp.select('game_date').head(5).to_series().to_list()}, "
            f"shifts dates={shifts.select('game_date').head(5).to_series().to_list()}; "
            "snapshot NOT published"
        )

    # Normalize game IDs to Int64 (the documented SDV join type). Null and
    # unparseable IDs are excluded from joining, not sorted or joined as data.
    prior_pbp = prior_pbp.with_columns(
        pl.col("game_id").cast(pl.Int64, strict=False).alias("game_id")
    )
    prior_shifts = prior_shifts.with_columns(
        pl.col("game_id").cast(pl.Int64, strict=False).alias("game_id")
    )
    pbp_ids = set(prior_pbp.get_column("game_id").drop_nulls().unique().to_list())
    shift_ids = set(prior_shifts.get_column("game_id").drop_nulls().unique().to_list())
    shared_ids = sorted(pbp_ids & shift_ids)
    print(
        f"PLAYER-STRENGTH IDS | PBP={len(pbp_ids)} | "
        f"shifts={len(shift_ids)} | shared={len(shared_ids)} | "
        f"PBP_null_rows={prior_pbp.get_column('game_id').null_count()} | "
        f"shift_null_rows={prior_shifts.get_column('game_id').null_count()}",
        flush=True,
    )
    if not shared_ids:
        raise RuntimeError(
            "PBP and shifts have no common usable historical game IDs; "
            f"PBP sample={sorted(pbp_ids)[:5]}, "
            f"shift sample={sorted(shift_ids)[:5]}; "
            "snapshot NOT published"
        )
    prior_pbp = prior_pbp.filter(pl.col("game_id").is_in(shared_ids))
    prior_shifts = prior_shifts.filter(pl.col("game_id").is_in(shared_ids))

    # The expensive computation is ONLY invoked by this separate script.
    print(
        f"PLAYER-STRENGTH CALCULATE | prior_pbp={prior_pbp.height} "
        f"prior_shifts={prior_shifts.height}", flush=True,
    )
    metrics = sdv.compute_lineup_team_metrics(prior_pbp, prior_shifts)
    if not metrics:
        raise RuntimeError("No team player-strength metrics produced; snapshot NOT published")

    completed_at = datetime.now(UTC)
    stamp = completed_at.strftime("%Y%m%dT%H%M%S%fZ")
    rows: list[dict[str, str]] = []
    populated = 0
    for team, metric_map in sorted(metrics.items()):
        name = str(team).strip()
        if not name:
            raise RuntimeError("Historical metrics include a blank team identity")
        row = {
            "schema_version": VERSION,
            "season": str(season),
            "history_exclusive_date": target_day.isoformat(),
            "snapshot_as_of_utc": completed_at.isoformat(),
            "source_prior_pbp_rows": str(prior_pbp.height),
            "source_prior_shift_rows": str(prior_shifts.height),
            "team": name,
        }
        for metric in sdv.LINEUP_TEAM_METRICS:
            val = normalized_metric(metric_map.get(metric), name, metric)
            populated += bool(val)
            row[metric] = val
        rows.append(row)

    if populated == 0:
        raise RuntimeError("All metrics are empty; snapshot NOT published")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    archive = SNAPSHOT_DIR / f"{stamp}_team_player_strength.csv"
    write_atomic_csv(archive, rows)
    write_atomic_csv(LATEST_FILE, rows)
    print(
        f"PLAYER-STRENGTH COMPLETE | teams={len(rows)} | metrics={populated} "
        f"| seconds={time.perf_counter()-started:.1f}\n"
        f"ARCHIVE={archive}\nLATEST={LATEST_FILE}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
