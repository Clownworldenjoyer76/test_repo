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
    sdv.OPTIONS.no_cache = args.no_cache
    print(f"PLAYER-STRENGTH START | day={target_day} | season={season}", flush=True)

    # Unlike the daily web slate, the full-season schedule supplies reliable
    # prior-game IDs for strictly historical PBP/shift filtering.
    schedule = nhl.load_nhl_schedule(season)
    if sdv.frame_empty(schedule):
        raise RuntimeError("Full-season NHL schedule is missing or empty")
    prior_ids = sdv.prior_game_ids_for_target(schedule, target_day)
    if not prior_ids:
        raise RuntimeError("No games strictly before snapshot date found in season schedule")

    failures: list[str] = []
    pbp = sdv.cached_season_pull(
        failures, "pbp", nhl.load_nhl_pbp_full, season, current=True,
    )
    shifts = sdv.cached_season_pull(
        failures, "shifts", nhl.load_nhl_shifts, season, current=True,
    )
    if failures or sdv.frame_empty(pbp) or sdv.frame_empty(shifts):
        raise RuntimeError(f"Missing season data: {failures}")

    prior_pbp = sdv.frame_strictly_before_game(
        pbp, target_day=target_day, prior_game_ids=prior_ids,
    )
    prior_shifts = sdv.frame_strictly_before_game(
        shifts, target_day=target_day, prior_game_ids=prior_ids,
    )
    if prior_pbp.is_empty() or prior_shifts.is_empty():
        raise RuntimeError(
            "Historical PBP/shifts are empty after strict date/ID filtering; "
            "snapshot NOT published"
        )

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
