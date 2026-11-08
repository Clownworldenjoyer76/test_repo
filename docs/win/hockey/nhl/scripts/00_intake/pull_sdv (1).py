#!/usr/bin/env python3
"""
Single-entry SportsDataverse NHL puller.

Default:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py

Historical:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --season 2024
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --start-season 2021 --end-season 2024

Optional category subset:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --categories schedule,goalie,odds

Default current run: all production families and SDV predictions; ESPN odds,
raw duplicates, roster/profile diagnostics disabled. Use --include-raw,
--include-diagnostics or --categories odds only when explicitly needed.

Current mode never requires a date. It uses the current America/New_York date.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import pickle
import platform
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from collections import defaultdict
import json
import shutil
import sys
import time
import traceback
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import sportsdataverse.nhl as nhl


BASE_DIR = Path("docs/win/hockey/nhl")
ROOT = BASE_DIR / "sdv"
OFFICIAL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
CATEGORIES = (
    "schedule",
    "team-strength",
    "goalie",
    "lineup-strength",
    "fatigue",
    "sdv_predictions",
    "odds",
)
NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass
class PullOptions:
    include_raw: bool = False
    include_diagnostics: bool = False
    max_workers: int = 3
    cache_dir: Path = Path.home() / ".cache" / "nhl_sdv_season_v2"
    no_cache: bool = False
    cache_current_hours: float = 8.0
    cache_historical_days: float = 30.0
    compare_baseline_dir: Path | None = None


OPTIONS = PullOptions()
TIMING_LOCK = threading.Lock()
TIMING_EVENTS: list[dict[str, Any]] = []
CACHE_COUNTS = {"hit": 0, "miss": 0, "write": 0, "invalid": 0}
TIMING_PATH = BASE_DIR / "errors" / "00_intake" / "pull_sdv_timing.json"

# Only applies to current-slate fatigue team schedule requests.
# Three total attempts: initial request plus two bounded retries.
FATIGUE_SCHEDULE_MAX_ATTEMPTS = 3
FATIGUE_SCHEDULE_RETRY_SECONDS = 1.0


def record_timing(name: str, elapsed: float, status: str = "ok") -> None:
    with TIMING_LOCK:
        TIMING_EVENTS.append({
            "name": name,
            "seconds": round(elapsed, 5),
            "status": status,
        })


@contextmanager
def timed_operation(name: str):
    started_at = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - started_at
        record_timing(name, duration, status)
        print(f"SDV TIMING | {name} | {duration:.2f}s | {status}", flush=True)


def timed_call(name: str, function, *args, **kwargs):
    with timed_operation(name):
        return function(*args, **kwargs)
    return None


def library_version() -> str:
    try:
        return importlib.metadata.version("sportsdataverse")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _cache_file(label: str, season: int) -> Path:
    return OPTIONS.cache_dir / f"{label}_season_{season}.pickle"


def _cache_fresh(metadata: dict[str, Any], label: str, season: int,
                 current: bool, now: datetime) -> bool:
    if (metadata.get("schema") != 2
            or metadata.get("label") != label
            or metadata.get("season") != season
            or metadata.get("python") != platform.python_version_tuple()[:2]
            or metadata.get("sportsdataverse") != library_version()):
        return False
    try:
        stamped = datetime.fromisoformat(metadata["fetched_at_utc"])
        if stamped.tzinfo is None or stamped > now:
            return False
        age_seconds = (now - stamped).total_seconds()
    except (ValueError, TypeError, KeyError):
        return False
    if current:
        # Cached in-season data is never reused across ET calendar days.
        # The season dataset can gain new game results overnight.
        return (
            stamped.astimezone(NY).date() == now.astimezone(NY).date()
            and age_seconds <= OPTIONS.cache_current_hours * 3600
        )
    return age_seconds <= OPTIONS.cache_historical_days * 86400


def cached_season_pull(failures: list[str], label: str, function,
                       season: int, *, current: bool):
    """Trust only same-version, same-season, freshness-checked cache entries.

    Existing game-level as-of validations remain entirely in their builders.
    Cached full-season source frames never bypass those validations.
    """
    path = _cache_file(label, season)
    now = datetime.now(timezone.utc)
    if not OPTIONS.no_cache and path.is_file():
        try:
            # Cache is created by this workflow, not loaded from user uploads.
            with path.open("rb") as stream:
                metadata, value = pickle.load(stream)
            if _cache_fresh(metadata, label, season, current, now) and not frame_empty(value):
                with TIMING_LOCK:
                    CACHE_COUNTS["hit"] += 1
                print(f"SDV CACHE HIT | {label} | season={season}", flush=True)
                return value
            with TIMING_LOCK:
                CACHE_COUNTS["invalid"] += 1
        except (OSError, EOFError, ValueError, TypeError, AttributeError, pickle.UnpicklingError):
            with TIMING_LOCK:
                CACHE_COUNTS["invalid"] += 1

    with TIMING_LOCK:
        CACHE_COUNTS["miss"] += 1
    print(f"SDV CACHE MISS | {label} | season={season}", flush=True)
    value = safe_pull(failures, "season-context", label, function, season)
    if value is None or frame_empty(value) or OPTIONS.no_cache:
        return value

    OPTIONS.cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": 2,
        "label": label,
        "season": season,
        "python": platform.python_version_tuple()[:2],
        "sportsdataverse": library_version(),
        "fetched_at_utc": now.isoformat(),
    }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".nhl_sdv_",
                                         suffix=".tmp", dir=OPTIONS.cache_dir,
                                         delete=False) as stream:
            tmp_path = Path(stream.name)
            pickle.dump((meta, value), stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        with TIMING_LOCK:
            CACHE_COUNTS["write"] += 1
    except (OSError, TypeError, ValueError, pickle.PicklingError) as exc:
        print(f"WARN | SDV cache write {label}: {exc}", file=sys.stderr)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return value


def write_timing_report(wall_seconds: float) -> None:
    TIMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | int]] = {}
    for event in TIMING_EVENTS:
        stat = summary.setdefault(event["name"], {"count": 0, "seconds": 0.0})
        stat["count"] += 1
        stat["seconds"] = round(stat["seconds"] + event["seconds"], 5)
    report = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall_seconds, 3),
        "cache": dict(CACHE_COUNTS),
        "summary": summary,
        "events": list(TIMING_EVENTS),
    }
    TIMING_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SDV TIMING | total | {wall_seconds:.2f}s", flush=True)
    for name, stat in sorted(summary.items(),
                             key=lambda item: item[1]["seconds"], reverse=True)[:15]:
        print(f"SDV TIMING SUMMARY | {name} | "
              f"{stat['seconds']:.2f}s | calls={stat['count']}", flush=True)
    print(f"SDV TIMING REPORT | {TIMING_PATH}", flush=True)


TEAM_STRENGTH_FIELDS = (
    "adj_xgf",
    "adj_xga",
    "adj_xg_net",
    "adj_gf",
    "adj_ga",
    "off_rank",
    "def_rank",
    "net_rank",
    "net_z",
)


GOALIE_STATUS_VALUES = (
    "projected",
    "expected",
    "confirmed",
    "unknown",
)

GOALIE_GSAX_FIELDS = (
    "player_id",
    "goalie",
    "shots",
    "xga",
    "ga",
    "gsax",
    "gsax_per_60",
)

GOALIE_PRODUCTION_CUTOFF_MINUTES = 60
GOALIE_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"


LINEUP_PRODUCTION_CUTOFF_MINUTES = 60
LINEUP_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"

LINEUP_TEAM_METRICS = (
    "skater_rapm",
    "skater_war",
    "pp_value",
    "pk_value",
    "forward_line_strength",
    "defense_pair_strength",
)

LINEUP_STATUS_VALUES = (
    "projected",
    "expected",
    "confirmed",
    "unknown",
)

PREGAME_FEATURE_EVALUATION = (
    (
        "nhl_xg",
        "xg_quality",
        "covered_elsewhere",
        "prior_pbp_safe",
        "VERIFIED",
        "Used upstream by team-strength/RAPM; do not duplicate as a separate lineup feature.",
    ),
    (
        "nhl_goalie_gsax",
        "goalie_performance",
        "covered_elsewhere",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Integrated in SDV-P3 goalie strength.",
    ),
    (
        "nhl_skater_rapm",
        "player_impact",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate leakage-safe prior-game skater xG RAPM to team/game level.",
    ),
    (
        "nhl_skater_war",
        "player_impact",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate leakage-safe prior-game WAR to team/game level.",
    ),
    (
        "nhl_special_teams_value",
        "power_play_penalty_kill",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate PP and PK value to team/game level.",
    ),
    (
        "nhl_unit_ratings",
        "forward_line_defense_pair",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate forward-line and defense-pair unit_value to team/game level.",
    ),
    (
        "nhl_penalty_value",
        "penalty_value",
        "evaluated_not_selected",
        "prior_pbp_safe",
        "VERIFIED",
        "WAR already contains a penalty component; separate production use would duplicate signal.",
    ),
    (
        "nhl_faceoff_value",
        "faceoff_value",
        "evaluated_not_selected",
        "prior_pbp_safe",
        "VERIFIED",
        "WAR already contains a faceoff component; separate production use would duplicate signal.",
    ),
    (
        "nhl_edge_skating_value",
        "edge_skating",
        "research_only_not_production_safe",
        "not_reconstructable_at_t60",
        "VERIFIED",
        "SportsDataverse requires caller-supplied season aggregate EDGE detail_frames; "
        "the live detail path is not implemented and the aggregate has no historical observation "
        "timestamp, so it cannot reproduce a T-60 historical information state.",
    ),
    (
        "nhl_expected_assists",
        "expected_assists",
        "research_only",
        "prior_pbp_safe",
        "VERIFIED",
        "SportsDataverse computes entirely from the supplied PBP, so the metric is leakage-safe "
        "when the pipeline supplies only games strictly before the target date; keep research-only "
        "until incremental predictive value is demonstrated.",
    ),
    (
        "nhl_zone_transitions",
        "zone_transitions",
        "research_only",
        "prior_pbp_safe",
        "VERIFIED",
        "SportsDataverse computes from the supplied PBP and is leakage-safe on prior-game input, "
        "but controlled-entry classification is a documented PBP heuristic without ground-truth "
        "microstat tags, so keep it research-only.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        help="NHL season start year, e.g. --season 2024. Repeat for multiple seasons.",
    )
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated category names, or 'all'.",
    )
    parser.add_argument("--include-raw", action="store_true",
                        help="Also fetch duplicate raw API responses for diagnostics")
    parser.add_argument("--include-diagnostics", action="store_true",
                        help="Fetch roster, goalie-profile, standings and season diagnostic data")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Bounded parallel team schedule requests (1-4, default 3)")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(os.environ.get("NHL_SDV_CACHE_DIR",
                                       str(Path.home() / ".cache" / "nhl_sdv_season_v2"))))
    parser.add_argument("--no-cache", action="store_true",
                        help="Force season source downloads, bypass persistent cache")
    parser.add_argument("--cache-current-hours", type=float, default=8.0)
    parser.add_argument("--cache-historical-days", type=float, default=30.0)
    parser.add_argument("--compare-baseline-dir", type=Path,
                        help="Compare latest production CSVs with an independently saved baseline")
    return parser.parse_args()


def requested_seasons(args: argparse.Namespace) -> list[int]:
    out: list[int] = []

    if args.season:
        out.extend(args.season)

    if args.start_season is not None or args.end_season is not None:
        if args.start_season is None or args.end_season is None:
            raise SystemExit(
                "--start-season and --end-season must be used together"
            )

        if args.end_season < args.start_season:
            raise SystemExit(
                "--end-season cannot be less than --start-season"
            )

        out.extend(
            range(
                args.start_season,
                args.end_season + 1,
            )
        )

    return sorted(set(out))


def requested_categories(args: argparse.Namespace) -> set[str]:
    if str(args.categories).strip().lower() == "all":
        return set(CATEGORIES) - {"odds"}

    values = {
        x.strip()
        for x in str(args.categories).split(",")
        if x.strip()
    }

    unknown = sorted(values - set(CATEGORIES))

    if unknown:
        raise SystemExit(
            f"Unknown categories: {unknown}"
        )

    return values


def season_start_for_day(day: date) -> int:
    return (
        day.year
        if day.month >= 7
        else day.year - 1
    )


def run_stamp() -> str:
    return datetime.now(NY).strftime(
        "%Y_%m_%dT%H%M%S_ET"
    )



def utc_iso(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_timestamp_utc(
    value: Any,
) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=UTC
        )

    return parsed.astimezone(
        UTC
    )


def conservative_game_date_cutoff_utc(
    game_day: date,
) -> datetime:
    return datetime.combine(
        game_day,
        dt_time.min,
        tzinfo=NY,
    ).astimezone(
        UTC
    )


def game_date_time_cutoff_utc(
    game_day: date,
    game_time: Any,
) -> datetime | None:
    text = str(
        game_time or ""
    ).strip()

    if not text:
        return None

    for fmt in (
        "%H:%M:%S",
        "%H:%M",
    ):
        try:
            parsed_time = datetime.strptime(
                text,
                fmt,
            ).time()
            return datetime.combine(
                game_day,
                parsed_time,
                tzinfo=NY,
            ).astimezone(
                UTC
            )
        except ValueError:
            continue

    return None


@lru_cache(maxsize=1)
def official_schedule_cutoff_lookup() -> dict[
    str,
    datetime,
]:
    lookup: dict[
        str,
        datetime,
    ] = {}

    if not OFFICIAL_SCHEDULE_DIR.exists():
        return lookup

    for path in sorted(
        OFFICIAL_SCHEDULE_DIR.glob(
            "*.csv"
        )
    ):
        try:
            pdf = pd.read_csv(
                path,
                dtype=str,
            ).fillna("")
        except (OSError, UnicodeError, ValueError):
            continue

        if (
            "game_id" not in pdf.columns
            or "start_time_utc" not in pdf.columns
        ):
            continue

        for _, row in pdf.iterrows():
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            cutoff = parse_timestamp_utc(
                row.get(
                    "start_time_utc",
                    "",
                )
            )

            if game_id and cutoff is not None:
                lookup[
                    game_id
                ] = cutoff

    return lookup


def schedule_games_with_cutoffs(
    schedule: Any,
) -> pd.DataFrame:
    if frame_empty(
        schedule
    ):
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "pregame_cutoff_utc",
                "pregame_cutoff_source",
            ]
        )

    pdf = as_pandas(
        schedule
    )

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )
    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )
    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )
    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        gid_col is None
        or date_col is None
        or home_col is None
        or away_col is None
    ):
        return pd.DataFrame()

    direct_utc_col = first_col(
        pdf,
        (
            "start_time_utc",
            "startTimeUTC",
            "start_utc",
        ),
    )

    datetime_col = first_col(
        pdf,
        (
            "game_datetime",
            "game_date_time",
            "start_datetime",
            "date_time",
        ),
    )

    game_time_col = first_col(
        pdf,
        (
            "game_time",
            "start_time_et",
            "start_time",
        ),
    )

    official_lookup = (
        official_schedule_cutoff_lookup()
    )

    rows: list[
        dict[str, str]
    ] = []

    for _, row in pdf.iterrows():
        game_id = str(
            row.get(
                gid_col,
                "",
            )
        ).strip()

        parsed_day = pd.to_datetime(
            row.get(
                date_col,
                "",
            ),
            errors="coerce",
        )

        if (
            not game_id
            or pd.isna(
                parsed_day
            )
        ):
            continue

        game_day = parsed_day.date()

        cutoff = (
            official_lookup.get(
                game_id
            )
        )
        source = (
            "official_nhl_schedule"
            if cutoff is not None
            else ""
        )

        if (
            cutoff is None
            and direct_utc_col is not None
        ):
            cutoff = parse_timestamp_utc(
                row.get(
                    direct_utc_col,
                    "",
                )
            )
            if cutoff is not None:
                source = (
                    f"schedule:{direct_utc_col}"
                )

        if (
            cutoff is None
            and datetime_col is not None
        ):
            raw = row.get(
                datetime_col,
                "",
            )
            parsed = pd.to_datetime(
                raw,
                errors="coerce",
                utc=True,
            )
            if not pd.isna(
                parsed
            ):
                cutoff = (
                    parsed.to_pydatetime()
                    .astimezone(
                        UTC
                    )
                )
                source = (
                    f"schedule:{datetime_col}"
                )

        if (
            cutoff is None
            and game_time_col is not None
        ):
            raw_game_time = row.get(
                game_time_col,
                "",
            )

            cutoff = (
                game_date_time_cutoff_utc(
                    game_day,
                    raw_game_time,
                )
            )

            if cutoff is not None:
                source = (
                    f"schedule:{game_time_col}_et"
                )
            else:
                parsed = pd.to_datetime(
                    raw_game_time,
                    errors="coerce",
                    utc=True,
                )

                if not pd.isna(
                    parsed
                ):
                    cutoff = (
                        parsed.to_pydatetime()
                        .astimezone(
                            UTC
                        )
                    )
                    source = (
                        f"schedule:{game_time_col}"
                    )

        if cutoff is None:
            cutoff = (
                conservative_game_date_cutoff_utc(
                    game_day
                )
            )
            source = (
                "conservative_game_date_start_et"
            )

        rows.append(
            {
                "game_id": game_id,
                "game_date": (
                    game_day.isoformat()
                ),
                "home_team": str(
                    row.get(
                        home_col,
                        "",
                    )
                ).strip(),
                "away_team": str(
                    row.get(
                        away_col,
                        "",
                    )
                ).strip(),
                "pregame_cutoff_utc": (
                    utc_iso(
                        cutoff
                    )
                ),
                "pregame_cutoff_source": source,
            }
        )

    return pd.DataFrame(
        rows
    )


def strict_historical_ratings_as_of_utc(
    pregame_cutoff_utc: datetime,
) -> datetime:
    return (
        pregame_cutoff_utc
        - timedelta(
            microseconds=1
        )
    )


def goalie_production_cutoff_utc(
    game_start_utc: datetime,
) -> datetime:
    return (
        game_start_utc.astimezone(UTC)
        - timedelta(
            minutes=GOALIE_PRODUCTION_CUTOFF_MINUTES
        )
    )




def lineup_production_cutoff_utc(
    game_start_utc: datetime,
) -> datetime:
    return (
        game_start_utc.astimezone(UTC)
        - timedelta(
            minutes=LINEUP_PRODUCTION_CUTOFF_MINUTES
        )
    )


def pregame_feature_evaluation() -> dict[str, Any]:
    rows = []

    for (
        function_name,
        family,
        decision,
        as_of_capability,
        evaluation_status,
        reason,
    ) in PREGAME_FEATURE_EVALUATION:
        rows.append(
            {
                "function": function_name,
                "family": family,
                "available": bool(
                    hasattr(
                        nhl,
                        function_name,
                    )
                ),
                "decision": decision,
                "as_of_capability": as_of_capability,
                "evaluation_status": evaluation_status,
                "reason": reason,
            }
        )

    return {
        "evaluation_status": "VERIFIED",
        "decision_cutoff_minutes_before_puck_drop": (
            LINEUP_PRODUCTION_CUTOFF_MINUTES
        ),
        "historical_input_rule": (
            "PBP and shifts supplied to pregame player/microstat calculations "
            "must contain only games strictly before the target game date."
        ),
        "missing_or_late_lineup_behavior": (
            "unknown status with lineup-dependent fields blank; "
            "never backfill observations after T-60"
        ),
        "families": rows,
    }


def ensure_dirs() -> None:
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in CATEGORIES:
        (ROOT / name).mkdir(
            parents=True,
            exist_ok=True,
        )


def is_polars_frame(obj: Any) -> bool:
    return isinstance(
        obj,
        pl.DataFrame,
    )


def is_pandas_frame(obj: Any) -> bool:
    return isinstance(
        obj,
        pd.DataFrame,
    )


def as_pandas(obj: Any) -> pd.DataFrame:
    if is_pandas_frame(obj):
        return obj.copy()

    if is_polars_frame(obj):
        return pd.DataFrame(
            obj.to_dicts()
        )

    raise TypeError(
        f"Not a DataFrame: {type(obj).__name__}"
    )


def csv_safe_value(value: Any) -> Any:
    if isinstance(
        value,
        (dict, list, tuple, set),
    ):
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )

    return value


def csv_safe_frame(obj: Any) -> pd.DataFrame:
    df = as_pandas(obj)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                csv_safe_value
            )

    return df


def _write_json(
    path: Path,
    obj: Any,
) -> None:
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    obj: Any,
) -> None:
    csv_safe_frame(obj).to_csv(
        path,
        index=False,
    )


def save_object(
    category: str,
    label: str,
    obj: Any,
    *,
    prefix: str,
    current: bool,
) -> list[Path]:
    out_dir = ROOT / category
    written: list[Path] = []

    if (
        isinstance(obj, dict)
        and obj
        and any(
            is_polars_frame(v)
            or is_pandas_frame(v)
            for v in obj.values()
        )
    ):
        for key, value in obj.items():
            written.extend(
                save_object(
                    category,
                    f"{label}_{key}",
                    value,
                    prefix=prefix,
                    current=current,
                )
            )

        return written

    if (
        is_polars_frame(obj)
        or is_pandas_frame(obj)
    ):
        production_features = {
            ("fatigue", "fatigue"),
            ("team-strength", "team_ratings_asof"),
            ("goalie", "game_goalie_features_asof"),
            ("lineup-strength", "game_lineup_features_asof"),
        }
        if (category, label) in production_features and len(obj.columns) == 0:
            print(
                f"SDV empty feature snapshot skipped: "
                f"{category}/{label} ({prefix})"
            )
            return written

        snap = (
            out_dir
            / f"{prefix}_{label}.csv"
        )

        _write_csv(
            snap,
            obj,
        )

        written.append(snap)

        if current:
            latest = (
                out_dir
                / f"latest_{label}.csv"
            )

            shutil.copyfile(
                snap,
                latest,
            )

            written.append(latest)

        return written

    snap = (
        out_dir
        / f"{prefix}_{label}.json"
    )

    _write_json(
        snap,
        obj,
    )

    written.append(snap)

    if current:
        latest = (
            out_dir
            / f"latest_{label}.json"
        )

        shutil.copyfile(
            snap,
            latest,
        )

        written.append(latest)

    return written


def safe_pull(
    failures: list[str],
    category: str,
    label: str,
    fn,
    *args,
    **kwargs,
):
    started_at = time.perf_counter()
    status = "ok"
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        status = "error"
        with TIMING_LOCK:
            failures.append(f"{category}/{label}: {exc}")
        print(f"WARN | {category}/{label} | {exc}", file=sys.stderr)
        return None
    finally:
        record_timing(f"request/{category}/{label}",
                      time.perf_counter() - started_at, status)



def frame_empty(obj: Any) -> bool:
    if obj is None:
        return True

    if is_polars_frame(obj):
        return obj.is_empty()

    if is_pandas_frame(obj):
        return obj.empty

    return False


def clean_war_rows(war: Any) -> Any:
    if is_polars_frame(war):
        if "player_id" not in war.columns:
            return war

        player_id = pl.col(
            "player_id"
        ).cast(
            pl.Int64,
            strict=False,
        )

        return war.filter(
            player_id.is_not_null()
            & (player_id != 0)
        )

    if is_pandas_frame(war):
        if "player_id" not in war.columns:
            return war

        player_id = pd.to_numeric(
            war["player_id"],
            errors="coerce",
        )

        return war.loc[
            player_id.notna()
            & player_id.ne(0)
        ].copy()

    return war


def first_col(
    df: Any,
    names: tuple[str, ...],
) -> str | None:
    cols = set(
        as_pandas(df).columns
    )

    for name in names:
        if name in cols:
            return name

    return None


def filter_exact_date(
    schedule: Any,
    target: date,
):
    if frame_empty(schedule):
        return schedule

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "schedule_date",
            "game_date",
            "date",
            "start_date",
        ),
    )

    if date_col is None:
        return schedule

    normalized = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dt.date

    return pl.from_pandas(
        pdf.loc[
            normalized == target
        ].reset_index(
            drop=True
        )
    )


def schedule_teams(
    schedule: Any,
) -> list[str]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(schedule)

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        home_col is None
        or away_col is None
    ):
        return []

    vals = pd.concat(
        [
            pdf[home_col],
            pdf[away_col],
        ],
        ignore_index=True,
    )

    vals = (
        vals
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        {
            x
            for x in vals
            if x
        }
    )


def historical_schedule_date_bounds(
    schedule: Any,
) -> tuple[
    date | None,
    date | None,
]:
    if frame_empty(schedule):
        return None, None

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    if date_col is None:
        return None, None

    d = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dropna()

    if d.empty:
        return None, None

    return (
        d.min().date(),
        d.max().date(),
    )


def historical_schedule_game_dates(
    schedule: Any,
) -> list[date]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )

    if date_col is None:
        return []

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dropna()

    return sorted(
        {
            value.date()
            for value in parsed
        }
    )


def current_or_historical_prefix(
    *,
    current: bool,
    season: int | None,
) -> str:
    if current:
        return run_stamp()

    assert season is not None

    return f"season_{season}"


def save_pair(
    failures: list[str],
    *,
    category: str,
    label: str,
    parsed_fn,
    raw_fn,
    prefix: str,
    current: bool,
):
    parsed = safe_pull(
        failures,
        category,
        f"{label}_parsed",
        parsed_fn,
    )

    if parsed is not None:
        save_object(
            category,
            label,
            parsed,
            prefix=prefix,
            current=current,
        )

    raw = (safe_pull(
        failures,
        category,
        f"{label}_raw",
        raw_fn,
    ) if OPTIONS.include_raw else None)

    if raw is not None:
        save_object(
            category,
            f"{label}_raw",
            raw,
            prefix=prefix,
            current=current,
        )

    return parsed


def normalize_loader_schedule_for_ratings(
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    if schedule.is_empty():
        return pl.DataFrame()

    required = {
        "game_id",
        "season",
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
    }

    if not required.issubset(
        set(schedule.columns)
    ):
        return pl.DataFrame()

    work = schedule

    if "game_type" in work.columns:
        work = work.filter(
            pl.col("game_type") == "R"
        )

    return work.select(
        pl.col("game_id"),
        pl.col("season").cast(
            pl.Int64
        ),
        pl.col("game_date")
        .cast(pl.Utf8)
        .str.strptime(
            pl.Date,
            "%Y-%m-%d",
            strict=False,
        )
        .alias("date"),
        pl.col("home_team_abbr")
        .cast(pl.Utf8)
        .alias("home_abbr"),
        pl.col("away_team_abbr")
        .cast(pl.Utf8)
        .alias("away_abbr"),
        pl.lit(False).alias(
            "neutral_site"
        ),
    )


def ratings_from_game_rates(
    game_rates: pl.DataFrame,
) -> pl.DataFrame:
    if game_rates.is_empty():
        return pl.DataFrame()

    const = nhl.get_constants(
        "nhl"
    )

    xg_adj = nhl.adjust_rate_opponent(
        game_rates,
        for_col="xgf",
        against_col="xga",
        hfa=const.hfa,
        avg=const.avg_xgf,
        shrink_k=const.shrink_k,
    )

    goal_adj = nhl.adjust_rate_opponent(
        game_rates,
        for_col="gf",
        against_col="ga",
        hfa=const.hfa,
        avg=(
            const.avg_total_goals
            / 2.0
        ),
        shrink_k=const.shrink_k,
    )

    if (
        xg_adj.is_empty()
        or goal_adj.is_empty()
    ):
        return pl.DataFrame()

    out = (
        xg_adj.join(
            goal_adj.select(
                "team",
                pl.col(
                    "adj_for"
                ).alias(
                    "adj_gf"
                ),
                pl.col(
                    "adj_against"
                ).alias(
                    "adj_ga"
                ),
            ),
            on="team",
            how="left",
        )
        .rename(
            {
                "adj_for": "adj_xgf",
                "adj_against": "adj_xga",
                "adj_net": "adj_xg_net",
            }
        )
    )

    net_mean = out[
        "adj_xg_net"
    ].mean()

    net_std = out[
        "adj_xg_net"
    ].std()

    return out.with_columns(
        pl.col("adj_xgf")
        .rank(
            method="ordinal",
            descending=True,
        )
        .cast(pl.Int64)
        .alias("off_rank"),

        pl.col("adj_xga")
        .rank(
            method="ordinal",
            descending=False,
        )
        .cast(pl.Int64)
        .alias("def_rank"),

        pl.col("adj_xg_net")
        .rank(
            method="ordinal",
            descending=True,
        )
        .cast(pl.Int64)
        .alias("net_rank"),

        (
            (
                (
                    pl.col(
                        "adj_xg_net"
                    )
                    - net_mean
                )
                / net_std
            )
            if net_std not in (
                None,
                0,
            )
            else pl.lit(0.0)
        ).alias(
            "net_z"
        ),
    ).select(
        "season",
        "team",
        "adj_xgf",
        "adj_xga",
        "adj_xg_net",
        "adj_gf",
        "adj_ga",
        "games",
        "off_rank",
        "def_rank",
        "net_rank",
        "net_z",
    )




def normalize_team_strength_asof(
    ratings: Any,
    *,
    schedule: Any,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if (
        frame_empty(ratings)
        or frame_empty(schedule)
    ):
        return pl.DataFrame()

    ratings_pdf = as_pandas(
        ratings
    )

    team_col = first_col(
        ratings_pdf,
        (
            "team",
            "team_abbr",
            "team_abbrev",
            "team_abbreviation",
        ),
    )

    if team_col is None:
        raise ValueError(
            "Team-strength ratings missing team identity column"
        )

    missing = [
        field
        for field in TEAM_STRENGTH_FIELDS
        if field not in ratings_pdf.columns
    ]

    if missing:
        raise ValueError(
            "Team-strength ratings missing required fields: "
            f"{missing}"
        )

    games = (
        schedule_games_with_cutoffs(
            schedule
        )
    )

    if games.empty:
        return pl.DataFrame()

    ratings_pdf = (
        ratings_pdf.copy()
    )
    ratings_pdf[
        "_team_key"
    ] = (
        ratings_pdf[
            team_col
        ]
        .astype(str)
        .str.strip()
    )

    ratings_by_team = {
        str(
            row[
                "_team_key"
            ]
        ).strip(): row
        for _, row in ratings_pdf.iterrows()
        if str(
            row[
                "_team_key"
            ]
        ).strip()
    }

    rows: list[
        dict[str, Any]
    ] = []

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    for _, game in games.iterrows():
        cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        if cutoff is None:
            continue

        # Current snapshots are only valid when they were created
        # strictly before the target game's pregame cutoff.
        if generated_at_utc >= cutoff:
            continue

        for team in (
            str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
        ):
            rating = (
                ratings_by_team.get(
                    team
                )
            )

            if rating is None:
                continue

            out = {
                "game_id": str(
                    game.get(
                        "game_id",
                        "",
                    )
                ).strip(),
                "game_date": str(
                    game.get(
                        "game_date",
                        "",
                    )
                ).strip(),
                "team": team,
                "pregame_cutoff_utc": (
                    utc_iso(
                        cutoff
                    )
                ),
                "ratings_as_of_utc": (
                    utc_iso(
                        generated_at_utc
                    )
                ),
                "pregame_cutoff_source": str(
                    game.get(
                        "pregame_cutoff_source",
                        "",
                    )
                ).strip(),
            }

            if "season" in ratings_pdf.columns:
                out[
                    "season"
                ] = pd.to_numeric(
                    rating.get(
                        "season"
                    ),
                    errors="coerce",
                )

            for field in (
                TEAM_STRENGTH_FIELDS
            ):
                out[
                    field
                ] = pd.to_numeric(
                    rating.get(
                        field
                    ),
                    errors="coerce",
                )

            rows.append(
                out
            )

    if not rows:
        return pl.DataFrame()

    out = (
        pd.DataFrame(
            rows
        )
        .drop_duplicates(
            subset=[
                "game_id",
                "team",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return pl.from_pandas(
        out
    )


def historical_team_strength(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    if (
        schedule.is_empty()
        or pbp.is_empty()
    ):
        return pl.DataFrame()

    rating_schedule = (
        normalize_loader_schedule_for_ratings(
            schedule
        )
    )

    if rating_schedule.is_empty():
        return pl.DataFrame()

    game_rates = (
        nhl.team_game_xg_rates(
            pbp,
            rating_schedule,
        )
    )

    if game_rates.is_empty():
        return pl.DataFrame()

    games = (
        schedule_games_with_cutoffs(
            schedule
        )
    )

    if games.empty:
        return pl.DataFrame()

    outputs: list[
        pl.DataFrame
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        d_str = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            game_day = date.fromisoformat(
                d_str
            )
        except ValueError:
            continue

        cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        if (
            not game_id
            or cutoff is None
        ):
            continue

        # Historical game rates are date-granular in the SDV helper.
        # Excluding the entire target date is intentionally conservative:
        # it guarantees the target game and all later games cannot enter
        # the reconstructed pregame rating snapshot.
        prior_rates = (
            game_rates.filter(
                pl.col("date")
                < pl.lit(
                    game_day
                )
            )
        )

        if prior_rates.is_empty():
            continue

        ratings = (
            ratings_from_game_rates(
                prior_rates
            )
        )

        if ratings.is_empty():
            continue

        target_teams = [
            str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
        ]

        target_teams = [
            team
            for team in target_teams
            if team
        ]

        if target_teams:
            ratings = ratings.filter(
                pl.col("team")
                .cast(pl.Utf8)
                .is_in(
                    target_teams
                )
            )

        if ratings.is_empty():
            continue

        ratings_as_of = (
            strict_historical_ratings_as_of_utc(
                cutoff
            )
        )

        outputs.append(
            ratings.with_columns(
                pl.lit(
                    game_id
                ).alias(
                    "game_id"
                ),
                pl.lit(
                    d_str
                ).alias(
                    "game_date"
                ),
                pl.lit(
                    utc_iso(
                        cutoff
                    )
                ).alias(
                    "pregame_cutoff_utc"
                ),
                pl.lit(
                    utc_iso(
                        ratings_as_of
                    )
                ).alias(
                    "ratings_as_of_utc"
                ),
                pl.lit(
                    str(
                        game.get(
                            "pregame_cutoff_source",
                            "",
                        )
                    ).strip()
                ).alias(
                    "pregame_cutoff_source"
                ),
            )
        )

    if not outputs:
        return pl.DataFrame()

    combined = pl.concat(
        outputs,
        how="diagonal_relaxed",
    )

    columns = [
        "game_id",
        "game_date",
        "team",
        "pregame_cutoff_utc",
        "ratings_as_of_utc",
        "pregame_cutoff_source",
        *TEAM_STRENGTH_FIELDS,
    ]

    if (
        "season"
        in combined.columns
    ):
        columns.insert(
            0,
            "season",
        )

    return combined.select(
        columns
    )

def prediction_games(
    schedule: Any,
) -> pl.DataFrame:
    if frame_empty(schedule):
        return pl.DataFrame()

    pdf = as_pandas(schedule)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    date_col = first_col(
        pdf,
        (
            "schedule_date",
            "game_date",
            "date",
        ),
    )

    if (
        gid_col is None
        or home_col is None
        or away_col is None
    ):
        return pl.DataFrame()

    neutral_col = first_col(
        pdf,
        (
            "neutral_site",
            "neutral",
        ),
    )

    out = pd.DataFrame(
        {
            "game_id": (
                pdf[gid_col]
                .astype(str)
            ),
            "home_team": (
                pdf[home_col]
                .astype(str)
            ),
            "away_team": (
                pdf[away_col]
                .astype(str)
            ),
            "neutral_site": (
                pdf[
                    neutral_col
                ]
                .fillna(False)
                .astype(bool)
                if neutral_col
                else False
            ),
        }
    )

    if date_col:
        out[
            "game_date"
        ] = pd.to_datetime(
            pdf[date_col],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    return pl.from_pandas(out)


def build_fatigue_from_team_schedule(
    team: str,
    team_schedule: Any,
    *,
    target_date: date | None = None,
) -> pl.DataFrame:
    if frame_empty(
        team_schedule
    ):
        return pl.DataFrame()

    pdf = as_pandas(
        team_schedule
    )

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    if date_col is None:
        return pl.DataFrame()

    dates = sorted(
        {
            x.date()
            for x in pd.to_datetime(
                pdf[date_col],
                errors="coerce",
            ).dropna()
        }
    )

    rows: list[
        dict[str, Any]
    ] = []

    for idx, game_day in enumerate(
        dates
    ):
        prior = dates[:idx]

        previous = (
            prior[-1]
            if prior
            else None
        )

        days_rest = (
            (
                game_day
                - previous
            ).days
            if previous
            else None
        )

        def count_inclusive(
            window_days: int,
        ) -> int:
            return 1 + sum(
                1
                for d in prior
                if (
                    0
                    < (
                        game_day
                        - d
                    ).days
                    < window_days
                )
            )

        games_2 = count_inclusive(
            2
        )

        games_4 = count_inclusive(
            4
        )

        games_6 = count_inclusive(
            6
        )

        games_7 = count_inclusive(
            7
        )

        rows.append(
            {
                "team": team,
                "game_date": (
                    game_day.isoformat()
                ),
                "previous_game_date": (
                    previous.isoformat()
                    if previous
                    else None
                ),
                "days_rest": (
                    days_rest
                ),
                "back_to_back": (
                    games_2 >= 2
                ),
                "games_in_4_days": (
                    games_4
                ),
                "three_in_four": (
                    games_4 >= 3
                ),
                "games_in_6_days": (
                    games_6
                ),
                "four_in_six": (
                    games_6 >= 4
                ),
                "games_in_7_days": (
                    games_7
                ),
            }
        )

    out = pl.DataFrame(rows)

    if (
        target_date is not None
        and not out.is_empty()
    ):
        out = out.filter(
            pl.col(
                "game_date"
            )
            == target_date.isoformat()
        )

    return out


def build_fatigue_from_league_schedule(
    schedule: Any,
) -> pl.DataFrame:
    if frame_empty(schedule):
        return pl.DataFrame()

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        date_col is None
        or home_col is None
        or away_col is None
    ):
        return pl.DataFrame()

    long_rows = []

    for _, row in pdf.iterrows():
        d = pd.to_datetime(
            row.get(
                date_col
            ),
            errors="coerce",
        )

        if pd.isna(d):
            continue

        long_rows.append(
            (
                str(
                    row.get(
                        home_col,
                        "",
                    )
                ).strip(),
                d.date(),
            )
        )

        long_rows.append(
            (
                str(
                    row.get(
                        away_col,
                        "",
                    )
                ).strip(),
                d.date(),
            )
        )

    by_team: dict[
        str,
        list[date],
    ] = {}

    for team, d in long_rows:
        if team:
            by_team.setdefault(
                team,
                [],
            ).append(d)

    frames = []

    for team, dates in by_team.items():
        fake = pd.DataFrame(
            {
                "game_date": sorted(
                    set(dates)
                )
            }
        )

        frames.append(
            build_fatigue_from_team_schedule(
                team,
                fake,
            )
        )

    return (
        pl.concat(
            frames,
            how="diagonal_relaxed",
        )
        if frames
        else pl.DataFrame()
    )


def historical_predictions(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    if (
        schedule.is_empty()
        or pbp.is_empty()
    ):
        return pl.DataFrame()

    rating_schedule = (
        normalize_loader_schedule_for_ratings(
            schedule
        )
    )

    if rating_schedule.is_empty():
        return pl.DataFrame()

    game_rates = (
        nhl.team_game_xg_rates(
            pbp,
            rating_schedule,
        )
    )

    if game_rates.is_empty():
        return pl.DataFrame()

    games = prediction_games(
        schedule
    )

    if (
        games.is_empty()
        or "game_date"
        not in games.columns
    ):
        return pl.DataFrame()

    dates = sorted(
        {
            d
            for d in games[
                "game_date"
            ].to_list()
            if (
                isinstance(
                    d,
                    str,
                )
                and d
            )
        }
    )

    outputs = []

    for d_str in dates:
        try:
            d = date.fromisoformat(
                d_str
            )
        except ValueError:
            continue

        prior_rates = (
            game_rates.filter(
                pl.col("date")
                < pl.lit(d)
            )
        )

        ratings = (
            ratings_from_game_rates(
                prior_rates
            )
        )

        if ratings.is_empty():
            continue

        games_day = (
            games.filter(
                pl.col(
                    "game_date"
                )
                == d_str
            )
            .select(
                "game_id",
                "home_team",
                "away_team",
                "neutral_site",
            )
        )

        if games_day.is_empty():
            continue

        pred = (
            nhl.nhl_predict_games(
                games_day,
                ratings,
            )
        )

        if not pred.is_empty():
            outputs.append(
                pred.with_columns(
                    pl.lit(
                        d_str
                    ).alias(
                        "game_date"
                    )
                )
            )

    return (
        pl.concat(
            outputs,
            how="diagonal_relaxed",
        )
        if outputs
        else pl.DataFrame()
    )


def pull_schedule(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
):
    if current:
        parsed = safe_pull(
            failures,
            "schedule",
            "nhl_web_schedule",
            nhl.nhl_web_schedule,
            date=target_date.isoformat(),
        )

        raw = (safe_pull(
            failures,
            "schedule",
            "nhl_web_schedule_raw",
            nhl.nhl_web_schedule,
            date=target_date.isoformat(),
            return_parsed=False,
        ) if OPTIONS.include_raw else None)

        if parsed is not None:
            save_object(
                "schedule",
                "nhl_schedule",
                parsed,
                prefix=prefix,
                current=True,
            )

        if raw is not None:
            save_object(
                "schedule",
                "nhl_schedule_raw",
                raw,
                prefix=prefix,
                current=True,
            )

        slate = (
            filter_exact_date(
                parsed,
                target_date,
            )
            if parsed is not None
            else pl.DataFrame()
        )

        if not frame_empty(
            slate
        ):
            save_object(
                "schedule",
                "nhl_schedule_slate",
                slate,
                prefix=prefix,
                current=True,
            )

        return parsed, slate

    parsed = safe_pull(
        failures,
        "schedule",
        f"load_nhl_schedule_{season}",
        nhl.load_nhl_schedule,
        season,
    )

    if parsed is not None:
        save_object(
            "schedule",
            "nhl_schedule",
            parsed,
            prefix=prefix,
            current=False,
        )

    return parsed, parsed



def pull_team_strength(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
    teams: list[str],
    schedule: Any,
):
    ratings = None

    if current:
        ratings = safe_pull(
            failures,
            "team-strength",
            "nhl_team_ratings",
            nhl.nhl_team_ratings,
            season,
            as_of_date=target_date,
        )

        if ratings is not None:
            save_object(
                "team-strength",
                "team_ratings",
                ratings,
                prefix=prefix,
                current=True,
            )

            ratings_asof = safe_pull(
                failures,
                "team-strength",
                "normalize_team_ratings_asof",
                normalize_team_strength_asof,
                ratings,
                schedule=schedule,
                generated_at_utc=datetime.now(
                    UTC
                ),
            )

            if ratings_asof is not None:
                if frame_empty(
                    ratings_asof
                ):
                    failures.append(
                        "team-strength/team_ratings_asof: "
                        "no current rating row satisfied the strict "
                        "ratings_as_of_utc < pregame_cutoff_utc rule"
                    )

                save_object(
                    "team-strength",
                    "team_ratings_asof",
                    ratings_asof,
                    prefix=prefix,
                    current=True,
                )

    if OPTIONS.include_diagnostics:
        standings_date = (
            target_date.isoformat()
        )

        standings = safe_pull(
            failures,
            "team-strength",
            "standings",
            nhl.nhl_standings,
            date=standings_date,
        )

        if standings is not None:
            save_object(
                "team-strength",
                "standings",
                standings,
                prefix=prefix,
                current=current,
            )

        standings_raw = (safe_pull(
            failures,
            "team-strength",
            "standings_raw",
            nhl.nhl_standings,
            date=standings_date,
            return_parsed=False,
        ) if OPTIONS.include_raw else None)

        if standings_raw is not None:
            save_object(
                "team-strength",
                "standings_raw",
                standings_raw,
                prefix=prefix,
                current=current,
            )

        for team in teams:
            parsed = safe_pull(
                failures,
                "team-strength",
                f"{team}_club_stats",
                nhl.nhl_club_stats,
                team=team,
                season=(
                    None
                    if current
                    else season
                ),
            )

            if parsed is not None:
                save_object(
                    "team-strength",
                    f"{team}_club_stats",
                    parsed,
                    prefix=prefix,
                    current=current,
                )

            raw = (safe_pull(
                failures,
                "team-strength",
                f"{team}_club_stats_raw",
                nhl.nhl_club_stats,
                team=team,
                season=(
                    None
                    if current
                    else season
                ),
                return_parsed=False,
            ) if OPTIONS.include_raw else None)

            if raw is not None:
                save_object(
                    "team-strength",
                    f"{team}_club_stats_raw",
                    raw,
                    prefix=prefix,
                    current=current,
                )

    return ratings

def pull_rosters(
    failures: list[str],
    *,
    current: bool,
    season: int,
    prefix: str,
    teams: list[str],
) -> dict[str, Any]:
    rosters: dict[
        str,
        Any,
    ] = {}

    for team in teams:
        parsed = safe_pull(
            failures,
            "lineup-strength",
            f"{team}_roster",
            nhl.nhl_roster,
            team=team,
            season=(
                None
                if current
                else season
            ),
        )

        if parsed is not None:
            rosters[
                team
            ] = parsed

            save_object(
                "lineup-strength",
                f"{team}_roster",
                parsed,
                prefix=prefix,
                current=current,
            )

        raw = (safe_pull(
            failures,
            "lineup-strength",
            f"{team}_roster_raw",
            nhl.nhl_roster,
            team=team,
            season=(
                None
                if current
                else season
            ),
            return_parsed=False,
        ) if OPTIONS.include_raw else None)

        if raw is not None:
            save_object(
                "lineup-strength",
                f"{team}_roster_raw",
                raw,
                prefix=prefix,
                current=current,
            )

    return rosters


def pull_goalie_live_profiles(
    failures: list[str],
    *,
    current: bool,
    prefix: str,
    rosters: dict[str, Any],
):
    for team, roster in rosters.items():
        if frame_empty(
            roster
        ):
            continue

        pdf = as_pandas(
            roster
        )

        if (
            "position_group"
            not in pdf.columns
        ):
            continue

        goalies = pdf.loc[
            pdf[
                "position_group"
            ]
            .astype(str)
            .str.lower()
            .eq("goalies")
        ].copy()

        if goalies.empty:
            continue

        save_object(
            "goalie",
            f"{team}_goalies_roster",
            goalies,
            prefix=prefix,
            current=current,
        )

        id_col = first_col(
            goalies,
            (
                "id",
                "player_id",
                "playerId",
            ),
        )

        if id_col is None:
            continue

        for value in (
            goalies[
                id_col
            ]
            .dropna()
            .unique()
            .tolist()
        ):
            try:
                player_id = int(
                    value
                )
            except (TypeError, ValueError, OverflowError):
                continue

            parsed = safe_pull(
                failures,
                "goalie",
                f"{team}_{player_id}_player_landing",
                nhl.nhl_player_landing,
                player_id=player_id,
            )

            if parsed is not None:
                save_object(
                    "goalie",
                    f"{team}_{player_id}_player_landing",
                    parsed,
                    prefix=prefix,
                    current=current,
                )

            raw = (safe_pull(
                failures,
                "goalie",
                f"{team}_{player_id}_player_landing_raw",
                nhl.nhl_player_landing,
                player_id=player_id,
                return_parsed=False,
            ) if OPTIONS.include_raw else None)

            if raw is not None:
                save_object(
                    "goalie",
                    f"{team}_{player_id}_player_landing_raw",
                    raw,
                    prefix=prefix,
                    current=current,
                )


def pull_season_context(
    failures: list[str],
    *,
    season: int,
    current: bool,
    need_shifts: bool = True,
):
    # Only PBP and shifts feed the production goalie/lineup calculations.
    # Other full-season datasets are retained behind diagnostic opt-in.
    context = {
        "pbp": cached_season_pull(
            failures, "pbp", nhl.load_nhl_pbp_full, season, current=current,
        ),
        "shifts": (
            cached_season_pull(
                failures, "shifts", nhl.load_nhl_shifts, season, current=current,
            ) if need_shifts else None
        ),
    }
    if OPTIONS.include_diagnostics:
        for label, function, key, category in (
            ("goalie_box", nhl.load_nhl_goalie_boxscores, "goalie_box", "goalie"),
            ("skater_box", nhl.load_nhl_skater_boxscores, "skater_box", "lineup-strength"),
            ("rosters", nhl.load_nhl_rosters, "rosters", "lineup-strength"),
        ):
            context[key] = cached_season_pull(
                failures, label, function, season, current=current,
            )
    return context



def save_season_context(
    *,
    current: bool,
    prefix: str,
    context: dict[str, Any],
):
    mapping = (
        (
            "goalie",
            "goalie_boxscores",
            context.get(
                "goalie_box"
            ),
        ),
        (
            "lineup-strength",
            "skater_boxscores",
            context.get(
                "skater_box"
            ),
        ),
        (
            "lineup-strength",
            "season_rosters",
            context.get(
                "rosters"
            ),
        ),
    )

    for (
        category,
        label,
        obj,
    ) in mapping:
        if obj is not None:
            save_object(
                category,
                label,
                obj,
                prefix=prefix,
                current=current,
            )



def normalize_player_id(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return ""

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass

    return text


def prior_game_ids_for_target(
    schedule: Any,
    target_day: date,
) -> set[str]:
    if frame_empty(schedule):
        return set()

    pdf = as_pandas(schedule)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )
    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )

    if gid_col is None or date_col is None:
        return set()

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    )

    return {
        str(game_id).strip()
        for game_id, game_date in zip(
            pdf[gid_col],
            parsed,
        )
        if (
            str(game_id).strip()
            and not pd.isna(game_date)
            and game_date.date() < target_day
        )
    }


def frame_strictly_before_game(
    frame: Any,
    *,
    target_day: date,
    prior_game_ids: set[str],
) -> pl.DataFrame:
    if frame_empty(frame):
        return pl.DataFrame()

    pdf = as_pandas(frame)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    if gid_col is not None and prior_game_ids:
        ids = (
            pdf[gid_col]
            .astype(str)
            .str.strip()
        )
        return pl.from_pandas(
            pdf.loc[
                ids.isin(prior_game_ids)
            ].reset_index(drop=True)
        )

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
            "game_datetime",
            "date_time",
        ),
    )

    if date_col is None:
        return pl.DataFrame()

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
        utc=False,
    )

    mask = parsed.notna() & (
        parsed.dt.date < target_day
    )

    return pl.from_pandas(
        pdf.loc[
            mask
        ].reset_index(drop=True)
    )


def research_pbp_strictly_before_target(
    schedule: Any,
    pbp: Any,
    *,
    target_day: date,
) -> pl.DataFrame:
    """Return leakage-safe PBP for research-only pregame microstats.

    The helper deliberately uses the same prior-game boundary as the production
    lineup/goalie feature builders.  Expected-assist and zone-transition
    research must call their SportsDataverse functions only on this filtered
    frame; the target date and every later game are excluded.
    """
    prior_game_ids = prior_game_ids_for_target(
        schedule,
        target_day,
    )

    return frame_strictly_before_game(
        pbp,
        target_day=target_day,
        prior_game_ids=prior_game_ids,
    )


def goalie_gsax_lookup(
    gsax: Any,
) -> dict[str, dict[str, Any]]:
    if frame_empty(gsax):
        return {}

    pdf = as_pandas(gsax)

    required = {
        "player_id",
        "goalie",
        "gsax",
    }

    if not required.issubset(
        set(pdf.columns)
    ):
        raise ValueError(
            "SportsDataverse goalie GSAx output missing "
            f"required columns: {sorted(required - set(pdf.columns))}"
        )

    out: dict[
        str,
        dict[str, Any],
    ] = {}

    for _, row in pdf.iterrows():
        player_id = normalize_player_id(
            row.get(
                "player_id"
            )
        )

        if not player_id:
            continue

        out[player_id] = {
            "player_id": player_id,
            "goalie": str(
                row.get(
                    "goalie",
                    "",
                )
            ).strip(),
            "gsax": pd.to_numeric(
                row.get(
                    "gsax"
                ),
                errors="coerce",
            ),
            "gsax_per_60": pd.to_numeric(
                row.get(
                    "gsax_per_60"
                ),
                errors="coerce",
            ),
        }

    return out


def goalie_usage_candidates(
    prior_pbp: Any,
    team: str,
) -> list[dict[str, Any]]:
    if frame_empty(prior_pbp):
        return []

    pdf = as_pandas(
        prior_pbp
    )

    game_id_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    records: list[
        dict[str, str]
    ] = []

    for side in (
        "home",
        "away",
    ):
        team_col = first_col(
            pdf,
            (
                f"{side}_abbr",
                f"{side}_team_abbr",
                f"{side}_team_abbrev",
                f"{side}_team",
            ),
        )
        goalie_id_col = first_col(
            pdf,
            (
                f"{side}_goalie_id",
                f"{side}_goalie_player_id",
            ),
        )
        goalie_name_col = first_col(
            pdf,
            (
                f"{side}_goalie",
                f"{side}_goalie_name",
            ),
        )

        if (
            team_col is None
            or goalie_id_col is None
        ):
            continue

        side_rows = pdf.loc[
            pdf[team_col]
            .astype(str)
            .str.strip()
            .eq(str(team).strip())
        ]

        for row_number, row in side_rows.iterrows():
            player_id = normalize_player_id(
                row.get(
                    goalie_id_col
                )
            )

            if not player_id:
                continue

            game_id = (
                str(
                    row.get(
                        game_id_col,
                        "",
                    )
                ).strip()
                if game_id_col is not None
                else str(row_number)
            )

            records.append(
                {
                    "player_id": player_id,
                    "goalie": (
                        str(
                            row.get(
                                goalie_name_col,
                                "",
                            )
                        ).strip()
                        if goalie_name_col is not None
                        else ""
                    ),
                    "game_id": game_id,
                }
            )

    if not records:
        return []

    usage = (
        pd.DataFrame(records)
        .drop_duplicates(
            subset=[
                "player_id",
                "game_id",
            ]
        )
    )

    names = (
        usage.groupby(
            "player_id",
            dropna=False,
        )["goalie"]
        .agg(
            lambda values: next(
                (
                    str(value).strip()
                    for value in values
                    if str(value).strip()
                ),
                "",
            )
        )
    )

    counts = (
        usage.groupby(
            "player_id",
            dropna=False,
        )["game_id"]
        .nunique()
        .rename(
            "prior_appearances"
        )
    )

    combined = (
        pd.concat(
            [
                names,
                counts,
            ],
            axis=1,
        )
        .reset_index()
    )

    return sorted(
        combined.to_dict(
            orient="records"
        ),
        key=lambda candidate_row: (
            -int(
                candidate_row.get(
                    "prior_appearances",
                    0,
                )
            ),
            str(
                candidate_row.get(
                    "goalie",
                    "",
                )
            ),
            str(
                candidate_row.get(
                    "player_id",
                    "",
                )
            ),
        ),
    )


def goalie_value(
    gsax_lookup: dict[
        str,
        dict[str, Any],
    ],
    candidate: dict[str, Any] | None,
) -> Any:
    if not candidate:
        return None

    player_id = normalize_player_id(
        candidate.get(
            "player_id"
        )
    )

    row = gsax_lookup.get(
        player_id
    )

    if row is None:
        return None

    value = row.get(
        "gsax"
    )

    if pd.isna(value):
        return None

    return float(value)


def goalie_name(
    gsax_lookup: dict[
        str,
        dict[str, Any],
    ],
    candidate: dict[str, Any] | None,
) -> str:
    if not candidate:
        return ""

    name = str(
        candidate.get(
            "goalie",
            "",
        )
    ).strip()

    if name:
        return name

    player_id = normalize_player_id(
        candidate.get(
            "player_id"
        )
    )

    return str(
        gsax_lookup.get(
            player_id,
            {},
        ).get(
            "goalie",
            "",
        )
    ).strip()


def build_game_goalie_features_asof(
    schedule: Any,
    pbp: Any,
    shifts: Any,
    *,
    current: bool,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if (
        frame_empty(schedule)
        or frame_empty(pbp)
    ):
        return pl.DataFrame()

    games = schedule_games_with_cutoffs(
        schedule
    )

    if games.empty:
        return pl.DataFrame()

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    cache: dict[
        str,
        tuple[
            pl.DataFrame,
            dict[str, dict[str, Any]],
        ],
    ] = {}

    rows: list[
        dict[str, Any]
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        game_date = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            target_day = date.fromisoformat(
                game_date
            )
        except ValueError:
            continue

        game_start_cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        cutoff_source = str(
            game.get(
                "pregame_cutoff_source",
                "",
            )
        ).strip()

        if (
            not game_id
            or game_start_cutoff is None
            or cutoff_source
            == "conservative_game_date_start_et"
        ):
            continue

        production_cutoff = (
            goalie_production_cutoff_utc(
                game_start_cutoff
            )
        )

        snapshot_as_of = (
            generated_at_utc
            if current
            else production_cutoff
        )

        # Production goalie information is frozen at T-60.
        # Current runs after T-60 cannot backfill later information
        # into the game-level goalie feature snapshot.
        if snapshot_as_of > production_cutoff:
            continue

        cache_key = target_day.isoformat()

        if cache_key not in cache:
            prior_game_ids = (
                prior_game_ids_for_target(
                    schedule,
                    target_day,
                )
            )

            prior_pbp = (
                frame_strictly_before_game(
                    pbp,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
            )

            prior_shifts = (
                frame_strictly_before_game(
                    shifts,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    shifts
                )
                else pl.DataFrame()
            )

            gsax = (
                nhl.nhl_goalie_gsax(
                    prior_pbp,
                    prior_shifts,
                )
                if not prior_pbp.is_empty()
                else pl.DataFrame()
            )

            cache[
                cache_key
            ] = (
                prior_pbp,
                goalie_gsax_lookup(
                    gsax
                ),
            )

        (
            prior_pbp,
            gsax_lookup,
        ) = cache[
            cache_key
        ]

        row: dict[
            str,
            Any
        ] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team": str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            "away_team": str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
            "pregame_cutoff_utc": utc_iso(
                game_start_cutoff
            ),
            "goalie_decision_cutoff_utc": utc_iso(
                production_cutoff
            ),
            "goalie_decision_cutoff_source": (
                GOALIE_PRODUCTION_CUTOFF_SOURCE
            ),
            "goalie_snapshot_as_of_utc": utc_iso(
                snapshot_as_of
            ),
            "pregame_cutoff_source": cutoff_source,
        }

        for side in (
            "home",
            "away",
        ):
            team = row[
                f"{side}_team"
            ]

            candidates = (
                goalie_usage_candidates(
                    prior_pbp,
                    team,
                )
            )

            starter = (
                candidates[0]
                if candidates
                else None
            )

            backup = (
                candidates[1]
                if len(candidates) > 1
                else None
            )

            status = (
                "projected"
                if starter is not None
                else "unknown"
            )

            row[
                f"{side}_expected_starter"
            ] = goalie_name(
                gsax_lookup,
                starter,
            )

            row[
                f"{side}_starter_gsax"
            ] = goalie_value(
                gsax_lookup,
                starter,
            )

            row[
                f"{side}_backup_gsax"
            ] = goalie_value(
                gsax_lookup,
                backup,
            )

            row[
                f"{side}_goalie_status"
            ] = status

            row[
                f"{side}_goalie_status_observed_at"
            ] = utc_iso(
                snapshot_as_of
            )

            row[
                f"{side}_goalie_status_source"
            ] = (
                "sportsdataverse_prior_goalie_usage_projection"
                if status == "projected"
                else "sportsdataverse_no_pregame_starter_evidence"
            )

        home_gsax = row.get(
            "home_starter_gsax"
        )

        away_gsax = row.get(
            "away_starter_gsax"
        )

        if (
            home_gsax is None
            or away_gsax is None
        ):
            row[
                "starter_gsax_differential"
            ] = None
        else:
            row[
                "starter_gsax_differential"
            ] = (
                float(home_gsax)
                - float(away_gsax)
            )

        rows.append(
            row
        )

    if not rows:
        return pl.DataFrame()

    return pl.from_pandas(
        pd.DataFrame(
            rows
        )
    )




def _shift_player_team_weights(
    prior_shifts: Any,
) -> dict[str, dict[str, float]]:
    """Allocate player value by observed historical shift changes per team.

    The SDV shift source contains event_team and comma-space-separated ids_on.
    Allocation by shift changes preserves the full player WAR total even when
    the player appeared for multiple teams (e.g. traded during the season).
    It must not infer membership from counts of 2-/3-person combinations.
    """
    from sportsdataverse.nhl.nhl_player_impact_constants import team_fullname_to_abbr

    if frame_empty(prior_shifts):
        raise RuntimeError("Cannot attribute player strength: empty shift source")
    required = {"event_team", "ids_on"}
    pdf = as_pandas(prior_shifts)
    if not required.issubset(pdf.columns):
        raise RuntimeError(
            f"Cannot attribute players: shifts missing {sorted(required - set(pdf.columns))}"
        )

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unknown_teams: set[str] = set()
    for team_full, id_text in zip(pdf["event_team"], pdf["ids_on"]):
        if pd.isna(id_text) or not str(id_text).strip():
            continue
        team_full = str(team_full).strip()
        team = team_fullname_to_abbr(team_full)
        if team is None:
            unknown_teams.add(team_full)
            continue
        # Use the same comma-space ID convention as SDV's own build_stints.
        for raw_id in str(id_text).split(", "):
            player_id = normalize_player_id(raw_id)
            if player_id and player_id != "0":
                counts[player_id][team] += 1

    if unknown_teams:
        raise RuntimeError(
            "Unknown historical shift team labels: "
            + repr(sorted(unknown_teams)[:12])
        )
    if not counts:
        raise RuntimeError("No player-team identities decoded from historical shifts")
    return {
        player_id: {
            team: count / sum(by_team.values())
            for team, count in sorted(by_team.items())
        }
        for player_id, by_team in counts.items()
    }


_WAR_CONFIG_LOCK = threading.RLock()


def corrected_nhl_skater_war(prior_pbp: Any, prior_shifts: Any) -> Any:
    """Correct the SDV WAR units mismatch without modifying installed packages.

    Upstream NHL RAPM coefficients express incremental xG/60 *above league
    average*. The released WAR formula subtracts absolute replacement xG/60
    (1.9 off, 2.6 xGA), which makes all teams strongly negative. Translate
    these rates into the centered RAPM coordinate system for just this call.
    All original source code, goals-per-win and other constants are preserved.
    """
    from sportsdataverse.nhl.nhl_player_impact_constants import LEAGUE_CONSTANTS

    with _WAR_CONFIG_LOCK:
        cfg = LEAGUE_CONSTANTS["nhl"]
        if not (
            0.0 < cfg.replacement_ev_off < cfg.league_xg_rate_ev
            < cfg.replacement_ev_def
            and cfg.goals_per_win > 0.0
        ):
            raise RuntimeError(
                "Unexpected SportsDataverse WAR constants; inspect upstream "
                "before applying centered replacement-rate correction"
            )
        corrected_off = cfg.replacement_ev_off - cfg.league_xg_rate_ev
        corrected_def = cfg.league_xg_rate_ev - cfg.replacement_ev_def
        revised_cfg = replace(
            cfg,
            replacement_ev_off=corrected_off,
            replacement_ev_def=corrected_def,
        )
        print(
            "WAR REPLACEMENT SCALE FIX | "
            f"off={cfg.replacement_ev_off}->{corrected_off:.6f} "
            f"def={cfg.replacement_ev_def}->{corrected_def:.6f}",
            flush=True,
        )
        try:
            LEAGUE_CONSTANTS["nhl"] = revised_cfg
            return nhl.nhl_skater_war(prior_pbp, prior_shifts)
        finally:
            LEAGUE_CONSTANTS["nhl"] = cfg


def _weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float | None:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    mask = (
        numeric_values.notna()
        & numeric_weights.notna()
        & numeric_weights.gt(0)
    )

    if not mask.any():
        valid = numeric_values.dropna()
        return (
            float(
                valid.mean()
            )
            if not valid.empty
            else None
        )

    denom = float(
        numeric_weights.loc[
            mask
        ].sum()
    )

    if denom <= 0:
        return None

    return float(
        (
            numeric_values.loc[
                mask
            ]
            * numeric_weights.loc[
                mask
            ]
        ).sum()
        / denom
    )


def _player_metric_by_team(
    frame: Any,
    player_team: dict[str, dict[str, float]],
    *,
    value_col: str,
    mode: str,
    weight_col: str | None = None,
) -> dict[str, float]:
    """Aggregate one per-player value using shift-based team shares.

    WAR/PP/PK sums are conserved when a player changes teams. RAPM is
    weighted by the relevant share of player TOI, not number of line units.
    """
    if frame_empty(frame):
        return {}
    pdf = as_pandas(frame)
    if "player_id" not in pdf.columns or value_col not in pdf.columns:
        return {}
    rows: list[dict[str, float | str]] = []
    missing_players: set[str] = set()
    for _, record in pdf.iterrows():
        player = normalize_player_id(record["player_id"])
        value = pd.to_numeric(record[value_col], errors="coerce")
        if not player or not pd.notna(value):
            continue
        allocation = player_team.get(player)
        if not allocation:
            missing_players.add(player)
            continue
        player_weight = (
            pd.to_numeric(record.get(weight_col), errors="coerce")
            if weight_col else None
        )
        for team, share in allocation.items():
            rows.append({
                "_team": team,
                "_value": float(value) * share if mode == "sum" else float(value),
                "_weight": (
                    float(player_weight) * share
                    if player_weight is not None and pd.notna(player_weight)
                    else float("nan")
                ),
            })

    if missing_players:
        print(
            f"WARN | {value_col}: {len(missing_players)} player IDs absent "
            f"from historical shift-team mapping; sample={sorted(missing_players)[:8]}",
            file=sys.stderr, flush=True,
        )
    if not rows:
        return {}
    work = pd.DataFrame(rows)
    output: dict[str, float] = {}
    for team, group in work.groupby("_team", dropna=False):
        if mode == "sum":
            output[str(team)] = float(group["_value"].sum())
        elif mode == "weighted_mean" and weight_col is not None:
            computed = _weighted_mean(group["_value"], group["_weight"])
            if computed is not None:
                output[str(team)] = computed
        else:
            output[str(team)] = float(group["_value"].mean())
    return output


def _unit_metric_by_team(
    frame: Any,
) -> dict[str, float]:
    if frame_empty(frame):
        return {}

    pdf = as_pandas(frame)

    required = {
        "team",
        "unit_value",
    }

    if not required.issubset(
        set(
            pdf.columns
        )
    ):
        return {}

    output: dict[
        str,
        float,
    ] = {}

    for team, group in pdf.groupby(
        "team",
        dropna=False,
    ):
        team_name = str(
            team
        ).strip()

        if not team_name:
            continue

        if "toi_minutes" in group.columns:
            value = _weighted_mean(
                group[
                    "unit_value"
                ],
                group[
                    "toi_minutes"
                ],
            )
        else:
            values = pd.to_numeric(
                group[
                    "unit_value"
                ],
                errors="coerce",
            ).dropna()

            value = (
                float(
                    values.mean()
                )
                if not values.empty
                else None
            )

        if value is not None:
            output[
                team_name
            ] = float(
                value
            )

    return output


def compute_lineup_team_metrics(
    prior_pbp: Any,
    prior_shifts: Any,
) -> dict[str, dict[str, float]]:
    if (
        frame_empty(
            prior_pbp
        )
        or frame_empty(
            prior_shifts
        )
    ):
        return {}

    rapm = nhl.nhl_skater_rapm(
        prior_pbp,
        prior_shifts,
    )

    war = clean_war_rows(
        corrected_nhl_skater_war(prior_pbp, prior_shifts)
    )

    special = nhl.nhl_special_teams_value(
        prior_pbp,
        prior_shifts,
    )

    forward_units = nhl.nhl_unit_ratings(
        prior_pbp,
        prior_shifts,
        unit_type="forward_line",
        min_toi=0.0,
    )

    defense_units = nhl.nhl_unit_ratings(
        prior_pbp,
        prior_shifts,
        unit_type="defense_pair",
        min_toi=0.0,
    )

    player_team = _shift_player_team_weights(prior_shifts)

    metric_maps = {
        "skater_rapm": (
            _player_metric_by_team(
                rapm,
                player_team,
                value_col="xg_rapm",
                mode="weighted_mean",
                weight_col="toi_minutes",
            )
        ),
        "skater_war": (
            _player_metric_by_team(
                war,
                player_team,
                value_col="war",
                mode="sum",
            )
        ),
        "pp_value": (
            _player_metric_by_team(
                special,
                player_team,
                value_col="pp_value",
                mode="sum",
            )
        ),
        "pk_value": (
            _player_metric_by_team(
                special,
                player_team,
                value_col="pk_value",
                mode="sum",
            )
        ),
        "forward_line_strength": (
            _unit_metric_by_team(
                forward_units
            )
        ),
        "defense_pair_strength": (
            _unit_metric_by_team(
                defense_units
            )
        ),
    }

    # WAR must reconcile to the player-level output. Never silently drop
    # unassigned player contributions from the published team totals.
    war_values = pd.to_numeric(as_pandas(war)["war"], errors="coerce")
    if war_values.isna().any():
        raise RuntimeError("Invalid player WAR in SportsDataverse output")
    raw_war_total = float(war_values.sum())
    team_war_total = float(sum(metric_maps["skater_war"].values()))
    if abs(raw_war_total - team_war_total) > 1e-6:
        raise RuntimeError(
            f"Player-to-team WAR reconciliation failed: "
            f"player_total={raw_war_total:.8f}, "
            f"allocated_team_total={team_war_total:.8f}, "
            f"difference={raw_war_total-team_war_total:.8f}; "
            "check unmapped historical shift player IDs"
        )
    print(
        f"WAR PLAYER-TEAM RECONCILED | player_total={raw_war_total:.8f} "
        f"team_total={team_war_total:.8f}", flush=True,
    )

    teams = sorted(
        {
            team
            for metric_map in metric_maps.values()
            for team in metric_map
        }
    )

    return {
        team: {
            metric: metric_maps[
                metric
            ].get(
                team
            )
            for metric in LINEUP_TEAM_METRICS
        }
        for team in teams
    }



@lru_cache(maxsize=1)
def load_player_strength_snapshots() -> list[dict[str, Any]]:
    """Read immutable, independently refreshed player-strength snapshots.

    Reject malformed schemas; do not silently substitute a different season,
    a future snapshot, or a snapshot containing games on/after the target date.
    """
    import csv
    import math

    folder = ROOT / "lineup-strength" / "player_strength_snapshots"
    paths = sorted(folder.glob("*_team_player_strength.csv"))
    if not paths:
        raise RuntimeError(
            f"No weekly player-strength snapshots at {folder}; "
            "run nhl_player_strength_manual workflow before daily NHL pipeline"
        )

    required = {
        "schema_version", "season", "history_exclusive_date",
        "snapshot_as_of_utc", "team", *LINEUP_TEAM_METRICS,
    }
    output: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not required.issubset(headers):
                raise RuntimeError(f"Invalid player-strength snapshot headers: {path}")
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"Empty player-strength snapshot: {path}")
        first = rows[0]
        if first.get("schema_version") != "1":
            raise RuntimeError(f"Unknown player-strength snapshot schema: {path}")
        try:
            season = int(first["season"])
            exclusive_date = date.fromisoformat(first["history_exclusive_date"])
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Invalid player-strength date/season: {path}") from exc
        asof = parse_timestamp_utc(first.get("snapshot_as_of_utc"))
        if asof is None:
            raise RuntimeError(f"Missing snapshot as-of timestamp: {path}")

        teams: dict[str, dict[str, float | None]] = {}
        for number, rec in enumerate(rows, start=2):
            if rec.get(None) is not None:
                raise RuntimeError(f"Extra CSV fields {path}:{number}")
            if (
                rec.get("schema_version") != "1"
                or rec.get("season") != str(season)
                or rec.get("history_exclusive_date") != exclusive_date.isoformat()
                or parse_timestamp_utc(rec.get("snapshot_as_of_utc")) != asof
            ):
                raise RuntimeError(f"Inconsistent snapshot metadata: {path}:{number}")
            team = str(rec.get("team") or "").strip()
            if not team or team in teams:
                raise RuntimeError(f"Missing/duplicate player team {team!r}: {path}:{number}")
            metrics: dict[str, float | None] = {}
            for metric in LINEUP_TEAM_METRICS:
                raw = str(rec.get(metric) or "").strip()
                if not raw:
                    metrics[metric] = None
                    continue
                try:
                    number_value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid {metric} for {team} in {path}:{number}"
                    ) from exc
                if not math.isfinite(number_value):
                    raise RuntimeError(f"Non-finite {metric} for {team}: {path}:{number}")
                metrics[metric] = number_value
            teams[team] = metrics
        # A complete league snapshot with negative WAR for every team is not
        # accepted as a production WAR feature. Preserve the other five
        # historical strength metrics; mark WAR unavailable without inventing
        # a replacement-level offset or changing the sign.
        war_values = [metrics.get("skater_war") for metrics in teams.values()]
        if (
            len(teams) == 32
            and len(war_values) == 32
            and all(value is not None and value < 0 for value in war_values)
        ):
            print(
                "WARN | lineup-strength/skater_war | all 32 teams have "
                "negative WAR in " + path.name + "; ignoring WAR until "
                "player-level source calculation is audited",
                file=sys.stderr,
                flush=True,
            )
            for metrics in teams.values():
                metrics["skater_war"] = None
        output.append({
            "asof": asof,
            "history_exclusive_date": exclusive_date,
            "season": season,
            "teams": teams,
            "source": path.name,
        })
    return sorted(output, key=lambda row: row["asof"], reverse=True)


def build_game_lineup_features_from_snapshots(
    schedule: Any, *, generated_at_utc: datetime,
) -> pl.DataFrame:
    """Produce original game-level lineup schema WITHOUT expensive recomputation.

    Snapshots contain historical team performance, not confirmed lineups.
    The snapshot creation timestamp must precede the game's fixed T-60 cutoff.
    Snapshot statistics must exclude the target date and subsequent dates.
    """
    if frame_empty(schedule):
        return pl.DataFrame()
    games = schedule_games_with_cutoffs(schedule)
    if games.empty:
        return pl.DataFrame()

    snapshots = load_player_strength_snapshots()
    generated_at_utc = generated_at_utc.astimezone(UTC)
    rows: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        game_id = str(game.get("game_id") or "").strip()
        game_date = str(game.get("game_date") or "").strip()
        try:
            game_day = date.fromisoformat(game_date)
        except ValueError:
            continue
        game_start = parse_timestamp_utc(game.get("pregame_cutoff_utc"))
        cutoff_source = str(game.get("pregame_cutoff_source") or "").strip()
        if (
            not game_id or game_start is None
            or cutoff_source == "conservative_game_date_start_et"
        ):
            continue
        decision_cutoff = lineup_production_cutoff_utc(game_start)
        if generated_at_utc > decision_cutoff:
            # Preserve original current-run T-60 behavior: do not backfill late.
            continue

        current_season = season_start_for_day(game_day)
        candidates = [
            snap for snap in snapshots
            if (
                snap["season"] == current_season
                and snap["asof"] <= decision_cutoff
                and snap["history_exclusive_date"] <= game_day
                and 0 <= (game_day - snap["history_exclusive_date"]).days <= 8
            )
        ]
        if not candidates:
            raise RuntimeError(
                f"No eligible <=8-day pregame player-strength snapshot "
                f"for game_id={game_id} date={game_date} "
                f"T-60={utc_iso(decision_cutoff)}; run the weekly workflow"
            )
        selected = candidates[0]
        row: dict[str, Any] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team": str(game.get("home_team") or "").strip(),
            "away_team": str(game.get("away_team") or "").strip(),
            "pregame_cutoff_utc": utc_iso(game_start),
            "lineup_decision_cutoff_utc": utc_iso(decision_cutoff),
            "lineup_decision_cutoff_source": LINEUP_PRODUCTION_CUTOFF_SOURCE,
            "lineup_snapshot_as_of_utc": utc_iso(selected["asof"]),
            "pregame_cutoff_source": cutoff_source,
        }
        for side in ("home", "away"):
            team = row[f"{side}_team"]
            if team not in selected["teams"]:
                raise RuntimeError(
                    f"Team {team!r} absent from player snapshot {selected['source']} "
                    f"for game_id={game_id}; no silent fallback permitted"
                )
            metrics = selected["teams"][team]
            for metric in LINEUP_TEAM_METRICS:
                row[f"{side}_{metric}"] = metrics[metric]
            row[f"{side}_lineup_status"] = "unknown"
            row[f"{side}_lineup_observed_at"] = ""
            row[f"{side}_lineup_source"] = (
                "sportsdataverse_cached_historical_player_strength_no_lineup_confirmation"
            )

        for metric in LINEUP_TEAM_METRICS:
            h = row[f"home_{metric}"]
            a = row[f"away_{metric}"]
            row[f"{metric}_differential"] = (
                float(h) - float(a)
                if h is not None and a is not None
                else None
            )
        rows.append(row)
    return pl.from_pandas(pd.DataFrame(rows)) if rows else pl.DataFrame()


def build_game_lineup_features_asof(
    schedule: Any,
    pbp: Any,
    shifts: Any,
    *,
    current: bool,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if frame_empty(
        schedule
    ):
        return pl.DataFrame()

    games = schedule_games_with_cutoffs(
        schedule
    )

    if games.empty:
        return pl.DataFrame()

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    cache: dict[
        str,
        dict[
            str,
            dict[str, float],
        ],
    ] = {}

    rows: list[
        dict[str, Any]
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        game_date = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            target_day = date.fromisoformat(
                game_date
            )
        except ValueError:
            continue

        game_start = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        cutoff_source = str(
            game.get(
                "pregame_cutoff_source",
                "",
            )
        ).strip()

        if (
            not game_id
            or game_start is None
            or cutoff_source
            == "conservative_game_date_start_et"
        ):
            continue

        decision_cutoff = (
            lineup_production_cutoff_utc(
                game_start
            )
        )

        snapshot_as_of = (
            generated_at_utc
            if current
            else decision_cutoff
        )

        # Production lineup/player information is frozen at T-60.
        # Later current runs cannot backfill information into the feature row.
        if snapshot_as_of > decision_cutoff:
            continue

        cache_key = target_day.isoformat()

        if cache_key not in cache:
            prior_game_ids = (
                prior_game_ids_for_target(
                    schedule,
                    target_day,
                )
            )

            prior_pbp = (
                frame_strictly_before_game(
                    pbp,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    pbp
                )
                else pl.DataFrame()
            )

            prior_shifts = (
                frame_strictly_before_game(
                    shifts,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    shifts
                )
                else pl.DataFrame()
            )

            cache[
                cache_key
            ] = (
                compute_lineup_team_metrics(
                    prior_pbp,
                    prior_shifts,
                )
            )

        team_metrics = cache[
            cache_key
        ]

        row: dict[
            str,
            Any,
        ] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team": str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            "away_team": str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
            "pregame_cutoff_utc": utc_iso(
                game_start
            ),
            "lineup_decision_cutoff_utc": utc_iso(
                decision_cutoff
            ),
            "lineup_decision_cutoff_source": (
                LINEUP_PRODUCTION_CUTOFF_SOURCE
            ),
            "lineup_snapshot_as_of_utc": utc_iso(
                snapshot_as_of
            ),
            "pregame_cutoff_source": cutoff_source,
        }

        for side in (
            "home",
            "away",
        ):
            team = row[
                f"{side}_team"
            ]

            metrics = team_metrics.get(
                team,
                {},
            )

            for metric in (
                LINEUP_TEAM_METRICS
            ):
                row[
                    f"{side}_{metric}"
                ] = metrics.get(
                    metric
                )

            # SportsDataverse provides the historical performance inputs,
            # not a timestamped NHL pregame lineup-confirmation feed.
            # Therefore actual lineup state remains unknown rather than
            # being inferred from postgame participation.
            row[
                f"{side}_lineup_status"
            ] = "unknown"

            row[
                f"{side}_lineup_observed_at"
            ] = ""

            row[
                f"{side}_lineup_source"
            ] = (
                "sportsdataverse_prior_game_player_pool_no_lineup_confirmation"
                if metrics
                else "sportsdataverse_no_pregame_lineup_evidence"
            )

        for metric in (
            LINEUP_TEAM_METRICS
        ):
            home_value = row.get(
                f"home_{metric}"
            )
            away_value = row.get(
                f"away_{metric}"
            )

            if (
                home_value is None
                or away_value is None
                or pd.isna(
                    home_value
                )
                or pd.isna(
                    away_value
                )
            ):
                row[
                    f"{metric}_differential"
                ] = None
            else:
                row[
                    f"{metric}_differential"
                ] = (
                    float(
                        home_value
                    )
                    - float(
                        away_value
                    )
                )

        rows.append(
            row
        )

    if not rows:
        return pl.DataFrame()

    return pl.from_pandas(
        pd.DataFrame(
            rows
        )
    )



def pull_advanced_player_strength(
    failures: list[str],
    *,
    current: bool,
    prefix: str,
    context: dict[str, Any],
):
    """Legacy goalie diagnostic only.

    Production player/lineup features are generated per game by
    build_game_lineup_features_asof under the fixed T-60 contract.
    """
    pbp = context.get(
        "pbp"
    )

    shifts = context.get(
        "shifts"
    )

    if (
        frame_empty(pbp)
        or frame_empty(shifts)
    ):
        return

    gsax = safe_pull(
        failures,
        "goalie",
        "goalie_gsax",
        nhl.nhl_goalie_gsax,
        pbp,
        shifts,
    )

    if gsax is not None:
        save_object(
            "goalie",
            "goalie_gsax",
            gsax,
            prefix=prefix,
            current=current,
        )


def _fetch_valid_fatigue_schedule(
    team: str,
    target_date: date,
) -> tuple[Any, dict[str, Any], pl.DataFrame]:
    """Fetch one NHL payload, parse THAT payload, and validate the slate game.

    No independently fetched parsed/raw pair can disagree because parsing
    and fatigue calculation both use the same verified raw response.
    Empty/malformed responses and exceptions are retried with bounded backoff.
    This function never writes to the repository.
    """
    target = target_date.isoformat()
    attempt_errors: list[str] = []

    for attempt in range(1, FATIGUE_SCHEDULE_MAX_ATTEMPTS + 1):
        started_at = time.perf_counter()
        stage = "fetch"
        try:
            raw = nhl.nhl_club_schedule_season(
                team=team,
                return_parsed=False,
            )

            stage = "validate_raw"
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Expected a raw JSON object, got {type(raw).__name__}"
                )
            games = raw.get("games")
            if not isinstance(games, list) or not games:
                raise ValueError(
                    "Raw schedule has no games array or it is empty "
                    "(the HTTP response status is not available here)"
                )

            target_games = [
                game for game in games
                if isinstance(game, dict)
                and game.get("gameDate") == target
                and team in (
                    (game.get("homeTeam") or {}).get("abbrev"),
                    (game.get("awayTeam") or {}).get("abbrev"),
                )
            ]
            if len(target_games) != 1:
                raise ValueError(
                    f"Expected one {team} game on {target} in raw "
                    f"schedule, found {len(target_games)}"
                )

            stage = "parse_raw"
            # Public SportsDataverse parser; do not make a second request.
            parsed = nhl.parse_nhl_web_club_schedule(raw)
            if not (is_polars_frame(parsed) or is_pandas_frame(parsed)):
                raise TypeError(
                    f"Parser returned {type(parsed).__name__}, not a DataFrame"
                )
            if frame_empty(parsed):
                raise ValueError(
                    "Parser returned no rows for a populated raw schedule"
                )
            parsed_pdf = as_pandas(parsed)
            if len(parsed_pdf) != len(games):
                raise ValueError(
                    f"Parser row count {len(parsed_pdf)} differs from "
                    f"raw games count {len(games)}"
                )
            date_col = first_col(
                parsed, ("game_date", "schedule_date", "date")
            )
            if date_col is None:
                raise ValueError("Parsed club schedule is missing a date column")
            game_dates = pd.to_datetime(
                parsed_pdf[date_col], errors="coerce"
            )
            if int((game_dates.dt.date == target_date).sum()) != 1:
                raise ValueError(
                    f"Parsed schedule does not contain exactly one "
                    f"game for {team} on {target}"
                )
            if "id" in parsed_pdf.columns:
                expected_id = str(target_games[0].get("id", ""))
                parsed_id = str(
                    parsed_pdf.loc[game_dates.dt.date == target_date, "id"].iloc[0]
                )
                if parsed_id != expected_id:
                    raise ValueError(
                        f"Parsed game ID {parsed_id!r} does not match "
                        f"raw game ID {expected_id!r}"
                    )

            stage = "calculate_fatigue"
            fatigue = build_fatigue_from_team_schedule(
                team, parsed, target_date=target_date
            )
            if frame_empty(fatigue) or fatigue.height != 1:
                raise ValueError(
                    f"Expected exactly one fatigue record for {team} "
                    f"on {target}, got "
                    f"{0 if frame_empty(fatigue) else fatigue.height}"
                )
            feature = fatigue.to_dicts()[0]
            if feature.get("team") != team or feature.get("game_date") != target:
                raise ValueError(
                    f"Fatigue record does not match {team} on {target}"
                )

            record_timing(
                f"request/fatigue/{team}_club_schedule_attempt_{attempt}",
                time.perf_counter() - started_at,
                "ok",
            )
            if attempt > 1:
                print(
                    f"SDV FATIGUE RECOVERED | {team} | "
                    f"date={target} | attempt={attempt}",
                    flush=True,
                )
            return parsed, raw, fatigue
        except Exception as exc:
            detail = (
                f"attempt={attempt}/{FATIGUE_SCHEDULE_MAX_ATTEMPTS} "
                f"stage={stage} {type(exc).__name__}: {exc}"
            )
            attempt_errors.append(detail)
            record_timing(
                f"request/fatigue/{team}_club_schedule_attempt_{attempt}",
                time.perf_counter() - started_at,
                "error",
            )
            print(
                f"WARN | fatigue/{team}_club_schedule | {detail}",
                file=sys.stderr, flush=True,
            )
            if attempt < FATIGUE_SCHEDULE_MAX_ATTEMPTS:
                time.sleep(
                    FATIGUE_SCHEDULE_RETRY_SECONDS * 2 ** (attempt - 1)
                )

    raise RuntimeError(
        f"fatigue/{team}_club_schedule: no valid schedule for "
        f"{target} after {FATIGUE_SCHEDULE_MAX_ATTEMPTS} "
        f"attempts; " + " | ".join(attempt_errors)
    )


def pull_fatigue(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    prefix: str,
    teams: list[str],
    schedule: Any,
):
    if not current:
        fatigue = build_fatigue_from_league_schedule(schedule)
        save_object(
            "fatigue", "fatigue", fatigue,
            prefix=prefix, current=False,
        )
        return

    if frame_empty(schedule):
        raise RuntimeError(
            "Cannot validate current fatigue coverage: "
            "NHL schedule is unavailable or empty"
        )
    if not teams:
        print(
            f"SDV FATIGUE | No scheduled teams for "
            f"{target_date.isoformat()}; no fatigue snapshot required",
            flush=True,
        )
        return

    if len(teams) != len(set(teams)):
        raise ValueError("Duplicate teams in the current NHL slate")

    results: dict[str, tuple[Any, dict[str, Any], pl.DataFrame]] = {}
    problems: list[str] = []

    # Keep existing concurrency bound. Every team gets one raw fetch
    # per attempt, then that SAME payload is parsed and validated.
    with ThreadPoolExecutor(max_workers=OPTIONS.max_workers) as pool:
        futures = {
            pool.submit(_fetch_valid_fatigue_schedule, team, target_date): team
            for team in teams
        }
        for future in as_completed(futures):
            team = futures[future]
            try:
                results[team] = future.result()
            except Exception as exc:
                problems.append(f"{team}: {exc}")

    if problems or len(results) != len(teams):
        missing = sorted(set(teams) - set(results))
        problem = (
            f"Required NHL fatigue schedules incomplete for "
            f"{target_date.isoformat()}: "
            f"expected={len(teams)}, valid={len(results)}, "
            f"missing={missing}; details: {'; '.join(sorted(problems))}"
        )
        with TIMING_LOCK:
            failures.append(problem)
        # Never publish partial daily or latest fatigue snapshots.
        raise RuntimeError(problem)

    fatigue = pl.concat(
        [results[team][2] for team in teams],
        how="diagonal_relaxed",
    )
    keys = [(str(row["team"]), str(row["game_date"]))
            for row in fatigue.to_dicts()]
    expected_keys = {(team, target_date.isoformat()) for team in teams}
    if len(keys) != len(teams) or set(keys) != expected_keys:
        raise RuntimeError(
            f"Fatigue output validation failed for {target_date}: "
            f"expected={sorted(expected_keys)}, actual={keys}"
        )

    # All scheduled teams were validated before writing any output.
    for team in teams:
        parsed, raw, _ = results[team]
        save_object(
            "fatigue", f"{team}_club_schedule", parsed,
            prefix=prefix, current=True,
        )
        if OPTIONS.include_raw:
            save_object(
                "fatigue", f"{team}_club_schedule_raw", raw,
                prefix=prefix, current=True,
            )

    save_object(
        "fatigue", "fatigue", fatigue,
        prefix=prefix, current=True,
    )
    print(
        f"SDV FATIGUE VERIFIED | date={target_date.isoformat()} "
        f"| teams={len(teams)} | rows={fatigue.height}",
        flush=True,
    )


def pull_predictions(
    failures: list[str],
    *,
    current: bool,
    prefix: str,
    schedule: Any,
    slate: Any,
    ratings: Any,
    context: dict[str, Any] | None,
):
    if current:
        if (
            frame_empty(slate)
            or frame_empty(ratings)
        ):
            save_object(
                "sdv_predictions",
                "predictions",
                pl.DataFrame(),
                prefix=prefix,
                current=True,
            )

            return

        games = prediction_games(
            slate
        )

        if games.is_empty():
            return

        if (
            "game_date"
            in games.columns
        ):
            games = games.select(
                "game_id",
                "home_team",
                "away_team",
                "neutral_site",
            )

        pred = safe_pull(
            failures,
            "sdv_predictions",
            "nhl_predict_games",
            nhl.nhl_predict_games,
            games,
            ratings,
        )

        if pred is not None:
            save_object(
                "sdv_predictions",
                "predictions",
                pred,
                prefix=prefix,
                current=True,
            )

        return

    if context is None:
        return

    pbp = context.get(
        "pbp"
    )

    if (
        frame_empty(schedule)
        or frame_empty(pbp)
    ):
        return

    pred = safe_pull(
        failures,
        "sdv_predictions",
        "historical_predictions",
        historical_predictions,
        schedule,
        pbp,
    )

    if pred is not None:
        save_object(
            "sdv_predictions",
            "predictions",
            pred,
            prefix=prefix,
            current=False,
        )


def espn_event_ids(
    schedule: Any,
) -> list[str]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(
        schedule
    )

    id_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    if id_col is None:
        return []

    return sorted(
        {
            str(v).strip()
            for v in (
                pdf[
                    id_col
                ]
                .dropna()
                .tolist()
            )
            if str(v).strip()
        }
    )


def pull_odds(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
):
    if current:
        query_dates = [
            target_date
        ]

    else:
        historical_schedule = (
            safe_pull(
                failures,
                "odds",
                f"load_nhl_schedule_{season}",
                nhl.load_nhl_schedule,
                season,
            )
        )

        query_dates = (
            historical_schedule_game_dates(
                historical_schedule
            )
        )

        if not query_dates:
            failures.append(
                "odds/historical_schedule_dates: "
                f"no game dates found for season={season}"
            )

            save_object(
                "odds",
                "espn_schedule",
                pl.DataFrame(),
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_scoreboard_raw",
                {},
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_odds",
                pl.DataFrame(),
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_odds_raw",
                {},
                prefix=prefix,
                current=False,
            )

            return

    schedule_frames = []
    scoreboard_raw_by_date: dict[
        str,
        Any,
    ] = {}

    for game_day in query_dates:
        dates_arg = int(
            game_day.strftime(
                "%Y%m%d"
            )
        )

        espn_schedule_day = (
            safe_pull(
                failures,
                "odds",
                f"espn_schedule_{dates_arg}",
                nhl.espn_nhl_schedule,
                dates=dates_arg,
                limit=5000,
            )
        )

        if (
            espn_schedule_day
            is not None
            and not frame_empty(
                espn_schedule_day
            )
        ):
            schedule_frames.append(
                pl.from_pandas(
                    as_pandas(
                        espn_schedule_day
                    )
                )
            )

        scoreboard_raw = (
            (safe_pull(
                failures,
                "odds",
                f"espn_scoreboard_raw_{dates_arg}",
                nhl.espn_nhl_scoreboard,
                dates=dates_arg,
                limit=5000,
                return_parsed=False,
            ) if OPTIONS.include_raw else None)
        )

        if scoreboard_raw is not None:
            scoreboard_raw_by_date[
                game_day.isoformat()
            ] = scoreboard_raw

        if not current:
            time.sleep(
                0.02
            )

    espn_schedule = (
        pl.concat(
            schedule_frames,
            how="diagonal_relaxed",
        )
        if schedule_frames
        else pl.DataFrame()
    )

    if (
        not espn_schedule.is_empty()
    ):
        id_col = first_col(
            espn_schedule,
            (
                "game_id",
                "id",
                "event_id",
            ),
        )

        if id_col is not None:
            pdf = as_pandas(
                espn_schedule
            )

            pdf[
                id_col
            ] = (
                pdf[
                    id_col
                ]
                .astype(str)
                .str.strip()
            )

            pdf = (
                pdf.loc[
                    pdf[
                        id_col
                    ].ne("")
                ]
                .drop_duplicates(
                    subset=[
                        id_col
                    ],
                    keep="first",
                )
                .reset_index(
                    drop=True
                )
            )

            espn_schedule = (
                pl.from_pandas(
                    pdf
                )
            )

    save_object(
        "odds",
        "espn_schedule",
        espn_schedule,
        prefix=prefix,
        current=current,
    )

    save_object(
        "odds",
        "espn_scoreboard_raw",
        scoreboard_raw_by_date,
        prefix=prefix,
        current=current,
    )

    event_ids = espn_event_ids(
        espn_schedule
    )

    print(
        f"ESPN odds discovery | "
        f"dates={len(query_dates)} | "
        f"events={len(event_ids)}"
    )

    odds_frames = []

    raw_by_event: dict[
        str,
        Any,
    ] = {}

    for event_id in event_ids:
        parsed = safe_pull(
            failures,
            "odds",
            f"espn_odds_{event_id}",
            nhl.espn_nhl_game_odds,
            event_id=event_id,
        )

        if (
            parsed is not None
            and not frame_empty(
                parsed
            )
        ):
            pdf = as_pandas(
                parsed
            )

            pdf.insert(
                0,
                "espn_event_id",
                event_id,
            )

            odds_frames.append(
                pl.from_pandas(
                    pdf
                )
            )

        raw = (safe_pull(
            failures,
            "odds",
            f"espn_odds_raw_{event_id}",
            nhl.espn_nhl_game_odds,
            event_id=event_id,
            return_parsed=False,
        ) if OPTIONS.include_raw else None)

        if raw is not None:
            raw_by_event[
                event_id
            ] = raw

        if not current:
            time.sleep(
                0.05
            )

    combined = (
        pl.concat(
            odds_frames,
            how="diagonal_relaxed",
        )
        if odds_frames
        else pl.DataFrame()
    )

    save_object(
        "odds",
        "espn_odds",
        combined,
        prefix=prefix,
        current=current,
    )

    save_object(
        "odds",
        "espn_odds_raw",
        raw_by_event,
        prefix=prefix,
        current=current,
    )


def run_one(
    *,
    current: bool,
    target_date: date,
    season: int,
    categories: set[str],
) -> list[str]:
    failures: list[str] = []

    prefix = (
        current_or_historical_prefix(
            current=current,
            season=(
                None
                if current
                else season
            ),
        )
    )

    schedule = pl.DataFrame()
    slate = pl.DataFrame()

    if (
        "schedule" in categories
        or "team-strength" in categories
        or "goalie" in categories
        or "lineup-strength" in categories
        or "fatigue" in categories
        or "sdv_predictions" in categories
    ):
        (
            schedule,
            slate,
        ) = timed_call("stage/schedule", pull_schedule, 
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
        )

    relevant_schedule = (
        slate
        if current
        else schedule
    )

    teams = schedule_teams(
        relevant_schedule
    )

    team_strength_date = (
        target_date
    )

    if not current:
        (
            _,
            max_day,
        ) = historical_schedule_date_bounds(
            schedule
        )

        if max_day is not None:
            team_strength_date = (
                max_day
            )

    ratings = None

    if (
        "team-strength"
        in categories
        or "sdv_predictions"
        in categories
    ):
        ratings = timed_call("stage/team_strength", pull_team_strength, 
            failures,
            current=current,
            target_date=(
                team_strength_date
                if not current
                else target_date
            ),
            season=season,
            prefix=prefix,
            teams=teams,
            schedule=relevant_schedule,
        )

    rosters: dict[
        str,
        Any,
    ] = {}

    if OPTIONS.include_diagnostics and (
        "lineup-strength" in categories or "goalie" in categories
    ):
        rosters = timed_call("stage/diagnostic_rosters", pull_rosters, 
            failures,
            current=current,
            season=season,
            prefix=prefix,
            teams=teams,
        )

    if OPTIONS.include_diagnostics and "goalie" in categories:
        timed_call("stage/diagnostic_goalie_profiles", pull_goalie_live_profiles, 
            failures,
            current=current,
            prefix=prefix,
            rosters=rosters,
        )

    need_context = bool(
        {"goalie", "lineup-strength"} & categories
        or (not current and {"sdv_predictions", "team-strength"} & categories)
    )

    context = None

    if need_context:
        context = (
            timed_call("stage/season_context", pull_season_context, 
                failures,
                season=season,
                current=current,
                need_shifts=bool({"goalie", "lineup-strength"} & categories),
            )
        )

        timed_call("stage/save_season_context", save_season_context, 
            current=current,
            prefix=prefix,
            context=context,
        )

    if "lineup-strength" in categories:
        save_object(
            "lineup-strength",
            "pregame_feature_evaluation",
            pregame_feature_evaluation(),
            prefix=prefix,
            current=current,
        )

    if "lineup-strength" in categories:
        if current:
            # Daily current-slate run: read independently refreshed historical
            # player-strength snapshots. No RAPM/WAR/unit ratings computed here.
            lineup_features = timed_call(
                "stage/lineup_strength_snapshot_read",
                build_game_lineup_features_from_snapshots,
                relevant_schedule,
                generated_at_utc=datetime.now(UTC),
            )
        elif context is not None:
            # Historical --season replay retains the original strictly-as-of
            # per-game calculation, preserving past reprocessing behavior.
            lineup_features = timed_call(
                "stage/historical_lineup_feature_calculation",
                safe_pull, failures, "lineup-strength",
                "game_lineup_features_asof",
                build_game_lineup_features_asof,
                relevant_schedule,
                context.get("pbp"), context.get("shifts"),
                current=False,
                generated_at_utc=datetime.now(UTC),
            )
        else:
            lineup_features = None

        if lineup_features is not None:
            save_object(
                "lineup-strength",
                "game_lineup_features_asof",
                lineup_features,
                prefix=prefix,
                current=current,
            )

    if (
        "goalie" in categories
        and context is not None
    ):
        goalie_features = timed_call(
            "stage/goalie_feature_calculation", safe_pull,
            failures,
            "goalie",
            "game_goalie_features_asof",
            build_game_goalie_features_asof,
            relevant_schedule,
            context.get(
                "pbp"
            ),
            context.get(
                "shifts"
            ),
            current=current,
            generated_at_utc=datetime.now(
                UTC
            ),
        )

        if goalie_features is not None:
            save_object(
                "goalie",
                "game_goalie_features_asof",
                goalie_features,
                prefix=prefix,
                current=current,
            )

    if (
        (not current)
        and (
            "team-strength"
            in categories
        )
    ):
        pbp = (
            context.get(
                "pbp"
            )
            if context is not None
            else None
        )

        if frame_empty(
            pbp
        ):
            pbp = safe_pull(
                failures,
                "team-strength",
                "load_nhl_pbp_full",
                nhl.load_nhl_pbp_full,
                season,
            )

        ratings_asof = safe_pull(
            failures,
            "team-strength",
            "historical_team_strength",
            historical_team_strength,
            schedule,
            pbp,
        )

        if ratings_asof is not None:
            save_object(
                "team-strength",
                "team_ratings_asof",
                ratings_asof,
                prefix=prefix,
                current=False,
            )

    if (
        context is not None
        and (
            "goalie"
            in categories
        )
        and (
            "lineup-strength"
            not in categories
        )
    ):
        # Legacy full-season player outputs are diagnostics only.
        # Production lineup/player features are generated above by
        # build_game_lineup_features_asof using the T-60 contract.
        pull_advanced_player_strength(
            failures,
            current=current,
            prefix=prefix,
            context=context,
        )

    if "fatigue" in categories:
        timed_call("stage/fatigue", pull_fatigue, 
            failures,
            current=current,
            target_date=target_date,
            prefix=prefix,
            teams=teams,
            schedule=schedule,
        )

    if (
        "sdv_predictions"
        in categories
    ):
        timed_call("stage/sdv_predictions", pull_predictions, 
            failures,
            current=current,
            prefix=prefix,
            schedule=schedule,
            slate=slate,
            ratings=ratings,
            context=context,
        )

    if "odds" in categories:
        timed_call("stage/espn_odds", pull_odds, 
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
        )

    return failures


def compare_baseline_outputs(baseline_dir: Path) -> None:
    """Compare five production CSVs against a preserved, same-slate baseline.

    The API can legitimately revise data between independent pulls. Only
    generated-at observation timestamps are ignored here; all model inputs,
    identities and fixed cutoff fields are compared exactly.
    """
    import csv

    targets = (
        ("fatigue", "latest_fatigue.csv", ("game_date", "team")),
        ("team-strength", "latest_team_ratings_asof.csv", ("game_id", "team")),
        ("goalie", "latest_game_goalie_features_asof.csv", ("game_id",)),
        ("lineup-strength", "latest_game_lineup_features_asof.csv", ("game_id",)),
        ("sdv_predictions", "latest_predictions.csv", ("game_id",)),
    )
    ignored = {
        "ratings_as_of_utc", "goalie_snapshot_as_of_utc",
        "lineup_snapshot_as_of_utc", "home_goalie_status_observed_at",
        "away_goalie_status_observed_at", "home_lineup_observed_at",
        "away_lineup_observed_at",
    }
    results = []
    problems = []
    for family, name, identity in targets:
        actual = ROOT / family / name
        expected = baseline_dir / family / name
        if not actual.is_file() or not expected.is_file():
            problems.append(f"Missing comparison file: {actual} or {expected}")
            continue
        def load_rows(path: Path):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                headers = list(reader.fieldnames or [])
                rows = list(reader)
            if not set(identity).issubset(headers):
                raise ValueError(f"Comparison keys missing: {path}")
            indexed = {}
            for row in rows:
                identity_key = tuple(row[k].strip() for k in identity)
                if identity_key in indexed:
                    raise ValueError(f"Duplicate comparison key {identity_key}: {path}")
                indexed[identity_key] = row
            return headers, indexed
        left_header, left = load_rows(expected)
        right_header, right = load_rows(actual)
        changes = []
        if left_header != right_header:
            changes.append("column sequence changed")
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                changes.append(f"row set changed: {key}")
                continue
            for field in sorted(set(left_header) | set(right_header)):
                if field in ignored:
                    continue
                a = (left[key].get(field) or "").strip()
                b = (right[key].get(field) or "").strip()
                if a != b:
                    changes.append(f"{key}: {field}: {a!r} => {b!r}")
        results.append({"family": family, "file": name,
                        "baseline_rows": len(left), "new_rows": len(right),
                        "difference_count": len(changes),
                        "differences": changes[:50]})
        print(f"SDV OUTPUT COMPARE | {family} | differences={len(changes)}")
        if changes:
            problems.append(f"{family}: {len(changes)} differences")
    output = BASE_DIR / "errors" / "00_intake" / "pull_sdv_output_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"comparisons": results, "problems": problems},
                                indent=2), encoding="utf-8")
    print(f"SDV OUTPUT COMPARE REPORT | {output}")
    if problems:
        raise RuntimeError("SDV baseline output mismatches: " + "; ".join(problems))


def main() -> None:
    ensure_dirs()

    args = parse_args()
    if not 1 <= args.max_workers <= 4:
        raise SystemExit("--max-workers must be between 1 and 4")
    if args.cache_current_hours <= 0 or args.cache_historical_days <= 0:
        raise SystemExit("Cache freshness durations must be positive")
    OPTIONS.include_raw = args.include_raw
    OPTIONS.include_diagnostics = args.include_diagnostics
    OPTIONS.max_workers = args.max_workers
    OPTIONS.cache_dir = args.cache_dir.expanduser()
    OPTIONS.no_cache = args.no_cache
    OPTIONS.cache_current_hours = args.cache_current_hours
    OPTIONS.cache_historical_days = args.cache_historical_days
    OPTIONS.compare_baseline_dir = args.compare_baseline_dir

    categories = (
        requested_categories(
            args
        )
    )

    seasons = (
        requested_seasons(
            args
        )
    )

    now = datetime.now(
        NY
    )

    target_date = now.date()

    all_failures: list[
        str
    ] = []

    if not seasons:
        current_season = (
            season_start_for_day(
                target_date
            )
        )

        print(
            f"SDV NHL current pull | "
            f"date={target_date.isoformat()} "
            f"| season={current_season} "
            f"| categories={sorted(categories)}"
        )

        all_failures.extend(
            run_one(
                current=True,
                target_date=target_date,
                season=current_season,
                categories=categories,
            )
        )

    else:
        for season in seasons:
            print(
                f"SDV NHL historical pull | "
                f"season={season} "
                f"| categories={sorted(categories)}"
            )

            all_failures.extend(
                run_one(
                    current=False,
                    target_date=target_date,
                    season=season,
                    categories=categories,
                )
            )

    if all_failures:
        print(
            "\nSDV pull completed with warnings:",
            file=sys.stderr,
        )

        for item in all_failures:
            print(
                f"  - {item}",
                file=sys.stderr,
            )

    if OPTIONS.compare_baseline_dir is not None:
        compare_baseline_outputs(OPTIONS.compare_baseline_dir)

    print(
        "SDV NHL pull complete."
    )


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        write_timing_report(time.perf_counter() - start)
