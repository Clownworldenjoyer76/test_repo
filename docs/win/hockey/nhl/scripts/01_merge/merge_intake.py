#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/01_merge/merge_intake.py

import csv
import re
import traceback
from pathlib import Path
from typing import Never
from datetime import datetime, UTC, timedelta
from zoneinfo import ZoneInfo


BASE_DIR = Path("docs/win/hockey/nhl")

GAMES_DIR = BASE_DIR / "00_intake" / "games"
SPORTSBOOK_DIR = BASE_DIR / "00_intake" / "sportsbook"
PREDICTIONS_DIR = BASE_DIR / "00_intake" / "predictions"
SDV_PREDICTIONS_PATH = BASE_DIR / "sdv" / "sdv_predictions" / "latest_predictions.csv"

FATIGUE_DIR = BASE_DIR / "sdv" / "fatigue"
TEAM_STRENGTH_DIR = BASE_DIR / "sdv" / "team-strength"
GOALIE_DIR = BASE_DIR / "sdv" / "goalie"
LINEUP_DIR = BASE_DIR / "sdv" / "lineup-strength"
TEAM_MAP_PATH = (
    BASE_DIR
    / "config"
    / "mapping"
    / "team_map_nhl.csv"
)

MERGE_DIR = BASE_DIR / "01_merge"
AUDIT_DIR = MERGE_DIR / "audit"

ERROR_DIR = BASE_DIR / "errors" / "01_merge"
ERROR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "merge_intake.txt"

MERGE_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

GAME_ID_RE = re.compile(r"^\d{10}$")
GAME_DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
ET = ZoneInfo("America/New_York")


TEAM_STRENGTH_VALUE_COLUMNS = [
    "adj_xgf",
    "adj_xga",
    "adj_xg_net",
    "adj_gf",
    "adj_ga",
    "off_rank",
    "def_rank",
    "net_rank",
    "net_z",
]

TEAM_STRENGTH_FEATURE_COLUMNS = [
    "home_adj_xgf",
    "away_adj_xgf",
    "adj_xgf_differential",
    "home_adj_xga",
    "away_adj_xga",
    "adj_xga_differential",
    "home_adj_xg_net",
    "away_adj_xg_net",
    "adj_xg_net_differential",
    "home_adj_gf",
    "away_adj_gf",
    "adj_gf_differential",
    "home_adj_ga",
    "away_adj_ga",
    "adj_ga_differential",
    "home_off_rank",
    "away_off_rank",
    "off_rank_differential",
    "home_def_rank",
    "away_def_rank",
    "def_rank_differential",
    "home_net_rank",
    "away_net_rank",
    "net_rank_differential",
    "home_net_z",
    "away_net_z",
    "net_z_differential",
]


GOALIE_STATUS_VALUES = {
    "projected",
    "expected",
    "confirmed",
    "unknown",
}

GOALIE_PRODUCTION_CUTOFF_MINUTES = 60
GOALIE_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"


LINEUP_PRODUCTION_CUTOFF_MINUTES = 60
LINEUP_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"

LINEUP_STATUS_VALUES = {
    "projected",
    "expected",
    "confirmed",
    "unknown",
}

LINEUP_NUMERIC_METRICS = [
    "skater_rapm",
    "skater_war",
    "pp_value",
    "pk_value",
    "forward_line_strength",
    "defense_pair_strength",
]

LINEUP_NUMERIC_FEATURE_COLUMNS = [
    column
    for metric in LINEUP_NUMERIC_METRICS
    for column in (
        f"home_{metric}",
        f"away_{metric}",
        f"{metric}_differential",
    )
]

LINEUP_METADATA_COLUMNS = [
    "home_lineup_status",
    "away_lineup_status",
    "home_lineup_observed_at",
    "away_lineup_observed_at",
    "home_lineup_source",
    "away_lineup_source",
]

LINEUP_FEATURE_COLUMNS = [
    *LINEUP_NUMERIC_FEATURE_COLUMNS,
    *LINEUP_METADATA_COLUMNS,
]

REQUIRED_LINEUP_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "pregame_cutoff_utc",
    "lineup_decision_cutoff_utc",
    "lineup_decision_cutoff_source",
    "lineup_snapshot_as_of_utc",
    "pregame_cutoff_source",
    *LINEUP_FEATURE_COLUMNS,
]

GOALIE_FEATURE_COLUMNS = [
    "home_expected_starter",
    "away_expected_starter",
    "home_starter_gsax",
    "away_starter_gsax",
    "home_backup_gsax",
    "away_backup_gsax",
    "starter_gsax_differential",
    "home_goalie_status",
    "away_goalie_status",
    "home_goalie_status_observed_at",
    "away_goalie_status_observed_at",
    "home_goalie_status_source",
    "away_goalie_status_source",
]

REQUIRED_GOALIE_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "pregame_cutoff_utc",
    "goalie_decision_cutoff_utc",
    "goalie_decision_cutoff_source",
    "goalie_snapshot_as_of_utc",
    "pregame_cutoff_source",
    *GOALIE_FEATURE_COLUMNS,
]


SDV_PREDICTION_COLUMNS = [
    "sdv_home_win_prob",
    "sdv_exp_margin",
    "sdv_exp_total",
]

REQUIRED_SDV_PREDICTION_COLUMNS = [
    "game_id",
    "home_win_prob",
    "exp_margin",
    "exp_total",
]


MERGED_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "game_id",
    "away_team",
    "home_team",
    "home_days_rest",
    "away_days_rest",
    "home_back_to_back",
    "away_back_to_back",
    "home_games_in_4_days",
    "away_games_in_4_days",
    "home_three_in_four",
    "away_three_in_four",
    "home_games_in_6_days",
    "away_games_in_6_days",
    "home_four_in_six",
    "away_four_in_six",
    "home_games_in_7_days",
    "away_games_in_7_days",
    "rest_differential",
    *TEAM_STRENGTH_FEATURE_COLUMNS,
    *GOALIE_FEATURE_COLUMNS,
    *LINEUP_FEATURE_COLUMNS,
    *SDV_PREDICTION_COLUMNS,
    "away_prob_moneyline",
    "home_prob_moneyline",
    "away_projected_goals",
    "home_projected_goals",
    "total_projected_goals",
    "away_puck_line",
    "home_puck_line",
    "total",
    "away_dk_moneyline_american",
    "home_dk_moneyline_american",
    "away_dk_moneyline_decimal",
    "home_dk_moneyline_decimal",
    "away_dk_puck_line_american",
    "home_dk_puck_line_american",
    "away_dk_puck_line_decimal",
    "home_dk_puck_line_decimal",
    "dk_total_over_american",
    "dk_total_under_american",
    "dk_total_over_decimal",
    "dk_total_under_decimal",
    "odds_source",
    "moneyline_provider_id",
    "moneyline_provider_name",
    "puck_line_provider_id",
    "puck_line_provider_name",
    "total_provider_id",
    "total_provider_name",
    "pulled_at",
]

AUDIT_COLUMNS = [
    "game_date",
    "game_id",
    "away_team",
    "home_team",
    "source_present_games",
    "source_present_sportsbook",
    "source_present_predictions",
    "source_present_sdv_predictions",
    "status",
]

REJECTION_COLUMNS = [
    "reason",
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
]

REQUIRED_GAMES_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

REQUIRED_SPORTSBOOK_COLUMNS = [
    "game_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
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
    "odds_source",
    "moneyline_provider_id",
    "moneyline_provider_name",
    "puck_line_provider_id",
    "puck_line_provider_name",
    "total_provider_id",
    "total_provider_name",
    "pulled_at",
]

REQUIRED_PREDICTION_COLUMNS = [
    "sport",
    "league",
    "game_id",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
    "home_prob_moneyline",
    "away_prob_moneyline",
    "away_projected_goals",
    "home_projected_goals",
    "total_projected_goals",
]

REQUIRED_FATIGUE_COLUMNS = [
    "team",
    "game_date",
    "days_rest",
    "back_to_back",
    "games_in_4_days",
    "three_in_four",
    "games_in_6_days",
    "four_in_six",
    "games_in_7_days",
]

FATIGUE_VALUE_COLUMNS = [
    "days_rest",
    "back_to_back",
    "games_in_4_days",
    "three_in_four",
    "games_in_6_days",
    "four_in_six",
    "games_in_7_days",
]

REQUIRED_TEAM_STRENGTH_COLUMNS = [
    "game_id",
    "team",
    "game_date",
    "pregame_cutoff_utc",
    "ratings_as_of_utc",
    "pregame_cutoff_source",
    *TEAM_STRENGTH_VALUE_COLUMNS,
]


with open(LOG_FILE, "w", encoding="utf-8") as startup_log:
    startup_log.write(
        f"=== merge_intake RUN "
        f"{datetime.now(UTC).isoformat()} ===\n"
    )


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(UTC).isoformat()} | {msg}\n"
        )


def fail(message: str) -> Never:
    log(f"FATAL: {message}")
    log("STATUS: FAILED")
    raise SystemExit(message)


def wipe_merge_outputs() -> None:
    removed_merge = 0
    removed_audit = 0

    for path in MERGE_DIR.glob("*.csv"):
        path.unlink()
        removed_merge += 1

    for path in AUDIT_DIR.glob("*.csv"):
        path.unlink()
        removed_audit += 1

    log(
        f"Wiped merge CSV outputs: {removed_merge}"
    )
    log(
        f"Wiped merge audit CSV outputs: {removed_audit}"
    )


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    log(
        f"WROTE {path} ({len(rows)} rows)"
    )


def validate_required_columns(
    path: Path,
    fieldnames: list[str],
    required_columns: list[str],
) -> None:
    missing = [
        col
        for col in required_columns
        if col not in fieldnames
    ]

    if missing:
        fail(
            f"{path} missing required columns: "
            f"{missing}"
        )


def row_date(
    row: dict[str, str],
) -> str:
    return str(
        row.get("game_date", "")
    ).strip()


def row_game_id(
    row: dict[str, str],
) -> str:
    return str(
        row.get("game_id", "")
    ).strip()


def parse_game_date(
    value: str,
):
    value = str(value).strip()

    if not GAME_DATE_RE.fullmatch(value):
        return None

    try:
        return datetime.strptime(
            value,
            "%Y_%m_%d",
        ).date()

    except ValueError:
        return None


def validate_source_rows(
    path: Path,
    rows: list[dict[str, str]],
    source_name: str,
) -> list[dict[str, str]]:
    seen_game_ids: set[str] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = row_game_id(row)
        game_date = row_date(row)

        if not game_id:
            fail(
                f"{source_name} file has blank "
                f"game_id: {path} row={row_number}"
            )

        if not GAME_ID_RE.fullmatch(game_id):
            fail(
                f"{source_name} file has "
                f"non-canonical game_id: "
                f"{path} row={row_number} "
                f"game_id={game_id!r}"
            )

        if game_id in seen_game_ids:
            fail(
                f"{source_name} file has duplicate "
                f"game_id: {path} "
                f"game_id={game_id}"
            )

        seen_game_ids.add(game_id)

        if parse_game_date(game_date) is None:
            fail(
                f"{source_name} file has invalid "
                f"game_date: {path} "
                f"row={row_number} "
                f"game_date={game_date!r}"
            )

        home_team = str(
            row.get("home_team", "")
        ).strip()

        away_team = str(
            row.get("away_team", "")
        ).strip()

        if not home_team or not away_team:
            fail(
                f"{source_name} file has blank "
                f"team identity: {path} "
                f"row={row_number} "
                f"game_id={game_id}"
            )

    return rows


def load_source_rows(
    source_name: str,
    directory: Path,
    pattern: str,
    required_columns: list[str],
) -> list[dict[str, str]]:
    all_rows: list[dict[str, str]] = []

    if source_name == "sportsbook":
        files = sorted(
            path
            for path in directory.glob("*.csv")
            if path.name.lower().startswith("nhl_")
        )
    else:
        files = sorted(
            directory.glob(pattern)
        )

    log(
        f"{source_name} files found: "
        f"{len(files)}"
    )

    if not files:
        fail(
            f"No {source_name} files found in "
            f"{directory} matching {pattern}"
        )

    for path in files:
        fieldnames, rows = load_csv(path)

        validate_required_columns(
            path,
            fieldnames,
            required_columns,
        )

        rows = validate_source_rows(
            path,
            rows,
            source_name,
        )

        for row in rows:
            row["_source_file"] = str(path)
            all_rows.append(row)

        log(
            f"Loaded {source_name} file: "
            f"{path} ({len(rows)} usable rows)"
        )

    return all_rows


def rows_by_date_game_id(
    rows: list[dict[str, str]],
    source_name: str,
) -> dict[
    str,
    dict[str, dict[str, str]],
]:
    grouped: dict[
        str,
        dict[str, dict[str, str]],
    ] = {}

    seen_game_ids: dict[str, str] = {}

    for row_number, row in enumerate(
        rows,
        start=1,
    ):
        game_date = row_date(row)
        game_id = row_game_id(row)

        if not game_date:
            fail(
                f"{source_name} row has blank "
                f"game_date: "
                f"source_file="
                f"{row.get('_source_file', '')} "
                f"row={row_number} "
                f"game_id={game_id}"
            )

        if not game_id:
            fail(
                f"{source_name} row reached "
                f"grouping with blank game_id: "
                f"source_file="
                f"{row.get('_source_file', '')} "
                f"row={row_number} "
                f"game_date={game_date}"
            )

        if game_id in seen_game_ids:
            fail(
                f"{source_name} has duplicate "
                f"game_id across daily files: "
                f"game_id={game_id} "
                f"first_file="
                f"{seen_game_ids[game_id]} "
                f"duplicate_file="
                f"{row.get('_source_file', '')}"
            )

        seen_game_ids[game_id] = row.get(
            "_source_file",
            "",
        )

        grouped.setdefault(
            game_date,
            {},
        )[game_id] = row

    return grouped


def rows_by_game_id(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    return {
        row_game_id(row): row
        for row in rows
    }


def validate_source_identity_against_games(
    source_name: str,
    source_rows: list[dict[str, str]],
    games_by_id: dict[str, dict[str, str]],
) -> None:
    identity_fields = [
        "sport",
        "league",
        "game_date",
        "game_time",
        "home_team",
        "away_team",
    ]

    problems: list[str] = []

    for row in source_rows:
        game_id = row_game_id(row)
        source_file = row.get(
            "_source_file",
            "",
        )

        game = games_by_id.get(game_id)

        if game is None:
            problems.append(
                f"orphan {source_name} row | "
                f"game_id={game_id} | "
                f"file={source_file}"
            )

            continue

        for field in identity_fields:
            source_value = str(
                row.get(field, "")
            ).strip()

            game_value = str(
                game.get(field, "")
            ).strip()

            if source_value != game_value:
                problems.append(
                    f"{source_name} identity mismatch | "
                    f"game_id={game_id} | "
                    f"field={field} | "
                    f"source={source_value!r} | "
                    f"games={game_value!r} | "
                    f"file={source_file}"
                )

    if problems:
        preview = problems[:25]

        if len(problems) > 25:
            preview.append(
                f"... plus "
                f"{len(problems) - 25} more"
            )

        fail(
            "\n".join(preview)
        )


def rejection_from_row(
    reason: str,
    row: dict[str, str],
) -> dict[str, str]:
    return {
        "reason": reason,
        "game_id": str(
            row.get("game_id", "")
        ).strip(),
        "sport": row.get(
            "sport",
            "",
        ),
        "league": row.get(
            "league",
            "",
        ),
        "game_date": row.get(
            "game_date",
            "",
        ),
        "game_time": row.get(
            "game_time",
            "",
        ),
        "away_team": row.get(
            "away_team",
            "",
        ),
        "home_team": row.get(
            "home_team",
            "",
        ),
    }


def normalize_fatigue_date(
    value: str,
) -> str:
    text = str(value).strip().replace(
        "-",
        "_",
    )

    if not GAME_DATE_RE.fullmatch(text):
        return ""

    try:
        return datetime.strptime(
            text,
            "%Y_%m_%d",
        ).strftime(
            "%Y_%m_%d"
        )

    except ValueError:
        return ""


def normalize_team_lookup_key(
    value: str,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def normalize_fatigue_bool(
    value: str,
) -> str:
    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return "1"

    if text in {
        "false",
        "0",
        "no",
    }:
        return "0"

    if text in {
        "",
        "none",
        "null",
        "nan",
    }:
        return ""

    raise ValueError(
        f"Invalid fatigue boolean value: {value!r}"
    )


def format_numeric(
    value: float,
) -> str:
    if float(value).is_integer():
        return str(
            int(value)
        )

    return (
        f"{float(value):.6f}"
        .rstrip("0")
        .rstrip(".")
    )



def parse_utc_timestamp(
    value: str,
) -> datetime | None:
    text = str(
        value or ""
    ).strip()

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


def canonical_game_cutoff_utc(
    game: dict[str, str],
) -> datetime | None:
    game_date = parse_game_date(
        str(
            game.get(
                "game_date",
                "",
            )
        ).strip()
    )

    game_time = str(
        game.get(
            "game_time",
            "",
        )
    ).strip()

    if (
        game_date is None
        or not game_time
    ):
        return None

    for fmt in (
        "%H:%M:%S",
        "%H:%M",
    ):
        try:
            parsed_time = datetime.strptime(
                game_time,
                fmt,
            ).time()
            return datetime.combine(
                game_date,
                parsed_time,
                tzinfo=ET,
            ).astimezone(
                UTC
            )
        except ValueError:
            continue

    return None


def canonical_goalie_decision_cutoff_utc(
    game: dict[str, str],
) -> datetime | None:
    game_start = canonical_game_cutoff_utc(
        game
    )

    if game_start is None:
        return None

    return (
        game_start
        - timedelta(
            minutes=GOALIE_PRODUCTION_CUTOFF_MINUTES
        )
    )




def canonical_lineup_decision_cutoff_utc(
    game: dict[str, str],
) -> datetime | None:
    game_start = canonical_game_cutoff_utc(
        game
    )

    if game_start is None:
        return None

    return (
        game_start
        - timedelta(
            minutes=LINEUP_PRODUCTION_CUTOFF_MINUTES
        )
    )


def validate_team_strength_timestamp_contract(
    *,
    game_id: str,
    stored_cutoff: datetime,
    ratings_as_of: datetime,
    canonical_cutoff: datetime | None,
    source_file: str,
) -> None:
    if not (
        ratings_as_of
        < stored_cutoff
    ):
        fail(
            "Unsafe SportsDataverse team-strength timestamp: "
            f"game_id={game_id} "
            f"ratings_as_of_utc={ratings_as_of.isoformat()} "
            f"pregame_cutoff_utc={stored_cutoff.isoformat()} "
            f"file={source_file}"
        )

    if (
        canonical_cutoff is not None
        and stored_cutoff
        > canonical_cutoff
    ):
        fail(
            "SportsDataverse team-strength cutoff exceeds "
            "canonical NHL game pregame cutoff: "
            f"game_id={game_id} "
            f"stored_cutoff_utc={stored_cutoff.isoformat()} "
            f"canonical_cutoff_utc={canonical_cutoff.isoformat()} "
            f"file={source_file}"
        )


def load_team_identity_map() -> dict[str, str]:
    if not TEAM_MAP_PATH.exists():
        fail(
            f"Missing NHL team mapping file: "
            f"{TEAM_MAP_PATH}"
        )

    fieldnames, rows = load_csv(
        TEAM_MAP_PATH
    )

    required = {
        "canonical_team",
        "nhl_abbrev",
    }

    missing = sorted(
        required
        - set(fieldnames)
    )

    if missing:
        fail(
            f"{TEAM_MAP_PATH} missing required "
            f"team identity mapping columns: {missing}"
        )

    lookup: dict[str, str] = {}

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        canonical = str(
            row.get(
                "canonical_team",
                "",
            )
        ).strip()

        abbrev = str(
            row.get(
                "nhl_abbrev",
                "",
            )
        ).strip()

        if (
            not canonical
            or canonical == "TBD"
            or not abbrev
        ):
            continue

        for raw_key in (
            canonical,
            abbrev,
        ):
            key = normalize_team_lookup_key(
                raw_key
            )

            if not key:
                continue

            prior = lookup.get(
                key
            )

            if (
                prior is not None
                and prior != canonical
            ):
                fail(
                    f"{TEAM_MAP_PATH} row "
                    f"{row_number} has conflicting "
                    f"team identity mapping for "
                    f"{raw_key!r}: "
                    f"{prior!r} != {canonical!r}"
                )

            lookup[
                key
            ] = canonical

    if not lookup:
        fail(
            f"No NHL team identity mappings loaded "
            f"from {TEAM_MAP_PATH}"
        )

    return lookup


def authoritative_fatigue_files() -> list[Path]:
    files = sorted(
        FATIGUE_DIR.glob(
            "season_*_fatigue.csv"
        )
    )

    latest = (
        FATIGUE_DIR
        / "latest_fatigue.csv"
    )

    if latest.is_file():
        files.append(
            latest
        )

    return files


def load_fatigue_index() -> dict[
    tuple[str, str],
    dict[str, str],
]:
    files = authoritative_fatigue_files()

    log(
        f"SportsDataverse fatigue files found: "
        f"{len(files)}"
    )

    if not files:
        log(
            "No SportsDataverse fatigue files "
            "available; fatigue features will "
            "remain blank."
        )

        return {}

    team_lookup = (
        load_team_identity_map()
    )

    index: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for path in files:
        fieldnames, rows = load_csv(
            path
        )

        validate_required_columns(
            path,
            fieldnames,
            REQUIRED_FATIGUE_COLUMNS,
        )

        loaded_rows = 0

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            game_date = (
                normalize_fatigue_date(
                    row.get(
                        "game_date",
                        "",
                    )
                )
            )

            team_raw = str(
                row.get(
                    "team",
                    "",
                )
            ).strip()

            team_key = (
                normalize_team_lookup_key(
                    team_raw
                )
            )

            canonical_team = (
                team_lookup.get(
                    team_key,
                    "",
                )
            )

            if not game_date:
                fail(
                    f"{path} row {row_number} "
                    "has invalid fatigue "
                    f"game_date="
                    f"{row.get('game_date', '')!r}"
                )

            if not canonical_team:
                fail(
                    f"{path} row {row_number} "
                    "has unmapped fatigue team="
                    f"{team_raw!r}"
                )

            normalized = {
                "team": canonical_team,
                "game_date": game_date,
                "days_rest": str(
                    row.get(
                        "days_rest",
                        "",
                    )
                ).strip(),
                "back_to_back": (
                    normalize_fatigue_bool(
                        row.get(
                            "back_to_back",
                            "",
                        )
                    )
                ),
                "games_in_4_days": str(
                    row.get(
                        "games_in_4_days",
                        "",
                    )
                ).strip(),
                "three_in_four": (
                    normalize_fatigue_bool(
                        row.get(
                            "three_in_four",
                            "",
                        )
                    )
                ),
                "games_in_6_days": str(
                    row.get(
                        "games_in_6_days",
                        "",
                    )
                ).strip(),
                "four_in_six": (
                    normalize_fatigue_bool(
                        row.get(
                            "four_in_six",
                            "",
                        )
                    )
                ),
                "games_in_7_days": str(
                    row.get(
                        "games_in_7_days",
                        "",
                    )
                ).strip(),
                "_source_file": str(
                    path
                ),
            }

            key = (
                game_date,
                canonical_team,
            )

            prior = index.get(
                key
            )

            if prior is not None:
                prior_values = {
                    field: prior.get(
                        field,
                        "",
                    )
                    for field
                    in FATIGUE_VALUE_COLUMNS
                }

                new_values = {
                    field: normalized.get(
                        field,
                        "",
                    )
                    for field
                    in FATIGUE_VALUE_COLUMNS
                }

                if (
                    prior_values
                    != new_values
                ):
                    fail(
                        "Conflicting SportsDataverse "
                        "fatigue rows for "
                        f"game_date={game_date} "
                        f"team={canonical_team}: "
                        f"{prior.get('_source_file', '')} "
                        f"vs {path}"
                    )

                continue

            index[
                key
            ] = normalized

            loaded_rows += 1

        log(
            f"Loaded fatigue file: "
            f"{path} ({loaded_rows} "
            "unique rows)"
        )

    return index



def authoritative_team_strength_files() -> list[Path]:
    files = sorted(
        TEAM_STRENGTH_DIR.glob(
            "season_*_team_ratings_asof.csv"
        )
    )

    latest = (
        TEAM_STRENGTH_DIR
        / "latest_team_ratings_asof.csv"
    )

    if latest.is_file():
        files.append(
            latest
        )

    return files



def load_team_strength_index() -> dict[
    tuple[str, str],
    dict[str, str],
]:
    files = (
        authoritative_team_strength_files()
    )

    log(
        "SportsDataverse strict-as-of team-strength "
        f"files found: {len(files)}"
    )

    if not files:
        log(
            "No SportsDataverse strict-as-of team-strength "
            "files available; team-strength features "
            "will remain blank."
        )

        return {}

    team_lookup = (
        load_team_identity_map()
    )

    index: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for path in files:
        fieldnames, rows = load_csv(
            path
        )

        validate_required_columns(
            path,
            fieldnames,
            REQUIRED_TEAM_STRENGTH_COLUMNS,
        )

        loaded_rows = 0

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            game_date = (
                normalize_fatigue_date(
                    row.get(
                        "game_date",
                        "",
                    )
                )
            )

            team_raw = str(
                row.get(
                    "team",
                    "",
                )
            ).strip()

            team_key = (
                normalize_team_lookup_key(
                    team_raw
                )
            )

            canonical_team = (
                team_lookup.get(
                    team_key,
                    "",
                )
            )

            if not GAME_ID_RE.fullmatch(
                game_id
            ):
                fail(
                    f"{path} row {row_number} "
                    "has invalid team-strength "
                    f"game_id={game_id!r}"
                )

            if not game_date:
                fail(
                    f"{path} row {row_number} "
                    "has invalid team-strength "
                    f"game_date="
                    f"{row.get('game_date', '')!r}"
                )

            if not canonical_team:
                fail(
                    f"{path} row {row_number} "
                    "has unmapped team-strength "
                    f"team={team_raw!r}"
                )

            stored_cutoff = (
                parse_utc_timestamp(
                    row.get(
                        "pregame_cutoff_utc",
                        "",
                    )
                )
            )

            ratings_as_of = (
                parse_utc_timestamp(
                    row.get(
                        "ratings_as_of_utc",
                        "",
                    )
                )
            )

            if stored_cutoff is None:
                fail(
                    f"{path} row {row_number} "
                    "has invalid pregame_cutoff_utc="
                    f"{row.get('pregame_cutoff_utc', '')!r}"
                )

            if ratings_as_of is None:
                fail(
                    f"{path} row {row_number} "
                    "has invalid ratings_as_of_utc="
                    f"{row.get('ratings_as_of_utc', '')!r}"
                )

            if not (
                ratings_as_of
                < stored_cutoff
            ):
                fail(
                    "Unsafe SportsDataverse team-strength "
                    "row: ratings_as_of_utc must be "
                    "strictly earlier than pregame_cutoff_utc "
                    f"path={path} row={row_number} "
                    f"game_id={game_id}"
                )

            normalized = {
                "game_id": game_id,
                "team": canonical_team,
                "game_date": game_date,
                "pregame_cutoff_utc": (
                    stored_cutoff.isoformat()
                ),
                "ratings_as_of_utc": (
                    ratings_as_of.isoformat()
                ),
                "pregame_cutoff_source": str(
                    row.get(
                        "pregame_cutoff_source",
                        "",
                    )
                ).strip(),
                "_source_file": str(
                    path
                ),
            }

            for field in (
                TEAM_STRENGTH_VALUE_COLUMNS
            ):
                normalized[
                    field
                ] = str(
                    row.get(
                        field,
                        "",
                    )
                ).strip()

            key = (
                game_id,
                canonical_team,
            )

            prior = index.get(
                key
            )

            if prior is not None:
                compare_fields = [
                    "game_date",
                    "pregame_cutoff_utc",
                    "ratings_as_of_utc",
                    *TEAM_STRENGTH_VALUE_COLUMNS,
                ]

                prior_values = {
                    field: prior.get(
                        field,
                        "",
                    )
                    for field
                    in compare_fields
                }

                new_values = {
                    field: normalized.get(
                        field,
                        "",
                    )
                    for field
                    in compare_fields
                }

                if (
                    prior_values
                    != new_values
                ):
                    fail(
                        "Conflicting SportsDataverse "
                        "team-strength rows for "
                        f"game_id={game_id} "
                        f"team={canonical_team}: "
                        f"{prior.get('_source_file', '')} "
                        f"vs {path}"
                    )

                continue

            index[
                key
            ] = normalized
            loaded_rows += 1

        log(
            f"Loaded team-strength file: "
            f"{path} ({loaded_rows} "
            "unique strict-as-of rows)"
        )

    return index


def authoritative_goalie_files() -> list[Path]:
    files = sorted(
        GOALIE_DIR.glob(
            "season_*_game_goalie_features_asof.csv"
        )
    )

    latest = (
        GOALIE_DIR
        / "latest_game_goalie_features_asof.csv"
    )

    if latest.is_file():
        files.append(
            latest
        )

    return files


def validate_goalie_timestamp_contract(
    *,
    game_id: str,
    stored_game_start: datetime,
    decision_cutoff: datetime,
    decision_cutoff_source: str,
    snapshot_as_of: datetime,
    observed_at: datetime,
    canonical_game_start: datetime | None,
    canonical_decision_cutoff: datetime | None,
    source_file: str,
    side: str,
) -> None:
    if (
        decision_cutoff_source
        != GOALIE_PRODUCTION_CUTOFF_SOURCE
    ):
        fail(
            "Invalid SportsDataverse goalie production cutoff source: "
            f"game_id={game_id} side={side} "
            f"source={decision_cutoff_source!r} "
            f"file={source_file}"
        )

    if not (
        decision_cutoff
        == (
            stored_game_start
            - timedelta(
                minutes=GOALIE_PRODUCTION_CUTOFF_MINUTES
            )
        )
    ):
        fail(
            "SportsDataverse goalie decision cutoff is not "
            "exactly 60 minutes before the stored game start: "
            f"game_id={game_id} side={side} "
            f"decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"game_start_utc={stored_game_start.isoformat()} "
            f"file={source_file}"
        )

    if snapshot_as_of > decision_cutoff:
        fail(
            "Unsafe SportsDataverse goalie snapshot timestamp: "
            f"game_id={game_id} side={side} "
            f"goalie_snapshot_as_of_utc={snapshot_as_of.isoformat()} "
            f"goalie_decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"file={source_file}"
        )

    if observed_at > decision_cutoff:
        fail(
            "Unsafe SportsDataverse goalie observation timestamp: "
            f"game_id={game_id} side={side} "
            f"observed_at={observed_at.isoformat()} "
            f"goalie_decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"file={source_file}"
        )

    if observed_at > snapshot_as_of:
        fail(
            "SportsDataverse goalie status observation occurs "
            "after the stored goalie snapshot: "
            f"game_id={game_id} side={side} "
            f"observed_at={observed_at.isoformat()} "
            f"snapshot_as_of={snapshot_as_of.isoformat()} "
            f"file={source_file}"
        )

    if (
        canonical_game_start is not None
        and stored_game_start
        != canonical_game_start
    ):
        fail(
            "SportsDataverse goalie game start does not match "
            "the canonical NHL game start: "
            f"game_id={game_id} side={side} "
            f"stored_game_start_utc={stored_game_start.isoformat()} "
            f"canonical_game_start_utc={canonical_game_start.isoformat()} "
            f"file={source_file}"
        )

    if (
        canonical_decision_cutoff is not None
        and decision_cutoff
        != canonical_decision_cutoff
    ):
        fail(
            "SportsDataverse goalie production cutoff does not match "
            "the canonical T-60 cutoff: "
            f"game_id={game_id} side={side} "
            f"stored_decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"canonical_decision_cutoff_utc="
            f"{canonical_decision_cutoff.isoformat()} "
            f"file={source_file}"
        )


def load_goalie_index() -> dict[
    str,
    dict[str, str],
]:
    files = authoritative_goalie_files()

    log(
        "SportsDataverse strict-as-of goalie "
        f"files found: {len(files)}"
    )

    if not files:
        log(
            "No SportsDataverse strict-as-of goalie files "
            "available; goalie features will remain unknown/blank."
        )
        return {}

    team_lookup = load_team_identity_map()

    index: dict[
        str,
        dict[str, str],
    ] = {}

    for path in files:
        fieldnames, rows = load_csv(
            path
        )

        validate_required_columns(
            path,
            fieldnames,
            REQUIRED_GOALIE_COLUMNS,
        )

        loaded_rows = 0

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            game_date = normalize_fatigue_date(
                row.get(
                    "game_date",
                    "",
                )
            )

            if not GAME_ID_RE.fullmatch(
                game_id
            ):
                fail(
                    f"{path} row {row_number} "
                    f"has invalid goalie game_id={game_id!r}"
                )

            if not game_date:
                fail(
                    f"{path} row {row_number} "
                    "has invalid goalie game_date="
                    f"{row.get('game_date', '')!r}"
                )

            stored_cutoff = parse_utc_timestamp(
                row.get(
                    "pregame_cutoff_utc",
                    "",
                )
            )

            decision_cutoff = parse_utc_timestamp(
                row.get(
                    "goalie_decision_cutoff_utc",
                    "",
                )
            )

            decision_cutoff_source = str(
                row.get(
                    "goalie_decision_cutoff_source",
                    "",
                )
            ).strip()

            snapshot_as_of = parse_utc_timestamp(
                row.get(
                    "goalie_snapshot_as_of_utc",
                    "",
                )
            )

            if (
                stored_cutoff is None
                or decision_cutoff is None
                or snapshot_as_of is None
            ):
                fail(
                    f"{path} row {row_number} "
                    "has invalid goalie strict-as-of timestamps"
                )

            normalized = {
                "game_id": game_id,
                "game_date": game_date,
                "pregame_cutoff_utc": stored_cutoff.isoformat(),
                "goalie_decision_cutoff_utc": (
                    decision_cutoff.isoformat()
                ),
                "goalie_decision_cutoff_source": (
                    decision_cutoff_source
                ),
                "goalie_snapshot_as_of_utc": snapshot_as_of.isoformat(),
                "pregame_cutoff_source": str(
                    row.get(
                        "pregame_cutoff_source",
                        "",
                    )
                ).strip(),
                "_source_file": str(
                    path
                ),
            }

            for side in (
                "home",
                "away",
            ):
                team_raw = str(
                    row.get(
                        f"{side}_team",
                        "",
                    )
                ).strip()

                canonical_team = team_lookup.get(
                    normalize_team_lookup_key(
                        team_raw
                    ),
                    "",
                )

                if not canonical_team:
                    fail(
                        f"{path} row {row_number} "
                        f"has unmapped goalie {side}_team={team_raw!r}"
                    )

                status = str(
                    row.get(
                        f"{side}_goalie_status",
                        "",
                    )
                ).strip().lower()

                if status not in GOALIE_STATUS_VALUES:
                    fail(
                        f"{path} row {row_number} "
                        f"has invalid {side}_goalie_status={status!r}"
                    )

                observed_at = parse_utc_timestamp(
                    row.get(
                        f"{side}_goalie_status_observed_at",
                        "",
                    )
                )

                if observed_at is None:
                    fail(
                        f"{path} row {row_number} "
                        f"has invalid {side}_goalie_status_observed_at"
                    )

                validate_goalie_timestamp_contract(
                    game_id=game_id,
                    stored_game_start=stored_cutoff,
                    decision_cutoff=decision_cutoff,
                    decision_cutoff_source=decision_cutoff_source,
                    snapshot_as_of=snapshot_as_of,
                    observed_at=observed_at,
                    canonical_game_start=None,
                    canonical_decision_cutoff=None,
                    source_file=str(path),
                    side=side,
                )

                normalized[
                    f"{side}_team"
                ] = canonical_team

                for field in (
                    f"{side}_expected_starter",
                    f"{side}_starter_gsax",
                    f"{side}_backup_gsax",
                    f"{side}_goalie_status_source",
                ):
                    normalized[field] = str(
                        row.get(
                            field,
                            "",
                        )
                    ).strip()

                normalized[
                    f"{side}_goalie_status"
                ] = status

                normalized[
                    f"{side}_goalie_status_observed_at"
                ] = observed_at.isoformat()

            normalized[
                "starter_gsax_differential"
            ] = str(
                row.get(
                    "starter_gsax_differential",
                    "",
                )
            ).strip()

            prior = index.get(
                game_id
            )

            if prior is not None:
                compare_fields = [
                    field
                    for field in REQUIRED_GOALIE_COLUMNS
                    if field not in {
                        "home_team",
                        "away_team",
                    }
                ]

                prior_values = {
                    field: prior.get(
                        field,
                        "",
                    )
                    for field in compare_fields
                }

                new_values = {
                    field: normalized.get(
                        field,
                        "",
                    )
                    for field in compare_fields
                }

                if prior_values != new_values:
                    fail(
                        "Conflicting SportsDataverse goalie rows for "
                        f"game_id={game_id}: "
                        f"{prior.get('_source_file', '')} vs {path}"
                    )

                continue

            index[
                game_id
            ] = normalized
            loaded_rows += 1

        log(
            f"Loaded goalie file: {path} "
            f"({loaded_rows} unique strict-as-of rows)"
        )

    return index


def empty_goalie_features() -> dict[str, str]:
    features = {
        field: ""
        for field in GOALIE_FEATURE_COLUMNS
    }

    features[
        "home_goalie_status"
    ] = "unknown"

    features[
        "away_goalie_status"
    ] = "unknown"

    return features


def goalie_features_for_game(
    game: dict[str, str],
    goalie_index: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, str]:
    game_id = str(
        game.get(
            "game_id",
            "",
        )
    ).strip()

    row = goalie_index.get(
        game_id
    )

    if row is None:
        return empty_goalie_features()

    game_date = normalize_fatigue_date(
        game.get(
            "game_date",
            "",
        )
    )

    if row.get(
        "game_date",
        "",
    ) != game_date:
        fail(
            "SportsDataverse goalie game_date mismatch: "
            f"game_id={game_id} "
            f"goalie_game_date={row.get('game_date', '')} "
            f"canonical_game_date={game_date}"
        )

    for side in (
        "home",
        "away",
    ):
        canonical_team = str(
            game.get(
                f"{side}_team",
                "",
            )
        ).strip()

        if row.get(
            f"{side}_team",
            "",
        ) != canonical_team:
            fail(
                "SportsDataverse goalie team identity mismatch: "
                f"game_id={game_id} side={side} "
                f"goalie_team={row.get(f'{side}_team', '')!r} "
                f"canonical_team={canonical_team!r}"
            )

    stored_cutoff = parse_utc_timestamp(
        row.get(
            "pregame_cutoff_utc",
            "",
        )
    )

    decision_cutoff = parse_utc_timestamp(
        row.get(
            "goalie_decision_cutoff_utc",
            "",
        )
    )

    decision_cutoff_source = str(
        row.get(
            "goalie_decision_cutoff_source",
            "",
        )
    ).strip()

    snapshot_as_of = parse_utc_timestamp(
        row.get(
            "goalie_snapshot_as_of_utc",
            "",
        )
    )

    canonical_game_start = canonical_game_cutoff_utc(
        game
    )

    canonical_decision_cutoff = (
        canonical_goalie_decision_cutoff_utc(
            game
        )
    )

    if (
        stored_cutoff is None
        or decision_cutoff is None
        or snapshot_as_of is None
    ):
        fail(
            "SportsDataverse goalie row missing valid strict-as-of "
            f"timestamps: game_id={game_id}"
        )

    for side in (
        "home",
        "away",
    ):
        observed_at = parse_utc_timestamp(
            row.get(
                f"{side}_goalie_status_observed_at",
                "",
            )
        )

        if observed_at is None:
            fail(
                "SportsDataverse goalie row missing valid "
                f"{side} observation timestamp: game_id={game_id}"
            )

        validate_goalie_timestamp_contract(
            game_id=game_id,
            stored_game_start=stored_cutoff,
            decision_cutoff=decision_cutoff,
            decision_cutoff_source=decision_cutoff_source,
            snapshot_as_of=snapshot_as_of,
            observed_at=observed_at,
            canonical_game_start=canonical_game_start,
            canonical_decision_cutoff=canonical_decision_cutoff,
            source_file=str(
                row.get(
                    "_source_file",
                    "",
                )
            ),
            side=side,
        )

    return {
        field: str(
            row.get(
                field,
                "",
            )
        ).strip()
        for field in GOALIE_FEATURE_COLUMNS
    }




def authoritative_lineup_files() -> list[Path]:
    files = sorted(
        LINEUP_DIR.glob(
            "season_*_game_lineup_features_asof.csv"
        )
    )

    latest = (
        LINEUP_DIR
        / "latest_game_lineup_features_asof.csv"
    )

    if latest.is_file():
        files.append(
            latest
        )

    return files


def validate_lineup_timestamp_contract(
    *,
    game_id: str,
    stored_game_start: datetime,
    decision_cutoff: datetime,
    decision_cutoff_source: str,
    snapshot_as_of: datetime,
    canonical_game_start: datetime | None,
    canonical_decision_cutoff: datetime | None,
    source_file: str,
) -> None:
    if (
        decision_cutoff_source
        != LINEUP_PRODUCTION_CUTOFF_SOURCE
    ):
        fail(
            "Invalid SportsDataverse lineup production cutoff source: "
            f"game_id={game_id} "
            f"source={decision_cutoff_source!r} "
            f"file={source_file}"
        )

    expected = (
        stored_game_start
        - timedelta(
            minutes=LINEUP_PRODUCTION_CUTOFF_MINUTES
        )
    )

    if decision_cutoff != expected:
        fail(
            "SportsDataverse lineup decision cutoff is not "
            "exactly 60 minutes before the stored game start: "
            f"game_id={game_id} "
            f"decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"game_start_utc={stored_game_start.isoformat()} "
            f"file={source_file}"
        )

    if snapshot_as_of > decision_cutoff:
        fail(
            "Unsafe SportsDataverse lineup snapshot timestamp: "
            f"game_id={game_id} "
            f"lineup_snapshot_as_of_utc={snapshot_as_of.isoformat()} "
            f"lineup_decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"file={source_file}"
        )

    if (
        canonical_game_start is not None
        and stored_game_start != canonical_game_start
    ):
        fail(
            "SportsDataverse lineup game start does not match "
            "canonical NHL game start: "
            f"game_id={game_id} "
            f"stored_game_start_utc={stored_game_start.isoformat()} "
            f"canonical_game_start_utc={canonical_game_start.isoformat()} "
            f"file={source_file}"
        )

    if (
        canonical_decision_cutoff is not None
        and decision_cutoff != canonical_decision_cutoff
    ):
        fail(
            "SportsDataverse lineup production cutoff does not match "
            "the canonical T-60 cutoff: "
            f"game_id={game_id} "
            f"stored_decision_cutoff_utc={decision_cutoff.isoformat()} "
            f"canonical_decision_cutoff_utc="
            f"{canonical_decision_cutoff.isoformat()} "
            f"file={source_file}"
        )


def load_lineup_index() -> dict[
    str,
    dict[str, str],
]:
    files = authoritative_lineup_files()

    log(
        "SportsDataverse strict-as-of lineup/player "
        f"files found: {len(files)}"
    )

    if not files:
        log(
            "No SportsDataverse strict-as-of lineup/player files "
            "available; lineup/player features will remain blank "
            "and lineup status will remain unknown."
        )
        return {}

    team_lookup = load_team_identity_map()

    index: dict[
        str,
        dict[str, str],
    ] = {}

    for path in files:
        fieldnames, rows = load_csv(
            path
        )

        validate_required_columns(
            path,
            fieldnames,
            REQUIRED_LINEUP_COLUMNS,
        )

        loaded_rows = 0

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            game_date = normalize_fatigue_date(
                row.get(
                    "game_date",
                    "",
                )
            )

            if not GAME_ID_RE.fullmatch(
                game_id
            ):
                fail(
                    f"{path} row {row_number} "
                    f"has invalid lineup game_id={game_id!r}"
                )

            if not game_date:
                fail(
                    f"{path} row {row_number} "
                    "has invalid lineup game_date="
                    f"{row.get('game_date', '')!r}"
                )

            stored_game_start = parse_utc_timestamp(
                row.get(
                    "pregame_cutoff_utc",
                    "",
                )
            )

            decision_cutoff = parse_utc_timestamp(
                row.get(
                    "lineup_decision_cutoff_utc",
                    "",
                )
            )

            decision_cutoff_source = str(
                row.get(
                    "lineup_decision_cutoff_source",
                    "",
                )
            ).strip()

            snapshot_as_of = parse_utc_timestamp(
                row.get(
                    "lineup_snapshot_as_of_utc",
                    "",
                )
            )

            if (
                stored_game_start is None
                or decision_cutoff is None
                or snapshot_as_of is None
            ):
                fail(
                    f"{path} row {row_number} "
                    "has invalid lineup strict-as-of timestamps"
                )

            validate_lineup_timestamp_contract(
                game_id=game_id,
                stored_game_start=stored_game_start,
                decision_cutoff=decision_cutoff,
                decision_cutoff_source=decision_cutoff_source,
                snapshot_as_of=snapshot_as_of,
                canonical_game_start=None,
                canonical_decision_cutoff=None,
                source_file=str(path),
            )

            normalized = {
                "game_id": game_id,
                "game_date": game_date,
                "pregame_cutoff_utc": (
                    stored_game_start.isoformat()
                ),
                "lineup_decision_cutoff_utc": (
                    decision_cutoff.isoformat()
                ),
                "lineup_decision_cutoff_source": (
                    decision_cutoff_source
                ),
                "lineup_snapshot_as_of_utc": (
                    snapshot_as_of.isoformat()
                ),
                "pregame_cutoff_source": str(
                    row.get(
                        "pregame_cutoff_source",
                        "",
                    )
                ).strip(),
                "_source_file": str(
                    path
                ),
            }

            for side in (
                "home",
                "away",
            ):
                team_raw = str(
                    row.get(
                        f"{side}_team",
                        "",
                    )
                ).strip()

                canonical_team = team_lookup.get(
                    normalize_team_lookup_key(
                        team_raw
                    ),
                    "",
                )

                if not canonical_team:
                    fail(
                        f"{path} row {row_number} "
                        f"has unmapped lineup {side}_team={team_raw!r}"
                    )

                normalized[
                    f"{side}_team"
                ] = canonical_team

                status = str(
                    row.get(
                        f"{side}_lineup_status",
                        "",
                    )
                ).strip().lower()

                if status not in LINEUP_STATUS_VALUES:
                    fail(
                        f"{path} row {row_number} "
                        f"has invalid {side}_lineup_status={status!r}"
                    )

                observed_text = str(
                    row.get(
                        f"{side}_lineup_observed_at",
                        "",
                    )
                ).strip()

                observed_at = (
                    parse_utc_timestamp(
                        observed_text
                    )
                    if observed_text
                    else None
                )

                if (
                    status != "unknown"
                    and observed_at is None
                ):
                    fail(
                        f"{path} row {row_number} "
                        f"{side}_lineup_status={status!r} "
                        "requires a valid observation timestamp"
                    )

                if observed_at is not None:
                    if observed_at > decision_cutoff:
                        fail(
                            "Unsafe SportsDataverse lineup observation "
                            f"game_id={game_id} side={side} "
                            f"observed_at={observed_at.isoformat()} "
                            f"decision_cutoff={decision_cutoff.isoformat()} "
                            f"file={path}"
                        )

                    if observed_at > snapshot_as_of:
                        fail(
                            "SportsDataverse lineup observation occurs "
                            "after the stored lineup snapshot: "
                            f"game_id={game_id} side={side} "
                            f"observed_at={observed_at.isoformat()} "
                            f"snapshot_as_of={snapshot_as_of.isoformat()} "
                            f"file={path}"
                        )

                normalized[
                    f"{side}_lineup_status"
                ] = status

                normalized[
                    f"{side}_lineup_observed_at"
                ] = (
                    observed_at.isoformat()
                    if observed_at is not None
                    else ""
                )

                normalized[
                    f"{side}_lineup_source"
                ] = str(
                    row.get(
                        f"{side}_lineup_source",
                        "",
                    )
                ).strip()

            for field in (
                LINEUP_NUMERIC_FEATURE_COLUMNS
            ):
                normalized[
                    field
                ] = str(
                    row.get(
                        field,
                        "",
                    )
                ).strip()

            prior = index.get(
                game_id
            )

            if prior is not None:
                compare_fields = [
                    "game_date",
                    "pregame_cutoff_utc",
                    "lineup_decision_cutoff_utc",
                    "lineup_decision_cutoff_source",
                    "lineup_snapshot_as_of_utc",
                    *LINEUP_FEATURE_COLUMNS,
                ]

                prior_values = {
                    field: prior.get(
                        field,
                        "",
                    )
                    for field in compare_fields
                }

                new_values = {
                    field: normalized.get(
                        field,
                        "",
                    )
                    for field in compare_fields
                }

                if prior_values != new_values:
                    fail(
                        "Conflicting SportsDataverse lineup rows for "
                        f"game_id={game_id}: "
                        f"{prior.get('_source_file', '')} vs {path}"
                    )

                continue

            index[
                game_id
            ] = normalized

            loaded_rows += 1

        log(
            f"Loaded lineup/player file: {path} "
            f"({loaded_rows} unique strict-as-of rows)"
        )

    return index


def empty_lineup_features() -> dict[str, str]:
    features = {
        field: ""
        for field in LINEUP_FEATURE_COLUMNS
    }

    features[
        "home_lineup_status"
    ] = "unknown"

    features[
        "away_lineup_status"
    ] = "unknown"

    return features


def lineup_features_for_game(
    game: dict[str, str],
    lineup_index: dict[
        str,
        dict[str, str],
    ],
) -> dict[str, str]:
    game_id = str(
        game.get(
            "game_id",
            "",
        )
    ).strip()

    row = lineup_index.get(
        game_id
    )

    if row is None:
        return empty_lineup_features()

    game_date = normalize_fatigue_date(
        game.get(
            "game_date",
            "",
        )
    )

    if row.get(
        "game_date",
        "",
    ) != game_date:
        fail(
            "SportsDataverse lineup game_date mismatch: "
            f"game_id={game_id} "
            f"lineup_game_date={row.get('game_date', '')} "
            f"canonical_game_date={game_date}"
        )

    for side in (
        "home",
        "away",
    ):
        canonical_team = str(
            game.get(
                f"{side}_team",
                "",
            )
        ).strip()

        if row.get(
            f"{side}_team",
            "",
        ) != canonical_team:
            fail(
                "SportsDataverse lineup team identity mismatch: "
                f"game_id={game_id} side={side} "
                f"lineup_team={row.get(f'{side}_team', '')!r} "
                f"canonical_team={canonical_team!r}"
            )

    stored_game_start = parse_utc_timestamp(
        row.get(
            "pregame_cutoff_utc",
            "",
        )
    )

    decision_cutoff = parse_utc_timestamp(
        row.get(
            "lineup_decision_cutoff_utc",
            "",
        )
    )

    snapshot_as_of = parse_utc_timestamp(
        row.get(
            "lineup_snapshot_as_of_utc",
            "",
        )
    )

    decision_cutoff_source = str(
        row.get(
            "lineup_decision_cutoff_source",
            "",
        )
    ).strip()

    canonical_game_start = canonical_game_cutoff_utc(
        game
    )

    canonical_decision_cutoff = (
        canonical_lineup_decision_cutoff_utc(
            game
        )
    )

    if (
        stored_game_start is None
        or decision_cutoff is None
        or snapshot_as_of is None
    ):
        fail(
            "SportsDataverse lineup row missing valid strict-as-of "
            f"timestamps: game_id={game_id}"
        )

    validate_lineup_timestamp_contract(
        game_id=game_id,
        stored_game_start=stored_game_start,
        decision_cutoff=decision_cutoff,
        decision_cutoff_source=decision_cutoff_source,
        snapshot_as_of=snapshot_as_of,
        canonical_game_start=canonical_game_start,
        canonical_decision_cutoff=canonical_decision_cutoff,
        source_file=str(
            row.get(
                "_source_file",
                "",
            )
        ),
    )

    for side in (
        "home",
        "away",
    ):
        status = str(
            row.get(
                f"{side}_lineup_status",
                "",
            )
        ).strip().lower()

        observed_text = str(
            row.get(
                f"{side}_lineup_observed_at",
                "",
            )
        ).strip()

        observed_at = (
            parse_utc_timestamp(
                observed_text
            )
            if observed_text
            else None
        )

        if status not in LINEUP_STATUS_VALUES:
            fail(
                "SportsDataverse lineup row has invalid status: "
                f"game_id={game_id} side={side} status={status!r}"
            )

        if (
            status != "unknown"
            and observed_at is None
        ):
            fail(
                "SportsDataverse lineup status requires observation "
                f"timestamp: game_id={game_id} side={side}"
            )

        if (
            observed_at is not None
            and (
                observed_at > decision_cutoff
                or observed_at > snapshot_as_of
            )
        ):
            fail(
                "Unsafe SportsDataverse lineup observation timestamp: "
                f"game_id={game_id} side={side} "
                f"observed_at={observed_at.isoformat()}"
            )

    return {
        field: str(
            row.get(
                field,
                "",
            )
        ).strip()
        for field in LINEUP_FEATURE_COLUMNS
    }


def numeric_difference(
    home_value: str,
    away_value: str,
) -> str:
    try:
        return format_numeric(
            float(home_value)
            - float(away_value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return ""



def team_strength_features_for_game(
    game: dict[str, str],
    team_strength_index: dict[
        tuple[str, str],
        dict[str, str],
    ],
) -> dict[str, str]:
    game_id = str(
        game.get(
            "game_id",
            "",
        )
    ).strip()

    game_date = (
        normalize_fatigue_date(
            game.get(
                "game_date",
                "",
            )
        )
    )

    home_team = str(
        game.get(
            "home_team",
            "",
        )
    ).strip()

    away_team = str(
        game.get(
            "away_team",
            "",
        )
    ).strip()

    home = team_strength_index.get(
        (
            game_id,
            home_team,
        ),
        {},
    )

    away = team_strength_index.get(
        (
            game_id,
            away_team,
        ),
        {},
    )

    canonical_cutoff = (
        canonical_game_cutoff_utc(
            game
        )
    )

    for side, row in (
        (
            "home",
            home,
        ),
        (
            "away",
            away,
        ),
    ):
        if not row:
            continue

        if (
            row.get(
                "game_date",
                "",
            )
            != game_date
        ):
            fail(
                "SportsDataverse team-strength "
                "game_date mismatch: "
                f"game_id={game_id} side={side} "
                f"ratings_game_date={row.get('game_date', '')} "
                f"canonical_game_date={game_date}"
            )

        stored_cutoff = (
            parse_utc_timestamp(
                row.get(
                    "pregame_cutoff_utc",
                    "",
                )
            )
        )
        ratings_as_of = (
            parse_utc_timestamp(
                row.get(
                    "ratings_as_of_utc",
                    "",
                )
            )
        )

        if (
            stored_cutoff is None
            or ratings_as_of is None
        ):
            fail(
                "SportsDataverse team-strength row "
                "missing valid strict-as-of timestamps: "
                f"game_id={game_id} side={side}"
            )

        validate_team_strength_timestamp_contract(
            game_id=game_id,
            stored_cutoff=stored_cutoff,
            ratings_as_of=ratings_as_of,
            canonical_cutoff=canonical_cutoff,
            source_file=str(
                row.get(
                    "_source_file",
                    "",
                )
            ),
        )

    features: dict[
        str,
        str,
    ] = {}

    for field in (
        TEAM_STRENGTH_VALUE_COLUMNS
    ):
        home_value = str(
            home.get(
                field,
                "",
            )
        ).strip()

        away_value = str(
            away.get(
                field,
                "",
            )
        ).strip()

        features[
            f"home_{field}"
        ] = home_value

        features[
            f"away_{field}"
        ] = away_value

        features[
            f"{field}_differential"
        ] = numeric_difference(
            home_value,
            away_value,
        )

    return features

def fatigue_features_for_game(
    game: dict[str, str],
    fatigue_index: dict[
        tuple[str, str],
        dict[str, str],
    ],
) -> dict[str, str]:
    game_date = (
        normalize_fatigue_date(
            game.get(
                "game_date",
                "",
            )
        )
    )

    home_team = str(
        game.get(
            "home_team",
            "",
        )
    ).strip()

    away_team = str(
        game.get(
            "away_team",
            "",
        )
    ).strip()

    home = fatigue_index.get(
        (
            game_date,
            home_team,
        ),
        {},
    )

    away = fatigue_index.get(
        (
            game_date,
            away_team,
        ),
        {},
    )

    features = {
        "home_days_rest": home.get(
            "days_rest",
            "",
        ),
        "away_days_rest": away.get(
            "days_rest",
            "",
        ),
        "home_back_to_back": home.get(
            "back_to_back",
            "",
        ),
        "away_back_to_back": away.get(
            "back_to_back",
            "",
        ),
        "home_games_in_4_days": home.get(
            "games_in_4_days",
            "",
        ),
        "away_games_in_4_days": away.get(
            "games_in_4_days",
            "",
        ),
        "home_three_in_four": home.get(
            "three_in_four",
            "",
        ),
        "away_three_in_four": away.get(
            "three_in_four",
            "",
        ),
        "home_games_in_6_days": home.get(
            "games_in_6_days",
            "",
        ),
        "away_games_in_6_days": away.get(
            "games_in_6_days",
            "",
        ),
        "home_four_in_six": home.get(
            "four_in_six",
            "",
        ),
        "away_four_in_six": away.get(
            "four_in_six",
            "",
        ),
        "home_games_in_7_days": home.get(
            "games_in_7_days",
            "",
        ),
        "away_games_in_7_days": away.get(
            "games_in_7_days",
            "",
        ),
        "rest_differential": "",
    }

    try:
        home_rest = float(
            features[
                "home_days_rest"
            ]
        )

        away_rest = float(
            features[
                "away_days_rest"
            ]
        )

        features[
            "rest_differential"
        ] = format_numeric(
            home_rest
            - away_rest
        )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return features



def load_sdv_prediction_index() -> dict[str, dict[str, str]]:
    archive_dir = SDV_PREDICTIONS_PATH.parent

    prediction_files = sorted(
        archive_dir.glob("*_ET_predictions.csv")
    )

    if SDV_PREDICTIONS_PATH.exists():
        prediction_files.append(
            SDV_PREDICTIONS_PATH
        )

    if not prediction_files:
        log(
            "No SportsDataverse prediction files found; "
            "SDV challenger fields will remain blank: "
            f"{archive_dir}"
        )
        return {}

    index: dict[str, dict[str, str]] = {}
    source_by_game_id: dict[str, Path] = {}

    mappings = {
        "sdv_home_win_prob": "home_win_prob",
        "sdv_exp_margin": "exp_margin",
        "sdv_exp_total": "exp_total",
    }

    for prediction_path in prediction_files:
        seen_in_file: set[str] = set()

        fieldnames, rows = load_csv(
            prediction_path
        )

        validate_required_columns(
            prediction_path,
            fieldnames,
            REQUIRED_SDV_PREDICTION_COLUMNS,
        )

        for row_number, row in enumerate(
            rows,
            start=2,
        ):
            game_id = str(
                row.get("game_id", "")
            ).strip()

            if not GAME_ID_RE.fullmatch(game_id):
                fail(
                    "SportsDataverse prediction file has "
                    "non-canonical game_id: "
                    f"{prediction_path} "
                    f"row={row_number} "
                    f"game_id={game_id!r}"
                )

            if game_id in seen_in_file:
                fail(
                    "SportsDataverse prediction file has duplicate game_id: "
                    f"{prediction_path} game_id={game_id}"
                )

            seen_in_file.add(game_id)

            normalized: dict[str, str] = {
                "game_id": game_id
            }

            for output_col, source_col in mappings.items():
                raw = str(
                    row.get(source_col, "")
                ).strip()

                if raw == "":
                    fail(
                        "SportsDataverse prediction row has "
                        "blank required value: "
                        f"{prediction_path} "
                        f"row={row_number} "
                        f"game_id={game_id} "
                        f"column={source_col}"
                    )

                try:
                    value = float(raw)
                except ValueError:
                    fail(
                        "SportsDataverse prediction row has "
                        "non-numeric required value: "
                        f"{prediction_path} "
                        f"row={row_number} "
                        f"game_id={game_id} "
                        f"column={source_col} "
                        f"value={raw!r}"
                    )
                else:
                    if (
                        output_col == "sdv_home_win_prob"
                        and not (0.0 <= value <= 1.0)
                    ):
                        fail(
                            "SportsDataverse home win probability "
                            "is outside [0,1]: "
                            f"{prediction_path} "
                            f"row={row_number} "
                            f"game_id={game_id} "
                            f"value={value}"
                        )

                    normalized[output_col] = (
                        format_numeric(value)
                    )

            previous_source = source_by_game_id.get(
                game_id
            )

            if previous_source is not None:
                log(
                    "SportsDataverse prediction override | "
                    f"game_id={game_id} | "
                    f"previous={previous_source.name} | "
                    f"new={prediction_path.name}"
                )

            index[game_id] = normalized
            source_by_game_id[game_id] = (
                prediction_path
            )

    log(
        "SportsDataverse prediction archive loaded: "
        f"files={len(prediction_files)} "
        f"games={len(index)} "
        f"dir={archive_dir}"
    )

    return index

def process_date(
    date_val: str,
    games_map: dict[str, dict[str, str]],
    sportsbook_map: dict[str, dict[str, str]],
    predictions_map: dict[str, dict[str, str]],
    fatigue_index: dict[
        tuple[str, str],
        dict[str, str],
    ],
    team_strength_index: dict[
        tuple[str, str],
        dict[str, str],
    ],
    goalie_index: dict[
        str,
        dict[str, str],
    ],
    lineup_index: dict[
        str,
        dict[str, str],
    ],
    sdv_prediction_index: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    int,
    int,
    int,
    int,
    bool,
]:
    merged_path = (
        MERGE_DIR
        / f"{date_val}_NHL_merged.csv"
    )

    audit_path = (
        AUDIT_DIR
        / f"{date_val}_NHL_merge_audit.csv"
    )

    rejected_sportsbook_path = (
        AUDIT_DIR
        / f"{date_val}_NHL_rejected_sportsbook.csv"
    )

    rejected_predictions_path = (
        AUDIT_DIR
        / f"{date_val}_NHL_rejected_predictions.csv"
    )

    log(
        f"Processing game_date: {date_val}"
    )

    log(
        f"Games rows for date: "
        f"{len(games_map)}"
    )

    log(
        f"Sportsbook rows for date: "
        f"{len(sportsbook_map)}"
    )

    log(
        f"Prediction rows for date: "
        f"{len(predictions_map)}"
    )

    hard_failure = False

    audit_rows: list[
        dict[str, str]
    ] = []

    rejected_sportsbook: list[
        dict[str, str]
    ] = []

    rejected_predictions: list[
        dict[str, str]
    ] = []

    merged_rows: list[
        dict[str, str]
    ] = []

    missing_source_games = 0

    for game_id, row in sportsbook_map.items():
        if game_id not in games_map:
            hard_failure = True

            rejected_sportsbook.append(
                rejection_from_row(
                    "sportsbook_row_not_found_in_games",
                    row,
                )
            )

    for game_id, row in predictions_map.items():
        if game_id not in games_map:
            hard_failure = True

            rejected_predictions.append(
                rejection_from_row(
                    "prediction_row_not_found_in_games",
                    row,
                )
            )

    for game_id, game in games_map.items():
        has_sportsbook = (
            game_id in sportsbook_map
        )

        has_prediction = (
            game_id in predictions_map
        )

        if (
            has_sportsbook
            and has_prediction
        ):
            status = "matched"

        elif (
            not has_sportsbook
            and not has_prediction
        ):
            status = (
                "missing_sportsbook_and_prediction"
            )

            missing_source_games += 1

        elif not has_sportsbook:
            status = "missing_sportsbook"
            missing_source_games += 1

        else:
            status = "missing_prediction"
            missing_source_games += 1

        audit_rows.append(
            {
                "game_date": game.get(
                    "game_date",
                    date_val,
                ),
                "game_id": game_id,
                "away_team": game.get(
                    "away_team",
                    "",
                ),
                "home_team": game.get(
                    "home_team",
                    "",
                ),
                "source_present_games": "1",
                "source_present_sportsbook": (
                    "1"
                    if has_sportsbook
                    else "0"
                ),
                "source_present_predictions": (
                    "1"
                    if has_prediction
                    else "0"
                ),
                "source_present_sdv_predictions": (
                    "1"
                    if game_id in sdv_prediction_index
                    else "0"
                ),
                "status": status,
            }
        )

        if status != "matched":
            continue

        sportsbook = sportsbook_map[
            game_id
        ]

        prediction = predictions_map[
            game_id
        ]

        fatigue_features = (
            fatigue_features_for_game(
                game,
                fatigue_index,
            )
        )

        team_strength_features = (
            team_strength_features_for_game(
                game,
                team_strength_index,
            )
        )

        goalie_features = (
            goalie_features_for_game(
                game,
                goalie_index,
            )
        )

        lineup_features = (
            lineup_features_for_game(
                game,
                lineup_index,
            )
        )

        sdv_prediction = sdv_prediction_index.get(
            game_id,
            {},
        )

        sdv_prediction_features = {
            column: sdv_prediction.get(column, "")
            for column in SDV_PREDICTION_COLUMNS
        }

        merged_rows.append(
            {
                "sport": game.get(
                    "sport",
                    "hockey",
                ),
                "league": game.get(
                    "league",
                    "nhl",
                ),
                "game_date": game.get(
                    "game_date",
                    date_val,
                ),
                "game_time": game.get(
                    "game_time",
                    "",
                ),
                "game_id": game_id,
                "away_team": game.get(
                    "away_team",
                    "",
                ),
                "home_team": game.get(
                    "home_team",
                    "",
                ),
                **fatigue_features,
                **team_strength_features,
                **goalie_features,
                **lineup_features,
                **sdv_prediction_features,
                "away_prob_moneyline": (
                    prediction.get(
                        "away_prob_moneyline",
                        "",
                    )
                ),
                "home_prob_moneyline": (
                    prediction.get(
                        "home_prob_moneyline",
                        "",
                    )
                ),
                "away_projected_goals": (
                    prediction.get(
                        "away_projected_goals",
                        "",
                    )
                ),
                "home_projected_goals": (
                    prediction.get(
                        "home_projected_goals",
                        "",
                    )
                ),
                "total_projected_goals": (
                    prediction.get(
                        "total_projected_goals",
                        "",
                    )
                ),
                "away_puck_line": (
                    sportsbook.get(
                        "away_puck_line",
                        "",
                    )
                ),
                "home_puck_line": (
                    sportsbook.get(
                        "home_puck_line",
                        "",
                    )
                ),
                "total": sportsbook.get(
                    "total",
                    "",
                ),
                "away_dk_moneyline_american": (
                    sportsbook.get(
                        "away_dk_moneyline_american",
                        "",
                    )
                ),
                "home_dk_moneyline_american": (
                    sportsbook.get(
                        "home_dk_moneyline_american",
                        "",
                    )
                ),
                "away_dk_moneyline_decimal": (
                    sportsbook.get(
                        "away_dk_moneyline_decimal",
                        "",
                    )
                ),
                "home_dk_moneyline_decimal": (
                    sportsbook.get(
                        "home_dk_moneyline_decimal",
                        "",
                    )
                ),
                "away_dk_puck_line_american": (
                    sportsbook.get(
                        "away_dk_puck_line_american",
                        "",
                    )
                ),
                "home_dk_puck_line_american": (
                    sportsbook.get(
                        "home_dk_puck_line_american",
                        "",
                    )
                ),
                "away_dk_puck_line_decimal": (
                    sportsbook.get(
                        "away_dk_puck_line_decimal",
                        "",
                    )
                ),
                "home_dk_puck_line_decimal": (
                    sportsbook.get(
                        "home_dk_puck_line_decimal",
                        "",
                    )
                ),
                "dk_total_over_american": (
                    sportsbook.get(
                        "dk_total_over_american",
                        "",
                    )
                ),
                "dk_total_under_american": (
                    sportsbook.get(
                        "dk_total_under_american",
                        "",
                    )
                ),
                "dk_total_over_decimal": (
                    sportsbook.get(
                        "dk_total_over_decimal",
                        "",
                    )
                ),
                "dk_total_under_decimal": (
                    sportsbook.get(
                        "dk_total_under_decimal",
                        "",
                    )
                ),
                "odds_source": sportsbook.get(
                    "odds_source",
                    "",
                ),
                "moneyline_provider_id": sportsbook.get(
                    "moneyline_provider_id",
                    "",
                ),
                "moneyline_provider_name": sportsbook.get(
                    "moneyline_provider_name",
                    "",
                ),
                "puck_line_provider_id": sportsbook.get(
                    "puck_line_provider_id",
                    "",
                ),
                "puck_line_provider_name": sportsbook.get(
                    "puck_line_provider_name",
                    "",
                ),
                "total_provider_id": sportsbook.get(
                    "total_provider_id",
                    "",
                ),
                "total_provider_name": sportsbook.get(
                    "total_provider_name",
                    "",
                ),
                "pulled_at": sportsbook.get(
                    "pulled_at",
                    "",
                ),
            }
        )

    write_csv(
        audit_path,
        AUDIT_COLUMNS,
        audit_rows,
    )

    write_csv(
        rejected_sportsbook_path,
        REJECTION_COLUMNS,
        rejected_sportsbook,
    )

    write_csv(
        rejected_predictions_path,
        REJECTION_COLUMNS,
        rejected_predictions,
    )

    if merged_rows:
        write_csv(
            merged_path,
            MERGED_COLUMNS,
            merged_rows,
        )
    else:
        log(
            f"No merged rows written for "
            f"{date_val}"
        )

    log(
        f"Date summary {date_val}: "
        f"games={len(games_map)} "
        f"sportsbook={len(sportsbook_map)} "
        f"predictions={len(predictions_map)} "
        f"merged={len(merged_rows)} "
        f"missing_source_games="
        f"{missing_source_games} "
        f"rejected_sportsbook="
        f"{len(rejected_sportsbook)} "
        f"rejected_predictions="
        f"{len(rejected_predictions)} "
        f"hard_failure={hard_failure}"
    )

    return (
        len(merged_rows),
        len(rejected_sportsbook),
        len(rejected_predictions),
        missing_source_games,
        hard_failure,
    )


def main() -> None:
    total_merged = 0
    total_rejected_sportsbook = 0
    total_rejected_predictions = 0
    total_missing_source_games = 0
    dates_with_missing_sources = 0
    dates_with_hard_failures = 0

    try:
        games_rows = load_source_rows(
            "games",
            GAMES_DIR,
            "*_nhl_games.csv",
            REQUIRED_GAMES_COLUMNS,
        )

        sportsbook_rows = load_source_rows(
            "sportsbook",
            SPORTSBOOK_DIR,
            "NHL_*.csv",
            REQUIRED_SPORTSBOOK_COLUMNS,
        )

        prediction_rows = load_source_rows(
            "predictions",
            PREDICTIONS_DIR,
            "hockey_*.csv",
            REQUIRED_PREDICTION_COLUMNS,
        )

        fatigue_index = (
            load_fatigue_index()
        )

        team_strength_index = (
            load_team_strength_index()
        )

        goalie_index = (
            load_goalie_index()
        )

        lineup_index = (
            load_lineup_index()
        )

        sdv_prediction_index = (
            load_sdv_prediction_index()
        )

        games_by_date = (
            rows_by_date_game_id(
                games_rows,
                "games",
            )
        )

        sportsbook_by_date = (
            rows_by_date_game_id(
                sportsbook_rows,
                "sportsbook",
            )
        )

        predictions_by_date = (
            rows_by_date_game_id(
                prediction_rows,
                "predictions",
            )
        )

        games_by_id = rows_by_game_id(
            games_rows
        )

        validate_source_identity_against_games(
            "sportsbook",
            sportsbook_rows,
            games_by_id,
        )

        validate_source_identity_against_games(
            "predictions",
            prediction_rows,
            games_by_id,
        )

        dates = sorted(
            games_by_date.keys()
        )

        log(
            f"Dates found from canonical "
            f"games rows: {len(dates)}"
        )

        if not dates:
            fail(
                "No Stage 00 games rows found."
            )

        extra_sportsbook_dates = sorted(
            set(sportsbook_by_date)
            - set(games_by_date)
        )

        if extra_sportsbook_dates:
            fail(
                "Sportsbook contains dates not "
                "present in canonical games: "
                f"{extra_sportsbook_dates}"
            )

        extra_prediction_dates = sorted(
            set(predictions_by_date)
            - set(games_by_date)
        )

        if extra_prediction_dates:
            fail(
                "Predictions contain dates not "
                "present in canonical games: "
                f"{extra_prediction_dates}"
            )

        wipe_merge_outputs()

        for date_val in dates:
            (
                merged_count,
                rejected_sportsbook_count,
                rejected_predictions_count,
                missing_source_count,
                hard_failure,
            ) = process_date(
                date_val,
                games_by_date.get(
                    date_val,
                    {},
                ),
                sportsbook_by_date.get(
                    date_val,
                    {},
                ),
                predictions_by_date.get(
                    date_val,
                    {},
                ),
                fatigue_index,
                team_strength_index,
                goalie_index,
                lineup_index,
                sdv_prediction_index,
            )

            total_merged += merged_count

            total_rejected_sportsbook += (
                rejected_sportsbook_count
            )

            total_rejected_predictions += (
                rejected_predictions_count
            )

            total_missing_source_games += (
                missing_source_count
            )

            if missing_source_count > 0:
                dates_with_missing_sources += 1

            if hard_failure:
                dates_with_hard_failures += 1

        log("--- SUMMARY ---")
        log(
            f"Dates processed: {len(dates)}"
        )
        log(
            "Dates with missing source games: "
            f"{dates_with_missing_sources}"
        )
        log(
            "Games missing prediction and/or "
            "sportsbook source: "
            f"{total_missing_source_games}"
        )
        log(
            "Dates with hard failures: "
            f"{dates_with_hard_failures}"
        )
        log(
            f"Rows merged: {total_merged}"
        )
        log(
            "Rejected sportsbook rows: "
            f"{total_rejected_sportsbook}"
        )
        log(
            "Rejected prediction rows: "
            f"{total_rejected_predictions}"
        )

        if dates_with_hard_failures > 0:
            fail(
                "Stage 01 merge failed for "
                f"{dates_with_hard_failures} "
                "date(s) due to invalid/orphan "
                "source rows. See audit and "
                "rejection CSVs."
            )

        log("STATUS: SUCCESS")

        print("STAGE 01 MERGE PASSED")
        print(
            f"dates_processed={len(dates)}"
        )
        print(
            f"rows_merged={total_merged}"
        )
        print(
            "games_missing_sources="
            f"{total_missing_source_games}"
        )
        print(
            "dates_with_missing_sources="
            f"{dates_with_missing_sources}"
        )
        print(
            "rejected_sportsbook_rows="
            f"{total_rejected_sportsbook}"
        )
        print(
            "rejected_prediction_rows="
            f"{total_rejected_predictions}"
        )

    except SystemExit:
        raise

    except Exception as exc:
        log(
            f"FATAL ERROR: {exc}\n"
            f"{traceback.format_exc()}"
        )
        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()