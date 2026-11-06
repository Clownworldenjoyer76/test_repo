#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/transform_hockey.py

import re
import traceback
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from team_map_common import (
    normalize_team_alias_key,
    strip_record,
)
from datetime import datetime

import pandas as pd


BASE_DIR = Path("docs/win/hockey/nhl")

INPUT_DIR = BASE_DIR / "00_intake" / "predictions" / "scraper"
OUTPUT_DIR = BASE_DIR / "00_intake" / "predictions"

MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"
NO_MAP_PATH = BASE_DIR / "config" / "mapping" / "no_map_nhl_pred.csv"

ERROR_DIR = BASE_DIR / "errors" / "00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "transform_hockey.txt"

OUTPUT_COLUMNS = [
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


with open(LOG_FILE, "w", encoding="utf-8") as startup_log:
    startup_log.write(f"=== transform_hockey RUN {datetime.now().isoformat()} ===\n")


def log(msg: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {msg}\n")


def load_team_map(source: str) -> dict[str, dict[str, str]]:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Missing team mapping file: {MAP_PATH}")

    df = pd.read_csv(MAP_PATH, dtype=str).fillna("")

    required_columns = {
        "league",
        "source",
        "alias",
        "canonical_team",
        "nhl_team_id",
        "nhl_abbrev",
    }
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"{MAP_PATH} missing required columns: {missing}"
        )

    allowed_sources = {source, "shared", "official_nhl"}
    mapping: dict[str, dict[str, str]] = {}
    identity_by_id: dict[str, tuple[str, str]] = {}

    for row_number, row in df.iterrows():
        league = str(row["league"]).strip().lower()
        row_source = str(row["source"]).strip().lower()

        if league != "nhl" or row_source not in allowed_sources:
            continue

        alias = str(row["alias"]).strip()
        canonical = str(row["canonical_team"]).strip()
        team_id = str(row["nhl_team_id"]).strip()
        abbrev = str(row["nhl_abbrev"]).strip().upper()

        if not alias or not canonical:
            continue

        if canonical != "TBD":
            if not team_id or not team_id.isdigit():
                raise ValueError(
                    f"{MAP_PATH} row {row_number + 2} has invalid "
                    f"nhl_team_id={team_id!r}"
                )

            if not re.fullmatch(r"[A-Z]{3}", abbrev):
                raise ValueError(
                    f"{MAP_PATH} row {row_number + 2} has invalid "
                    f"nhl_abbrev={abbrev!r}"
                )

            prior_identity = identity_by_id.get(team_id)
            identity_value = (canonical, abbrev)

            if prior_identity is not None and prior_identity != identity_value:
                raise ValueError(
                    f"{MAP_PATH} has conflicting identity for "
                    f"nhl_team_id={team_id}: "
                    f"{prior_identity} != {identity_value}"
                )

            identity_by_id[team_id] = identity_value

        identity = {
            "canonical_team": canonical,
            "nhl_team_id": team_id,
            "nhl_abbrev": abbrev,
        }

        key = normalize_team_alias_key(alias)
        prior = mapping.get(key)

        if prior is not None and prior != identity:
            raise ValueError(
                f"{MAP_PATH} has conflicting {source} mapping for "
                f"alias={alias!r}: {prior} != {identity}"
            )

        mapping[key] = identity

    if not mapping:
        raise ValueError(
            f"No NHL mappings loaded for source={source} from {MAP_PATH}"
        )

    stable_ids = {
        identity["nhl_team_id"]
        for identity in mapping.values()
        if identity["nhl_team_id"]
    }

    log(f"Loaded team map: {MAP_PATH}")
    log(f"Team map source: {source}")
    log(f"Team aliases loaded: {len(mapping)}")
    log(f"Stable NHL team IDs loaded: {len(stable_ids)}")

    return mapping


def normalize_team(
    name: str,
    team_map: dict[str, dict[str, str]],
    no_map_records: list,
    source_file: str,
) -> str:
    stripped = strip_record(name)
    key = normalize_team_alias_key(stripped)
    identity = team_map.get(key)

    if identity is not None:
        return identity["canonical_team"]

    no_map_records.append(
        {
            "source_file": source_file,
            "raw_team": name,
            "stripped_team": stripped,
            "normalized_attempt": key,
        }
    )

    return stripped

def parse_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(str(date_str).strip(), "%m/%d/%Y %I:%M %p")
        return dt.strftime("%Y_%m_%d")
    except (TypeError, ValueError, OverflowError):
        return str(date_str).strip().replace("/", "_").replace(" ", "_")


def parse_time(date_str: str) -> str:
    parts = str(date_str).strip().split(" ")
    if len(parts) >= 2:
        return " ".join(parts[1:])
    return ""


def parse_probability(value) -> str:
    try:
        parsed = float(str(value).replace("%", "").strip()) / 100
        return f"{parsed:.6f}"
    except (TypeError, ValueError, OverflowError):
        return ""


def parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return ""


def write_no_map_file(no_map_records: list) -> None:
    NO_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not no_map_records:
        pd.DataFrame(
            columns=[
                "source_file",
                "raw_team",
                "stripped_team",
                "normalized_attempt",
            ]
        ).to_csv(NO_MAP_PATH, index=False)
        log(f"WROTE no-map file with 0 rows: {NO_MAP_PATH}")
        return

    no_map_df = pd.DataFrame(no_map_records).drop_duplicates()
    no_map_df.to_csv(NO_MAP_PATH, index=False)
    log(f"WROTE no-map file: {NO_MAP_PATH} ({len(no_map_df)} rows)")


def transform_prediction_file(
    input_path: Path,
    team_map: dict,
    no_map_records: list,
    files_written: list,
) -> None:
    log(f"Processing prediction input: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        log(f"WARNING: input file empty: {input_path}")
        return

    required_columns = [
        "date_time",
        "team1",
        "team2",
        "team1_win_pct",
        "team2_win_pct",
        "proj_score_1",
        "proj_score_2",
        "score1",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        log(f"WARNING: skipping {input_path}; missing columns: {missing_columns}")
        return

    df["team1_clean"] = df["team1"].apply(
        lambda value: normalize_team(value, team_map, no_map_records, str(input_path))
    )
    df["team2_clean"] = df["team2"].apply(
        lambda value: normalize_team(value, team_map, no_map_records, str(input_path))
    )

    df["game_date"] = df["date_time"].apply(parse_date)
    df["game_time"] = df["date_time"].apply(parse_time)

    upcoming = df[
        df["score1"].isna()
        | (df["score1"].astype(str).str.strip() == "")
    ].copy()

    if upcoming.empty:
        log(f"No upcoming NHL games found in {input_path}.")
        return

    for date_val, group in upcoming.groupby("game_date"):
        output_rows = []

        for _, row in group.iterrows():
            away_team = row["team1_clean"]
            home_team = row["team2_clean"]

            away_projected_goals = parse_float(row["proj_score_1"])
            home_projected_goals = parse_float(row["proj_score_2"])

            if away_projected_goals != "" and home_projected_goals != "":
                total_projected_goals = round(
                    away_projected_goals + home_projected_goals,
                    2,
                )
            else:
                total_projected_goals = ""

            output_rows.append(
                {
                    "sport": "hockey",
                    "league": "nhl",
                    # Official NHL game_id is assigned only by reconcile_game_ids.py.
                    "game_id": "",
                    "game_date": date_val,
                    "game_time": row["game_time"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_prob_moneyline": parse_probability(row["team2_win_pct"]),
                    "away_prob_moneyline": parse_probability(row["team1_win_pct"]),
                    "away_projected_goals": away_projected_goals,
                    "home_projected_goals": home_projected_goals,
                    "total_projected_goals": total_projected_goals,
                }
            )

        output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"hockey_{date_val}.csv"
        output.to_csv(output_path, index=False)

        files_written.append((str(output_path), len(output)))
        log(
            f"WROTE unreconciled prediction output: {output_path} "
            f"({len(output)} rows); game_id left blank for reconcile_game_ids.py"
        )


def main():
    files_written = []

    try:
        log(f"Input directory: {INPUT_DIR}")
        log(f"Output directory: {OUTPUT_DIR}")
        log(f"Mapping file: {MAP_PATH}")
        log(f"No-map file: {NO_MAP_PATH}")
        log("Official NHL game_id assignment is deferred to reconcile_game_ids.py")

        team_map = load_team_map("dratings")
        no_map_records = []

        input_files = sorted(INPUT_DIR.glob("*_nhl_predictions.csv"))

        if not input_files:
            log(f"WARNING: no prediction input files found in {INPUT_DIR}")

        for input_path in input_files:
            transform_prediction_file(
                input_path=input_path,
                team_map=team_map,
                no_map_records=no_map_records,
                files_written=files_written,
            )

        write_no_map_file(no_map_records)

        log("--- SUMMARY ---")
        log(f"Input files processed: {len(input_files)}")
        log(f"Files written: {len(files_written)}")
        for path, count in files_written:
            log(f"  FILE: {path} ({count} rows)")
        log(f"No-map records: {len(no_map_records)}")
        log("STATUS: SUCCESS")

    except Exception as e:
        log(f"FATAL ERROR: {e}\n{traceback.format_exc()}")
        log("STATUS: FAILED")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()