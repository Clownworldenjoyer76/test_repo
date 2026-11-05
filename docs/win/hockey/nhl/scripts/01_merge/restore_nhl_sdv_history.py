
#!/usr/bin/env python3
"""
Restore archived pregame SportsDataverse NHL features, run the existing Stage 01
merge unchanged, and audit missing-feature coverage.

Run from the repository root:
    python docs/win/hockey/nhl/scripts/01_merge/restore_nhl_sdv_history.py prepare
    python docs/win/hockey/nhl/scripts/01_merge/merge_intake.py
    python docs/win/hockey/nhl/scripts/01_merge/restore_nhl_sdv_history.py audit

Or invoke with "all" to do all three steps in one command.

No probability, pricing, filtering, or betting formulas are modified.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NoReturn

ROOT = Path("docs/win/hockey/nhl")
SDV = ROOT / "sdv"
MERGED = ROOT / "01_merge"
AUDIT = MERGED / "audit" / "sdv_feature_coverage.csv"
MAP = ROOT / "config" / "mapping" / "team_map_nhl.csv"
ORIGINAL_MERGE = ROOT / "scripts" / "01_merge" / "merge_intake.py"

# Category, archive suffix, record identity, and source fields for coverage.
FAMILIES = {
    "fatigue": (
        "fatigue", "fatigue", ("game_date", "team"),
        ("days_rest", "back_to_back", "games_in_4_days", "three_in_four",
         "games_in_6_days", "four_in_six", "games_in_7_days"),
    ),
    "team-strength": (
        "team-strength", "team_ratings_asof", ("game_id", "team"),
        ("adj_xgf", "adj_xga", "adj_xg_net", "adj_gf", "adj_ga",
         "off_rank", "def_rank", "net_rank", "net_z"),
    ),
    "goalie": (
        "goalie", "game_goalie_features_asof", ("game_id",),
        ("home_expected_starter", "away_expected_starter",
         "home_starter_gsax", "away_starter_gsax",
         "home_backup_gsax", "away_backup_gsax",
         "home_goalie_status", "away_goalie_status",
         "home_goalie_status_observed_at", "away_goalie_status_observed_at",
         "home_goalie_status_source", "away_goalie_status_source"),
    ),
    "lineup-strength": (
        "lineup-strength", "game_lineup_features_asof", ("game_id",),
        tuple(f"{side}_{feature}" for side in ("home", "away") for feature in (
            "skater_rapm", "skater_war", "pp_value", "pk_value",
            "forward_line_strength", "defense_pair_strength",
            "lineup_status", "lineup_observed_at", "lineup_source")),
    ),
}


def error(message: str) -> NoReturn:
    raise RuntimeError(message)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or [])
        if not header or len(header) != len(set(header)):
            error(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        error(f"Row has more values than headers: {path}")
    return header, rows


def norm_date(value: str) -> str:
    return (value or "").strip().replace("-", "_")


def utc_stamp(value: str, where: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(
            (value or "").strip().replace("Z", "+00:00")
        )
    except ValueError:
        error(f"Invalid timestamp {value!r}: {where}")
    else:
        if stamp.tzinfo is None:
            error(f"Missing UTC timezone in timestamp {value!r}: {where}")
        return stamp.astimezone(timezone.utc)


def validate_source_row(
    family: str,
    row: dict[str, str],
    where: str,
) -> None:
    if family == "fatigue":
        previous = norm_date(row.get("previous_game_date", ""))
        current = norm_date(row.get("game_date", ""))
        if previous and previous >= current:
            error(f"Unsafe previous_game_date: {where}")
        return

    start = utc_stamp(row.get("pregame_cutoff_utc", ""), where)

    if family == "team-strength":
        ratings = utc_stamp(row.get("ratings_as_of_utc", ""), where)
        if ratings >= start:
            error(f"Post-cutoff team rating rejected: {where}")
        return

    label = "goalie" if family == "goalie" else "lineup"
    cutoff = utc_stamp(
        row.get(f"{label}_decision_cutoff_utc", ""), where
    )
    snapshot = utc_stamp(
        row.get(f"{label}_snapshot_as_of_utc", ""), where
    )

    if cutoff != start - timedelta(minutes=60) or snapshot > cutoff:
        error(f"Unsafe {label} decision/snapshot cutoff: {where}")

    if (
        row.get(f"{label}_decision_cutoff_source")
        != "fixed_60_minutes_before_puck_drop"
    ):
        error(f"Invalid {label} production cutoff source: {where}")

    for side in ("home", "away"):
        status_key = f"{side}_{label}_status"
        observed_key = (
            f"{side}_goalie_status_observed_at"
            if label == "goalie"
            else f"{side}_lineup_observed_at"
        )
        observation = (row.get(observed_key) or "").strip()
        status = (row.get(status_key) or "").strip().lower()

        if status not in {"unknown", "projected", "expected", "confirmed"}:
            error(f"Invalid {side} {label} status: {where}")

        if status != "unknown" and not observation:
            error(f"Missing {side} {label} observation: {where}")

        if observation:
            observed = utc_stamp(observation, where)
            if observed > snapshot or observed > cutoff:
                error(f"Post-cutoff {side} {label} observation: {where}")


def make_key(
    row: dict[str, str],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    result = tuple(
        norm_date(row.get(field, ""))
        if field == "game_date"
        else (row.get(field) or "").strip()
        for field in fields
    )
    if any(not part for part in result):
        error(f"Incomplete identity {fields}: {row}")
    return result


def checked_rows(
    path: Path,
    family: str,
    fields: tuple[str, ...],
    value_fields: tuple[str, ...],
    filename_date: str = "",
) -> tuple[list[str], list[dict[str, str]]]:
    header, rows = read_csv(path)
    needed = set(fields) | set(value_fields) | {"game_date"}

    if family == "team-strength":
        needed |= {"pregame_cutoff_utc", "ratings_as_of_utc"}

    if family in ("goalie", "lineup-strength"):
        label = "goalie" if family == "goalie" else "lineup"
        needed |= {
            "pregame_cutoff_utc",
            f"{label}_decision_cutoff_utc",
            f"{label}_decision_cutoff_source",
            f"{label}_snapshot_as_of_utc",
        }

    missing = needed - set(header)
    if missing:
        error(f"Missing {family} source columns in {path}: {sorted(missing)}")

    encountered: dict[tuple[str, ...], dict[str, str]] = {}

    for line, row in enumerate(rows, start=2):
        where = f"{path}:{line}"

        if not norm_date(row.get("game_date", "")):
            error(f"Missing game_date: {where}")

        if (
            filename_date
            and norm_date(row.get("game_date", "")) != filename_date
        ):
            error(f"Snapshot filename/game_date mismatch: {where}")

        validate_source_row(family, row, where)
        key = make_key(row, fields)

        if key in encountered and encountered[key] != row:
            error(f"Conflicting records within {path}: {key}")

        encountered[key] = row

    return header, list(encountered.values())


def atomic_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def prepare_history() -> None:
    for family, (
        directory, suffix, key_fields, value_fields
    ) in FAMILIES.items():
        folder = SDV / directory

        if not folder.is_dir():
            error(f"Missing SDV source folder: {folder}")

        latest_path = folder / f"latest_{suffix}.csv"
        latest_dates: set[str] = set()

        if latest_path.is_file():
            _, latest_rows = checked_rows(
                latest_path, family, key_fields, value_fields
            )
            latest_dates = {
                norm_date(row["game_date"]) for row in latest_rows
            }

        filename_pattern = re.compile(
            rf"^(\d{{4}}_\d{{2}}_\d{{2}})T\d{{6}}_ET_"
            rf"{re.escape(suffix)}\.csv$"
        )

        by_date: dict[str, list[Path]] = defaultdict(list)

        for path in sorted(folder.glob(f"*_ET_{suffix}.csv")):
            matched = filename_pattern.fullmatch(path.name)

            if matched and not path.read_text(encoding="utf-8-sig").strip():
                print(f"SDV empty historical snapshot skipped: {path}")
                continue

            if (
                matched
                and matched.group(1) not in latest_dates
            ):
                by_date[matched.group(1)].append(path)

        by_year: dict[
            str, dict[tuple[str, ...], dict[str, str]]
        ] = defaultdict(dict)

        headers: dict[str, list[str]] = defaultdict(list)
        duplicate_days = 0

        for date_key, candidates in sorted(by_date.items()):
            alternatives = []

            for path in candidates:
                header, rows = checked_rows(
                    path,
                    family,
                    key_fields,
                    value_fields,
                    date_key,
                )
                alternatives.append((path, header, rows))

            # Most complete slate wins; earliest file breaks ties.
            selected = min(
                alternatives,
                key=lambda item: (-len(item[2]), item[0].name),
            )

            path, header, rows = selected

            if len(alternatives) > 1:
                duplicate_days += 1
                print(
                    f"SDV snapshot selection {family} {date_key}: "
                    f"{path.name} ({len(rows)} rows, "
                    f"{len(alternatives)} candidates)"
                )

            year = date_key[:4]

            for column in header:
                if column not in headers[year]:
                    headers[year].append(column)

            for row in rows:
                key = make_key(row, key_fields)

                if (
                    key in by_year[year]
                    and by_year[year][key] != row
                ):
                    error(
                        f"Duplicate history keys across dates: "
                        f"{family} {key}"
                    )

                by_year[year][key] = row

        for year, archive_records in sorted(by_year.items()):
            output = folder / f"season_{year}_{suffix}.csv"
            merged_records = dict(archive_records)

            if output.exists():
                existing_header, existing_rows = checked_rows(
                    output, family, key_fields, value_fields
                )

                for col in existing_header:
                    if col not in headers[year]:
                        headers[year].append(col)

                for row in existing_rows:
                    # Do not duplicate the current latest slate.
                    if norm_date(row["game_date"]) in latest_dates:
                        continue

                    key = make_key(row, key_fields)
                    prior = merged_records.get(key)

                    if prior is not None and any(
                        (prior.get(col) or "")
                        != (row.get(col) or "")
                        for col in set(prior) | set(row)
                    ):
                        error(
                            f"Existing season data conflicts "
                            f"with archive: {output} {key}"
                        )

                    merged_records.setdefault(key, row)

            output_rows = [
                merged_records[key]
                for key in sorted(merged_records)
            ]

            atomic_csv(output, headers[year], output_rows)

            print(
                f"SDV history {family}: "
                f"{len(output_rows)} historical rows => {output}"
            )

        print(
            f"SDV history {family}: {len(by_date)} dates, "
            f"{duplicate_days} dates with multiple snapshots, "
            f"latest dates excluded: {sorted(latest_dates)}"
        )


def team_map() -> dict[str, str]:
    _, rows = read_csv(MAP)
    result = {}

    for row in rows:
        canonical = (row.get("canonical_team") or "").strip()
        abbreviation = (row.get("nhl_abbrev") or "").strip()

        if canonical and abbreviation and canonical != "TBD":
            result[abbreviation.upper()] = canonical
            result[canonical.upper()] = canonical

    if not result:
        error(f"No NHL team mappings: {MAP}")

    return result


def input_index(
    family: str,
    identities: dict[str, str],
) -> dict[tuple[str, ...], dict[str, str]]:
    directory, suffix, key_fields, values = FAMILIES[family]
    folder = SDV / directory

    paths = sorted(folder.glob(f"season_*_{suffix}.csv"))
    latest = folder / f"latest_{suffix}.csv"

    if latest.exists():
        paths.append(latest)

    index: dict[tuple[str, ...], dict[str, str]] = {}

    for path in paths:
        _, rows = checked_rows(
            path, family, key_fields, values
        )

        for row in rows:
            identity = list(make_key(row, key_fields))

            if "team" in key_fields:
                position = key_fields.index("team")
                team = identities.get(
                    identity[position].upper()
                )

                if not team:
                    error(
                        f"Unmapped {family} source team "
                        f"{identity[position]!r}: {path}"
                    )

                identity[position] = team

            key = tuple(identity)

            if key in index and index[key] != row:
                error(
                    f"Conflicting {family} season/latest "
                    f"data for {key}"
                )

            index[key] = row

    return index


def audit_history() -> None:
    identities = team_map()

    indexed = {
        family: input_index(family, identities)
        for family in FAMILIES
    }

    columns = [
        "game_date",
        "game_id",
        "home_team",
        "away_team",
        "feature_family",
        "status",
        "source_records",
        "missing_source_fields",
        "missing_merged_fields",
        "unknown_statuses",
    ]

    audit_rows = []
    totals = Counter()
    failed = 0

    merged_files = sorted(
        MERGED.glob("*_NHL_merged.csv")
    )

    if not merged_files:
        error(
            f"Missing NHL Stage 01 merged files in {MERGED}"
        )

    for path in merged_files:
        header, games = read_csv(path)

        for required in (
            "game_id",
            "game_date",
            "home_team",
            "away_team",
        ):
            if required not in header:
                error(
                    f"Merged file missing {required}: {path}"
                )

        for game in games:
            game_id = game["game_id"].strip()
            date_key = norm_date(game["game_date"])
            home = game["home_team"].strip()
            away = game["away_team"].strip()

            for family in FAMILIES:
                _, _, _, value_fields = FAMILIES[family]

                if family == "fatigue":
                    sources = [
                        (
                            side,
                            indexed[family].get(
                                (date_key, team)
                            ),
                        )
                        for side, team in (
                            ("home", home),
                            ("away", away),
                        )
                    ]

                elif family == "team-strength":
                    sources = [
                        (
                            side,
                            indexed[family].get(
                                (game_id, team)
                            ),
                        )
                        for side, team in (
                            ("home", home),
                            ("away", away),
                        )
                    ]

                else:
                    source = indexed[family].get(
                        (game_id,)
                    )
                    sources = [("", source)]

                source_missing = []
                merged_missing = []
                records = 0

                for side, source in sources:
                    if source is None:
                        source_missing.append(
                            side or "game"
                        )
                        continue

                    records += 1

                    if family in (
                        "fatigue",
                        "team-strength",
                    ):
                        mappings = (
                            (col, f"{side}_{col}")
                            for col in value_fields
                        )
                    else:
                        mappings = (
                            (col, col)
                            for col in value_fields
                        )

                    for source_col, target_col in mappings:
                        source_value = (
                            source.get(source_col) or ""
                        ).strip()

                        merged_value = (
                            game.get(target_col) or ""
                        ).strip()

                        if not source_value:
                            # Unknown status legitimately has no
                            # observation timestamp.
                            status_column = (
                                source_col
                                .replace(
                                    "_goalie_status_observed_at",
                                    "_goalie_status",
                                )
                                .replace(
                                    "_lineup_observed_at",
                                    "_lineup_status",
                                )
                            )

                            is_unknown_observation = (
                                source_col.endswith(
                                    (
                                        "_lineup_observed_at",
                                        "_goalie_status_observed_at",
                                    )
                                )
                                and (
                                    source.get(status_column) or ""
                                ).strip().lower() == "unknown"
                            )

                            if not is_unknown_observation:
                                source_missing.append(
                                    f"{side + '_' if side else ''}"
                                    f"{source_col}"
                                )

                        elif not merged_value:
                            merged_missing.append(target_col)

                statuses = []

                if family == "goalie":
                    statuses = [
                        side
                        for side in ("home", "away")
                        if (
                            game.get(f"{side}_goalie_status") or ""
                        ).lower() == "unknown"
                    ]

                if family == "lineup-strength":
                    statuses = [
                        side
                        for side in ("home", "away")
                        if (
                            game.get(f"{side}_lineup_status") or ""
                        ).lower() == "unknown"
                    ]

                status = (
                    "MERGE_DATA_LOSS"
                    if merged_missing
                    else "SOURCE_MISSING"
                    if source_missing
                    else "COMPLETE"
                )

                if merged_missing:
                    failed += 1

                totals[(family, status)] += 1

                audit_rows.append({
                    "game_date": date_key,
                    "game_id": game_id,
                    "home_team": home,
                    "away_team": away,
                    "feature_family": family,
                    "status": status,
                    "source_records": records,
                    "missing_source_fields": ";".join(
                        source_missing
                    ),
                    "missing_merged_fields": ";".join(
                        merged_missing
                    ),
                    "unknown_statuses": ";".join(
                        statuses
                    ),
                })

    atomic_csv(AUDIT, columns, audit_rows)

    for record in audit_rows:
        if record["status"] == "MERGE_DATA_LOSS":
            print(
                f"SDV MERGE DATA LOSS: "
                f"game_id={record['game_id']} "
                f"family={record['feature_family']} "
                f"fields={record['missing_merged_fields']}",
                file=sys.stderr,
            )

    for family in FAMILIES:
        parts = ", ".join(
            f"{state}={totals[(family, state)]}"
            for state in (
                "COMPLETE",
                "SOURCE_MISSING",
                "MERGE_DATA_LOSS",
            )
        )
        print(
            f"SDV feature coverage {family}: {parts}"
        )

    print(
        f"Full per-game coverage audit: {AUDIT}"
    )

    if failed:
        error(
            f"{failed} game/family pairs lost "
            f"available source fields during merge; "
            f"see {AUDIT}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "action",
        choices=("prepare", "audit", "all"),
    )
    args = parser.parse_args()

    if not ORIGINAL_MERGE.is_file():
        error(
            f"Run from repository root; "
            f"cannot find {ORIGINAL_MERGE}"
        )

    if args.action in ("prepare", "all"):
        prepare_history()

    if args.action == "all":
        print(
            "Running original NHL merge_intake.py "
            "without model or odds changes"
        )
        subprocess.run(
            [sys.executable, str(ORIGINAL_MERGE)],
            check=True,
        )

    if args.action in ("audit", "all"):
        audit_history()

    print(
        f"NHL historical SDV feature action "
        f"{args.action}: PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(
            f"NHL SDV HISTORY ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
