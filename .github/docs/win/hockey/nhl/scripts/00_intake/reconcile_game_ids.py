#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/reconcile_game_ids.py

from __future__ import annotations

import csv
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# noinspection PyPep8
from team_map_common import (
    normalize_team_alias_key,
    parse_nhl_team_map_row,
    register_team_identity,
    strip_record,
)
from zoneinfo import ZoneInfo


BASE_DIR = Path("docs/win/hockey/nhl")
SPORTSBOOK_DIR = BASE_DIR / "00_intake" / "sportsbook"
PREDICTIONS_DIR = BASE_DIR / "00_intake" / "predictions"
NHL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
RECONCILED_DIR = BASE_DIR / "00_intake" / "reconciled"
AUDIT_DIR = RECONCILED_DIR / "audit"
MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nhl.csv"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "reconcile_game_ids.txt"

ET = ZoneInfo("America/New_York")

RECONCILED_COLUMNS = [
    "game_id",
    "sportsbook_event_id",
    "sport",
    "league",
    "game_date",
    "game_time",
    "home_team",
    "away_team",
]

AUDIT_COLUMNS = [
    "source",
    "source_file",
    "source_row",
    "sportsbook_event_id",
    "previous_game_id",
    "game_date",
    "away_team",
    "home_team",
    "schedule_match_count",
    "official_game_id",
    "orientation_corrected",
    "official_away_team",
    "official_home_team",
    "status",
]

RECONCILED_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    LOG_FILE.write_text(
        f"=== reconcile_game_ids RUN {datetime.now(ET).isoformat()} ===\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(ET).isoformat()} | {message}\n")


def normalize_date(value: str) -> str:
    text = str(value).strip().replace("-", "_").replace("/", "_")

    for fmt in ("%Y_%m_%d", "%m_%d_%Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y_%m_%d")
        except ValueError:
            continue

    return ""


def load_team_map() -> dict:
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing team mapping file: {MAP_PATH}"
        )

    by_source: dict[
        str,
        dict[str, dict[str, str]],
    ] = defaultdict(dict)
    by_id: dict[str, dict[str, str]] = {}
    by_abbrev: dict[str, dict[str, str]] = {}
    supported_sources = {
        "dratings",
        "sportsbook",
        "official_nhl",
        "shared",
        "final_scores",
    }

    with MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "league",
            "source",
            "alias",
            "canonical_team",
            "nhl_team_id",
            "nhl_abbrev",
        }
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)

        if missing:
            raise ValueError(
                f"{MAP_PATH} missing required columns: {missing}"
            )

        for row_number, row in enumerate(reader, start=2):
            parsed = parse_nhl_team_map_row(
                row,
                row_number,
                MAP_PATH,
                supported_sources=supported_sources,
            )
            if parsed is None:
                continue

            (
                source,
                alias,
                team_id,
                abbrev,
                identity,
            ) = parsed

            register_team_identity(
                by_id,
                team_id,
                identity,
                MAP_PATH,
            )

            if abbrev:
                prior_abbrev = by_abbrev.get(abbrev)
                if (
                    prior_abbrev is not None
                    and prior_abbrev != identity
                ):
                    raise ValueError(
                        f"{MAP_PATH} has conflicting identity for "
                        f"nhl_abbrev={abbrev}: "
                        f"{prior_abbrev} != {identity}"
                    )
                by_abbrev[abbrev] = identity

            key = normalize_team_alias_key(alias)
            prior_alias = by_source[source].get(key)

            if (
                prior_alias is not None
                and prior_alias != identity
            ):
                raise ValueError(
                    f"{MAP_PATH} has conflicting mapping for "
                    f"source={source} alias={alias!r}: "
                    f"{prior_alias} != {identity}"
                )

            by_source[source][key] = identity

    if not by_source.get("official_nhl"):
        raise ValueError(
            f"No official_nhl mappings loaded from {MAP_PATH}"
        )

    if not by_source.get("dratings"):
        raise ValueError(
            f"No dratings mappings loaded from {MAP_PATH}"
        )

    if not by_source.get("sportsbook"):
        raise ValueError(
            f"No sportsbook mappings loaded from {MAP_PATH}"
        )

    if len(by_id) != 32:
        raise ValueError(
            f"Expected 32 stable NHL team IDs in {MAP_PATH}; "
            f"found {len(by_id)}"
        )

    return {
        "by_source": dict(by_source),
        "by_id": by_id,
        "by_abbrev": by_abbrev,
    }


def mapping_source(source: str) -> str:
    if source == "prediction":
        return "dratings"
    if source == "sportsbook":
        return "sportsbook"
    if source == "official_nhl":
        return "official_nhl"
    raise ValueError(f"Unknown team mapping source: {source}")


def resolve_team_identity(
    value: str,
    source: str,
    team_map: dict,
) -> dict[str, str] | None:
    raw = strip_record(value)

    if not raw:
        return None

    key = normalize_team_alias_key(raw)
    source_key = mapping_source(source)
    by_source = team_map["by_source"]

    for candidate_source in (
        source_key,
        "shared",
        "official_nhl",
    ):
        identity = by_source.get(
            candidate_source,
            {},
        ).get(key)

        if identity is not None:
            return identity

    return None


def canonical_team(
    value: str,
    source: str,
    team_map: dict,
) -> str:
    raw = strip_record(value)

    if not raw:
        return ""

    identity = resolve_team_identity(
        raw,
        source,
        team_map,
    )

    if identity is None:
        return raw

    return identity["canonical_team"]


def team_match_key(
    value: str,
    source: str,
    team_map: dict,
) -> str:
    identity = resolve_team_identity(
        value,
        source,
        team_map,
    )

    if identity is None:
        return ""

    return identity["nhl_team_id"]


def official_schedule_identity(
    row: dict[str, str],
    side: str,
    team_map: dict,
    path: Path,
    row_number: int,
) -> dict[str, str]:
    team_id = str(
        row.get(f"{side}_team_id", "")
    ).strip()
    abbrev = str(
        row.get(f"{side}_team_abbrev", "")
    ).strip().upper()
    name = str(
        row.get(f"{side}_team", "")
    ).strip()

    if not team_id or not abbrev or not name:
        raise ValueError(
            f"{path} row {row_number} has incomplete official "
            f"{side} team identity"
        )

    identity_by_id = team_map["by_id"].get(team_id)
    identity_by_abbrev = team_map["by_abbrev"].get(abbrev)
    identity_by_name = resolve_team_identity(
        name,
        "official_nhl",
        team_map,
    )

    if identity_by_id is None:
        raise ValueError(
            f"{path} row {row_number} has unmapped official "
            f"{side}_team_id={team_id}"
        )

    if identity_by_abbrev is None:
        raise ValueError(
            f"{path} row {row_number} has unmapped official "
            f"{side}_team_abbrev={abbrev}"
        )

    if identity_by_name is None:
        raise ValueError(
            f"{path} row {row_number} has unmapped official "
            f"{side}_team={name!r}"
        )

    if not (
        identity_by_id
        == identity_by_abbrev
        == identity_by_name
    ):
        raise ValueError(
            f"{path} row {row_number} official {side} identity "
            f"disagrees across id/abbrev/name"
        )

    return identity_by_id

def load_csv(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [
            dict(row)
            for row in reader
        ]

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

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_field_order(
    fieldnames: list[str],
    leading: list[str],
) -> list[str]:
    ordered: list[str] = []

    for field in leading:
        if field not in ordered:
            ordered.append(field)

    for field in fieldnames:
        if field not in ordered:
            ordered.append(field)

    return ordered


def source_date_from_filename(
    path: Path,
    prefix: str,
) -> str:
    stem = path.stem

    if not stem.lower().startswith(
        prefix.lower()
    ):
        return ""

    return normalize_date(
        stem[len(prefix) :]
    )


def schedule_path_for_date(
    date_value: str,
) -> Path:
    return (
        NHL_SCHEDULE_DIR
        / f"NHL_{date_value}.csv"
    )


def load_schedule_for_date(
    date_value: str,
    team_map: dict,
) -> tuple[
    list[dict[str, str]],
    dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
]:
    path = schedule_path_for_date(
        date_value
    )

    if not path.exists():
        return [], {}

    fieldnames, rows = load_csv(path)

    required = [
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "home_team_abbrev",
        "away_team_abbrev",
        "home_team_id",
        "away_team_id",
    ]

    missing = [
        field
        for field in required
        if field not in fieldnames
    ]

    if missing:
        raise ValueError(
            f"{path} missing required columns: "
            f"{missing}"
        )

    index: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    seen_game_ids: set[str] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        game_id = str(
            row.get("game_id", "")
        ).strip()

        game_date = normalize_date(
            row.get("game_date", "")
        )

        home_identity = official_schedule_identity(
            row,
            "home",
            team_map,
            path,
            row_number,
        )

        away_identity = official_schedule_identity(
            row,
            "away",
            team_map,
            path,
            row_number,
        )

        home_key = home_identity["nhl_team_id"]
        away_key = away_identity["nhl_team_id"]

        if (
            not game_id
            or not game_date
            or not home_key
            or not away_key
        ):
            raise ValueError(
                f"{path} row {row_number} "
                "has incomplete official "
                "schedule identity"
            )

        if game_id in seen_game_ids:
            raise ValueError(
                f"{path} contains duplicate "
                f"official game_id={game_id}"
            )

        seen_game_ids.add(game_id)

        row["game_date"] = game_date
        row["home_team"] = home_identity["canonical_team"]
        row["away_team"] = away_identity["canonical_team"]
        row["home_team_id"] = home_key
        row["away_team_id"] = away_key
        row["home_team_abbrev"] = home_identity["nhl_abbrev"]
        row["away_team_abbrev"] = away_identity["nhl_abbrev"]

        index[
            (
                game_date,
                home_key,
                away_key,
            )
        ].append(row)

    return rows, index


def source_match_key(
    row: dict[str, str],
    source: str,
    team_map: dict,
) -> tuple[str, str, str] | None:
    game_date = normalize_date(
        row.get("game_date", "")
    )

    home_key = team_match_key(
        row.get("home_team", ""),
        source,
        team_map,
    )

    away_key = team_match_key(
        row.get("away_team", ""),
        source,
        team_map,
    )

    if (
        not game_date
        or not home_key
        or not away_key
    ):
        return None

    return (
        game_date,
        home_key,
        away_key,
    )

def reverse_match_key(
    key: tuple[str, str, str],
) -> tuple[str, str, str]:
    game_date, home_key, away_key = key

    return (
        game_date,
        away_key,
        home_key,
    )


def swap_home_away_fields(
    row: dict[str, str],
) -> dict[str, str]:
    swapped = dict(row)

    for field in list(row):
        if not field.startswith("home_"):
            continue

        suffix = field[len("home_") :]
        away_field = f"away_{suffix}"

        if away_field not in row:
            continue

        swapped[field] = row.get(
            away_field,
            "",
        )

        swapped[away_field] = row.get(
            field,
            "",
        )

    return swapped


def audit_row(
    *,
    source: str,
    source_file: Path,
    source_row: int,
    row: dict[str, str],
    sportsbook_event_id: str,
    schedule_match_count: int,
    official_game_id: str,
    status: str,
    orientation_corrected: bool = False,
    official_away_team: str = "",
    official_home_team: str = "",
) -> dict[str, str]:
    return {
        "source": source,
        "source_file": str(source_file),
        "source_row": str(source_row),
        "sportsbook_event_id": (
            sportsbook_event_id
        ),
        "previous_game_id": str(
            row.get("game_id", "")
        ).strip(),
        "game_date": (
            normalize_date(
                row.get("game_date", "")
            )
            or str(
                row.get(
                    "game_date",
                    "",
                )
            ).strip()
        ),
        "away_team": str(
            row.get("away_team", "")
        ).strip(),
        "home_team": str(
            row.get("home_team", "")
        ).strip(),
        "schedule_match_count": str(
            schedule_match_count
        ),
        "official_game_id": (
            official_game_id
        ),
        "orientation_corrected": (
            "yes"
            if orientation_corrected
            else "no"
        ),
        "official_away_team": (
            official_away_team
        ),
        "official_home_team": (
            official_home_team
        ),
        "status": status,
    }


def match_source_rows_to_schedule(
    *,
    source: str,
    source_file: Path,
    rows: list[dict[str, str]],
    schedule_index: dict[
        tuple[str, str, str],
        list[dict[str, str]],
    ],
    team_map: dict[str, str],
) -> tuple[
    list[dict],
    list[dict[str, str]],
]:
    matched: list[dict] = []
    audits: list[dict[str, str]] = []

    seen_sportsbook_event_ids: set[str] = (
        set()
    )

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        sportsbook_event_id = ""

        if source == "sportsbook":
            sportsbook_event_id = str(
                row.get(
                    "sportsbook_event_id"
                )
                or row.get("game_id")
                or ""
            ).strip()

            if not sportsbook_event_id:
                audits.append(
                    audit_row(
                        source=source,
                        source_file=source_file,
                        source_row=row_number,
                        row=row,
                        sportsbook_event_id="",
                        schedule_match_count=0,
                        official_game_id="",
                        status=(
                            "missing_"
                            "sportsbook_event_id"
                        ),
                    )
                )
                continue

            if (
                sportsbook_event_id
                in seen_sportsbook_event_ids
            ):
                audits.append(
                    audit_row(
                        source=source,
                        source_file=source_file,
                        source_row=row_number,
                        row=row,
                        sportsbook_event_id=(
                            sportsbook_event_id
                        ),
                        schedule_match_count=0,
                        official_game_id="",
                        status=(
                            "duplicate_"
                            "sportsbook_event_id"
                        ),
                    )
                )
                continue

            seen_sportsbook_event_ids.add(
                sportsbook_event_id
            )

        key = source_match_key(
            row,
            source,
            team_map,
        )

        if key is None:
            audits.append(
                audit_row(
                    source=source,
                    source_file=source_file,
                    source_row=row_number,
                    row=row,
                    sportsbook_event_id=(
                        sportsbook_event_id
                    ),
                    schedule_match_count=0,
                    official_game_id="",
                    status=(
                        "invalid_date_or_"
                        "team_identity"
                    ),
                )
            )
            continue

        schedule_matches = (
            schedule_index.get(
                key,
                [],
            )
        )

        orientation_corrected = False

        if len(schedule_matches) == 0:
            reversed_matches = (
                schedule_index.get(
                    reverse_match_key(key),
                    [],
                )
            )

            if len(reversed_matches) == 1:
                schedule_matches = (
                    reversed_matches
                )
                orientation_corrected = True

            elif len(reversed_matches) > 1:
                audits.append(
                    audit_row(
                        source=source,
                        source_file=source_file,
                        source_row=row_number,
                        row=row,
                        sportsbook_event_id=(
                            sportsbook_event_id
                        ),
                        schedule_match_count=len(
                            reversed_matches
                        ),
                        official_game_id="",
                        status=(
                            "multiple_official_"
                            "schedule_matches"
                        ),
                    )
                )
                continue

        if len(schedule_matches) != 1:
            status = (
                "no_official_schedule_match"
                if len(schedule_matches) == 0
                else (
                    "multiple_official_"
                    "schedule_matches"
                )
            )

            audits.append(
                audit_row(
                    source=source,
                    source_file=source_file,
                    source_row=row_number,
                    row=row,
                    sportsbook_event_id=(
                        sportsbook_event_id
                    ),
                    schedule_match_count=len(
                        schedule_matches
                    ),
                    official_game_id="",
                    status=status,
                )
            )
            continue

        schedule_row = schedule_matches[0]

        official_game_id = str(
            schedule_row.get(
                "game_id",
                "",
            )
        ).strip()

        official_away_team = canonical_team(
            schedule_row.get(
                "away_team",
                "",
            ),
            "official_nhl",
            team_map,
        )

        official_home_team = canonical_team(
            schedule_row.get(
                "home_team",
                "",
            ),
            "official_nhl",
            team_map,
        )

        matched.append(
            {
                "source": source,
                "source_file": source_file,
                "source_row": row_number,
                "row": row,
                "sportsbook_event_id": (
                    sportsbook_event_id
                ),
                "official_game_id": (
                    official_game_id
                ),
                "schedule_row": schedule_row,
                "orientation_corrected": (
                    orientation_corrected
                ),
            }
        )

        audits.append(
            audit_row(
                source=source,
                source_file=source_file,
                source_row=row_number,
                row=row,
                sportsbook_event_id=(
                    sportsbook_event_id
                ),
                schedule_match_count=1,
                official_game_id=(
                    official_game_id
                ),
                orientation_corrected=(
                    orientation_corrected
                ),
                official_away_team=(
                    official_away_team
                ),
                official_home_team=(
                    official_home_team
                ),
                status=(
                    "official_schedule_match"
                ),
            )
        )

    return matched, audits


def mark_final_status(
    audits: list[dict[str, str]],
    source: str,
    source_row: int,
    status: str,
    sportsbook_event_id: str = "",
) -> None:
    for row in audits:
        if row["source"] != source:
            continue

        if (
            row["source_row"]
            != str(source_row)
        ):
            continue

        row["status"] = status

        if sportsbook_event_id:
            row[
                "sportsbook_event_id"
            ] = sportsbook_event_id

        return


def reconcile_date(
    date_value: str,
    sportsbook_path: Path | None,
    prediction_path: Path | None,
    team_map: dict[str, str],
) -> dict:
    schedule_rows, schedule_index = (
        load_schedule_for_date(
            date_value,
            team_map,
        )
    )

    sportsbook_fieldnames: list[str] = []
    sportsbook_rows: list[
        dict[str, str]
    ] = []

    prediction_fieldnames: list[str] = []
    prediction_rows: list[
        dict[str, str]
    ] = []

    if sportsbook_path is not None:
        (
            sportsbook_fieldnames,
            sportsbook_rows,
        ) = load_csv(
            sportsbook_path
        )

    if prediction_path is not None:
        (
            prediction_fieldnames,
            prediction_rows,
        ) = load_csv(
            prediction_path
        )

    sportsbook_matched: list[dict] = []
    prediction_matched: list[dict] = []
    audits: list[dict[str, str]] = []

    if sportsbook_path is not None:
        (
            sportsbook_matched,
            sportsbook_audits,
        ) = match_source_rows_to_schedule(
            source="sportsbook",
            source_file=sportsbook_path,
            rows=sportsbook_rows,
            schedule_index=schedule_index,
            team_map=team_map,
        )

        audits.extend(
            sportsbook_audits
        )

    if prediction_path is not None:
        (
            prediction_matched,
            prediction_audits,
        ) = match_source_rows_to_schedule(
            source="prediction",
            source_file=prediction_path,
            rows=prediction_rows,
            schedule_index=schedule_index,
            team_map=team_map,
        )

        audits.extend(
            prediction_audits
        )

    sportsbook_by_game: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    prediction_by_game: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for item in sportsbook_matched:
        sportsbook_by_game[
            item["official_game_id"]
        ].append(item)

    for item in prediction_matched:
        prediction_by_game[
            item["official_game_id"]
        ].append(item)

    referenced_game_ids = sorted(
        set(sportsbook_by_game)
        | set(prediction_by_game)
    )

    reconciled_games: list[dict] = []
    reconciled_sportsbook: list[
        dict[str, str]
    ] = []
    reconciled_predictions: list[
        dict[str, str]
    ] = []

    for game_id in referenced_game_ids:
        sportsbook_items = (
            sportsbook_by_game.get(
                game_id,
                [],
            )
        )

        prediction_items = (
            prediction_by_game.get(
                game_id,
                [],
            )
        )

        if (
            len(sportsbook_items) != 1
            or len(prediction_items) != 1
        ):
            if len(sportsbook_items) == 0:
                for item in prediction_items:
                    mark_final_status(
                        audits,
                        "prediction",
                        item["source_row"],
                        (
                            "missing_sportsbook_"
                            "for_official_game"
                        ),
                    )

            elif len(sportsbook_items) > 1:
                for item in sportsbook_items:
                    mark_final_status(
                        audits,
                        "sportsbook",
                        item["source_row"],
                        (
                            "multiple_"
                            "sportsbook_rows_"
                            "for_official_game"
                        ),
                    )

                for item in prediction_items:
                    mark_final_status(
                        audits,
                        "prediction",
                        item["source_row"],
                        (
                            "multiple_"
                            "sportsbook_rows_"
                            "for_official_game"
                        ),
                    )

            if len(prediction_items) == 0:
                for item in sportsbook_items:
                    mark_final_status(
                        audits,
                        "sportsbook",
                        item["source_row"],
                        (
                            "missing_prediction_"
                            "for_official_game"
                        ),
                    )

            elif len(prediction_items) > 1:
                for item in prediction_items:
                    mark_final_status(
                        audits,
                        "prediction",
                        item["source_row"],
                        (
                            "multiple_"
                            "prediction_rows_"
                            "for_official_game"
                        ),
                    )

                for item in sportsbook_items:
                    mark_final_status(
                        audits,
                        "sportsbook",
                        item["source_row"],
                        (
                            "multiple_"
                            "prediction_rows_"
                            "for_official_game"
                        ),
                    )

            continue

        sportsbook_item = (
            sportsbook_items[0]
        )

        prediction_item = (
            prediction_items[0]
        )

        schedule_row = (
            sportsbook_item[
                "schedule_row"
            ]
        )

        sportsbook_event_id = (
            sportsbook_item[
                "sportsbook_event_id"
            ]
        )

        canonical_home = canonical_team(
            schedule_row.get(
                "home_team",
                "",
            ),
            "official_nhl",
            team_map,
        )

        canonical_away = canonical_team(
            schedule_row.get(
                "away_team",
                "",
            ),
            "official_nhl",
            team_map,
        )

        official_time = str(
            schedule_row.get(
                "game_time",
                "",
            )
        ).strip()

        sportsbook_source_row = dict(
            sportsbook_item["row"]
        )

        if sportsbook_item[
            "orientation_corrected"
        ]:
            sportsbook_source_row = (
                swap_home_away_fields(
                    sportsbook_source_row
                )
            )

        prediction_source_row = dict(
            prediction_item["row"]
        )

        if prediction_item[
            "orientation_corrected"
        ]:
            prediction_source_row = (
                swap_home_away_fields(
                    prediction_source_row
                )
            )

        sportsbook_output = dict(
            sportsbook_source_row
        )

        sportsbook_output[
            "game_id"
        ] = game_id

        sportsbook_output[
            "sportsbook_event_id"
        ] = sportsbook_event_id

        sportsbook_output[
            "sport"
        ] = "hockey"

        sportsbook_output[
            "league"
        ] = "nhl"

        sportsbook_output[
            "game_date"
        ] = date_value

        sportsbook_output[
            "game_time"
        ] = official_time

        sportsbook_output[
            "home_team"
        ] = canonical_home

        sportsbook_output[
            "away_team"
        ] = canonical_away

        prediction_output = dict(
            prediction_source_row
        )

        prediction_output[
            "game_id"
        ] = game_id

        prediction_output[
            "sport"
        ] = "hockey"

        prediction_output[
            "league"
        ] = "nhl"

        prediction_output[
            "game_date"
        ] = date_value

        prediction_output[
            "game_time"
        ] = official_time

        prediction_output[
            "home_team"
        ] = canonical_home

        prediction_output[
            "away_team"
        ] = canonical_away

        reconciled_sportsbook.append(
            sportsbook_output
        )

        reconciled_predictions.append(
            prediction_output
        )

        reconciled_games.append(
            {
                "game_id": game_id,
                "sportsbook_event_id": (
                    sportsbook_event_id
                ),
                "sport": "hockey",
                "league": "nhl",
                "game_date": date_value,
                "game_time": official_time,
                "home_team": canonical_home,
                "away_team": canonical_away,
            }
        )

        mark_final_status(
            audits,
            "sportsbook",
            sportsbook_item[
                "source_row"
            ],
            "reconciled",
            sportsbook_event_id=(
                sportsbook_event_id
            ),
        )

        mark_final_status(
            audits,
            "prediction",
            prediction_item[
                "source_row"
            ],
            "reconciled",
            sportsbook_event_id=(
                sportsbook_event_id
            ),
        )

    failure_rows = [
        row
        for row in audits
        if row["status"] != "reconciled"
    ]

    orientation_corrections = [
        row
        for row in audits
        if (
            row["status"] == "reconciled"
            and row[
                "orientation_corrected"
            ]
            == "yes"
        )
    ]

    return {
        "date": date_value,
        "schedule_rows": schedule_rows,
        "sportsbook_path": sportsbook_path,
        "prediction_path": prediction_path,
        "sportsbook_fieldnames": (
            sportsbook_fieldnames
        ),
        "prediction_fieldnames": (
            prediction_fieldnames
        ),
        "sportsbook_input_count": len(
            sportsbook_rows
        ),
        "prediction_input_count": len(
            prediction_rows
        ),
        "reconciled_sportsbook": (
            reconciled_sportsbook
        ),
        "reconciled_predictions": (
            reconciled_predictions
        ),
        "reconciled_games": (
            reconciled_games
        ),
        "audits": audits,
        "failure_rows": failure_rows,
        "orientation_corrections": (
            orientation_corrections
        ),
    }


def register_source_file(
    files: dict[str, Path],
    date_value: str,
    path: Path,
    source_name: str,
) -> None:
    existing = files.get(
        date_value
    )

    if (
        existing is not None
        and existing != path
    ):
        raise ValueError(
            f"Multiple {source_name} files "
            f"found for {date_value}: "
            f"{existing} and {path}"
        )

    files[date_value] = path


def discover_source_files() -> tuple[
    dict[str, Path],
    dict[str, Path],
]:
    sportsbook_files: dict[
        str,
        Path,
    ] = {}

    prediction_files: dict[
        str,
        Path,
    ] = {}

    if SPORTSBOOK_DIR.exists():
        for path in sorted(
            SPORTSBOOK_DIR.glob(
                "*.csv"
            )
        ):
            if not path.stem.lower().startswith(
                "nhl_"
            ):
                continue

            date_value = (
                source_date_from_filename(
                    path,
                    "NHL_",
                )
            )

            if date_value:
                register_source_file(
                    sportsbook_files,
                    date_value,
                    path,
                    "sportsbook",
                )

    if PREDICTIONS_DIR.exists():
        for path in sorted(
            PREDICTIONS_DIR.glob(
                "*.csv"
            )
        ):
            if not path.stem.lower().startswith(
                "hockey_"
            ):
                continue

            date_value = (
                source_date_from_filename(
                    path,
                    "hockey_",
                )
            )

            if date_value:
                register_source_file(
                    prediction_files,
                    date_value,
                    path,
                    "prediction",
                )

    return (
        sportsbook_files,
        prediction_files,
    )


def write_audit(
    result: dict,
) -> Path:
    date_value = result["date"]

    path = (
        AUDIT_DIR
        / (
            f"NHL_{date_value}"
            "_reconciliation.csv"
        )
    )

    write_csv(
        path,
        AUDIT_COLUMNS,
        result["audits"],
    )

    return path


def commit_reconciled_result(
    result: dict,
) -> None:
    date_value = result["date"]

    sportsbook_path: Path | None = (
        result["sportsbook_path"]
    )

    prediction_path: Path | None = (
        result["prediction_path"]
    )

    if sportsbook_path is not None:
        sportsbook_fields = (
            ensure_field_order(
                result[
                    "sportsbook_fieldnames"
                ],
                [
                    "game_id",
                    "sportsbook_event_id",
                ],
            )
        )

        write_csv(
            sportsbook_path,
            sportsbook_fields,
            result[
                "reconciled_sportsbook"
            ],
        )

    if prediction_path is not None:
        prediction_fields = (
            ensure_field_order(
                result[
                    "prediction_fieldnames"
                ],
                ["game_id"],
            )
        )

        write_csv(
            prediction_path,
            prediction_fields,
            result[
                "reconciled_predictions"
            ],
        )

    reconciled_path = (
        RECONCILED_DIR
        / f"NHL_{date_value}.csv"
    )

    write_csv(
        reconciled_path,
        RECONCILED_COLUMNS,
        result["reconciled_games"],
    )


def main() -> None:
    reset_log()

    try:
        team_map = load_team_map()

        log(
            "Stable NHL team identities loaded: "
            f"{len(team_map['by_id'])}"
        )

        (
            sportsbook_files,
            prediction_files,
        ) = discover_source_files()

        dates = sorted(
            set(sportsbook_files)
            | set(prediction_files)
        )

        log(
            "Sportsbook files found: "
            f"{len(sportsbook_files)}"
        )

        log(
            "Prediction files found: "
            f"{len(prediction_files)}"
        )

        log(
            "Dates to reconcile: "
            f"{len(dates)}"
        )

        if not dates:
            raise RuntimeError(
                "No sportsbook or prediction "
                "files found to reconcile"
            )

        results: list[dict] = []
        total_failures = 0

        for date_value in dates:
            result = reconcile_date(
                date_value=date_value,
                sportsbook_path=(
                    sportsbook_files.get(
                        date_value
                    )
                ),
                prediction_path=(
                    prediction_files.get(
                        date_value
                    )
                ),
                team_map=team_map,
            )

            results.append(result)

            audit_path = write_audit(
                result
            )

            total_failures += len(
                result["failure_rows"]
            )

            log(
                f"DATE {date_value} "
                f"| official_schedule="
                f"{len(result['schedule_rows'])} "
                f"| sportsbook="
                f"{result['sportsbook_input_count']} "
                f"| predictions="
                f"{result['prediction_input_count']} "
                f"| reconciled="
                f"{len(result['reconciled_games'])} "
                f"| orientation_corrections="
                f"{len(result['orientation_corrections'])} "
                f"| failures="
                f"{len(result['failure_rows'])} "
                f"| audit={audit_path}"
            )

            for correction in result[
                "orientation_corrections"
            ]:
                log(
                    "ORIENTATION CORRECTED | "
                    f"source="
                    f"{correction['source']} | "
                    f"file="
                    f"{correction['source_file']} | "
                    f"row="
                    f"{correction['source_row']} | "
                    f"date="
                    f"{correction['game_date']} | "
                    f"original_away="
                    f"{correction['away_team']} | "
                    f"original_home="
                    f"{correction['home_team']} | "
                    f"official_away="
                    f"{correction['official_away_team']} | "
                    f"official_home="
                    f"{correction['official_home_team']} | "
                    f"official_game_id="
                    f"{correction['official_game_id']}"
                )

            for failure in result[
                "failure_rows"
            ]:
                log(
                    "REJECTED | "
                    f"source="
                    f"{failure['source']} | "
                    f"file="
                    f"{failure['source_file']} | "
                    f"row="
                    f"{failure['source_row']} | "
                    f"date="
                    f"{failure['game_date']} | "
                    f"away="
                    f"{failure['away_team']} | "
                    f"home="
                    f"{failure['home_team']} | "
                    f"sportsbook_event_id="
                    f"{failure['sportsbook_event_id']} | "
                    f"status="
                    f"{failure['status']}"
                )

        if total_failures:
            log("--- SUMMARY ---")

            log(
                "Dates processed: "
                f"{len(results)}"
            )

            log(
                "Rejected/unreconciled "
                "source rows: "
                f"{total_failures}"
            )

            log(
                "No sportsbook, prediction, "
                "or canonical reconciled files "
                "were rewritten."
            )

            log("STATUS: FAILED")

            raise SystemExit(
                "NHL game ID reconciliation "
                f"failed for {total_failures} "
                "source rows. See "
                "reconciliation audit files."
            )

        total_games = 0
        total_orientation_corrections = 0

        for result in results:
            commit_reconciled_result(
                result
            )

            total_games += len(
                result["reconciled_games"]
            )

            total_orientation_corrections += (
                len(
                    result[
                        "orientation_corrections"
                    ]
                )
            )

        log("--- SUMMARY ---")

        log(
            "Dates processed: "
            f"{len(results)}"
        )

        log(
            "Exactly reconciled games: "
            f"{total_games}"
        )

        log(
            "Home/away source rows "
            "corrected to official NHL "
            "orientation: "
            f"{total_orientation_corrections}"
        )

        log(
            "All sportsbook game_id "
            "values are official NHL "
            "game IDs."
        )

        log(
            "All sportsbook provider IDs "
            "are preserved as "
            "sportsbook_event_id."
        )

        log(
            "All D-Ratings prediction "
            "game_id values are official "
            "NHL game IDs."
        )

        log("STATUS: SUCCESS")

        print(
            "NHL game ID reconciliation "
            "complete."
        )

    except SystemExit:
        raise

    except Exception as exc:
        log(
            f"FATAL ERROR: {exc}"
        )

        log(
            traceback.format_exc()
        )

        log("STATUS: FAILED")
        raise


if __name__ == "__main__":
    main()