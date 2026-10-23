#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_raw_roster.py

"""
Pull configured-season CFB rosters from ESPN and publish one validated
raw roster CSV for the pipeline's accepted team universe.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/team_master.csv

Output:
    docs/win/football/cfb/data/raw/raw_roster.csv
"""

from __future__ import annotations

from http.client import HTTPException

import csv
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request



SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    load_current_week_config,
    write_csv_rows_durable,
)
from type_support import ScalarValue


CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

TEAM_MASTER_PATH = (
    CFB_ROOT
    / "data"
    / "master"
    / "team_master.csv"
)

OUTPUT_PATH = (
    CFB_ROOT
    / "data"
    / "raw"
    / "raw_roster.csv"
)

REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = (
    "cfb-raw-roster-v2-2026-09-15"
)

ROSTER_URL_TEMPLATE = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/college-football/"
    "teams/{team_id}/roster"
)
ESPN_SITE_HOST = "site.api.espn.com"

REQUIRED_OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "team_id",
    "id",
    "displayName",
]

ATHLETE_TEAM_SEASON_REF_PATTERN = re.compile(
    r"/seasons/(\d+)/teams/"
)

ATHLETE_TEAM_REF_KEY_PATTERN = re.compile(
    r"^teams\.\d+\.\$ref$"
)

@dataclass
class RuntimeState:
    request_count: int = 0
    request_failures: list[dict[str, str]] = field(
        default_factory=list
    )


class RosterValidationError(
    RuntimeError
):
    pass


def load_target_team_ids() -> list[str]:
    if not TEAM_MASTER_PATH.exists():
        raise FileNotFoundError(
            "Missing team master: "
            f"{TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        if "team_id" not in fieldnames:
            raise ValueError(
                "team_master.csv missing "
                "required column: team_id"
            )

        team_ids: set[str] = set()

        for index, row in enumerate(
            reader
        ):
            team_id = str(
                row.get(
                    "team_id",
                    "",
                )
            ).strip()

            if not team_id:
                raise ValueError(
                    "team_master.csv contains "
                    "blank team_id at row "
                    f"{index}"
                )

            if (
                not team_id.isdigit()
                or int(team_id) <= 0
            ):
                raise ValueError(
                    "team_master.csv contains "
                    "invalid team_id="
                    f"{team_id!r}"
                )

            team_ids.add(
                team_id
            )

    if not team_ids:
        raise ValueError(
            "team_master.csv contains "
            "no accepted team IDs"
        )

    return sorted(
        team_ids,
        key=int,
    )


def roster_url(
    team_id: str,
    season: int,
) -> str:
    base = (
        ROSTER_URL_TEMPLATE.format(
            team_id=team_id
        )
    )

    return (
        f"{base}?"
        + urlencode(
            {
                "season": season,
            }
        )
    )


def fetch_json(
    url: str,
    *,
    label: str,
    state: RuntimeState,
    timeout: int = 30,
) -> dict:
    state.request_count += 1

    request = Request(
        url,
        headers={
            "User-Agent": (
                "cfb-pull-raw-roster/2.0"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_SITE_HOST},
            timeout=timeout,
        ) as response:
            status = response.status

            body = (
                response.read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        error_body = ""

        try:
            error_body = (
                exc.read()
                .decode("utf-8")
            )
        except (HTTPException, OSError, UnicodeError, ValueError):
            pass

        failure = {
            "label": label,
            "url": url,
            "http_status": str(
                exc.code
            ),
            "error": (
                error_body
                or str(exc)
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} request failed: "
            f"status={exc.code}, "
            f"error={failure['error']}"
        ) from exc

    except URLError as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} request failed: "
            f"{exc}"
        ) from exc

    except Exception as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} request failed: "
            f"{exc}"
        ) from exc

    if (
        status < 200
        or status >= 300
    ):
        failure = {
            "label": label,
            "url": url,
            "http_status": str(
                status
            ),
            "error": body,
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} request failed: "
            f"status={status}"
        )

    try:
        payload = json.loads(
            body
        )
    except Exception as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                f"JSON parse failed: {exc}"
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} returned malformed JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        failure = {
            "label": label,
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                "JSON root is not an object"
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"{label} returned non-object JSON"
        )

    return payload


def flatten(
    obj: object,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, ScalarValue]:
    items: dict[
        str,
        object,
    ] = {}

    if isinstance(
        obj,
        dict,
    ):
        for key, value in obj.items():
            new_key = (
                f"{parent_key}{sep}{key}"
                if parent_key
                else str(key)
            )

            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )

    elif isinstance(
        obj,
        list,
    ):
        for index, value in enumerate(
            obj
        ):
            new_key = (
                f"{parent_key}{sep}{index}"
                if parent_key
                else str(index)
            )

            items.update(
                flatten(
                    value,
                    new_key,
                    sep,
                )
            )

    else:
        if not (
            obj is None
            or isinstance(
                obj,
                (str, int, float, bool),
            )
        ):
            raise TypeError(
                "Unexpected non-scalar roster value "
                f"at {parent_key!r}: {type(obj).__name__}"
            )

        if parent_key:
            items[
                parent_key
            ] = obj

    return items


def payload_season_year(
    payload: dict,
) -> int | None:
    season = payload.get(
        "season"
    )

    if isinstance(
        season,
        dict,
    ):
        raw = (
            season.get("year")
            or season.get("season")
        )

        if raw is not None:
            try:
                return int(
                    str(raw).strip()
                )
            except ValueError as exc:
                raise RosterValidationError(
                    "Roster payload season "
                    f"metadata is invalid: {raw!r}"
                ) from exc

    elif season is not None:
        try:
            return int(
                str(season).strip()
            )
        except ValueError:
            pass

    return None


def payload_team_id(
    payload: dict,
) -> str:
    team = payload.get(
        "team"
    )

    if not isinstance(
        team,
        dict,
    ):
        return ""

    return str(
        team.get(
            "id",
            "",
        )
    ).strip()


def extract_athletes(
    payload: dict,
    *,
    team_id: str,
    season: int,
) -> tuple[
    list[dict],
    bool,
]:
    explicit_season = (
        payload_season_year(
            payload
        )
    )

    if (
        explicit_season is not None
        and explicit_season != season
    ):
        raise RosterValidationError(
            "Roster payload season mismatch "
            f"for team_id={team_id}: "
            f"expected={season}, "
            f"actual={explicit_season}"
        )

    response_team_id = (
        payload_team_id(
            payload
        )
    )

    if (
        response_team_id
        and response_team_id != team_id
    ):
        raise RosterValidationError(
            "Roster payload team identity "
            f"mismatch for team_id={team_id}: "
            f"response_team_id="
            f"{response_team_id}"
        )

    groups = payload.get(
        "athletes"
    )

    if not isinstance(
        groups,
        list,
    ):
        raise RosterValidationError(
            "Roster payload athletes field "
            "is not a list for "
            f"team_id={team_id}"
        )

    athletes: list[dict] = []

    for group_index, group in enumerate(
        groups
    ):
        if not isinstance(
            group,
            dict,
        ):
            raise RosterValidationError(
                "Roster athletes contains "
                "non-object group for "
                f"team_id={team_id}, "
                f"group_index={group_index}"
            )

        items = group.get(
            "items"
        )

        if not isinstance(
            items,
            list,
        ):
            raise RosterValidationError(
                "Roster group items field "
                "is not a list for "
                f"team_id={team_id}, "
                f"group_index={group_index}"
            )

        for athlete_index, athlete in enumerate(
            items
        ):
            if not isinstance(
                athlete,
                dict,
            ):
                raise RosterValidationError(
                    "Roster group contains "
                    "non-object athlete for "
                    f"team_id={team_id}, "
                    f"group_index={group_index}, "
                    f"athlete_index={athlete_index}"
                )

            athletes.append(
                athlete
            )

    if not athletes:
        raise RosterValidationError(
            "Roster payload contains no "
            "athletes for "
            f"team_id={team_id}"
        )

    return (
        athletes,
        explicit_season is not None,
    )


def validate_athlete_season_refs(
    row: dict[str, ScalarValue],
    *,
    team_id: str,
    athlete_id: str,
    season: int,
) -> bool:
    observed_seasons: set[int] = set()

    for key, value in row.items():
        if not ATHLETE_TEAM_REF_KEY_PATTERN.match(
            str(key)
        ):
            continue

        text = str(
            value or ""
        )

        match = (
            ATHLETE_TEAM_SEASON_REF_PATTERN.search(
                text
            )
        )

        if match:
            observed_seasons.add(
                int(
                    match.group(1)
                )
            )

    if not observed_seasons:
        return False

    if observed_seasons != {
        season
    }:
        raise RosterValidationError(
            "Athlete team references contain "
            "season mismatch for "
            f"team_id={team_id}, "
            f"athlete_id={athlete_id}: "
            f"expected={season}, "
            f"observed="
            f"{sorted(observed_seasons)}"
        )

    return True


def build_raw_rows(
    target_team_ids: list[str],
    season: int,
    season_type: int,
    state: RuntimeState,
) -> tuple[
    list[dict[str, ScalarValue]],
    set[str],
    dict[str, int],
    list[dict[str, str]],
    list[str],
    int,
    int,
]:
    rows: list[
        dict[str, ScalarValue]
    ] = []

    columns: set[str] = set(
        REQUIRED_OUTPUT_COLUMNS
    )

    team_row_counts: dict[
        str,
        int,
    ] = {}

    processing_failures: list[
        dict[str, str]
    ] = []

    empty_roster_teams: list[
        str
    ] = []

    payload_season_metadata_count = 0
    athlete_season_ref_count = 0

    seen_team_athlete_keys: set[
        tuple[str, str]
    ] = set()

    athlete_team_assignments: dict[
        str,
        str,
    ] = {}

    for team_id in target_team_ids:
        url = roster_url(
            team_id,
            season,
        )

        try:
            payload = fetch_json(
                url,
                label=(
                    f"roster team_id={team_id}"
                ),
                state=state,
            )

            (
                athletes,
                has_payload_season,
            ) = extract_athletes(
                payload,
                team_id=team_id,
                season=season,
            )

        except Exception as exc:
            processing_failures.append(
                {
                    "team_id": team_id,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                }
            )

            continue

        if has_payload_season:
            payload_season_metadata_count += 1

        if not athletes:
            empty_roster_teams.append(
                team_id
            )

            continue

        team_row_counts[
            team_id
        ] = 0

        for athlete in athletes:
            flat_row = flatten(
                athlete
            )

            athlete_id = str(
                flat_row.get(
                    "id",
                    "",
                )
            ).strip()

            display_name = str(
                flat_row.get(
                    "displayName",
                    "",
                )
            ).strip()

            if not athlete_id:
                processing_failures.append(
                    {
                        "team_id": team_id,
                        "error_type": (
                            "AthleteIdentityError"
                        ),
                        "message": (
                            "Roster athlete has "
                            "blank id"
                        ),
                    }
                )

                continue

            if not display_name:
                processing_failures.append(
                    {
                        "team_id": team_id,
                        "error_type": (
                            "AthleteIdentityError"
                        ),
                        "message": (
                            "Roster athlete has "
                            "blank displayName for "
                            f"athlete_id={athlete_id}"
                        ),
                    }
                )

                continue

            key = (
                team_id,
                athlete_id,
            )

            if (
                key
                in seen_team_athlete_keys
            ):
                processing_failures.append(
                    {
                        "team_id": team_id,
                        "error_type": (
                            "DuplicateAthleteError"
                        ),
                        "message": (
                            "Duplicate athlete within "
                            "team roster: "
                            f"athlete_id={athlete_id}"
                        ),
                    }
                )

                continue

            assigned_team = (
                athlete_team_assignments.get(
                    athlete_id
                )
            )

            if (
                assigned_team is not None
                and assigned_team != team_id
            ):
                processing_failures.append(
                    {
                        "team_id": team_id,
                        "error_type": (
                            "AthleteTeamConflictError"
                        ),
                        "message": (
                            "Athlete appears on "
                            "multiple team rosters: "
                            f"athlete_id={athlete_id}, "
                            f"first_team={assigned_team}, "
                            f"second_team={team_id}"
                        ),
                    }
                )

                continue

            try:
                has_season_ref = (
                    validate_athlete_season_refs(
                        flat_row,
                        team_id=team_id,
                        athlete_id=athlete_id,
                        season=season,
                    )
                )
            except Exception as exc:
                processing_failures.append(
                    {
                        "team_id": team_id,
                        "error_type": (
                            type(exc).__name__
                        ),
                        "message": str(exc),
                    }
                )

                continue

            if has_season_ref:
                athlete_season_ref_count += 1

            flat_row[
                "season"
            ] = season

            flat_row[
                "season_type"
            ] = season_type

            flat_row[
                "team_id"
            ] = team_id

            seen_team_athlete_keys.add(
                key
            )

            athlete_team_assignments[
                athlete_id
            ] = team_id

            rows.append(
                flat_row
            )

            columns.update(
                flat_row.keys()
            )

            team_row_counts[
                team_id
            ] += 1

    return (
        rows,
        columns,
        team_row_counts,
        processing_failures,
        empty_roster_teams,
        payload_season_metadata_count,
        athlete_season_ref_count,
    )



def _require_raw_roster_rows(
    rows: list[dict[str, ScalarValue]],
) -> None:
    if not rows:
        raise ValueError(
            "Raw roster output would be empty"
        )


def _require_raw_roster_columns(
    missing_required_columns: list[str],
) -> None:
    if missing_required_columns:
        raise ValueError(
            "Raw roster output missing "
            "required columns: "
            f"{missing_required_columns}"
        )


def _validate_raw_roster_team_coverage(
    expected_teams: set[str],
    represented_teams: set[str],
) -> None:
    if represented_teams != expected_teams:
        missing = sorted(
            expected_teams
            - represented_teams,
            key=int,
        )

        foreign = sorted(
            represented_teams
            - expected_teams,
            key=int,
        )

        raise ValueError(
            "Raw roster team coverage "
            "does not match authoritative "
            "team set. "
            f"missing={missing[:50]}, "
            f"foreign={foreign[:50]}"
        )


def validate_final_rows(
    rows: list[
        dict[str, ScalarValue]
    ],
    columns: set[str],
    target_team_ids: list[str],
    team_row_counts: dict[
        str,
        int,
    ],
    season: int,
    season_type: int,
) -> None:
    _require_raw_roster_rows(rows)

    missing_required_columns = [
        column
        for column
        in REQUIRED_OUTPUT_COLUMNS
        if column not in columns
    ]

    _require_raw_roster_columns(
        missing_required_columns
    )

    expected_teams = set(
        target_team_ids
    )

    represented_teams = set(
        team_row_counts
    )

    _validate_raw_roster_team_coverage(
        expected_teams,
        represented_teams,
    )

    zero_row_teams = sorted(
        team_id
        for team_id
        in target_team_ids
        if team_row_counts.get(
            team_id,
            0,
        ) <= 0
    )

    if zero_row_teams:
        raise ValueError(
            "Raw roster has target teams "
            "with zero athlete rows: "
            f"{zero_row_teams[:50]}"
        )

    seen_keys: set[
        tuple[str, str]
    ] = set()

    athlete_team: dict[
        str,
        str,
    ] = {}

    for index, row in enumerate(
        rows
    ):
        team_id = str(
            row.get(
                "team_id",
                "",
            )
        ).strip()

        athlete_id = str(
            row.get(
                "id",
                "",
            )
        ).strip()

        display_name = str(
            row.get(
                "displayName",
                "",
            )
        ).strip()

        if team_id not in expected_teams:
            raise ValueError(
                "Raw roster row references "
                "foreign team_id at row "
                f"{index}: {team_id!r}"
            )

        if not athlete_id:
            raise ValueError(
                "Raw roster row has blank "
                f"athlete id at row {index}"
            )

        if not display_name:
            raise ValueError(
                "Raw roster row has blank "
                "displayName at row "
                f"{index}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ) != str(
            season
        ):
            raise ValueError(
                "Raw roster season mismatch "
                f"at row {index}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ) != str(
            season_type
        ):
            raise ValueError(
                "Raw roster season_type "
                f"mismatch at row {index}"
            )

        key = (
            team_id,
            athlete_id,
        )

        if key in seen_keys:
            raise ValueError(
                "Raw roster contains "
                "duplicate athlete key: "
                f"{key}"
            )

        seen_keys.add(
            key
        )

        prior_team = (
            athlete_team.get(
                athlete_id
            )
        )

        if (
            prior_team is not None
            and prior_team != team_id
        ):
            raise ValueError(
                "Raw roster athlete appears "
                "on multiple teams: "
                f"athlete_id={athlete_id}, "
                f"teams={prior_team},{team_id}"
            )

        athlete_team[
            athlete_id
        ] = team_id


def output_fieldnames(
    columns: set[str],
) -> list[str]:
    required = list(
        REQUIRED_OUTPUT_COLUMNS
    )

    remaining = sorted(
        column
        for column in columns
        if column not in required
    )

    return (
        required
        + remaining
    )


def temporary_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(
    path: Path,
    rows: list[
        dict[str, ScalarValue]
    ],
    fieldnames: list[str],
) -> None:
    write_csv_rows_durable(
        path,
        rows,
        fieldnames,
        project_columns=True,
    )

def validate_staged_csv(
    path: Path,
    *,
    expected_rows: int,
    expected_fieldnames: list[str],
    target_team_ids: list[str],
    season: int,
    season_type: int,
) -> None:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        actual_fieldnames = (
            reader.fieldnames
            or []
        )

        if (
            actual_fieldnames
            != expected_fieldnames
        ):
            raise ValueError(
                "Serialized raw roster "
                "header mismatch"
            )

        rows = list(
            reader
        )

    if len(rows) != expected_rows:
        raise ValueError(
            "Serialized raw roster row "
            "count mismatch: "
            f"expected={expected_rows}, "
            f"actual={len(rows)}"
        )

    counts: dict[
        str,
        int,
    ] = {}

    seen: set[
        tuple[str, str]
    ] = set()

    target_set = set(
        target_team_ids
    )

    for index, row in enumerate(
        rows
    ):
        team_id = str(
            row.get(
                "team_id",
                "",
            )
        ).strip()

        athlete_id = str(
            row.get(
                "id",
                "",
            )
        ).strip()

        display_name = str(
            row.get(
                "displayName",
                "",
            )
        ).strip()

        if team_id not in target_set:
            raise ValueError(
                "Serialized raw roster "
                "contains foreign team at "
                f"row {index}: {team_id!r}"
            )

        if not athlete_id:
            raise ValueError(
                "Serialized raw roster "
                "contains blank athlete id "
                f"at row {index}"
            )

        if not display_name:
            raise ValueError(
                "Serialized raw roster "
                "contains blank displayName "
                f"at row {index}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ) != str(
            season
        ):
            raise ValueError(
                "Serialized raw roster "
                "season mismatch at row "
                f"{index}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ) != str(
            season_type
        ):
            raise ValueError(
                "Serialized raw roster "
                "season_type mismatch at row "
                f"{index}"
            )

        key = (
            team_id,
            athlete_id,
        )

        if key in seen:
            raise ValueError(
                "Serialized raw roster "
                "contains duplicate key: "
                f"{key}"
            )

        seen.add(
            key
        )

        counts[
            team_id
        ] = (
            counts.get(
                team_id,
                0,
            )
            + 1
        )

    if set(counts) != target_set:
        missing = sorted(
            target_set
            - set(counts),
            key=int,
        )

        raise ValueError(
            "Serialized raw roster "
            "team coverage mismatch. "
            f"missing={missing[:50]}"
        )


def publish_atomic(
    rows: list[
        dict[str, ScalarValue]
    ],
    fieldnames: list[str],
    target_team_ids: list[str],
    season: int,
    season_type: int,
) -> bool:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temporary_path(
        OUTPUT_PATH
    )

    try:
        write_staged_csv(
            temp_path,
            rows,
            fieldnames,
        )

        validate_staged_csv(
            temp_path,
            expected_rows=len(
                rows
            ),
            expected_fieldnames=fieldnames,
            target_team_ids=(
                target_team_ids
            ),
            season=season,
            season_type=season_type,
        )

        if (
            OUTPUT_PATH.exists()
            and OUTPUT_PATH.read_bytes()
            == temp_path.read_bytes()
        ):
            return False

        os.replace(
            temp_path,
            OUTPUT_PATH,
        )

        return True

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass


def run(
    report: PipelineReporter,
    state: RuntimeState,
) -> int:
    (
        season,
        season_type,
        week,
    ) = load_current_week_config(CURRENT_WEEK_CONFIG_PATH)

    report.season = season
    report.week = week

    report.set_detail(
        "season_type",
        season_type,
    )

    target_team_ids = (
        load_target_team_ids()
    )

    report.set_detail(
        "target_team_count",
        len(
            target_team_ids
        ),
    )

    (
        rows,
        columns,
        team_row_counts,
        processing_failures,
        empty_roster_teams,
        payload_season_metadata_count,
        athlete_season_ref_count,
    ) = build_raw_rows(
        target_team_ids,
        season,
        season_type,
        state,
    )

    requested_teams = len(
        target_team_ids
    )

    successful_teams = len(
        team_row_counts
    )

    represented_teams = len(
        {
            str(
                row.get(
                    "team_id",
                    "",
                )
            ).strip()
            for row in rows
        }
    )

    unique_athletes = len(
        {
            str(
                row.get(
                    "id",
                    "",
                )
            ).strip()
            for row in rows
            if str(
                row.get(
                    "id",
                    "",
                )
            ).strip()
        }
    )

    report.set_rows(
        rows_in=requested_teams,
    )

    report.update_details(
        {
            "team_source": (
                "team_master.csv"
            ),
            "teams_requested": (
                requested_teams
            ),
            "teams_successfully_returned": (
                successful_teams
            ),
            "teams_represented_in_rows": (
                represented_teams
            ),
            "empty_roster_team_count": (
                len(
                    empty_roster_teams
                )
            ),
            "empty_roster_team_ids": (
                empty_roster_teams
            ),
            "processing_failure_count": (
                len(
                    processing_failures
                )
            ),
            "processing_failure_details": (
                processing_failures
            ),
            "athlete_rows_built": (
                len(
                    rows
                )
            ),
            "unique_athlete_ids": (
                unique_athletes
            ),
            "output_column_count": (
                len(
                    columns
                )
            ),
            "payloads_with_explicit_season_metadata": (
                payload_season_metadata_count
            ),
            "athletes_with_season_scoped_team_refs": (
                athlete_season_ref_count
            ),
            "output_modified": False,
        }
    )

    if processing_failures:
        raise RuntimeError(
            "One or more roster teams "
            "failed validation or retrieval; "
            "refusing to modify raw roster. "
            f"failures="
            f"{len(processing_failures)}"
        )

    if empty_roster_teams:
        raise RuntimeError(
            "One or more target teams "
            "returned an empty roster; "
            "refusing to modify raw roster. "
            f"teams="
            f"{empty_roster_teams[:50]}"
        )

    validate_final_rows(
        rows,
        columns,
        target_team_ids,
        team_row_counts,
        season,
        season_type,
    )

    fieldnames = output_fieldnames(
        columns
    )

    output_modified = publish_atomic(
        rows,
        fieldnames,
        target_team_ids,
        season,
        season_type,
    )

    report.set_rows(
        rows_out=len(
            rows
        ),
    )

    report.update_details(
        {
            "output_columns": (
                fieldnames
            ),
            "output_column_count": (
                len(
                    fieldnames
                )
            ),
            "output_modified": (
                output_modified
            ),
            "output_path": str(
                OUTPUT_PATH
            ),
        }
    )

    print(
        "pull_raw_roster.py completed"
    )

    print(
        f"season={season} "
        f"season_type={season_type} "
        f"week={week}"
    )

    print(
        f"target_teams="
        f"{len(target_team_ids)}"
    )

    print(
        f"teams_returned="
        f"{successful_teams}"
    )

    print(
        f"athlete_rows="
        f"{len(rows)}"
    )

    print(
        f"unique_athletes="
        f"{unique_athletes}"
    )

    print(
        f"columns="
        f"{len(fieldnames)}"
    )

    print(
        f"output_modified="
        f"{output_modified}"
    )

    print(
        f"output={OUTPUT_PATH}"
    )

    return 0


def main() -> int:
    state = RuntimeState()

    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": (
                SCRIPT_VERSION
            ),
            "source": (
                "ESPN Site API"
            ),
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.add_input(
            TEAM_MASTER_PATH
        )

        report.add_output(
            OUTPUT_PATH
        )

        report.set_detail(
            "output_modified",
            False,
        )

        try:
            return run(
                report,
                state,
            )

        finally:
            report.update_details(
                {
                    "espn_request_count": (
                        state.request_count
                    ),
                    "espn_request_failure_count": (
                        len(
                            state.request_failures
                        )
                    ),
                    "espn_request_failure_details": (
                        list(
                            state.request_failures
                        )
                    ),
                }
            )

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(
        main()
    )