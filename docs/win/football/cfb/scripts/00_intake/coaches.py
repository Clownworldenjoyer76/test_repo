#!/usr/bin/env python3
"""
coaches.py

Pull the configured-season head coach for every authoritative CFB team from
ESPN and publish one validated coaches_master.csv.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/league_master.csv

Output:
    docs/win/football/cfb/data/master/coaches_master.csv
"""

from __future__ import annotations

from http.client import HTTPException

import csv
import json
import os
import re
import ssl
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request



SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from pipeline_shared import load_current_week_config


CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
LEAGUE_MASTER_PATH = CFB_ROOT / "data" / "master" / "league_master.csv"
OUTPUT_PATH = CFB_ROOT / "data" / "master" / "coaches_master.csv"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-coaches-v5-2026-10-23-retries"

COACHES_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football/seasons/{season}/teams/{team_id}/coaches"
)

HEADER = [
    "sport",
    "league",
    "name",
    "team",
    "team_id",
    "experience",
    "career_record",
    "post_season_career_record",
    "id",
    "uid",
]

ESPN_CORE_HOST = "sports.core.api.espn.com"
COACH_REF_SEASON_PATTERN = re.compile(
    r"/seasons/(\d+)(?:[/?#]|$)"
)
COACH_TEAM_REF_PATTERN = re.compile(
    r"/teams/(\d+)(?:[/?#]|$)"
)

_REQUEST_COUNTS = {
    "coach_list": 0,
    "coach_ref": 0,
    "person_ref": 0,
    "career_record": 0,
}
_REQUEST_FAILURES: list[dict[str, str]] = []

MAX_REQUEST_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}

_RETRY_STATS = {
    "request_attempt_count": 0,
    "request_retry_count": 0,
    "recovered_transient_requests": 0,
    "exhausted_transient_failures": 0,
}
_REQUEST_RETRY_DETAILS: list[dict[str, str]] = []


class CoachValidationError(RuntimeError):
    pass


def reset_runtime_state() -> None:
    for key in _REQUEST_COUNTS:
        _REQUEST_COUNTS[key] = 0
    _REQUEST_FAILURES.clear()
    _REQUEST_RETRY_DETAILS.clear()
    for key in _RETRY_STATS:
        _RETRY_STATS[key] = 0


def _require_league_master_path() -> None:
    if not LEAGUE_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing league master: {LEAGUE_MASTER_PATH}"
        )


def _load_authoritative_team_rows(
    reader: csv.DictReader,
    teams: dict[str, str],
    *,
    season: int,
    season_type: int,
) -> None:
    for row_number, row in enumerate(
        reader,
        start=2,
    ):
        if None in row:
            raise ValueError(
                "league_master.csv contains a malformed row at "
                f"CSV line {row_number}"
            )

        team_id = str(
            row.get("team_id") or ""
        ).strip()
        team_abbr = str(
            row.get("team_abbr") or ""
        ).strip()
        row_season = str(
            row.get("season") or ""
        ).strip()
        row_season_type = str(
            row.get("season_type") or ""
        ).strip()

        if not team_id:
            raise ValueError(
                "league_master.csv has blank team_id at "
                f"CSV line {row_number}"
            )

        if not team_id.isdigit() or int(team_id) <= 0:
            raise ValueError(
                "league_master.csv has invalid team_id at "
                f"CSV line {row_number}: {team_id!r}"
            )

        if not team_abbr:
            raise ValueError(
                "league_master.csv has blank team_abbr at "
                f"CSV line {row_number}"
            )

        if row_season != str(season):
            raise ValueError(
                "league_master.csv season mismatch at "
                f"CSV line {row_number}: expected={season}, "
                f"actual={row_season!r}"
            )

        if row_season_type != str(season_type):
            raise ValueError(
                "league_master.csv season_type mismatch at "
                f"CSV line {row_number}: expected={season_type}, "
                f"actual={row_season_type!r}"
            )

        if team_id in teams:
            prior_abbr = teams[team_id]
            if prior_abbr != team_abbr:
                raise ValueError(
                    "league_master.csv contains conflicting "
                    "abbreviations for "
                    f"team_id={team_id}: "
                    f"{prior_abbr!r} vs {team_abbr!r}"
                )

            raise ValueError(
                "league_master.csv contains duplicate "
                f"team_id={team_id}"
            )

        teams[team_id] = team_abbr


def _require_authoritative_teams(
    teams: dict[str, str],
) -> None:
    if not teams:
        raise ValueError(
            "league_master.csv contains no authoritative teams"
        )


def load_authoritative_teams(
    *,
    season: int,
    season_type: int,
) -> list[tuple[str, str]]:
    _require_league_master_path()

    with LEAGUE_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"team_id", "team_abbr", "season", "season_type"}
        missing = sorted(required - set(fieldnames))

        if missing:
            raise ValueError(
                f"league_master.csv missing required columns: {missing}"
            )

        teams: dict[str, str] = {}

        _load_authoritative_team_rows(
            reader,
            teams,
            season=season,
            season_type=season_type,
        )

    _require_authoritative_teams(teams)

    return sorted(teams.items(), key=lambda item: int(item[0]))


def validate_espn_ref(url: str, *, label: str) -> str:
    text = str(url or "").strip()

    if not text:
        raise CoachValidationError(
            f"{label} is blank"
        )

    parsed = urlparse(
        text
    )

    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != ESPN_CORE_HOST
    ):
        raise CoachValidationError(
            f"{label} is not an approved ESPN Core URL: "
            f"{text!r}"
        )

    if parsed.scheme == "http":
        return urlunparse(
            ("https", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )

    return text


def _is_retryable_http_status(status: int) -> bool:
    return status in RETRYABLE_HTTP_STATUSES or 500 <= status <= 599


def _is_retryable_transport_error(exc: BaseException) -> bool:
    cause = exc.reason if isinstance(exc, URLError) else exc
    return (
        isinstance(cause, (OSError, HTTPException))
        and not isinstance(cause, ssl.SSLError)
    )


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8")
    except (HTTPException, OSError, UnicodeError, ValueError):
        return ""


def _record_request_failure(
    *,
    request_kind: str,
    label: str,
    url: str,
    http_status: str,
    error: str,
    attempt: int,
) -> dict[str, str]:
    failure = {
        "request_kind": request_kind,
        "label": label,
        "url": url,
        "http_status": http_status,
        "error": error,
        "attempt": str(attempt),
        "max_attempts": str(MAX_REQUEST_ATTEMPTS),
    }
    _REQUEST_FAILURES.append(failure)
    return failure


def _retry_request(
    *,
    request_kind: str,
    label: str,
    url: str,
    http_status: str,
    error: str,
    attempt: int,
) -> None:
    # Only called when another attempt is available.
    delay_seconds = RETRY_BACKOFF_SECONDS[attempt - 1]
    detail = {
        "request_kind": request_kind,
        "label": label,
        "url": url,
        "http_status": http_status,
        "error": error,
        "attempt": str(attempt),
        "next_attempt": str(attempt + 1),
        "max_attempts": str(MAX_REQUEST_ATTEMPTS),
        "delay_seconds": str(delay_seconds),
    }
    _RETRY_STATS["request_retry_count"] += 1
    _REQUEST_RETRY_DETAILS.append(detail)
    print(
        "COACHES_ESPN_REQUEST_RETRY " + json.dumps(detail, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    time.sleep(delay_seconds)


def fetch_json(
    url: str,
    *,
    request_kind: str,
    label: str,
    timeout: int = 30,
) -> dict:
    if request_kind not in _REQUEST_COUNTS:
        raise ValueError(f"Unknown request kind: {request_kind}")

    if request_kind != "coach_list":
        url = validate_espn_ref(url, label=f"{label} URL")

    _REQUEST_COUNTS[request_kind] += 1

    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        _RETRY_STATS["request_attempt_count"] += 1
        request = Request(
            url,
            headers={
                "User-Agent": "cfb-coaches/2.0",
                "Accept": "application/json",
            },
        )

        try:
            with open_https(
                request,
                allowed_hosts={ESPN_CORE_HOST},
                timeout=timeout,
            ) as response:
                status = int(response.status)
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = _read_http_error_body(exc) or str(exc)
            transient = _is_retryable_http_status(exc.code)
            if transient and attempt < MAX_REQUEST_ATTEMPTS:
                _retry_request(
                    request_kind=request_kind,
                    label=label,
                    url=url,
                    http_status=str(exc.code),
                    error=error_body,
                    attempt=attempt,
                )
                continue
            if transient:
                _RETRY_STATS["exhausted_transient_failures"] += 1
            failure = _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status=str(exc.code),
                error=error_body,
                attempt=attempt,
            )
            raise RuntimeError(
                f"{label} request failed: status={exc.code}, "
                f"attempt={attempt}/{MAX_REQUEST_ATTEMPTS}, "
                f"error={failure['error']}"
            ) from exc
        except (URLError, HTTPException, OSError) as exc:
            transient = _is_retryable_transport_error(exc)
            if transient and attempt < MAX_REQUEST_ATTEMPTS:
                _retry_request(
                    request_kind=request_kind,
                    label=label,
                    url=url,
                    http_status="",
                    error=str(exc),
                    attempt=attempt,
                )
                continue
            if transient:
                _RETRY_STATS["exhausted_transient_failures"] += 1
            _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status="",
                error=str(exc),
                attempt=attempt,
            )
            raise RuntimeError(
                f"{label} request failed: "
                f"attempt={attempt}/{MAX_REQUEST_ATTEMPTS}, error={exc}"
            ) from exc
        except Exception as exc:
            _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status="",
                error=str(exc),
                attempt=attempt,
            )
            raise RuntimeError(f"{label} request failed: {exc}") from exc

        if status < 200 or status >= 300:
            transient = _is_retryable_http_status(status)
            if transient and attempt < MAX_REQUEST_ATTEMPTS:
                _retry_request(
                    request_kind=request_kind,
                    label=label,
                    url=url,
                    http_status=str(status),
                    error=body,
                    attempt=attempt,
                )
                continue
            if transient:
                _RETRY_STATS["exhausted_transient_failures"] += 1
            _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status=str(status),
                error=body,
                attempt=attempt,
            )
            raise RuntimeError(
                f"{label} request failed: status={status}, "
                f"attempt={attempt}/{MAX_REQUEST_ATTEMPTS}"
            )

        try:
            payload = json.loads(body)
        except (ValueError, TypeError) as exc:
            _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status=str(status),
                error=f"JSON parse failed: {exc}",
                attempt=attempt,
            )
            raise RuntimeError(f"{label} returned malformed JSON") from exc

        if not isinstance(payload, dict):
            _record_request_failure(
                request_kind=request_kind,
                label=label,
                url=url,
                http_status=str(status),
                error="JSON root is not an object",
                attempt=attempt,
            )
            raise RuntimeError(f"{label} returned non-object JSON")

        if attempt > 1:
            _RETRY_STATS["recovered_transient_requests"] += 1
        return payload

    raise AssertionError("Unreachable coaches HTTP retry state")


def extract_season_from_coach_ref(
    coach_ref: str,
    *,
    team_id: str,
    item_index: int,
) -> int:
    parsed = urlparse(coach_ref)

    match = COACH_REF_SEASON_PATTERN.search(
        parsed.path
    )

    if not match:
        raise CoachValidationError(
            "Coach-list $ref does not contain season identity for "
            f"team_id={team_id}, item_index={item_index}: "
            f"{coach_ref!r}"
        )

    return int(
        match.group(1)
    )


def extract_team_id_from_coach(
    coach: dict,
) -> str:
    team_obj = coach.get("team")

    if not isinstance(team_obj, dict):
        raise CoachValidationError(
            "Coach payload missing team object"
        )

    team_ref = validate_espn_ref(
        team_obj.get("$ref", ""),
        label="coach team $ref",
    )

    parsed = urlparse(
        team_ref
    )

    match = COACH_TEAM_REF_PATTERN.search(
        parsed.path
    )

    if not match:
        raise CoachValidationError(
            "Coach team $ref does not contain team identity: "
            f"{team_ref!r}"
        )

    return match.group(1)


def collect_role_markers(obj: object) -> list[str]:
    if not isinstance(obj, dict):
        return []

    markers: list[str] = []
    role_keys = {
        "role",
        "position",
        "title",
        "job",
        "designation",
        "coachType",
        "coach_type",
    }

    def collect(value: object) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                markers.append(text)
        elif isinstance(value, dict):
            for marker_key in (
                "name",
                "displayName",
                "shortName",
                "abbreviation",
                "type",
                "description",
                "text",
                "value",
            ):
                if marker_key in value:
                    collect(value.get(marker_key))
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    for key in role_keys:
        if key in obj:
            collect(obj.get(key))

    return markers


def is_head_coach_marker(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).lower())
    return normalized in {"headcoach", "headfootballcoach", "hc"}


def select_head_coach(
    candidates: list[tuple[dict, dict]],
    *,
    team_id: str,
) -> tuple[dict, str]:
    if not candidates:
        raise CoachValidationError(
            f"No resolved coach candidates for team_id={team_id}"
        )

    marked: list[dict] = []
    for item, coach in candidates:
        markers = collect_role_markers(item) + collect_role_markers(coach)
        if any(is_head_coach_marker(marker) for marker in markers):
            marked.append(coach)

    if len(marked) == 1:
        return marked[0], "explicit_role"

    if len(marked) > 1:
        ids = [str(coach.get("id") or "").strip() for coach in marked]
        raise CoachValidationError(
            "Multiple candidates identify as head coach for "
            f"team_id={team_id}: coach_ids={ids}"
        )

    if len(candidates) == 1:
        return candidates[0][1], "sole_provider_candidate"

    ids = [
        str(coach.get("id") or "").strip()
        for _, coach in candidates
    ]
    raise CoachValidationError(
        "Multiple coach candidates returned without exactly one head-coach "
        f"role marker for team_id={team_id}: coach_ids={ids}"
    )


def resolve_team_head_coach(
    *,
    team_id: str,
    team_abbr: str,
    season: int,
) -> tuple[dict, str]:
    url = COACHES_URL_TEMPLATE.format(season=season, team_id=team_id)
    payload = fetch_json(
        url,
        request_kind="coach_list",
        label=f"coach list team_id={team_id}",
    )

    items = payload.get("items")
    if not isinstance(items, list):
        raise CoachValidationError(
            "Coach-list payload items field is not a list for "
            f"team={team_abbr}, team_id={team_id}"
        )
    if not items:
        raise CoachValidationError(
            "Coach-list payload contains no coach candidates for "
            f"team={team_abbr}, team_id={team_id}"
        )

    candidates: list[tuple[dict, dict]] = []
    seen_refs: set[str] = set()

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise CoachValidationError(
                "Coach-list payload contains non-object item for "
                f"team_id={team_id}, item_index={item_index}"
            )

        coach_ref = validate_espn_ref(
            item.get("$ref", ""),
            label=f"coach $ref team_id={team_id} item_index={item_index}",
        )

        coach_ref_season = extract_season_from_coach_ref(
            coach_ref,
            team_id=team_id,
            item_index=item_index,
        )

        if coach_ref_season != season:
            raise CoachValidationError(
                "Coach-list $ref season mismatch for "
                f"team_id={team_id}, item_index={item_index}: "
                f"expected={season}, actual={coach_ref_season}"
            )

        if coach_ref in seen_refs:
            raise CoachValidationError(
                "Coach-list payload contains duplicate coach $ref for "
                f"team_id={team_id}: {coach_ref}"
            )
        seen_refs.add(coach_ref)

        coach = fetch_json(
            coach_ref,
            request_kind="coach_ref",
            label=f"coach detail team_id={team_id} item_index={item_index}",
        )

        coach_team_id = extract_team_id_from_coach(
            coach
        )

        if coach_team_id != team_id:
            raise CoachValidationError(
                "Coach payload team mismatch for "
                f"requested_team_id={team_id}: "
                f"coach_team_id={coach_team_id}"
            )

        candidates.append((item, coach))

    return select_head_coach(candidates, team_id=team_id)


def _populate_career_record_values(
    career_records: list,
    *,
    team_id: str,
    coach_id: str,
    record_values: dict[str, set[str]],
) -> None:
    for record_index, record_ref_obj in enumerate(career_records):
        if not isinstance(record_ref_obj, dict):
            raise CoachValidationError(
                "careerRecords contains non-object entry for "
                f"team_id={team_id}, coach_id={coach_id}, "
                f"record_index={record_index}"
            )

        record_ref = validate_espn_ref(
            record_ref_obj.get("$ref", ""),
            label=(
                f"career record $ref team_id={team_id} "
                f"coach_id={coach_id} record_index={record_index}"
            ),
        )
        record = fetch_json(
            record_ref,
            request_kind="career_record",
            label=(
                f"career record team_id={team_id} "
                f"coach_id={coach_id} record_index={record_index}"
            ),
        )

        record_type = str(record.get("type") or "").strip()
        summary = str(record.get("summary") or "").strip()
        if record_type in record_values and summary:
            record_values[record_type].add(summary)


def get_career_records(
    coach: dict,
    *,
    team_id: str,
    coach_id: str,
) -> tuple[str, str, dict[str, int]]:
    diagnostics = {
        "missing_person_ref": 0,
        "missing_career_records_collection": 0,
        "missing_total_record": 0,
        "missing_postseason_record": 0,
    }

    person_obj = coach.get("person")
    if not isinstance(person_obj, dict):
        diagnostics = {key: 1 for key in diagnostics}
        return "", "", diagnostics

    person_ref = str(person_obj.get("$ref") or "").strip()
    if not person_ref:
        diagnostics = {key: 1 for key in diagnostics}
        return "", "", diagnostics

    person = fetch_json(
        person_ref,
        request_kind="person_ref",
        label=f"coach person team_id={team_id} coach_id={coach_id}",
    )

    career_records = person.get("careerRecords")
    if career_records is None:
        diagnostics["missing_career_records_collection"] = 1
        diagnostics["missing_total_record"] = 1
        diagnostics["missing_postseason_record"] = 1
        return "", "", diagnostics

    if not isinstance(career_records, list):
        raise CoachValidationError(
            "Coach person careerRecords is not a list for "
            f"team_id={team_id}, coach_id={coach_id}"
        )

    if not career_records:
        diagnostics["missing_career_records_collection"] = 1
        diagnostics["missing_total_record"] = 1
        diagnostics["missing_postseason_record"] = 1
        return "", "", diagnostics

    record_values: dict[str, set[str]] = {
        "Total": set(),
        "Post Season": set(),
    }

    _populate_career_record_values(
        career_records,
        team_id=team_id,
        coach_id=coach_id,
        record_values=record_values,
    )

    for record_type, summaries in record_values.items():
        if len(summaries) > 1:
            raise CoachValidationError(
                "Conflicting coach career record summaries for "
                f"team_id={team_id}, coach_id={coach_id}, "
                f"record_type={record_type!r}: {sorted(summaries)}"
            )

    career_record = next(iter(record_values["Total"]), "")
    postseason_record = next(iter(record_values["Post Season"]), "")

    if not career_record:
        diagnostics["missing_total_record"] = 1
    if not postseason_record:
        diagnostics["missing_postseason_record"] = 1

    return career_record, postseason_record, diagnostics


def empty_diagnostics() -> dict[str, int]:
    return {
        "explicit_role_selections": 0,
        "sole_provider_candidate_selections": 0,
        "missing_person_ref_count": 0,
        "missing_career_records_collection_count": 0,
        "missing_total_career_record_count": 0,
        "missing_postseason_career_record_count": 0,
    }


def build_rows(
    *,
    teams: list[tuple[str, str]],
    season: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    diagnostics = empty_diagnostics()

    for team_id, team_abbr in teams:
        try:
            coach, selection_method = resolve_team_head_coach(
                team_id=team_id,
                team_abbr=team_abbr,
                season=season,
            )

            coach_id = str(coach.get("id") or "").strip()
            uid = str(coach.get("uid") or "").strip()
            first_name = str(coach.get("firstName") or "").strip()
            last_name = str(coach.get("lastName") or "").strip()
            name = f"{first_name} {last_name}".strip()

            if not coach_id:
                raise CoachValidationError(
                    f"Resolved head coach has blank id for team_id={team_id}"
                )
            if not name:
                raise CoachValidationError(
                    "Resolved head coach has blank name for "
                    f"team_id={team_id}, coach_id={coach_id}"
                )

            career_record, postseason_record, record_diag = get_career_records(
                coach,
                team_id=team_id,
                coach_id=coach_id,
            )

            if selection_method == "explicit_role":
                diagnostics["explicit_role_selections"] += 1
            else:
                diagnostics["sole_provider_candidate_selections"] += 1

            diagnostics["missing_person_ref_count"] += (
                record_diag["missing_person_ref"]
            )
            diagnostics["missing_career_records_collection_count"] += (
                record_diag["missing_career_records_collection"]
            )
            diagnostics["missing_total_career_record_count"] += (
                record_diag["missing_total_record"]
            )
            diagnostics["missing_postseason_career_record_count"] += (
                record_diag["missing_postseason_record"]
            )

            experience_raw = coach.get("experience")
            experience = (
                "" if experience_raw is None else str(experience_raw).strip()
            )

            rows.append(
                {
                    "sport": "football",
                    "league": "college-football",
                    "name": name,
                    "team": team_abbr,
                    "team_id": team_id,
                    "experience": experience,
                    "career_record": career_record,
                    "post_season_career_record": postseason_record,
                    "id": coach_id,
                    "uid": uid,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    return rows, failures, diagnostics



def _require_coach_rows(
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        raise ValueError(
            "Coaches output would be empty"
        )


def _require_coach_row_count(
    rows: list[dict[str, str]],
    expected_team_ids: set[str],
) -> None:
    if len(rows) != len(expected_team_ids):
        raise ValueError(
            "Coaches row count does not match "
            "authoritative team count: "
            f"rows={len(rows)}, "
            f"teams={len(expected_team_ids)}"
        )


def _validate_coach_id(
    coach_id: str,
    *,
    team_id: str,
) -> None:
    if not coach_id.isdigit() or int(coach_id) <= 0:
        raise ValueError(
            "Coaches output contains invalid coach id for "
            f"team_id={team_id}: {coach_id!r}"
        )


def _validate_coach_team_coverage(
    seen_team_ids: set[str],
    expected_team_ids: set[str],
) -> None:
    if seen_team_ids != expected_team_ids:
        missing = sorted(
            expected_team_ids
            - seen_team_ids,
            key=int,
        )
        foreign = sorted(
            seen_team_ids
            - expected_team_ids,
            key=int,
        )
        raise ValueError(
            "Coaches output team coverage mismatch. "
            f"missing={missing[:50]}, "
            f"foreign={foreign[:50]}"
        )


def validate_final_rows(
    rows: list[dict[str, str]],
    *,
    teams: list[tuple[str, str]],
) -> None:
    _require_coach_rows(rows)

    expected_by_team = dict(teams)
    expected_team_ids = set(expected_by_team)

    _require_coach_row_count(
        rows,
        expected_team_ids,
    )

    seen_team_ids: set[str] = set()
    coach_team_assignments: dict[str, str] = {}

    for index, row in enumerate(rows):
        if set(row) != set(HEADER):
            raise ValueError(f"Coaches row schema mismatch at row_index={index}")

        team_id = str(row.get("team_id") or "").strip()
        team_abbr = str(row.get("team") or "").strip()
        coach_id = str(row.get("id") or "").strip()
        name = str(row.get("name") or "").strip()

        if team_id not in expected_team_ids:
            raise ValueError(
                "Coaches output references foreign team_id at "
                f"row_index={index}: {team_id!r}"
            )
        if team_abbr != expected_by_team[team_id]:
            raise ValueError(
                "Coaches output abbreviation mismatch for "
                f"team_id={team_id}: expected={expected_by_team[team_id]!r}, "
                f"actual={team_abbr!r}"
            )
        if not coach_id:
            raise ValueError(
                f"Coaches output contains blank coach id for team_id={team_id}"
            )
        _validate_coach_id(
            coach_id,
            team_id=team_id,
        )
        if not name:
            raise ValueError(
                f"Coaches output contains blank coach name for team_id={team_id}"
            )
        if team_id in seen_team_ids:
            raise ValueError(f"Coaches output contains duplicate team_id={team_id}")

        prior_team = coach_team_assignments.get(coach_id)
        if prior_team is not None and prior_team != team_id:
            raise ValueError(
                "One coach id is assigned to multiple authoritative teams: "
                f"coach_id={coach_id}, teams={prior_team},{team_id}"
            )

        seen_team_ids.add(team_id)
        coach_team_assignments[coach_id] = team_id

    _validate_coach_team_coverage(
        seen_team_ids,
        expected_team_ids,
    )


def temporary_path(final_path: Path) -> Path:
    return final_path.with_name(
        f".{final_path.name}.{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def validate_staged_csv(
    path: Path,
    *,
    expected_rows: list[dict[str, str]],
    teams: list[tuple[str, str]],
) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames != HEADER:
            raise ValueError("Serialized coaches_master.csv header mismatch")
        rows = list(reader)

    if rows != expected_rows:
        raise ValueError(
            "Serialized coaches_master.csv does not exactly match validated rows"
        )

    validate_final_rows(rows, teams=teams)


def publish_atomic(
    rows: list[dict[str, str]],
    *,
    teams: list[tuple[str, str]],
) -> bool:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path(OUTPUT_PATH)

    try:
        write_staged_csv(temp_path, rows)
        validate_staged_csv(
            temp_path,
            expected_rows=rows,
            teams=teams,
        )

        if (
            OUTPUT_PATH.exists()
            and OUTPUT_PATH.read_bytes()
            == temp_path.read_bytes()
        ):
            return False

        os.replace(temp_path, OUTPUT_PATH)
        return True
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def update_report_diagnostics(
    report: PipelineReporter,
    *,
    teams: list[tuple[str, str]],
    rows: list[dict[str, str]],
    failures: list[dict[str, str]],
    diagnostics: dict[str, int],
    output_modified: bool | None,
) -> None:
    resolved_team_ids = {
        str(row.get("team_id") or "").strip()
        for row in rows
        if str(row.get("team_id") or "").strip()
    }
    expected_team_ids = {team_id for team_id, _ in teams}
    missing_team_ids = sorted(
        expected_team_ids - resolved_team_ids,
        key=int,
    )
    unique_coach_ids = {
        str(row.get("id") or "").strip()
        for row in rows
        if str(row.get("id") or "").strip()
    }

    request_failures_by_kind = {key: 0 for key in _REQUEST_COUNTS}
    for failure in _REQUEST_FAILURES:
        request_kind = failure.get("request_kind", "")
        if request_kind in request_failures_by_kind:
            request_failures_by_kind[request_kind] += 1

    details: dict[str, object] = {
        "authoritative_team_count": len(teams),
        "teams_requested": len(teams),
        "teams_successfully_resolved": len(resolved_team_ids),
        "missing_team_count": len(missing_team_ids),
        "missing_team_ids": missing_team_ids,
        "processing_failure_count": len(failures),
        "processing_failure_details": failures,
        "final_row_count": len(rows),
        "unique_team_count": len(resolved_team_ids),
        "unique_coach_id_count": len(unique_coach_ids),
        "coach_list_request_count": _REQUEST_COUNTS["coach_list"],
        "coach_ref_request_count": _REQUEST_COUNTS["coach_ref"],
        "person_ref_request_count": _REQUEST_COUNTS["person_ref"],
        "career_record_request_count": _REQUEST_COUNTS["career_record"],
        "coach_list_request_failure_count": (
            request_failures_by_kind["coach_list"]
        ),
        "coach_ref_request_failure_count": (
            request_failures_by_kind["coach_ref"]
        ),
        "person_ref_request_failure_count": (
            request_failures_by_kind["person_ref"]
        ),
        "career_record_request_failure_count": (
            request_failures_by_kind["career_record"]
        ),
        "espn_request_failure_count": len(_REQUEST_FAILURES),
        "espn_request_failure_details": _REQUEST_FAILURES,
        "espn_request_attempt_count": _RETRY_STATS["request_attempt_count"],
        "espn_request_retry_count": _RETRY_STATS["request_retry_count"],
        "espn_recovered_transient_request_count": _RETRY_STATS[
            "recovered_transient_requests"
        ],
        "espn_exhausted_transient_failure_count": _RETRY_STATS[
            "exhausted_transient_failures"
        ],
        "espn_request_retry_details": _REQUEST_RETRY_DETAILS,
        "explicit_role_selections": diagnostics["explicit_role_selections"],
        "sole_provider_candidate_selections": diagnostics[
            "sole_provider_candidate_selections"
        ],
        "missing_person_ref_count": diagnostics["missing_person_ref_count"],
        "missing_career_records_collection_count": diagnostics[
            "missing_career_records_collection_count"
        ],
        "missing_total_career_record_count": diagnostics[
            "missing_total_career_record_count"
        ],
        "missing_postseason_career_record_count": diagnostics[
            "missing_postseason_career_record_count"
        ],
        "output_path": str(OUTPUT_PATH),
    }

    if output_modified is not None:
        details["output_modified"] = output_modified

    report.update_details(details)


def run(report: PipelineReporter) -> int:
    reset_runtime_state()

    season, season_type, week = load_current_week_config(CURRENT_WEEK_CONFIG_PATH)
    report.season = season
    report.week = week
    report.set_detail("season_type", season_type)

    teams = load_authoritative_teams(
        season=season,
        season_type=season_type,
    )
    report.set_rows(rows_in=len(teams))

    rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    diagnostics = empty_diagnostics()

    try:
        rows, failures, diagnostics = build_rows(
            teams=teams,
            season=season,
        )

        if failures:
            failure_examples: list[str] = []

            for failure in failures[:5]:
                message = " ".join(
                    str(
                        failure.get(
                            "message",
                            "",
                        )
                    ).split()
                )

                failure_examples.append(
                    f"{failure.get('team_abbr', '')}"
                    f"({failure.get('team_id', '')}): "
                    f"{failure.get('error_type', '')}: "
                    f"{message}"
                )

            raise RuntimeError(
                "Failed to resolve complete authoritative "
                "head-coach coverage: "
                f"failure_count={len(failures)}; "
                f"examples={' | '.join(failure_examples)}"
            )

        validate_final_rows(rows, teams=teams)
        output_modified = publish_atomic(rows, teams=teams)

        report.set_rows(
            rows_in=len(teams),
            rows_out=len(rows),
        )
        update_report_diagnostics(
            report,
            teams=teams,
            rows=rows,
            failures=failures,
            diagnostics=diagnostics,
            output_modified=output_modified,
        )
        return 0
    except Exception:
        update_report_diagnostics(
            report,
            teams=teams,
            rows=rows,
            failures=failures,
            diagnostics=diagnostics,
            output_modified=None,
        )
        raise


def main() -> int:
    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "ESPN Core API",
        },
    ) as report:
        report.set_detail("output_modified", False)
        report.add_input(CURRENT_WEEK_CONFIG_PATH)
        report.add_input(LEAGUE_MASTER_PATH)
        report.add_output(OUTPUT_PATH)
        return run(report)

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(main())
