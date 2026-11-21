#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/nhl_odds_pull.py

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import sportsdataverse.nhl as nhl


ET = ZoneInfo("America/New_York")

JSON_OUT_DIR = Path("docs/win/hockey/nhl/odds")
SNAPSHOT_ROOT = JSON_OUT_DIR / "snapshots"

PROVIDER_PRIORITY = [
    "Titanbets",
    "MGM",
    "Bet365",
    "Caesars Sportsbook",
    "SugarHouse",
]

JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)


def frame_records(obj: Any) -> list[dict]:
    if obj is None:
        return []

    if hasattr(obj, "to_dicts"):
        return [
            dict(row)
            for row in obj.to_dicts()
            if isinstance(row, dict)
        ]

    if hasattr(obj, "to_dict"):
        try:
            rows = obj.to_dict(orient="records")
            return [
                dict(row)
                for row in rows
                if isinstance(row, dict)
            ]
        except TypeError:
            pass

    if isinstance(obj, list):
        return [
            dict(row)
            for row in obj
            if isinstance(row, dict)
        ]

    if isinstance(obj, dict):
        return [dict(obj)]

    raise TypeError(
        f"Unsupported ESPN parsed object: "
        f"{type(obj).__name__}"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if (
        isinstance(
            value,
            (str, int, float, bool),
        )
        or value is None
    ):
        return value

    return str(value)


def normalize_status(event: dict) -> str:
    status = event.get("status", {})

    status_type = (
        status.get("type", {})
        if isinstance(status, dict)
        else {}
    )

    state = str(
        status_type.get(
            "state",
            "",
        )
    ).strip().lower()

    name = str(
        status_type.get(
            "name",
            "",
        )
    ).strip().lower()

    completed = bool(
        status_type.get(
            "completed",
            False,
        )
    )

    if completed or state == "post":
        return "post"

    if state == "in":
        return "in"

    if state == "pre":
        return "pending"

    if (
        "scheduled" in name
        or "pre" in name
    ):
        return "pending"

    return state or name


def competitor_name(
    competition: dict,
    side: str,
) -> str:
    competitors = competition.get(
        "competitors",
        [],
    )

    if not isinstance(
        competitors,
        list,
    ):
        return ""

    for competitor in competitors:
        if not isinstance(
            competitor,
            dict,
        ):
            continue

        if (
            str(
                competitor.get(
                    "homeAway",
                    "",
                )
            )
            .strip()
            .lower()
            != side
        ):
            continue

        team = competitor.get(
            "team",
            {},
        )

        if not isinstance(
            team,
            dict,
        ):
            continue

        for key in (
            "displayName",
            "shortDisplayName",
            "name",
        ):
            value = str(
                team.get(
                    key,
                    "",
                )
            ).strip()

            if value:
                return value

    return ""


def normalize_scoreboard_events(
    scoreboard_raw: dict,
) -> list[dict]:
    raw_events = scoreboard_raw.get(
        "events",
        [],
    )

    if not isinstance(
        raw_events,
        list,
    ):
        return []

    events = []

    for event in raw_events:
        if not isinstance(
            event,
            dict,
        ):
            continue

        event_id = str(
            event.get(
                "id",
                "",
            )
        ).strip()

        if not event_id:
            continue

        competitions = event.get(
            "competitions",
            [],
        )

        competition = (
            competitions[0]
            if (
                isinstance(
                    competitions,
                    list,
                )
                and competitions
                and isinstance(
                    competitions[0],
                    dict,
                )
            )
            else {}
        )

        events.append(
            {
                "id": event_id,
                "date": str(
                    competition.get(
                        "date"
                    )
                    or event.get(
                        "date"
                    )
                    or ""
                ).strip(),
                "status": normalize_status(
                    event
                ),
                "home": competitor_name(
                    competition,
                    "home",
                ),
                "away": competitor_name(
                    competition,
                    "away",
                ),
            }
        )

    return events


def fetch_scoreboard(
    run_day: datetime,
) -> dict:
    dates_arg = int(
        run_day.strftime(
            "%Y%m%d"
        )
    )

    raw = nhl.espn_nhl_scoreboard(
        dates=dates_arg,
        limit=5000,
        return_parsed=False,
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise RuntimeError(
            "ESPN NHL scoreboard returned "
            "a non-dict payload"
        )

    return raw


def fetch_event_odds(
    event_id: str,
) -> tuple[
    list[dict],
    Any,
]:
    parsed = nhl.espn_nhl_game_odds(
        event_id=event_id,
    )

    rows = frame_records(
        parsed
    )

    normalized_rows = []

    for row in rows:
        out = dict(row)

        out[
            "espn_event_id"
        ] = event_id

        normalized_rows.append(
            out
        )

    raw = nhl.espn_nhl_game_odds(
        event_id=event_id,
        return_parsed=False,
    )

    return (
        normalized_rows,
        raw,
    )


def write_json(
    path: Path,
    output: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(
                output
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    generated_at = datetime.now(
        ET
    )

    run_date = (
        generated_at.strftime(
            "%Y_%m_%d"
        )
    )

    snapshot_timestamp = (
        generated_at.strftime(
            "%Y_%m_%d_%H_%M_%S_%f_ET"
        )
    )

    latest_path = (
        JSON_OUT_DIR
        / f"{run_date}.json"
    )

    snapshot_path = (
        SNAPSHOT_ROOT
        / run_date
        / f"{snapshot_timestamp}.json"
    )

    scoreboard_raw = (
        fetch_scoreboard(
            generated_at
        )
    )

    events = (
        normalize_scoreboard_events(
            scoreboard_raw
        )
    )

    odds: list[dict] = []

    odds_raw_by_event: dict[
        str,
        Any,
    ] = {}

    warnings: list[str] = []

    for event in events:
        event_id = str(
            event.get(
                "id",
                "",
            )
        ).strip()

        if not event_id:
            continue

        try:
            parsed_rows, raw = (
                fetch_event_odds(
                    event_id
                )
            )

            odds.extend(
                parsed_rows
            )

            odds_raw_by_event[
                event_id
            ] = raw

        except Exception as exc:
            warning = (
                f"event_id={event_id} "
                f"ESPN odds pull failed: "
                f"{exc}"
            )

            warnings.append(
                warning
            )

            print(
                f"WARNING {warning}"
            )

        time.sleep(
            0.05
        )

    output = {
        "run_date": run_date,
        "generated_at_et": (
            generated_at.isoformat()
        ),
        "source": "espn",
        "league": "nhl",
        "bookmaker_strategy": {
            "type": (
                "priority_fallback_per_market"
            ),
            "priority": (
                PROVIDER_PRIORITY
            ),
            "exclude_live_odds_providers": (
                True
            ),
        },
        "events": events,
        "odds": odds,
        "scoreboard_raw": (
            scoreboard_raw
        ),
        "odds_raw_by_event": (
            odds_raw_by_event
        ),
        "warnings": warnings,
    }

    write_json(
        latest_path,
        output,
    )

    write_json(
        snapshot_path,
        output,
    )

    print(
        f"ESPN NHL events="
        f"{len(events)} "
        f"provider_rows="
        f"{len(odds)} "
        f"warnings="
        f"{len(warnings)}"
    )

    print(
        f"WROTE LATEST "
        f"{latest_path}"
    )

    print(
        f"WROTE SNAPSHOT "
        f"{snapshot_path}"
    )


if __name__ == "__main__":
    main()