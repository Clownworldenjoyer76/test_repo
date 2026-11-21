#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/00_intake/transform_hockey_odds.py

from __future__ import annotations

import csv
import json
import math
import traceback
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SCRIPTS_DIR = str(Path(__file__).resolve().parents[1])
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from odds_price_common import american_to_decimal as _price_decimal


ET = ZoneInfo(
    "America/New_York"
)

TOTAL_MIN = 5.5
TOTAL_MAX = 7.5

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

ODDS_DIR = (
    BASE_DIR
    / "odds"
)

SPORTSBOOK_DIR = (
    BASE_DIR
    / "00_intake"
    / "sportsbook"
)

ERROR_DIR = (
    BASE_DIR
    / "errors"
    / "00_intake"
)

LOG_FILE = (
    ERROR_DIR
    / "transform_hockey_odds.txt"
)

SPORTSBOOK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROVIDER_PRIORITY = [
    "Titanbets",
    "MGM",
    "Bet365",
    "Caesars Sportsbook",
    "SugarHouse",
]

FIELDS = [
    "sportsbook_event_id",
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


def reset_log() -> None:
    with open(
        LOG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"=== transform_hockey_odds RUN "
            f"{datetime.now(ET).isoformat()} ===\n"
        )


def log(
    msg: str,
) -> None:
    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{datetime.now(ET).isoformat()} "
            f"| {msg}\n"
        )


def blank(
    value,
) -> bool:
    if value is None:
        return True

    return (
        str(value)
        .strip()
        .lower()
        in {
            "",
            "none",
            "null",
            "nan",
            "n/a",
        }
    )


def numeric(
    value,
) -> float | None:
    if blank(
        value
    ):
        return None

    try:
        out = float(
            value
        )
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        math.isnan(out)
        or math.isinf(out)
    ):
        return None

    return out


def clean_line(
    value,
) -> str:
    number = numeric(
        value
    )

    if number is None:
        return ""

    if number.is_integer():
        return str(
            int(number)
        )

    return str(
        number
    )


def clean_american(
    value,
) -> str:
    number = numeric(
        value
    )

    if (
        number is None
        or number == 0
    ):
        return ""

    rounded = int(
        round(number)
    )

    if rounded > 0:
        return f"+{rounded}"

    return str(
        rounded
    )


def clean_decimal(
    value,
) -> str:
    number = numeric(value)
    if number is None or number <= 1:
        return ""
    # Keep enough digits for EV/Kelly and settlement to use the same payout.
    return format(number, ".15g")


def american_to_decimal(
    value,
) -> str:
    return clean_decimal(_price_decimal(value))


def first_value(
    row: dict,
    *keys: str,
):
    for key in keys:
        value = row.get(
            key
        )

        if not blank(
            value
        ):
            return value

    return ""


def canonical_market_decimal(
    row: dict,
    american: str,
    raw_decimal_key: str,
) -> str:
    """Use exact American price; log rounded ESPN decimal without trusting it."""
    canonical = american_to_decimal(american)
    if not canonical:
        return ""

    raw_fraction = first_value(
        row,
        raw_decimal_key.replace("_decimal", "_fraction"),
    )
    if not blank(raw_fraction):
        try:
            numerator, denominator = str(raw_fraction).split("/", 1)
            fractional_decimal = 1 + float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            log(f"PRICE REJECTED {raw_decimal_key}: invalid fractional={raw_fraction}")
            return ""
        if (
            not math.isfinite(fractional_decimal)
            or abs(fractional_decimal - float(canonical)) > 1e-6
        ):
            log(
                f"PRICE REJECTED {raw_decimal_key}: "
                f"american={american} fraction={raw_fraction}"
            )
            return ""

    raw_decimal = numeric(first_value(row, raw_decimal_key))
    if raw_decimal is not None and abs(raw_decimal - float(canonical)) > 1e-6:
        log(
            f"PRICE RECONCILED {raw_decimal_key}: "
            f"american={american} raw_decimal={raw_decimal} "
            f"canonical_decimal={canonical}"
        )

    return canonical


def to_et_date_time(
    date_str: str,
) -> tuple[
    str,
    str,
]:
    if not date_str:
        return "", ""

    try:
        dt = datetime.fromisoformat(
            str(
                date_str
            ).replace(
                "Z",
                "+00:00",
            )
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=ET
            )

        dt_et = dt.astimezone(
            ET
        )

        return (
            dt_et.strftime(
                "%Y_%m_%d"
            ),
            dt_et.strftime(
                "%H:%M"
            ),
        )

    except (TypeError, ValueError, OverflowError):
        return "", ""


def get_status(
    event: dict,
) -> str:
    return str(
        event.get(
            "status",
            "",
        )
    ).strip().lower()


def provider_name(
    row: dict,
) -> str:
    return str(
        row.get(
            "provider_name",
            "",
        )
    ).strip()


def provider_id(
    row: dict,
) -> str:
    return str(
        row.get(
            "provider_id",
            "",
        )
    ).strip()


def is_live_provider(
    row: dict,
) -> bool:
    return (
        "live odds"
        in provider_name(
            row
        ).lower()
    )


def provider_group_rank(
    row: dict,
) -> int:
    name = (
        provider_name(
            row
        ).lower()
    )

    if name == "titanbets":
        return 0

    if name == "mgm":
        return 1

    if name == "bet365":
        return 2

    if name.startswith(
        "caesars sportsbook"
    ):
        return 3

    if name == "sugarhouse":
        return 4

    return 5


def provider_priority_value(
    row: dict,
) -> int:
    value = numeric(
        row.get(
            "provider_priority"
        )
    )

    if value is None:
        return 999999

    return int(
        value
    )


def ordered_provider_rows(
    rows: list[dict],
) -> list[dict]:
    usable = [
        row
        for row in rows
        if (
            isinstance(
                row,
                dict,
            )
            and not is_live_provider(
                row
            )
        )
    ]

    return sorted(
        usable,
        key=lambda row: (
            provider_group_rank(
                row
            ),
            provider_priority_value(
                row
            ),
            provider_name(
                row
            ).lower(),
            provider_id(
                row
            ),
        ),
    )


def moneyline_from_provider(
    row: dict,
) -> dict | None:
    home_american = (
        clean_american(
            first_value(
                row,
                "home_team_odds_current_money_line_american",
                "home_team_odds_money_line",
            )
        )
    )

    away_american = (
        clean_american(
            first_value(
                row,
                "away_team_odds_current_money_line_american",
                "away_team_odds_money_line",
            )
        )
    )

    if (
        not home_american
        or not away_american
    ):
        return None

    home_decimal = canonical_market_decimal(
        row,
        home_american,
        "home_team_odds_current_money_line_decimal",
    )

    away_decimal = canonical_market_decimal(
        row,
        away_american,
        "away_team_odds_current_money_line_decimal",
    )

    if not away_decimal or not home_decimal:
        return None

    return {
        "home_american": (
            home_american
        ),
        "away_american": (
            away_american
        ),
        "home_decimal": (
            home_decimal
        ),
        "away_decimal": (
            away_decimal
        ),
    }


def puck_line_from_provider(
    row: dict,
) -> dict | None:
    home_line = numeric(
        first_value(
            row,
            "home_team_odds_current_point_spread_american",
            "home_team_odds_current_point_spread_alternate_display_value",
        )
    )

    away_line = numeric(
        first_value(
            row,
            "away_team_odds_current_point_spread_american",
            "away_team_odds_current_point_spread_alternate_display_value",
        )
    )

    if (
        home_line is None
        and away_line is not None
    ):
        home_line = (
            -away_line
        )

    if (
        away_line is None
        and home_line is not None
    ):
        away_line = (
            -home_line
        )

    if (
        home_line is None
        or away_line is None
    ):
        return None

    if (
        abs(
            abs(home_line)
            - 1.5
        )
        > 1e-9
        or abs(
            abs(away_line)
            - 1.5
        )
        > 1e-9
    ):
        return None

    home_american = (
        clean_american(
            first_value(
                row,
                "home_team_odds_current_spread_american",
                "home_team_odds_spread_odds",
            )
        )
    )

    away_american = (
        clean_american(
            first_value(
                row,
                "away_team_odds_current_spread_american",
                "away_team_odds_spread_odds",
            )
        )
    )

    if (
        not home_american
        or not away_american
    ):
        return None

    home_decimal = canonical_market_decimal(
        row,
        home_american,
        "home_team_odds_current_spread_decimal",
    )

    away_decimal = canonical_market_decimal(
        row,
        away_american,
        "away_team_odds_current_spread_decimal",
    )

    if not away_decimal or not home_decimal:
        return None

    return {
        "home_line": clean_line(
            home_line
        ),
        "away_line": clean_line(
            away_line
        ),
        "home_american": (
            home_american
        ),
        "away_american": (
            away_american
        ),
        "home_decimal": (
            home_decimal
        ),
        "away_decimal": (
            away_decimal
        ),
    }


def total_from_provider(
    row: dict,
) -> dict | None:
    total = numeric(
        first_value(
            row,
            "current_total_american",
            "current_total_alternate_display_value",
            "over_under",
        )
    )

    if (
        total is None
        or total < TOTAL_MIN
        or total > TOTAL_MAX
    ):
        return None

    over_american = (
        clean_american(
            first_value(
                row,
                "current_over_american",
                "over_odds",
            )
        )
    )

    under_american = (
        clean_american(
            first_value(
                row,
                "current_under_american",
                "under_odds",
            )
        )
    )

    if (
        not over_american
        or not under_american
    ):
        return None

    over_decimal = canonical_market_decimal(
        row,
        over_american,
        "current_over_decimal",
    )

    under_decimal = canonical_market_decimal(
        row,
        under_american,
        "current_under_decimal",
    )

    if not under_decimal or not over_decimal:
        return None

    return {
        "total": clean_line(
            total
        ),
        "over_american": (
            over_american
        ),
        "under_american": (
            under_american
        ),
        "over_decimal": (
            over_decimal
        ),
        "under_decimal": (
            under_decimal
        ),
    }


def select_market(
    sportsbook_event_id: str,
    rows: list[dict],
    market_name: str,
    parser,
    counters: dict,
) -> tuple[
    dict,
    dict,
]:
    ordered = (
        ordered_provider_rows(
            rows
        )
    )

    for index, provider_row in enumerate(
        ordered
    ):
        parsed = parser(
            provider_row
        )

        if parsed is None:
            continue

        name = (
            provider_name(
                provider_row
            )
            or provider_id(
                provider_row
            )
        )

        counters[
            f"{market_name}_provider:{name}"
        ] += 1

        if index > 0:
            counters[
                f"{market_name}_fallback_used"
            ] += 1

        return (
            parsed,
            provider_row,
        )

    counters[
        "warnings"
    ] += 1

    log(
        f"WARNING sportsbook_event_id="
        f"{sportsbook_event_id} "
        f"no usable {market_name} "
        f"from any non-live ESPN provider"
    )

    return {}, {}


def build_row(
    event: dict,
    provider_rows: list[dict],
    pulled_at: str,
    counters: dict,
) -> dict:
    sportsbook_event_id = str(
        event.get(
            "id",
            "",
        )
    ).strip()

    moneyline, ml_provider = (
        select_market(
            sportsbook_event_id,
            provider_rows,
            "moneyline",
            moneyline_from_provider,
            counters,
        )
    )

    puck_line, pl_provider = (
        select_market(
            sportsbook_event_id,
            provider_rows,
            "puck_line",
            puck_line_from_provider,
            counters,
        )
    )

    total, total_provider = (
        select_market(
            sportsbook_event_id,
            provider_rows,
            "total",
            total_from_provider,
            counters,
        )
    )

    game_date, game_time = (
        to_et_date_time(
            str(
                event.get(
                    "date",
                    "",
                )
            )
        )
    )

    return {
        "sportsbook_event_id": (
            sportsbook_event_id
        ),
        "sport": "hockey",
        "league": "nhl",
        "game_date": (
            game_date
        ),
        "game_time": (
            game_time
        ),
        "home_team": str(
            event.get(
                "home",
                "",
            )
        ).strip(),
        "away_team": str(
            event.get(
                "away",
                "",
            )
        ).strip(),
        "home_dk_moneyline_american": (
            moneyline.get(
                "home_american",
                "",
            )
        ),
        "away_dk_moneyline_american": (
            moneyline.get(
                "away_american",
                "",
            )
        ),
        "home_puck_line": (
            puck_line.get(
                "home_line",
                "",
            )
        ),
        "away_puck_line": (
            puck_line.get(
                "away_line",
                "",
            )
        ),
        "total": (
            total.get(
                "total",
                "",
            )
        ),
        "home_dk_puck_line_american": (
            puck_line.get(
                "home_american",
                "",
            )
        ),
        "away_dk_puck_line_american": (
            puck_line.get(
                "away_american",
                "",
            )
        ),
        "dk_total_over_american": (
            total.get(
                "over_american",
                "",
            )
        ),
        "dk_total_under_american": (
            total.get(
                "under_american",
                "",
            )
        ),
        "home_dk_moneyline_decimal": (
            moneyline.get(
                "home_decimal",
                "",
            )
        ),
        "away_dk_moneyline_decimal": (
            moneyline.get(
                "away_decimal",
                "",
            )
        ),
        "home_dk_puck_line_decimal": (
            puck_line.get(
                "home_decimal",
                "",
            )
        ),
        "away_dk_puck_line_decimal": (
            puck_line.get(
                "away_decimal",
                "",
            )
        ),
        "dk_total_over_decimal": (
            total.get(
                "over_decimal",
                "",
            )
        ),
        "dk_total_under_decimal": (
            total.get(
                "under_decimal",
                "",
            )
        ),
        "odds_source": "espn",
        "moneyline_provider_id": (
            provider_id(
                ml_provider
            )
        ),
        "moneyline_provider_name": (
            provider_name(
                ml_provider
            )
        ),
        "puck_line_provider_id": (
            provider_id(
                pl_provider
            )
        ),
        "puck_line_provider_name": (
            provider_name(
                pl_provider
            )
        ),
        "total_provider_id": (
            provider_id(
                total_provider
            )
        ),
        "total_provider_name": (
            provider_name(
                total_provider
            )
        ),
        "pulled_at": (
            pulled_at
        ),
    }


def read_existing_csv(
    path: Path,
) -> dict[
    str,
    dict,
]:
    if not path.exists():
        return {}

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(
            f
        )

        rows = {}

        for row in reader:
            sportsbook_event_id = str(
                row.get(
                    "sportsbook_event_id"
                )
                or row.get(
                    "game_id"
                )
                or ""
            ).strip()

            if not sportsbook_event_id:
                continue

            normalized_row = {
                field: row.get(
                    field,
                    "",
                )
                for field in FIELDS
            }

            normalized_row[
                "sportsbook_event_id"
            ] = sportsbook_event_id

            rows[
                sportsbook_event_id
            ] = normalized_row

    return rows


def row_changed(
    existing: dict,
    new: dict,
) -> bool:
    return any(
        str(
            existing.get(
                field,
                "",
            )
        )
        != str(
            new.get(
                field,
                "",
            )
        )
        for field in FIELDS
    )


def write_csv(
    path: Path,
    rows_by_event_id: dict[
        str,
        dict,
    ],
) -> None:
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for row in (
            rows_by_event_id.values()
        ):
            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in FIELDS
                }
            )


def main() -> None:
    reset_log()

    counters = defaultdict(
        int
    )

    try:
        json_files = sorted(
            ODDS_DIR.glob(
                "*.json"
            )
        )

        log(
            f"Odds input directory: "
            f"{ODDS_DIR}"
        )

        log(
            f"JSON files found: "
            f"{len(json_files)}"
        )

        new_rows_by_date = (
            defaultdict(
                dict
            )
        )

        status_by_date_event = {}

        for json_file in json_files:
            with open(
                json_file,
                "r",
                encoding="utf-8",
            ) as f:
                payload = json.load(
                    f
                )

            source = str(
                payload.get(
                    "source",
                    "",
                )
            ).strip().lower()

            if source != "espn":
                counters[
                    "legacy_json_files_skipped"
                ] += 1

                log(
                    f"SKIPPED "
                    f"{json_file} "
                    f"source="
                    f"{source or 'unknown'}"
                )

                continue

            counters[
                "json_files_processed"
            ] += 1

            events = payload.get(
                "events",
                [],
            )

            odds = payload.get(
                "odds",
                [],
            )

            pulled_at = str(
                payload.get(
                    "generated_at_et",
                    "",
                )
            ).strip()

            if not isinstance(
                events,
                list,
            ):
                counters[
                    "warnings"
                ] += 1

                log(
                    f"WARNING "
                    f"{json_file} "
                    f"events was not a list"
                )

                events = []

            if not isinstance(
                odds,
                list,
            ):
                counters[
                    "warnings"
                ] += 1

                log(
                    f"WARNING "
                    f"{json_file} "
                    f"odds was not a list"
                )

                odds = []

            counters[
                "events_found"
            ] += len(
                events
            )

            counters[
                "odds_provider_rows_found"
            ] += len(
                odds
            )

            events_by_id = {
                str(
                    event.get(
                        "id",
                        "",
                    )
                ).strip(): event
                for event in events
                if (
                    isinstance(
                        event,
                        dict,
                    )
                    and str(
                        event.get(
                            "id",
                            "",
                        )
                    ).strip()
                )
            }

            provider_rows_by_event = (
                defaultdict(
                    list
                )
            )

            for provider_row in odds:
                if not isinstance(
                    provider_row,
                    dict,
                ):
                    counters[
                        "warnings"
                    ] += 1
                    continue

                event_id = str(
                    provider_row.get(
                        "espn_event_id"
                    )
                    or provider_row.get(
                        "event_id"
                    )
                    or provider_row.get(
                        "id"
                    )
                    or ""
                ).strip()

                if not event_id:
                    counters[
                        "warnings"
                    ] += 1

                    log(
                        f"WARNING "
                        f"{json_file} "
                        f"provider row missing "
                        f"ESPN event ID"
                    )

                    continue

                provider_rows_by_event[
                    event_id
                ].append(
                    provider_row
                )

            for (
                sportsbook_event_id,
                event,
            ) in events_by_id.items():
                provider_rows = (
                    provider_rows_by_event.get(
                        sportsbook_event_id,
                        [],
                    )
                )

                if not provider_rows:
                    counters[
                        "warnings"
                    ] += 1

                    log(
                        f"WARNING event "
                        f"sportsbook_event_id="
                        f"{sportsbook_event_id} "
                        f"has no ESPN "
                        f"provider rows"
                    )

                    continue

                row = build_row(
                    event,
                    provider_rows,
                    pulled_at,
                    counters,
                )

                game_date = str(
                    row.get(
                        "game_date",
                        "",
                    )
                ).strip()

                if not game_date:
                    counters[
                        "warnings"
                    ] += 1

                    log(
                        f"WARNING "
                        f"sportsbook_event_id="
                        f"{sportsbook_event_id} "
                        f"skipped because "
                        f"game_date was blank"
                    )

                    continue

                new_rows_by_date[
                    game_date
                ][
                    sportsbook_event_id
                ] = row

                status_by_date_event[
                    (
                        game_date,
                        sportsbook_event_id,
                    )
                ] = get_status(
                    event
                )

                counters[
                    "rows_built"
                ] += 1

            log(
                f"READ {json_file} "
                f"| events={len(events)} "
                f"| provider_rows="
                f"{len(odds)}"
            )

        for (
            game_date,
            new_rows,
        ) in sorted(
            new_rows_by_date.items()
        ):
            csv_path = (
                SPORTSBOOK_DIR
                / f"NHL_{game_date}.csv"
            )

            existing_rows = (
                read_existing_csv(
                    csv_path
                )
            )

            merged_rows = dict(
                existing_rows
            )

            for (
                sportsbook_event_id,
                new_row,
            ) in new_rows.items():
                current_status = (
                    status_by_date_event.get(
                        (
                            game_date,
                            sportsbook_event_id,
                        ),
                        "",
                    )
                )

                existing_row = (
                    existing_rows.get(
                        sportsbook_event_id
                    )
                )

                if (
                    existing_row
                    and current_status
                    not in (
                        "",
                        "pending",
                    )
                ):
                    merged_rows[
                        sportsbook_event_id
                    ] = existing_row

                    counters[
                        "rows_preserved_status_changed"
                    ] += 1

                    continue

                if (
                    not existing_row
                    and current_status
                    not in (
                        "",
                        "pending",
                    )
                ):
                    counters[
                        "rows_skipped_non_pending_no_existing"
                    ] += 1

                    counters[
                        "warnings"
                    ] += 1

                    continue

                if not existing_row:
                    merged_rows[
                        sportsbook_event_id
                    ] = new_row

                    counters[
                        "rows_added"
                    ] += 1

                    continue

                if row_changed(
                    existing_row,
                    new_row,
                ):
                    merged_rows[
                        sportsbook_event_id
                    ] = new_row

                    counters[
                        "rows_updated"
                    ] += 1

                else:
                    merged_rows[
                        sportsbook_event_id
                    ] = existing_row

                    counters[
                        "rows_unchanged"
                    ] += 1

            write_csv(
                csv_path,
                merged_rows,
            )

            counters[
                "csv_files_written"
            ] += 1

            log(
                f"WROTE "
                f"{csv_path} "
                f"rows="
                f"{len(merged_rows)}"
            )

        log(
            f"Provider priority: "
            f"{' -> '.join(PROVIDER_PRIORITY)} "
            f"-> others"
        )

        log(
            "Live Odds providers excluded "
            "from production selection"
        )

        log(
            "--- SUMMARY ---"
        )

        for key in sorted(
            counters
        ):
            log(
                f"{key}: "
                f"{counters[key]}"
            )

        log(
            "STATUS: SUCCESS"
        )

        print(
            "NHL odds transform complete."
        )

    except Exception as exc:
        counters[
            "errors"
        ] += 1

        log(
            f"FATAL ERROR: "
            f"{exc}"
        )

        log(
            traceback.format_exc()
        )

        log(
            "STATUS: FAILED"
        )

        raise


if __name__ == "__main__":
    main()