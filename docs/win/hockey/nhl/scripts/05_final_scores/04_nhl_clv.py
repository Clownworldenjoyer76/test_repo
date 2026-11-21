#!/usr/bin/env python3
# docs/win/hockey/nhl/scripts/05_final_scores/04_nhl_clv.py

from __future__ import annotations

import csv
import json
import math
import re
import sys
import traceback
from datetime import datetime, UTC
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


NHL_ROOT = Path("docs/win/hockey/nhl")
SELECT_DIR = NHL_ROOT / "04_select"
SPORTSBOOK_DIR = NHL_ROOT / "00_intake" / "sportsbook"
SNAPSHOT_ROOT = NHL_ROOT / "odds" / "snapshots"
OUTPUT_DIR = NHL_ROOT / "05_final_scores" / "clv"
ERROR_DIR = NHL_ROOT / "05_final_scores" / "errors"
LOG_FILE = ERROR_DIR / "04_nhl_clv.txt"

SUMMARY_FILE = OUTPUT_DIR / "NHL_clv.csv"
HISTORY_FILE = OUTPUT_DIR / "NHL_clv_snapshot_history.csv"

ET = ZoneInfo("America/New_York")
GAME_ID_RE = re.compile(r"^\d{10}$")
TOTAL_MIN = 5.5
TOTAL_MAX = 7.5

PROVIDER_PRIORITY = [
    "Titanbets",
    "MGM",
    "Bet365",
    "Caesars Sportsbook",
    "SugarHouse",
]

SELECT_REQUIRED_COLUMNS = [
    "sport",
    "league",
    "game_date",
    "game_time",
    "game_id",
    "away_team",
    "home_team",
    "market_type",
    "bet_side",
    "line",
    "take_bet",
    "dk_odds_american",
    "dk_odds_decimal",
    "model_prob",
    "edge",
    "ev",
    "kelly",
    "selected_provider_id",
    "selected_provider_name",
    "odds_source",
    "pulled_at",
]

SPORTSBOOK_REQUIRED_COLUMNS = [
    "game_id",
    "sportsbook_event_id",
]

SUMMARY_COLUMNS = [
    *SELECT_REQUIRED_COLUMNS,
    "source_select_file",
    "sportsbook_event_id",
    "game_start_utc",
    "selected_pulled_at_utc",
    "later_pregame_snapshot_count",
    "closing_snapshot_at_utc",
    "closing_provider_id",
    "closing_provider_name",
    "closing_line",
    "closing_odds_american",
    "closing_odds_decimal",
    "selected_implied_probability",
    "closing_implied_probability",
    "price_comparable",
    "clv_implied_probability",
    "clv_decimal_ratio",
    "line_clv",
    "clv_status",
]

HISTORY_COLUMNS = [
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "market_type",
    "bet_side",
    "take_bet",
    "sportsbook_event_id",
    "selected_provider_id",
    "selected_provider_name",
    "selected_line",
    "selected_odds_american",
    "selected_odds_decimal",
    "selected_pulled_at_utc",
    "game_start_utc",
    "snapshot_at_utc",
    "reference_provider_id",
    "reference_provider_name",
    "reference_line",
    "reference_odds_american",
    "reference_odds_decimal",
    "price_comparable",
    "snapshot_clv_implied_probability",
    "snapshot_clv_decimal_ratio",
    "snapshot_line_clv",
    "is_closing_reference",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    ensure_dirs()
    LOG_FILE.write_text(
        f"=== 04_nhl_clv RUN {now()} ===\n",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{now()} | {message}\n")


def blank(value) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {
        "",
        "none",
        "null",
        "nan",
        "n/a",
    }


def text(value) -> str:
    if blank(value):
        return ""
    return str(value).strip()


def numeric(value) -> float | None:
    if blank(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def parse_timestamp(value) -> datetime | None:
    value = text(value)
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(UTC)


def parse_game_start(game_date, game_time) -> datetime | None:
    date_text = text(game_date).replace("-", "_")
    time_text = text(game_time)
    for fmt in ("%Y_%m_%d %H:%M:%S", "%Y_%m_%d %H:%M"):
        try:
            local = datetime.strptime(
                f"{date_text} {time_text}",
                fmt,
            ).replace(tzinfo=ET)
            return local.astimezone(UTC)
        except ValueError:
            continue
    return None


def american_to_decimal(value) -> float | None:
    american = numeric(value)
    if american is None or american == 0:
        return None
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def first_value(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if not blank(value):
            return value
    return None


def provider_name(row: dict) -> str:
    return text(row.get("provider_name"))


def provider_id(row: dict) -> str:
    return text(row.get("provider_id"))


def is_live_provider(row: dict) -> bool:
    return "live odds" in provider_name(row).lower()


def provider_group_rank(row: dict) -> int:
    name = provider_name(row).lower()
    if name == "titanbets":
        return 0
    if name == "mgm":
        return 1
    if name == "bet365":
        return 2
    if name.startswith("caesars sportsbook"):
        return 3
    if name == "sugarhouse":
        return 4
    return 5


def provider_priority_value(row: dict) -> int:
    value = numeric(row.get("provider_priority"))
    return int(value) if value is not None else 999999


def ordered_provider_rows(rows: list[dict]) -> list[dict]:
    usable = [
        row
        for row in rows
        if isinstance(row, dict) and not is_live_provider(row)
    ]
    return sorted(
        usable,
        key=lambda row: (
            provider_group_rank(row),
            provider_priority_value(row),
            provider_name(row).lower(),
            provider_id(row),
        ),
    )


def market_price(
    row: dict,
    american_keys: tuple[str, ...],
) -> tuple[float, float] | None:
    # ESPN's rounded decimal quote is display-only; the American quote
    # determines the canonical price used by selection and settlement.
    american = numeric(first_value(row, *american_keys))
    decimal = american_to_decimal(american)
    if american is None or decimal is None or decimal <= 1:
        return None
    return american, decimal


def moneyline_market(
    row: dict,
) -> dict | None:
    sides = {}

    for side in ("home", "away"):
        price = market_price(
            row,
            (
                f"{side}_team_odds_current_money_line_american",
                f"{side}_team_odds_money_line",
            ),
        )

        if price is None:
            return None

        american, decimal = price

        sides[side] = {
            "line": None,
            "american": american,
            "decimal": decimal,
        }

    return sides


def puck_line_market(
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

    if home_line is None and away_line is not None:
        home_line = -away_line
    if away_line is None and home_line is not None:
        away_line = -home_line
    if home_line is None or away_line is None:
        return None

    if (
        abs(abs(home_line) - 1.5) > 1e-9
        or abs(abs(away_line) - 1.5) > 1e-9
    ):
        return None

    sides = {}

    for side, line in (
        ("home", home_line),
        ("away", away_line),
    ):
        price = market_price(
            row,
            (
                f"{side}_team_odds_current_spread_american",
                f"{side}_team_odds_spread_odds",
            ),
        )

        if price is None:
            return None

        american, decimal = price

        sides[side] = {
            "line": line,
            "american": american,
            "decimal": decimal,
        }

    return sides


def total_market(
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

    sides = {}

    for side in ("over", "under"):
        price = market_price(
            row,
            (
                f"current_{side}_american",
                f"{side}_odds",
            ),
        )

        if price is None:
            return None

        american, decimal = price

        sides[side] = {
            "line": total,
            "american": american,
            "decimal": decimal,
        }

    return sides


def reference_market(
    rows: list[dict],
    market_type: str,
    bet_side: str,
) -> dict | None:
    if market_type == "moneyline":
        parser = moneyline_market
    elif market_type == "puck_line":
        parser = puck_line_market
    elif market_type == "total":
        parser = total_market
    else:
        return None

    for provider_row in ordered_provider_rows(rows):
        parsed = parser(provider_row)
        if parsed is None or bet_side not in parsed:
            continue
        return {
            **parsed[bet_side],
            "provider_id": provider_id(provider_row),
            "provider_name": provider_name(provider_row),
        }
    return None


def normalize_market(value) -> str:
    value = text(value).lower()
    if value in {"moneyline", "ml"}:
        return "moneyline"
    if value in {"puck_line", "puckline", "spread"}:
        return "puck_line"
    if value in {"total", "totals"}:
        return "total"
    return value


def normalize_side(value) -> str:
    return text(value).lower()


def validate_game_ids(df: pd.DataFrame, label: str) -> None:
    bad = [
        text(value)
        for value in df["game_id"]
        if not GAME_ID_RE.fullmatch(text(value))
    ]
    if bad:
        raise RuntimeError(
            f"{label} contains non-canonical game_id values: {bad[:10]}"
        )


def load_selected_bets() -> pd.DataFrame:
    files = sorted(SELECT_DIR.glob("*_NHL.csv"))
    if not files:
        raise FileNotFoundError(f"No Stage 04 files found in {SELECT_DIR}")

    parts = []
    for path in files:
        df = pd.read_csv(path, dtype=str)
        missing = [c for c in SELECT_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing required columns: {missing}")
        if df.empty:
            continue
        df = df.copy()
        df["source_select_file"] = path.name
        df["game_id"] = df["game_id"].astype(str).str.strip()
        df["market_type"] = df["market_type"].map(normalize_market)
        df["bet_side"] = df["bet_side"].map(normalize_side)
        validate_game_ids(df, str(path))
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=[*SELECT_REQUIRED_COLUMNS, "source_select_file"])
    return pd.concat(parts, ignore_index=True)


def sportsbook_files() -> list[Path]:
    return sorted(
        path
        for path in SPORTSBOOK_DIR.glob("*.csv")
        if path.name.lower().startswith("nhl_")
    )


def load_event_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sportsbook_files():
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            missing = [c for c in SPORTSBOOK_REQUIRED_COLUMNS if c not in fields]
            if missing:
                raise RuntimeError(f"{path} missing required columns: {missing}")
            for row in reader:
                game_id = text(row.get("game_id"))
                event_id = text(row.get("sportsbook_event_id"))
                if not game_id or not event_id:
                    continue
                if not GAME_ID_RE.fullmatch(game_id):
                    raise RuntimeError(
                        f"{path} contains non-canonical game_id={game_id!r}"
                    )
                prior = mapping.get(game_id)
                if prior is not None and prior != event_id:
                    raise RuntimeError(
                        "Conflicting sportsbook_event_id mapping for "
                        f"game_id={game_id}: {prior} vs {event_id}"
                    )
                mapping[game_id] = event_id
    return mapping


def snapshot_files() -> list[Path]:
    return sorted(SNAPSHOT_ROOT.glob("*/*.json"))


def load_snapshots() -> list[dict]:
    snapshots = []
    for path in snapshot_files():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed reading snapshot {path}: {exc}") from exc

        snapshot_at = parse_timestamp(raw.get("generated_at_et"))
        if snapshot_at is None:
            raise RuntimeError(
                f"Snapshot missing valid generated_at_et: {path}"
            )

        source = text(raw.get("source")).lower()
        events = raw.get("events", [])
        event_status: dict[str, str] = {}
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict):
                    event_status[text(event.get("id"))] = text(
                        event.get("status")
                    ).lower()

        rows_by_event: dict[str, list[dict]] = {}
        odds = raw.get("odds", [])
        if isinstance(odds, list):
            for row in odds:
                if not isinstance(row, dict):
                    continue
                event_id = text(
                    row.get("espn_event_id") or row.get("event_id")
                )
                if event_id:
                    rows_by_event.setdefault(event_id, []).append(row)

        snapshots.append(
            {
                "path": path,
                "snapshot_at": snapshot_at,
                "source": source,
                "event_status": event_status,
                "rows_by_event": rows_by_event,
            }
        )

    snapshots.sort(key=lambda item: item["snapshot_at"])
    return snapshots


def same_line(market_type: str, selected_line, reference_line) -> bool:
    if market_type == "moneyline":
        return True
    selected = numeric(selected_line)
    reference = numeric(reference_line)
    if selected is None or reference is None:
        return False
    return abs(selected - reference) < 1e-9


def implied_probability(decimal_value) -> float | None:
    decimal = numeric(decimal_value)
    if decimal is None or decimal <= 1:
        return None
    return 1.0 / decimal


def price_clv(selected_decimal, reference_decimal, comparable: bool):
    if not comparable:
        return None, None
    selected = numeric(selected_decimal)
    reference = numeric(reference_decimal)
    if selected is None or reference is None or selected <= 1 or reference <= 1:
        return None, None
    selected_prob = 1.0 / selected
    reference_prob = 1.0 / reference
    return (
        reference_prob - selected_prob,
        selected / reference - 1.0,
    )


def favorable_line_clv(
    market_type: str,
    bet_side: str,
    selected_line,
    reference_line,
) -> float | None:
    if market_type == "moneyline":
        return None
    selected = numeric(selected_line)
    reference = numeric(reference_line)
    if selected is None or reference is None:
        return None
    if market_type == "puck_line":
        return selected - reference
    if market_type == "total":
        if bet_side == "over":
            return reference - selected
        if bet_side == "under":
            return selected - reference
    return None


def reference_rows_for_bet(
    *,
    snapshots: list[dict],
    event_id: str,
    source: str,
    market_type: str,
    bet_side: str,
    game_start: datetime,
) -> list[dict]:
    references = []
    for snapshot in snapshots:
        snapshot_at = snapshot["snapshot_at"]
        if snapshot_at >= game_start:
            continue
        if snapshot["source"] != source:
            continue
        status = snapshot["event_status"].get(event_id, "")
        if status in {"in", "post"}:
            continue
        provider_rows = snapshot["rows_by_event"].get(event_id, [])
        if not provider_rows:
            continue
        market = reference_market(provider_rows, market_type, bet_side)
        if market is None:
            continue
        references.append(
            {
                "snapshot_at": snapshot_at,
                **market,
            }
        )
    return references


def _selected_bet_block_reason(
    source: str,
    event_id: str,
    game_start: datetime | None,
    selected_pulled_at: datetime | None,
    selected_decimal: float | None,
) -> str | None:
    if source != "espn":
        return "unsupported_odds_source"
    if not event_id:
        return "sportsbook_event_id_missing"
    if game_start is None:
        return "invalid_game_start"
    if selected_pulled_at is None:
        return "invalid_selected_pulled_at"
    if selected_decimal is None or selected_decimal <= 1:
        return "invalid_selected_odds"
    return None

def process_bets(
    bets: pd.DataFrame,
    event_map: dict[str, str],
    snapshots: list[dict],
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    history_rows: list[dict] = []

    for _, bet_row in bets.iterrows():
        bet = bet_row.to_dict()
        game_id = text(bet.get("game_id"))
        market_type = normalize_market(bet.get("market_type"))
        bet_side = normalize_side(bet.get("bet_side"))
        source = text(bet.get("odds_source")).lower()
        event_id = event_map.get(game_id, "")
        game_start = parse_game_start(
            bet.get("game_date"),
            bet.get("game_time"),
        )
        selected_pulled_at = parse_timestamp(
            bet.get("pulled_at")
        )
        selected_american = numeric(
            bet.get("dk_odds_american")
        )
        selected_decimal = american_to_decimal(selected_american)
        # For legacy selections, the raw rounded decimal is retained in the
        # immutable selection CSV; all CLV price comparisons are revalued
        # against the exact American payout.

        selected_line = numeric(bet.get("line"))

        base_summary = {
            col: text(bet.get(col))
            for col in SELECT_REQUIRED_COLUMNS
        }
        base_summary["source_select_file"] = text(
            bet.get("source_select_file")
        )
        base_summary["sportsbook_event_id"] = event_id
        base_summary["game_start_utc"] = (
            game_start.isoformat() if game_start else ""
        )
        base_summary["selected_pulled_at_utc"] = (
            selected_pulled_at.isoformat() if selected_pulled_at else ""
        )
        base_summary["selected_implied_probability"] = (
            implied_probability(selected_decimal)
        )

        blocked_reason = _selected_bet_block_reason(
            source, event_id, game_start, selected_pulled_at, selected_decimal
        )
        if blocked_reason is not None:
            base_summary["clv_status"] = blocked_reason
            summary_rows.append(base_summary)
            continue

        references = reference_rows_for_bet(
            snapshots=snapshots,
            event_id=event_id,
            source=source,
            market_type=market_type,
            bet_side=bet_side,
            game_start=game_start,
        )

        closing = references[-1] if references else None
        later = [
            ref
            for ref in references
            if ref["snapshot_at"] > selected_pulled_at
        ]

        base_summary["later_pregame_snapshot_count"] = len(later)

        for ref in later:
            comparable = same_line(
                market_type,
                selected_line,
                ref.get("line"),
            )
            clv_prob, clv_ratio = price_clv(
                selected_decimal,
                ref.get("decimal"),
                comparable,
            )
            history_rows.append(
                {
                    "game_id": game_id,
                    "game_date": text(bet.get("game_date")),
                    "away_team": text(bet.get("away_team")),
                    "home_team": text(bet.get("home_team")),
                    "market_type": market_type,
                    "bet_side": bet_side,
                    "take_bet": text(bet.get("take_bet")),
                    "sportsbook_event_id": event_id,
                    "selected_provider_id": text(
                        bet.get("selected_provider_id")
                    ),
                    "selected_provider_name": text(
                        bet.get("selected_provider_name")
                    ),
                    "selected_line": selected_line,
                    "selected_odds_american": selected_american,
                    "selected_odds_decimal": selected_decimal,
                    "selected_pulled_at_utc": selected_pulled_at.isoformat(),
                    "game_start_utc": game_start.isoformat(),
                    "snapshot_at_utc": ref["snapshot_at"].isoformat(),
                    "reference_provider_id": ref.get("provider_id", ""),
                    "reference_provider_name": ref.get("provider_name", ""),
                    "reference_line": ref.get("line"),
                    "reference_odds_american": ref.get("american"),
                    "reference_odds_decimal": ref.get("decimal"),
                    "price_comparable": int(comparable),
                    "snapshot_clv_implied_probability": clv_prob,
                    "snapshot_clv_decimal_ratio": clv_ratio,
                    "snapshot_line_clv": favorable_line_clv(
                        market_type,
                        bet_side,
                        selected_line,
                        ref.get("line"),
                    ),
                    "is_closing_reference": int(ref is closing),
                }
            )

        if closing is None:
            base_summary["clv_status"] = "no_pregame_reference_snapshot"
            summary_rows.append(base_summary)
            continue

        comparable = same_line(
            market_type,
            selected_line,
            closing.get("line"),
        )
        clv_prob, clv_ratio = price_clv(
            selected_decimal,
            closing.get("decimal"),
            comparable,
        )

        base_summary.update(
            {
                "closing_snapshot_at_utc": closing["snapshot_at"].isoformat(),
                "closing_provider_id": closing.get("provider_id", ""),
                "closing_provider_name": closing.get("provider_name", ""),
                "closing_line": closing.get("line"),
                "closing_odds_american": closing.get("american"),
                "closing_odds_decimal": closing.get("decimal"),
                "closing_implied_probability": implied_probability(
                    closing.get("decimal")
                ),
                "price_comparable": int(comparable),
                "clv_implied_probability": clv_prob,
                "clv_decimal_ratio": clv_ratio,
                "line_clv": favorable_line_clv(
                    market_type,
                    bet_side,
                    selected_line,
                    closing.get("line"),
                ),
            }
        )

        if closing["snapshot_at"] <= selected_pulled_at:
            base_summary["clv_status"] = "no_later_closing_snapshot"
        elif comparable:
            base_summary["clv_status"] = "evaluated"
        else:
            base_summary["clv_status"] = "line_changed_price_not_comparable"

        summary_rows.append(base_summary)

    return summary_rows, history_rows


def write_outputs(summary_rows: list[dict], history_rows: list[dict]) -> None:
    summary_df = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_COLUMNS)
    history_df = pd.DataFrame(history_rows).reindex(columns=HISTORY_COLUMNS)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            [
                "game_date",
                "game_time",
                "game_id",
                "market_type",
                "bet_side",
            ],
            kind="mergesort",
        )

    if not history_df.empty:
        history_df = history_df.sort_values(
            [
                "game_date",
                "game_id",
                "market_type",
                "bet_side",
                "snapshot_at_utc",
            ],
            kind="mergesort",
        )

    summary_df.to_csv(SUMMARY_FILE, index=False)
    history_df.to_csv(HISTORY_FILE, index=False)

    log(f"WROTE {SUMMARY_FILE} rows={len(summary_df)}")
    log(f"WROTE {HISTORY_FILE} rows={len(history_df)}")


def main() -> None:
    reset_log()

    try:
        bets = load_selected_bets()
        event_map = load_event_map()
        snapshots = load_snapshots()

        log(f"selected_bets={len(bets)}")
        log(f"game_event_mappings={len(event_map)}")
        log(f"immutable_snapshots={len(snapshots)}")

        if bets.empty:
            write_outputs([], [])
            log("STATUS: SUCCESS | no selected bets")
            print("NHL CLV complete: no selected bets.")
            return

        summary_rows, history_rows = process_bets(
            bets,
            event_map,
            snapshots,
        )
        write_outputs(summary_rows, history_rows)

        status_counts = (
            pd.Series(
                [row.get("clv_status", "") for row in summary_rows]
            )
            .value_counts(dropna=False)
            .to_dict()
        )
        log(f"clv_status_counts={status_counts}")
        log("STATUS: SUCCESS")
        print("NHL CLV complete.")

    except Exception as exc:
        log(
            f"STATUS: FAILED | {exc}\n"
            f"{traceback.format_exc()}"
        )
        print(f"NHL CLV failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
