#!/usr/bin/env python3
"""Summarize NHL graded props by market, OVER/UNDER, and individual ranges.

Install at:
  G:\\hockey\\docs\\win\\hockey\\nhl\\prop_engine\\scripts\\04_final\\grade_by_market_range.py

Read:
  <prop_engine>/04_final/YYYY_MM_DD_nhl_props_graded.csv (every dated file)

Write (replace each run):
  <prop_engine>/04_final/by_market_range/by_line_range.csv
  <prop_engine>/04_final/by_market_range/by_probability_range.csv
  <prop_engine>/04_final/by_market_range/by_odds_range.csv

Line buckets are 1 wide [0,1), [1,2), ...; probability buckets are
0.05 wide [0,0.05), ..., [0.95,1.00]; odds buckets are 50 American
odds points wide, e.g. [-150,-101] and [+100,+149]. Missing odds
are reported in their own MISSING bucket. All percentages and units
use only decided/settled bets, not PENDING, NO_PLAY or UNVERIFIED.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "04_final"
OUTPUT_DIR = INPUT_DIR / "by_market_range"
FILE_RE = re.compile(r"(?P<date>\d{4}_\d{2}_\d{2})_nhl_props_graded\.csv\Z")
MARKETS = {
    "shots_on_goal", "points", "assists", "goals", "blocked_shots",
    "power_play_points", "saves", "goals_against",
}
STATUSES = ("WIN", "LOSS", "PUSH", "NO_PLAY", "PENDING", "UNVERIFIED")
REQUIRED = {
    "game_date", "game_id", "player_id", "market", "bet_side", "line",
    "selected_probability", "selected_odds", "result", "net_units",
}
METRICS = (
    "selected", "win", "loss", "push", "no_play", "pending", "unverified",
    "decided", "win_rate", "priced_settled", "priced_net_units", "priced_roi",
)
COLUMNS = (
    "market", "bet_side", "range_start", "range_end", "range_label", *METRICS,
)


def decimal_value(raw: str, field: str) -> Decimal:
    try:
        value = Decimal(str(raw).strip())
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"Nonfinite {field}: {raw!r}")
    return value


def line_bucket(row: dict[str, str]) -> tuple[int, str, str, str]:
    value = decimal_value(row["line"], "line")
    if value < 0:
        raise ValueError(f"Negative line: {value}")
    low = int(value // 1)
    return low, str(low), str(low + 1), f"{low} to <{low + 1}"


def probability_bucket(row: dict[str, str]) -> tuple[int, str, str, str]:
    value = decimal_value(row["selected_probability"], "selected_probability")
    if not Decimal(0) <= value <= Decimal(1):
        raise ValueError(f"Probability outside [0,1]: {value}")
    index = min(int(value / Decimal("0.05")), 19)
    low = Decimal(index) * Decimal("0.05")
    high = low + Decimal("0.05")
    a, b = f"{low:.2f}", f"{high:.2f}"
    return index, a, b, (f"{a} to {b}" if index == 19 else f"{a} to <{b}")


def odds_bucket(row: dict[str, str]) -> tuple[int | None, str, str, str]:
    raw = (row["selected_odds"] or "").strip()
    if not raw:
        return None, "", "", "MISSING"
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise ValueError(f"Invalid selected_odds: {raw!r}")
    n = int(raw)
    if -99 <= n <= 99:
        raise ValueError(f"Invalid American odds: {raw!r}")
    low = n // 50 * 50
    high = low + 49
    fmt = lambda i: f"{i:+d}"
    return low, str(low), str(high), f"{fmt(low)} to {fmt(high)}"


def load_rows(input_dir: Path) -> list[dict[str, str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Graded-bets folder not found: {input_dir}")
    paths = sorted(p for p in input_dir.iterdir() if p.is_file() and FILE_RE.fullmatch(p.name))
    if not paths:
        raise FileNotFoundError(f"No YYYY_MM_DD_nhl_props_graded.csv files in {input_dir}")
    all_rows: list[dict[str, str]] = []
    identities: set[tuple[str, str, str, str, str, str]] = set()
    for path in paths:
        match = FILE_RE.fullmatch(path.name)
        assert match is not None
        file_date = date.fromisoformat(match.group("date").replace("_", "-")).isoformat()
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or (missing := REQUIRED - set(reader.fieldnames)):
                raise ValueError(f"{path.name}: missing columns {sorted(missing) if reader.fieldnames else sorted(REQUIRED)}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"{path.name}: duplicate CSV column name")
            count = 0
            for number, row in enumerate(reader, 2):
                try:
                    if any(value is None for value in row.values()):
                        raise ValueError("Malformed CSV row")
                    game_day = date.fromisoformat(row["game_date"].replace("_", "-")).isoformat()
                    if game_day != file_date:
                        raise ValueError("Row date differs from filename date")
                    if row["market"] not in MARKETS:
                        raise ValueError(f"Unknown market: {row['market']!r}")
                    if row["bet_side"] not in ("OVER", "UNDER"):
                        raise ValueError(f"Invalid bet_side: {row['bet_side']!r}")
                    if row["result"] not in STATUSES:
                        raise ValueError(f"Invalid result: {row['result']!r}")
                    if not row["game_id"].isdigit() or not row["player_id"].isdigit():
                        raise ValueError("Invalid NHL game/player identity")
                    line_bucket(row)
                    probability_bucket(row)
                    odds_bucket(row)
                    settled = row["result"] in ("WIN", "LOSS", "PUSH")
                    raw_units = (row["net_units"] or "").strip()
                    priced = bool((row["selected_odds"] or "").strip())
                    if raw_units:
                        if not (priced and settled):
                            raise ValueError("Unexpected net_units for an unpriced/unsettled bet")
                        units = decimal_value(raw_units, "net_units")
                        if row["result"] == "WIN" and units <= 0:
                            raise ValueError("WIN net_units must be positive")
                        if row["result"] == "LOSS" and units != -1:
                            raise ValueError("LOSS net_units must be -1")
                        if row["result"] == "PUSH" and units != 0:
                            raise ValueError("PUSH net_units must be zero")
                    elif priced and settled:
                        raise ValueError("Priced settled bet has blank net_units")
                    identity = (game_day, row["game_id"], row["player_id"],
                                row["market"], row["bet_side"], str(decimal_value(row["line"], "line")))
                    if identity in identities:
                        raise ValueError("Duplicate graded selection")
                    identities.add(identity)
                except (ValueError, InvalidOperation) as exc:
                    raise ValueError(f"{path.name}:{number}: {exc}") from exc
                all_rows.append(row)
                count += 1
        print(f"READ | {path.name} | graded_selections={count}", flush=True)
    return all_rows


def summarize(rows: list[dict[str, str]], bucket_fn) -> list[dict[str, object]]:
    groups = defaultdict(list)
    bucket_info = {}
    for row in rows:
        sort_key, start, end, label = bucket_fn(row)
        group_key = row["market"], row["bet_side"], sort_key
        groups[group_key].append(row)
        bucket_info[group_key] = start, end, label
    output = []
    for key in sorted(groups, key=lambda x: (x[0], x[1], x[2] is None, x[2] if x[2] is not None else 0)):
        market, side, _ = key
        members = groups[key]
        result_counts = {status: sum(row["result"] == status for row in members) for status in STATUSES}
        decided = result_counts["WIN"] + result_counts["LOSS"]
        priced = [row for row in members if (row["net_units"] or "").strip()]
        net = sum((decimal_value(r["net_units"], "net_units") for r in priced), Decimal(0))
        start, end, label = bucket_info[key]
        output.append({
            "market": market, "bet_side": side,
            "range_start": start, "range_end": end, "range_label": label,
            "selected": len(members),
            "win": result_counts["WIN"], "loss": result_counts["LOSS"],
            "push": result_counts["PUSH"], "no_play": result_counts["NO_PLAY"],
            "pending": result_counts["PENDING"], "unverified": result_counts["UNVERIFIED"],
            "decided": decided,
            "win_rate": str(Decimal(result_counts["WIN"]) / Decimal(decided)) if decided else "",
            "priced_settled": len(priced),
            "priced_net_units": str(net) if priced else "",
            "priced_roi": str(net / len(priced)) if priced else "",
        })
    if sum(int(record["selected"]) for record in output) != len(rows):
        raise AssertionError("Range summary dropped graded selections")
    return output


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                   help="Folder containing dated *_nhl_props_graded.csv files")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="Folder for three range summaries")
    args = p.parse_args(argv)
    rows = load_rows(args.input_dir)
    out = (
        ("by_line_range.csv", line_bucket),
        ("by_probability_range.csv", probability_bucket),
        ("by_odds_range.csv", odds_bucket),
    )
    for name, bucketer in out:
        summarized = summarize(rows, bucketer)
        destination = args.output_dir / name
        atomic_csv(destination, summarized)
        print(f"WROTE | {destination} | groups={len(summarized)} | source_selections={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR | {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
