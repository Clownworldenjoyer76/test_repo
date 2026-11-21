#!/usr/bin/env python3
"""Grade the CURRENT NHL prop selections against completed player-game results.

Inputs (read only):
  03_select/YYYY_MM_DD_nhl_props.csv
  data/nhl_historical_player_game_targets_2009_2026.csv

Outputs (overwritten on every run, no locked bets):
  04_final/YYYY_MM_DD_nhl_props_graded.csv
  04_final/summary_by_date.csv
  04_final/summary_by_market.csv

By default, re-grade ALL dated selection CSVs; --date selects one date.
Only official completed outcomes count. Never treat a missing outcome as zero.
Official power-play-points totals are verified via the EXISTING NHL verifier.
An absent sportsbook price never prevents WIN/LOSS/PUSH grading; unit
profitability is blank when the selected side's price is missing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # .../prop_engine/scripts/04_final/grader.py
SELECTIONS = ROOT / "03_select"
DEFAULT_TARGETS = ROOT / "data" / "nhl_historical_player_game_targets_2009_2026.csv"
DEFAULT_OUTPUT = ROOT / "04_final"
MARKETS = {
    "shots_on_goal": "SKATER",
    "points": "SKATER",
    "assists": "SKATER",
    "goals": "SKATER",
    "blocked_shots": "SKATER",
    "power_play_points": "SKATER",
    "saves": "GOALIE",
    "goals_against": "GOALIE",
}
SELECTION_COLUMNS = (
    "game_date", "game_id", "player_id", "team", "opponent_team",
    "home_away", "game_start_utc", "over_probability", "under_probability",
    "line", "market", "bet_side", "over_odds", "under_odds",
)
GRADE_COLUMNS = (
    *SELECTION_COLUMNS, "actual", "result", "selected_probability",
    "selected_odds", "net_units", "detail",
)
SUMMARY_COLUMNS = (
    "selected", "win", "loss", "push", "no_play", "pending", "unverified",
    "decided", "win_rate", "priced_decided", "priced_net_units", "priced_roi",
)
DATE_SUMMARY_COLUMNS = ("game_date", *SUMMARY_COLUMNS)
MARKET_SUMMARY_COLUMNS = ("market", "bet_side", *SUMMARY_COLUMNS)
VALID_DATE_FILE = re.compile(r"(\d{4}_\d{2}_\d{2})_nhl_props\.csv")
VALID_ID = re.compile(r"\d+")


def iso_date(value: str) -> str:
    raw = str(value).strip().replace("_", "-")
    return date.fromisoformat(raw).isoformat()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing required headers: {sorted(missing)}")
        return list(reader)


def selection_dates(selection_dir: Path) -> list[str]:
    days = []
    for p in selection_dir.glob("*_nhl_props.csv"):
        match = VALID_DATE_FILE.fullmatch(p.name)
        if match:
            days.append(iso_date(match.group(1)))
    return sorted(set(days))


def finite_number(raw: str, field: str) -> float:
    try:
        value = float(str(raw).strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {field}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field}: {raw!r}")
    return value


def american_odds(raw: str) -> int | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    if not re.fullmatch(r"[+-]?\d+", raw):
        raise ValueError(f"Invalid American odds: {raw!r}")
    odds = int(raw)
    if -99 <= odds <= 99:
        raise ValueError(f"Invalid American odds: {raw!r}")
    return odds


def validate_selections(rows: list[dict[str, str]], day: str) -> None:
    seen: set[tuple[str, str, str, str, str]] = set()
    for r in rows:
        if iso_date(r["game_date"]) != day:
            raise ValueError(f"Selected date does not match {day}: {r['game_date']}")
        gid, pid = r["game_id"].strip(), r["player_id"].strip()
        market, side = r["market"].strip(), r["bet_side"].strip().upper()
        if not VALID_ID.fullmatch(gid) or not VALID_ID.fullmatch(pid):
            raise ValueError(f"Invalid selection IDs: {gid}/{pid}")
        if market not in MARKETS or side not in ("OVER", "UNDER"):
            raise ValueError(f"Invalid market/side for {gid}/{pid}: {market}/{side}")
        line = finite_number(r["line"], "betting line")
        if line < 0:
            raise ValueError(f"Negative betting line for {gid}/{pid}/{market}")
        over = finite_number(r["over_probability"], "over_probability")
        under = finite_number(r["under_probability"], "under_probability")
        if not (0 <= over <= 1 and 0 <= under <= 1 and over + under <= 1 + 1e-8):
            raise ValueError(f"Invalid probabilities for {gid}/{pid}/{market}")
        selected_price = r["over_odds"] if side == "OVER" else r["under_odds"]
        american_odds(selected_price)
        key = (gid, pid, market, str(line), side)
        if key in seen:
            raise ValueError(f"Duplicate selected bet {key}")
        seen.add(key)


def load_outcomes(path: Path, selected_games: dict[str, str]):
    """One pass over the target file, restricted to selected game IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Historical player-game targets missing: {path}")
    required = {
        "game_id", "game_date", "player_id", "season", "position_group",
        "participated", "training_target_available", "shots_on_goal", "points",
        "assists", "goals", "blocked_shots", "saves", "goals_against",
    }
    actual: dict[tuple[str, str], dict[str, str]] = {}
    played_skaters: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Targets missing columns: {sorted(missing)}")
        for row in reader:
            gid = row["game_id"].strip()
            if gid not in selected_games:
                continue
            if iso_date(row["game_date"]) != selected_games[gid]:
                raise ValueError(f"Outcome game date mismatch: {gid}")
            key = (gid, row["player_id"].strip())
            if key in actual:
                raise ValueError(f"Duplicate target player-game {key}")
            actual[key] = row
            if row["position_group"] == "SKATER" and row["participated"] == "1":
                played_skaters[gid] += 1
    # A recorded game without played skater results is not a completed target game.
    completed_games = {gid for gid, count in played_skaters.items() if count > 0}
    return actual, completed_games


def official_ppp_for_day(rows: dict[tuple[str, str], dict[str, str]],
                         output_dir: Path, day: str) -> dict[tuple[str, str], int]:
    """Reuse the project's existing verified NHL PPP source, not power_play_goals."""
    if not rows:
        return {}
    from importlib import import_module
    model_dir = ROOT / "scripts" / "01_predict"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    verifier = import_module("predict_player_stats").official_ppp_values
    by_season: dict[str, dict] = defaultdict(dict)
    for key, value in rows.items():
        by_season[value["season"]][key] = value
    verified = {}
    for season, eligible in by_season.items():
        cache = output_dir / (
            f"official_ppp_grader_{day.replace('-', '_')}_{season.replace('-', '_')}.json.gz"
        )
        verified.update(verifier(eligible, season, cache))
    return verified


def grade_one(r: dict[str, str], actual: dict[tuple[str, str], dict[str, str]],
              completed: set[str], ppp: dict[tuple[str, str], int],
              ppp_error: str | None = None) -> dict[str, object]:
    gid, pid = r["game_id"].strip(), r["player_id"].strip()
    market, side = r["market"].strip(), r["bet_side"].strip().upper()
    odds_raw = r["over_odds"] if side == "OVER" else r["under_odds"]
    odds = american_odds(odds_raw)
    selected_prob = finite_number(r["over_probability"] if side == "OVER"
                                  else r["under_probability"], "selected probability")
    line = finite_number(r["line"], "line")
    actual_value: object = ""
    result, detail = "PENDING", "Game results not in historical targets"
    source = actual.get((gid, pid))
    if gid in completed:
        if source is None:
            result, detail = "UNVERIFIED", "Player-game outcome not in historical targets"
        elif source["position_group"] != MARKETS[market]:
            result, detail = "UNVERIFIED", "Outcome position group does not match market"
        elif source["participated"] != "1":
            result, detail = "NO_PLAY", "Player did not participate"
        elif source["training_target_available"] != "1":
            result, detail = "UNVERIFIED", "Completed player target not available"
        elif market == "power_play_points":
            key = (gid, pid)
            if key in ppp:
                actual_value = ppp[key]
                result, detail = decide(float(actual_value), line, side), ""
            else:
                result = "UNVERIFIED"
                detail = "Verified official power-play-points result unavailable"
                if ppp_error:
                    detail += f": {ppp_error}"
        else:
            value = source.get(market, "")
            if value is None or str(value).strip() == "":
                result, detail = "UNVERIFIED", f"Missing completed {market} statistic"
            else:
                n = finite_number(value, f"actual {market}")
                if n < 0 or not n.is_integer():
                    raise ValueError(f"Invalid actual count {gid}/{pid}/{market}={value!r}")
                actual_value = int(n)
                result, detail = decide(n, line, side), ""
    net_units: object = ""
    if odds is not None and result in ("WIN", "LOSS", "PUSH"):
        if result == "WIN":
            net_units = odds / 100 if odds > 0 else 100 / abs(odds)
        elif result == "LOSS":
            net_units = -1.0
        else:
            net_units = 0.0
    return {
        **{column: r[column] for column in SELECTION_COLUMNS},
        "actual": actual_value,
        "result": result,
        "selected_probability": selected_prob,
        "selected_odds": odds_raw,
        "net_units": net_units,
        "detail": detail,
    }


def decide(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "PUSH"
    success = actual > line if side == "OVER" else actual < line
    return "WIN" if success else "LOSS"


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(r["result"]) for r in rows)
    won, lost, pushed = (counts[s] for s in ("WIN", "LOSS", "PUSH"))
    decided = won + lost
    priced = [r for r in rows if r["net_units"] != ""]
    units = sum(float(r["net_units"]) for r in priced)
    out = {
        "selected": len(rows), "win": won, "loss": lost, "push": pushed,
        "no_play": counts["NO_PLAY"], "pending": counts["PENDING"],
        "unverified": counts["UNVERIFIED"], "decided": decided,
        "win_rate": won / decided if decided else "",
        "priced_decided": len(priced),
        "priced_net_units": units if priced else "",
        "priced_roi": units / len(priced) if priced else "",
    }
    if sum(counts.values()) != len(rows):
        raise ValueError("Unexpected grader result status")
    return out


def atomic_csv(path: Path, header: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as f:
            temp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def grade_dates(dates: list[str], *, selection_dir: Path = SELECTIONS,
                targets: Path = DEFAULT_TARGETS, output_dir: Path = DEFAULT_OUTPUT,
                ppp_provider=None) -> tuple[list[dict], list[dict], bool]:
    """Grades all requested dates; output is based solely on current selections."""
    selected: dict[str, list[dict[str, str]]] = {}
    games: dict[str, str] = {}
    for day in dates:
        stamp = day.replace("-", "_")
        path = selection_dir / f"{stamp}_nhl_props.csv"
        rows = read_csv(path, set(SELECTION_COLUMNS))
        validate_selections(rows, day)
        for r in rows:
            gid = r["game_id"].strip()
            if gid in games and games[gid] != day:
                raise ValueError(f"Same game {gid} in two selection dates")
            games[gid] = day
        selected[day] = rows
    actual, completed = load_outcomes(targets, games)
    all_graded = []
    date_summaries = []
    had_ppp_failure = False
    for day in dates:
        rows = selected[day]
        ppp_requested = {}
        for r in rows:
            gid, pid = r["game_id"].strip(), r["player_id"].strip()
            candidate = actual.get((gid, pid))
            if (r["market"] == "power_play_points" and gid in completed
                and candidate is not None and candidate["position_group"] == "SKATER"
                and candidate["participated"] == "1"
                and candidate["training_target_available"] == "1"):
                ppp_requested[(gid, pid)] = candidate
        ppp = {}
        ppp_error = None
        if ppp_requested:
            try:
                ppp = (ppp_provider or official_ppp_for_day)(ppp_requested, output_dir, day)
                if set(ppp) != set(ppp_requested):
                    raise ValueError(f"Official PPP incomplete: {len(set(ppp_requested)-set(ppp))} missing")
            except Exception as exc:
                had_ppp_failure = True
                ppp_error = str(exc)
                print(f"ERROR | {day} | official PPP verification failed: {exc}", file=sys.stderr)
        graded = [grade_one(r, actual, completed, ppp, ppp_error) for r in rows]
        output = output_dir / f"{day.replace('-', '_')}_nhl_props_graded.csv"
        atomic_csv(output, GRADE_COLUMNS, graded)
        row = {"game_date": day, **summary(graded)}
        date_summaries.append(row)
        all_graded.extend(graded)
        print(f"WROTE {output} | selected={row['selected']} win={row['win']} "
              f"loss={row['loss']} push={row['push']} no_play={row['no_play']} "
              f"pending={row['pending']} unverified={row['unverified']}")
    market_rows = []
    market_groups = defaultdict(list)
    for r in all_graded:
        market_groups[(r["market"], r["bet_side"])].append(r)
    for market, side in sorted(market_groups):
        market_rows.append({"market": market, "bet_side": side,
                            **summary(market_groups[(market, side)])})
    atomic_csv(output_dir / "summary_by_date.csv", DATE_SUMMARY_COLUMNS, date_summaries)
    atomic_csv(output_dir / "summary_by_market.csv", MARKET_SUMMARY_COLUMNS, market_rows)
    print(f"SUMMARY | {output_dir / 'summary_by_date.csv'} | "
          f"{output_dir / 'summary_by_market.csv'}")
    return date_summaries, market_rows, had_ppp_failure


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--date", help="YYYY-MM-DD or YYYY_MM_DD (one date)")
    modes.add_argument("--all-dates", action="store_true", help="Explicitly grade all dates (the default)")
    parser.add_argument("--selections-dir", type=Path, default=SELECTIONS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    dates = [iso_date(args.date)] if args.date else selection_dates(args.selections_dir)
    if not dates:
        parser.error(f"No dated selection files found in {args.selections_dir}")
    _, _, ppp_failed = grade_dates(dates, selection_dir=args.selections_dir,
                                   targets=args.targets, output_dir=args.output_dir)
    return 1 if ppp_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, csv.Error) as error:
        print(f"ERROR | {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
