#!/usr/bin/env python3
"""Manual DRatings NHL completed-game reference rows (NOT official NHL scores).

Updates the existing 18-field D-Ratings JSON contract without replacing NHL API
final scores. Scans all raw snapshots to avoid creating duplicate completed-game
checks for a matchup. Original pregame prediction fields stay immutable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from hockey_manual_predictions import (
    ET, FIELDS, RAW_DIR, atomic_json, load_raw, load_team_ids, matchup,
    parse_finals, team_id, when_et,
)


def date_of(row: dict) -> str:
    return when_et(row["date_time"], "et").strftime("%Y_%m_%d")


def is_final(row: dict) -> bool:
    return str(row.get("game_status", "")).strip().lower() in (
        "completed", "final", "final ot", "final so", "ended", "closed"
    )


def aligned_scores(incoming: dict, existing: dict, identities: dict[str, str]) -> tuple[str, str]:
    i1 = team_id(incoming["team1"], identities)
    e1 = team_id(existing["team1"], identities)
    if i1 == e1:
        return incoming["score1"], incoming["score2"]
    return incoming["score2"], incoming["score1"]


def apply_finals(rows: list[dict[str, str]], identities: dict[str, str]) -> dict[str, int]:
    if not rows:
        raise ValueError("No final-score rows provided")
    incoming = {}
    for row in rows:
        date = date_of(row)
        key = (date, matchup(row, identities))
        if key in incoming:
            raise ValueError(f"Duplicate pasted final game: {date}, {row['team1']} / {row['team2']}")
        incoming[key] = row

    # Load snapshots once. Existing scraper snapshots may already contain
    # completed results for the previous calendar day; don't duplicate them.
    snapshots = {p: load_raw(p) for p in sorted(RAW_DIR.glob("*_nhl_raw.json"))}
    existing: dict[tuple, list[tuple[Path, int]]] = defaultdict(list)
    requested_dates = {key[0] for key in incoming}
    for path, games in snapshots.items():
        for i, game in enumerate(games):
            try:
                game_date = date_of(game)
            except (KeyError, ValueError):
                # Unrelated historical bad rows cannot block manual score repair.
                continue
            if game_date not in requested_dates:
                continue
            try:
                k = (game_date, matchup(game, identities))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Cannot identify existing raw game {path} row {i+1}: {exc}") from exc
            if k in incoming:
                existing[k].append((path, i))
    changes: dict[Path, list[dict]] = {}
    updated = 0
    already_final = 0
    added = 0
    for key, row in incoming.items():
        matches = existing.get(key, [])
        complete = [(p, i) for p, i in matches if is_final(snapshots[p][i])]
        if complete:
            # Existing completed DRatings checks must never be overwritten. A
            # contradictory result requires investigation, not silent rewrite.
            for p, i in complete:
                saved = snapshots[p][i]
                a, b = aligned_scores(row, saved, identities)
                if str(saved.get("score1", "")) != a or str(saved.get("score2", "")) != b:
                    raise ValueError(f"Conflicting DRatings final for {key} in {p}")
            already_final += 1
            continue
        if matches:
            # Update just one existing upcoming row to avoid duplicate
            # completed-game comparisons in transform_final_scores.py.
            date = key[0]
            preferred = RAW_DIR / f"{date}_nhl_raw.json"
            p, i = next((entry for entry in matches if entry[0] == preferred), matches[0])
            if p not in changes:
                changes[p] = [dict(x) for x in snapshots[p]]
            old = changes[p][i]
            a, b = aligned_scores(row, old, identities)
            if str(old.get("score1", "")).strip() or str(old.get("score2", "")).strip():
                raise ValueError(f"Unrecognized scored record status for {key} in {p}")
            old["score1"] = a
            old["score2"] = b
            old["game_status"] = "completed"
            updated += 1
        else:
            path = RAW_DIR / f"{key[0]}_nhl_raw.json"
            if path not in changes:
                changes[path] = [dict(x) for x in snapshots.get(path, [])]
            changes[path].append({k: row.get(k, "") for k in FIELDS})
            added += 1

    # All validation happens above; write only affected snapshots.
    for p, data in sorted(changes.items()):
        atomic_json(p, data)
        print(f"WROTE {p}: rows={len(data)}")
    return {"parsed": len(rows), "added": added, "updated": updated,
            "already_recorded": already_final, "files": len(changes)}


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-file", type=Path, required=True)
    p.add_argument("--time-basis", choices=("et", "utc"), default="et")
    args = p.parse_args(argv)
    try:
        identities = load_team_ids()
        text = args.raw_file.read_text(encoding="utf-8-sig")
        rows = parse_finals(text, args.time_basis)
        stats = apply_finals(rows, identities)
        print("SUCCESS NHL DRatings final references: " + " ".join(f"{k}={v}" for k, v in stats.items()))
        print("NHL API remains authoritative for official grading scores")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
