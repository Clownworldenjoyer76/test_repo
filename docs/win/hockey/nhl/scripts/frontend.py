#!/usr/bin/env python3
"""Combine NHL slate sources and final-score/graded-bet sources into CSVs."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
NHL = ROOT / "docs/win/hockey/nhl"
OUTPUT = ROOT / "frontend/data/nhl_slate.csv"
FINAL_OUTPUT = ROOT / "frontend/data/nhl/nhl_final.csv"
TEAM_MAP = NHL / "config/mapping/team_map_nhl.csv"


def read_csv(path):
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def load_team_map():
    if not TEAM_MAP.is_file():
        raise FileNotFoundError(f"NHL team mapping file not found: {TEAM_MAP}")

    columns, rows = read_csv(TEAM_MAP)
    if not {"league", "alias", "canonical_team"}.issubset(columns):
        raise ValueError(f"NHL team mapping file has missing columns: {TEAM_MAP}")

    aliases = {}
    canonical_teams = set()

    for row in rows:
        if (row.get("league") or "").strip().casefold() != "nhl":
            continue

        canonical = (row.get("canonical_team") or "").strip()
        if not canonical or canonical.casefold() == "tbd":
            continue

        canonical_teams.add(canonical)
        for name in ((row.get("alias") or "").strip(), canonical):
            if not name:
                continue
            key = name.casefold()
            previous = aliases.get(key)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"Conflicting NHL team mappings for {name!r}: "
                    f"{previous!r} and {canonical!r}"
                )
            aliases[key] = canonical

    if not canonical_teams:
        raise ValueError(f"No NHL canonical team names found in {TEAM_MAP}")

    return aliases, sorted(canonical_teams)


def normalize_team_name(name, aliases, canonical_teams):
    cleaned = (name or "").strip()
    if not cleaned:
        return cleaned

    key = cleaned.casefold()
    if key in aliases:
        return aliases[key]

    # The supplied map lists full names but not nickname-only score aliases.
    # Accept a nickname only when it uniquely matches a mapped full name.
    matches = [
        canonical
        for canonical in canonical_teams
        if canonical.casefold().endswith(" " + key)
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Cannot uniquely normalize NHL team {cleaned!r} using {TEAM_MAP}; "
        f"matches: {matches}"
    )


def main():
    headers = []
    output_rows = []
    games_dir = NHL / "00_intake/games"

    for games_path in sorted(games_dir.glob("????_??_??_nhl_games.csv")):
        date = games_path.name.removesuffix("_nhl_games.csv")
        sources = [
            games_path,
            NHL / f"00_intake/sportsbook/NHL_{date}.csv",
            NHL / f"00_intake/predictions/hockey_{date}.csv",
            NHL / f"04_select/{date}_NHL.csv",
        ]

        files = [read_csv(path) for path in sources]
        for columns, _ in files:
            for column in columns:
                if column not in headers:
                    headers.append(column)

        games = {row["game_id"]: row.copy() for row in files[0][1]}

        for _, source_rows in files[1:3]:
            for row in source_rows:
                game = games.get(row["game_id"])
                if game is not None:
                    for column, value in row.items():
                        if value and not game.get(column):
                            game[column] = value

        picks = defaultdict(list)
        for row in files[3][1]:
            picks[row["game_id"]].append(row)

        for game in games.values():
            selected = picks.get(game["game_id"])
            if not selected:
                output_rows.append(game)
            else:
                for pick in selected:
                    combined = game.copy()
                    for column, value in pick.items():
                        if value and not combined.get(column):
                            combined[column] = value
                    output_rows.append(combined)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)



def load_game_times():
    """Index game times from the slate this script already produces."""
    _, slate_rows = read_csv(OUTPUT)
    game_times = {}

    for row in slate_rows:
        date = (row.get("game_date") or "").strip()
        game_id = (row.get("game_id") or "").strip()
        game_time = (row.get("game_time") or "").strip()
        if not (date and game_id and game_time):
            continue

        key = (date, game_id)
        existing = game_times.get(key)
        if existing is not None and existing != game_time:
            raise ValueError(
                f"Conflicting NHL game times for {date} game {game_id}: "
                f"{existing!r} and {game_time!r}"
            )
        game_times[key] = game_time

    return game_times


def has_final_score(row):
    """Both teams must have recorded, nonnegative integer final scores."""
    try:
        return all(
            int(str(row.get(column, "")).strip()) >= 0
            for column in ("away_score", "home_score")
        )
    except (ValueError, TypeError):
        return False


def build_nhl_final():
    scores_dir = NHL / "05_final_scores/final_scores"
    graded_dir = NHL / "05_final_scores/graded"
    score_suffix = "_NHL_final_scores.csv"
    graded_suffix = "_results_NHL.csv"

    score_files = {
        path.name.removesuffix(score_suffix): path
        for path in scores_dir.glob(f"????_??_??{score_suffix}")
    }
    graded_files = {
        path.name.removesuffix(graded_suffix): path
        for path in graded_dir.glob(f"????_??_??{graded_suffix}")
    }

    headers = []
    output_rows = []
    game_times = load_game_times()

    for date in sorted(score_files.keys() | graded_files.keys()):
        score_columns, scores = read_csv(
            score_files.get(date, scores_dir / f"{date}{score_suffix}")
        )
        graded_columns, graded = read_csv(
            graded_files.get(date, graded_dir / f"{date}{graded_suffix}")
        )

        for column in score_columns + graded_columns:
            if column not in headers:
                headers.append(column)

        grades_by_game = defaultdict(list)
        for bet in graded:
            grades_by_game[bet["game_id"]].append(bet)

        scored_game_ids = set()
        for score in scores:
            game_id = score["game_id"]
            scored_game_ids.add(game_id)
            bets = grades_by_game.get(game_id)
            if not bets:
                output_rows.append(score)
                continue

            for bet in bets:
                combined = score.copy()
                combined.update({key: value for key, value in bet.items() if value})
                output_rows.append(combined)

        # Retain graded bets even if their matching final-scores row is absent.
        for game_id, bets in grades_by_game.items():
            if game_id not in scored_game_ids:
                output_rows.extend(bets)

    # Score-only rows do not carry graded-bet metadata. Fill the scheduled
    # time from nhl_slate.csv and mark games final only when both scores exist.
    for column in ("game_time", "is_final"):
        if column not in headers:
            headers.append(column)

    aliases, canonical_teams = load_team_map()
    for row in output_rows:
        if not (row.get("game_time") or "").strip():
            key = (
                (row.get("game_date") or "").strip(),
                (row.get("game_id") or "").strip(),
            )
            row["game_time"] = game_times.get(key, "")

        if not (row.get("is_final") or "").strip() and has_final_score(row):
            row["is_final"] = "True"

        for field in ("away_team", "home_team"):
            if field in row:
                row[field] = normalize_team_name(
                    row[field], aliases, canonical_teams
                )

    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
    build_nhl_final()
