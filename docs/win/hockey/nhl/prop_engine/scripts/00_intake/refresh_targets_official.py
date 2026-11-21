#!/usr/bin/env python3
"""Incrementally update prop-engine player-game targets from FINAL official NHL boxscores.

Used by build_historical_targets.py in its daily mode. No SportsDataverse
refresh or downloaded boxscore archive is assumed. The pre-existing historical
CSV is kept; only recent finished games are added/reconciled. Failed network
requests and invalid/limited boxscores never publish partial CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

PROP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = PROP_ROOT / 'data' / 'nhl_historical_player_game_targets_2009_2026.csv'
SCORE_API = 'https://api-web.nhle.com/v1/score/'
BOXSCORE_API = 'https://api-web.nhle.com/v1/gamecenter/{}/boxscore'
FINISHED_STATES = {'OFF', 'FINAL'}
ET = ZoneInfo('America/New_York')
COUNT_FIELDS = ('shots_on_goal', 'points', 'assists', 'goals', 'blocked_shots',
                'power_play_goals', 'saves', 'goals_against')
OPTIONAL_FIELDS = ('pim', 'plus_minus', 'hits', 'faceoff_winning_pctg', 'shifts',
                   'giveaways', 'takeaways', 'even_strength_shots_against',
                   'power_play_shots_against', 'shorthanded_shots_against',
                   'save_shots_against', 'save_pctg', 'even_strength_goals_against',
                   'power_play_goals_against', 'shorthanded_goals_against',
                   'shots_against', 'decision')
REQUIRED_TARGET_FIELDS = {'season', 'game_id', 'game_date', 'player_id',
                          'position_group', 'participated', 'training_target_available',
                          *COUNT_FIELDS, *OPTIONAL_FIELDS}


def get_json(url: str) -> dict:
    last_error = None
    for attempt in range(4):
        try:
            req = Request(url, headers={'User-Agent': 'NHLPropEngine/1.0',
                                        'Accept': 'application/json'})
            with urlopen(req, timeout=30) as response:
                data = json.load(response)
            if not isinstance(data, dict):
                raise ValueError(f'Expected NHL JSON object: {url}')
            return data
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                break
            if attempt < 3:
                time.sleep(attempt + 1)
    raise RuntimeError(f'OFFICIAL NHL DATA UNAVAILABLE: {url}: {last_error}')


def nonnegative_int(value, field: str) -> str:
    if value is None or isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        raise ValueError(f'Missing/invalid official NHL {field}: {value!r}')
    return str(int(value))


def optional_number(value, field: str) -> str:
    if value is None or value == '':
        return ''
    try:
        n = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid NHL {field}: {value!r}') from exc
    if not (float('-inf') < n < float('inf')):
        raise ValueError(f'Nonfinite NHL {field}')
    return str(value)


def split_shots(value, field: str) -> tuple[str, str]:
    """Official goalie strings such as '27/29': saves and shots against."""
    if not isinstance(value, str) or not re.fullmatch(r'\d+\s*/\s*\d+', value):
        raise ValueError(f'Invalid official NHL {field}: {value!r}')
    saves, attempts = (int(x.strip()) for x in value.split('/'))
    if saves > attempts:
        raise ValueError(f'Impossible official goalie {field}: {value!r}')
    return str(saves), str(attempts)


def localized_name(value) -> str:
    if isinstance(value, dict):
        return str(value.get('default') or '').strip()
    return str(value or '').strip()


def toi_minutes(value: object) -> int:
    if value is None or not re.fullmatch(r'\d{1,3}:\d{2}', str(value)):
        raise ValueError(f'Missing/invalid NHL time on ice: {value!r}')
    mm, ss = (int(x) for x in str(value).split(':'))
    if ss >= 60:
        raise ValueError(f'Invalid NHL TOI seconds: {value!r}')
    return mm*60+ss


def field_string(value) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def official_player_rows(game: dict, box: dict, required_fields: list[str]) -> list[dict]:
    """Create one canonical target row per verified NHL player/game identity."""
    gid = nonnegative_int(game.get('id'), 'game id')
    if nonnegative_int(box.get('id'), 'boxscore game id') != gid:
        raise ValueError(f'Official NHL boxscore game ID differs for {gid}')
    if str(box.get('gameState', '')).upper() not in FINISHED_STATES:
        raise ValueError(f'Official boxscore not FINAL for {gid}')
    if box.get('limitedScoring'):
        raise ValueError(f'Official NHL boxscore limitedScoring: {gid}')
    stats = box.get('playerByGameStats')
    if not isinstance(stats, dict):
        raise ValueError(f'No official NHL player statistics for {gid}')
    day = str(game['gameDate'])
    date.fromisoformat(day)
    season_id = str(game.get('season'))
    if not re.fullmatch(r'20\d{6}', season_id):
        raise ValueError(f'Invalid season {season_id!r} in {gid}')
    start_season = int(season_id[:4])
    if int(season_id[4:]) != start_season + 1:
        raise ValueError(f'Season year mismatch for {gid}')
    season = f'{start_season}-{(start_season+1)%100:02d}'
    game_type = int(game.get('gameType', -1))
    if game_type not in (2, 3) or gid[:4] != str(start_season) or gid[4:6] != f'{game_type:02d}':
        raise ValueError(f'Invalid NHL season/game type ID for {gid}')
    output = []
    seen = set()
    team_counts = Counter()
    for place, other in [('home', 'away'), ('away', 'home')]:
        team, opponent = game.get(place+'Team'), game.get(other+'Team')
        if not isinstance(team, dict) or not isinstance(opponent, dict):
            raise ValueError(f'Invalid official teams in {gid}')
        team_id = nonnegative_int(team.get('id'), 'team id')
        abbrev, opposite = str(team.get('abbrev', '')).strip(), str(opponent.get('abbrev', '')).strip()
        if not (abbrev and opposite and abbrev != opposite):
            raise ValueError(f'Invalid team abbreviations in {gid}')
        group = stats.get(place+'Team')
        if not isinstance(group, dict):
            raise ValueError(f'Missing {place} player stats {gid}')
        for section, role in (('forwards', 'SKATER'), ('defense', 'SKATER'), ('goalies', 'GOALIE')):
            players = group.get(section)
            if not isinstance(players, list):
                raise ValueError(f'Missing player group {place}/{section}/{gid}')
            for p in players:
                if not isinstance(p, dict):
                    raise ValueError(f'Invalid player in {gid}')
                pid = nonnegative_int(p.get('playerId'), 'player id')
                key = (gid, pid)
                if key in seen:
                    raise ValueError(f'Duplicate NHL official player {key}')
                seen.add(key)
                toi = p.get('toi')
                elapsed = toi_minutes(toi)
                played = bool(elapsed)
                if played:
                    team_counts[(place, role)] += 1
                row = {name: '' for name in required_fields}
                row.update({
                    'season': season, 'game_id': gid, 'game_date': day,
                    'source_player_game_date': day, 'source_date_offset_days': '0',
                    'source_date_match_status': 'MATCH', 'game_type': str(game_type),
                    'player_id': pid, 'player_name': localized_name(p.get('name')),
                    'sweater_number': field_string(p.get('sweaterNumber')),
                    'team_id': team_id, 'team': abbrev, 'opponent_team': opposite,
                    'home_away': place.upper(), 'position': str(p.get('position') or ('G' if role == 'GOALIE' else '')).upper(),
                    'position_group': role, 'toi': str(toi),
                    'participation_status': 'PLAYED' if played else 'ZERO_TOI',
                    'participated': '1' if played else '0',
                    'training_target_available': '1' if played else '0',
                })
                if role == 'SKATER' and played:
                    for output_key, api_key in (
                        ('shots_on_goal', 'sog'), ('points', 'points'),
                        ('assists', 'assists'), ('goals', 'goals'),
                        ('blocked_shots', 'blockedShots'),
                        ('power_play_goals', 'powerPlayGoals')):
                        row[output_key] = nonnegative_int(p.get(api_key), output_key)
                    if int(row['goals'])+int(row['assists']) != int(row['points']):
                        raise ValueError(f'NHL official goals + assists != points for {gid}/{pid}')
                    for key, api_key in (
                        ('pim', 'pim'), ('plus_minus', 'plusMinus'), ('hits', 'hits'),
                        ('faceoff_winning_pctg', 'faceoffWinningPctg'),
                        ('shifts', 'shifts'), ('giveaways', 'giveaways'),
                        ('takeaways', 'takeaways')):
                        row[key] = optional_number(p.get(api_key), key)
                elif role == 'GOALIE' and played:
                    row['goals_against'] = nonnegative_int(p.get('goalsAgainst'), 'goalsAgainst')
                    if p.get('saves') is not None:
                        row['saves'] = nonnegative_int(p.get('saves'), 'saves')
                        if p.get('saveShotsAgainst'):
                            other_saves, _ = split_shots(p['saveShotsAgainst'], 'saveShotsAgainst')
                            if other_saves != row['saves']:
                                raise ValueError(f'Conflicting NHL saves data {gid}/{pid}')
                    else:
                        row['saves'], _ = split_shots(p.get('saveShotsAgainst'), 'saveShotsAgainst')
                    for key, api_key in (
                        ('even_strength_shots_against', 'evenStrengthShotsAgainst'),
                        ('power_play_shots_against', 'powerPlayShotsAgainst'),
                        ('shorthanded_shots_against', 'shorthandedShotsAgainst'),
                        ('save_shots_against', 'saveShotsAgainst'), ('save_pctg', 'savePctg'),
                        ('even_strength_goals_against', 'evenStrengthGoalsAgainst'),
                        ('power_play_goals_against', 'powerPlayGoalsAgainst'),
                        ('shorthanded_goals_against', 'shorthandedGoalsAgainst'),
                        ('shots_against', 'shotsAgainst'), ('decision', 'decision')):
                        row[key] = field_string(p.get(api_key))
                    row['goalie_starter_raw'] = field_string(p.get('starter'))
                output.append(row)
    if team_counts[('home', 'SKATER')] < 10 or team_counts[('away', 'SKATER')] < 10:
        raise ValueError(f'Insufficient finished skater results for {gid}: {team_counts}')
    if not output:
        raise ValueError(f'Empty official NHL player-game results for {gid}')
    return output


def inspect_existing(target: Path) -> tuple[list[str], date]:
    if not target.is_file():
        raise FileNotFoundError(f'Expected existing historical target file: {target}')
    latest = date(2000, 1, 1)
    with target.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or ())
        if not REQUIRED_TARGET_FIELDS.issubset(fields):
            raise ValueError(f'Missing target fields: {sorted(REQUIRED_TARGET_FIELDS-set(fields))}')
        for r in reader:
            d = date.fromisoformat(r['game_date'])
            if d > latest:
                latest = d
    return fields, latest


def run(targets: Path = DEFAULT_TARGETS, since: date | None = None,
        *, today: date | None = None, get=get_json):
    # Only final results strictly prior to today can enter the training targets.
    today = today or datetime.now(ET).date()
    end = today - timedelta(days=1)
    fields, latest = inspect_existing(targets)
    start = since or (latest - timedelta(days=3))
    if start > end:
        print(f'OFFICIAL NHL TARGET REFRESH | no completed prior dates to check | latest={latest}')
        return 0, 0
    newest_rows: dict[tuple[str,str], dict] = {}
    completed_ids = set()
    checked_dates = 0
    day = start
    while day <= end:
        score = get(SCORE_API + day.isoformat())
        games = score.get('games')
        if not isinstance(games, list):
            raise ValueError(f'Official NHL scoreboard missing games array for {day}')
        checked_dates += 1
        for game in games:
            if not isinstance(game, dict):
                raise ValueError(f'Bad official NHL score game record for {day}')
            if str(game.get('gameDate', '')) != day.isoformat():
                raise ValueError(f'Official score returned unexpected game date {game.get("gameDate")} for {day}')
            if int(game.get('gameType', -1)) not in (2, 3):
                continue
            gid = nonnegative_int(game.get('id'), 'score game id')
            state = str(game.get('gameState', '')).upper()
            if state not in FINISHED_STATES:
                print(f'NOT FINAL | {day} | {gid} | state={state} (will retry next run)')
                continue
            box = get(BOXSCORE_API.format(gid))
            fetched = official_player_rows(game, box, fields)
            completed_ids.add(gid)
            for row in fetched:
                key = row['game_id'], row['player_id']
                if key in newest_rows:
                    raise ValueError(f'Duplicate retrieved NHL player-game: {key}')
                newest_rows[key] = row
        day += timedelta(days=1)
    if not newest_rows:
        print(f'OFFICIAL NHL TARGET REFRESH | dates_checked={checked_dates} '
              f'| completed_games=0 | added_player_games=0 | corrected_player_games=0')
        return 0, 0
    # Only load/check recent player-game keys: never materialize 17 seasons
    # of targets in RAM and never overwrite nonstatistical source metadata.
    prior: dict[tuple[str,str], dict] = {}
    with targets.open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            key = (r['game_id'], r['player_id'])
            if key in newest_rows:
                if key in prior:
                    raise ValueError(f'Duplicate historical player-game: {key}')
                prior[key] = r
    additions: dict[tuple[str,str], dict] = {}
    corrections: dict[tuple[str,str], dict] = {}
    for key, row in newest_rows.items():
        old = prior.get(key)
        if old is None:
            additions[key] = row
            continue
        if old['position_group'] != row['position_group']:
            raise ValueError(f'Official NHL position group conflict: {key}')
        new = old.copy()
        for col in (*COUNT_FIELDS, 'toi', 'participated',
                    'training_target_available', 'participation_status'):
            new[col] = row[col]
        if new != old:
            corrections[key] = new
    if additions or corrections:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', newline='',
                                             dir=targets.parent, suffix='.tmp',
                                             delete=False) as f:
                tmp = Path(f.name)
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
                writer.writeheader()
                with targets.open('r', encoding='utf-8-sig', newline='') as oldfile:
                    for old in csv.DictReader(oldfile):
                        key = old['game_id'], old['player_id']
                        writer.writerow(corrections.get(key, old))
                for key in sorted(additions):
                    writer.writerow(additions[key])
            os.replace(tmp, targets)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    print(f'OFFICIAL NHL TARGET REFRESH | dates_checked={checked_dates} '
          f'| completed_games={len(completed_ids)} | added_player_games={len(additions)} '
          f'| corrected_player_games={len(corrections)} | target={targets}')
    return len(additions), len(corrections)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, default=DEFAULT_TARGETS)
    parser.add_argument('--since', type=date.fromisoformat,
                        help='Recheck results since YYYY-MM-DD; default latest target day minus 3')
    args = parser.parse_args(argv)
    try:
        run(args.targets, args.since)
    except (OSError, ValueError, TypeError, RuntimeError, csv.Error) as e:
        print(f'FAILED OFFICIAL NHL TARGET REFRESH | {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
