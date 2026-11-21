#!/usr/bin/env python3
"""Run the NHL player props pipeline in its specified eight-step order.

1. Refresh completed player stats from official NHL results via build_historical_targets.
2. Today's lineups and starting goalies.
3. Today's ESPN sportsbook lines (official NHL player IDs).
4. Today's 8-market predictions.
5. Recalculate ALL dates' selections with current markets.yaml.
6. Grade forecasts from fully completed historical dates only.
7. Grade ALL selection dates (unfinished ones remain PENDING).
8. Summarize graded bets by market, side, line, probability, and odds ranges.

No step writes to Downloads. Any failed step stops the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
TARGETS = ROOT / 'data' / 'nhl_historical_player_game_targets_2009_2026.csv'


def run_script(label: str, script: Path, *args: str) -> None:
    if not script.is_file():
        raise FileNotFoundError(f'{label} missing script: {script}')
    cmd = [sys.executable, '-B', str(script), *map(str, args)]
    print(f'\n=== {label} ===', flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise RuntimeError(f'{label} FAILED with exit code {result.returncode}; pipeline stopped')


def historical_prediction_files(today: str):
    """Only send whole, already completed slates to compare_prediction_accuracy."""
    candidates = []
    for path in sorted((ROOT / '01_predictions').glob('*_nhl_player_predictions.csv')):
        day = path.name[:10].replace('_','-')
        if day >= today:
            continue
        games = set()
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'game_id','game_date'}.issubset(reader.fieldnames or []):
                raise ValueError(f'Prediction missing game identity: {path}')
            for row in reader:
                if row['game_date'] != day:
                    raise ValueError(f'Prediction date conflict: {path}')
                games.add(row['game_id'])
        if games:
            candidates.append((path,games))
    if not candidates:
        return []
    needed = set().union(*(games for _,games in candidates))
    count = defaultdict(lambda: Counter())
    with TARGETS.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if (row['game_id'] in needed and row['position_group'] == 'SKATER'
                    and row['participated'] == '1'):
                count[row['game_id']][row['home_away']] += 1
    ready=[]
    for path,games in candidates:
        if all(count[game]['HOME']>=10 and count[game]['AWAY']>=10 for game in games):
            ready.append(path)
        else:
            missing=[gid for gid in games if count[gid]['HOME']<10 or count[gid]['AWAY']<10]
            print(f'SKIPPED INCOMPLETE PREDICTION EVALUATION | {path.name} '
                  f'| games_without_completed_player_stats={sorted(missing)}',flush=True)
    return ready


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grade-only', action='store_true',
                        help='After games finish, only run steps 1, 6, 7 and 8')
    args=parser.parse_args(argv)
    today=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    if not TARGETS.is_file():
        raise FileNotFoundError(f'Missing existing historical targets: {TARGETS}')
    run_script('1/8 - Update completed NHL player-game statistics',
               SCRIPTS/'00_intake'/'build_historical_targets.py','--refresh-daily')
    if not args.grade_only:
        run_script('2/8 - Capture today\'s lineups and goalies',
                   SCRIPTS/'00_intake'/'todays_starters.py')
        run_script('3/8 - Capture today\'s sportsbook props',
                   SCRIPTS/'00_intake'/'get_props.py')
        run_script('4/8 - Generate player predictions',
                   SCRIPTS/'01_predict'/'predict_player_stats.py',
                   '--slate-date',today,'--targets',str(TARGETS))
        run_script('5/8 - Recompute all dated selections',
                   SCRIPTS/'03_select'/'select_nhl_props.py','--all-dates')
    completed = historical_prediction_files(today)
    if completed:
        run_script('6/8 - Evaluate completed-game predictions',
                   SCRIPTS/'02_evaluate'/'compare_prediction_accuracy.py',
                   '--predictions',*completed,'--targets',str(TARGETS))
    else:
        print('6/8 - No fully completed prior-date predictions; evaluation skipped',flush=True)
    run_script('7/8 - Grade selected OVER/UNDER bets',
               SCRIPTS/'04_final'/'grader.py','--all-dates')
    run_script('8/8 - Summarize graded bets by market and ranges',
               SCRIPTS/'04_final'/'grade_by_market_range.py')
    print('\nPIPELINE COMPLETE | current selection files, grading and range summaries refreshed',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as err:
        print(f'PIPELINE FAILED | {err}',file=sys.stderr)
        raise SystemExit(1)
