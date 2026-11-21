#!/usr/bin/env python3
"""Grade pregame NHL player-stat forecasts against completed official outcomes.

NO ODDS / ROI / EV. Baselines are the EXISTING feature builder's prior
five- and ten-game means on the identical played player-market observations.
Does not treat unplayed pregame candidates as zero-stat performances.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROP_ROOT = Path(__file__).resolve().parents[2]
PREDICT_DIR = PROP_ROOT / 'scripts' / '01_predict'
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))

from predict_player_stats import (THRESHOLDS, iter_csv, parse_day, atomic_csv,
                                  official_ppp_values)

OUT_DIR = PROP_ROOT / '02_evaluation'
REQUIRED_PRED = ('game_date', 'game_id', 'player_id', 'market', 'position_group',
                 'predicted_mean', 'baseline_last5', 'baseline_last10',
                 'over_probability', 'threshold', 'baseline_last5_over_probability',
                 'baseline_last10_over_probability', 'candidate_source',
                 'prediction_as_of_utc')
MARKET_FIELDS = ('market', 'n_played', 'model_mae', 'model_bias', 'model_brier',
                 'model_over_accuracy', 'n_last5', 'paired_model_mae_last5',
                 'baseline_last5_mae', 'mae_gain_last5',
                 'paired_model_brier_last5', 'baseline_last5_brier', 'brier_gain_last5',
                 'n_last10', 'paired_model_mae_last10', 'baseline_last10_mae',
                 'mae_gain_last10', 'paired_model_brier_last10',
                 'baseline_last10_brier', 'brier_gain_last10')
DAY_MARKET_FIELDS = ('game_date', *MARKET_FIELDS)
SELECTION_FIELDS = ('game_date', 'position_group', 'games', 'actual_participants',
                    'pregame_candidates', 'played_candidates', 'unplayed_candidates',
                    'missing_played', 'precision', 'recall')
SCORED_FIELDS = ('game_date', 'game_id', 'player_id', 'position_group', 'market',
                 'actual', 'predicted_mean', 'baseline_last5', 'baseline_last10',
                 'absolute_error', 'squared_error', 'actual_over',
                 'over_probability', 'baseline_last5_over_probability',
                 'baseline_last10_over_probability', 'was_played')
EXTERNAL_FIELDS = ('market', 'n_overlap', 'model_mae', 'existing_mae', 'mae_gain',
                   'n_probability_overlap', 'model_brier', 'existing_brier', 'brier_gain')


def numeric(value, label: str, allow_blank=False):
    raw = str(value if value is not None else '').strip()
    if not raw:
        if allow_blank:
            return float('nan')
        raise ValueError(f'Missing numeric {label}')
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f'Nonfinite numeric {label}')
    return val


def mean_or_blank(values):
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)) if len(a) else ''


def load_predictions(paths: list[Path]):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob('*_nhl_player_predictions.csv')))
        else:
            files.append(path)
    files = sorted(set(files))
    if not files:
        raise ValueError('No dated NHL prediction files found')
    rows = [r for path in files for r in iter_csv(path)]
    if not rows:
        raise ValueError('Empty predictions CSV')
    if set(REQUIRED_PRED) - set(rows[0]):
        raise ValueError(f'Missing prediction columns: {sorted(set(REQUIRED_PRED) - set(rows[0]))}')
    keys = set()
    games = set()
    days = set()
    for r in rows:
        market = r['market']
        if market not in THRESHOLDS:
            raise ValueError(f'Unknown market {market}')
        role, threshold = THRESHOLDS[market]
        if r['position_group'] != role or numeric(r['threshold'], 'threshold') != threshold:
            raise ValueError(f'Market/role/threshold inconsistency: {market}')
        if r['candidate_source'] not in ('LINE_COMBINATIONS', 'DAILY_FACEOFF'):
            raise ValueError('Postgame candidates are not valid for the evaluation')
        key = (r['game_id'], r['player_id'], market)
        if key in keys:
            raise ValueError(f'Duplicate prediction {key}')
        keys.add(key)
        games.add(r['game_id'])
        days.add(parse_day(r['game_date']))
        if numeric(r['predicted_mean'], 'predicted_mean') < 0:
            raise ValueError('Negative count prediction')
        if not 0 <= numeric(r['over_probability'], 'over_probability') <= 1:
            raise ValueError('Invalid model over probability')
    if set(x['market'] for x in rows) != set(THRESHOLDS):
        raise ValueError('Prediction file must score all eight markets')
    return rows, games, days


def load_outcomes(targets: Path, game_ids: set, days: set):
    actual = {}
    seen_games = set()
    for r in iter_csv(targets):
        gid = str(r['game_id']).strip()
        if gid not in game_ids:
            continue
        d = parse_day(r['game_date'])
        if d not in days:
            raise ValueError(f'Prediction game has different outcome date {gid}')
        key = (gid, str(r['player_id']).strip())
        if key in actual:
            raise ValueError(f'Duplicate outcome player-game {key}')
        actual[key] = r
        seen_games.add(gid)
    if seen_games != game_ids:
        raise ValueError(f'No outcome data for {sorted(game_ids-seen_games)}')
    # A game with zero played skaters has not yet supplied verified results.
    for game_id in game_ids:
        played = [r for (gid, _), r in actual.items()
                  if gid == game_id and r['participated'] == '1' and r['position_group'] == 'SKATER']
        if not played:
            raise ValueError(f'Incomplete target game: no played skaters {game_id}')
    return actual


def metrics(items, market):
    y = np.asarray([r['actual'] for r in items], dtype=np.float64)
    mu = np.asarray([r['predicted_mean'] for r in items], dtype=np.float64)
    pred_prob = np.asarray([r['over_probability'] for r in items], dtype=np.float64)
    over = np.asarray([r['actual_over'] for r in items], dtype=np.float64)
    row = {'market': market, 'n_played': len(items),
           'model_mae': mean_or_blank(np.abs(mu-y)),
           'model_bias': mean_or_blank(mu-y),
           'model_brier': mean_or_blank((pred_prob-over)**2),
           'model_over_accuracy': mean_or_blank((pred_prob >= 0.5) == over)}
    for period in ('last5', 'last10'):
        x = np.asarray([r['baseline_'+period] for r in items], dtype=np.float64)
        pp = np.asarray([r['baseline_'+period+'_over_probability'] for r in items], dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(pp) & (x >= 0) & (pp >= 0) & (pp <= 1)
        base_abs = np.abs(x[ok]-y[ok])
        model_abs = np.abs(mu[ok]-y[ok])
        base_brier = (pp[ok]-over[ok])**2
        model_brier = (pred_prob[ok]-over[ok])**2
        row[f'n_{period}'] = int(np.sum(ok))
        row[f'paired_model_mae_{period}'] = mean_or_blank(model_abs)
        row[f'baseline_{period}_mae'] = mean_or_blank(base_abs)
        row[f'mae_gain_{period}'] = mean_or_blank(base_abs-model_abs)
        row[f'paired_model_brier_{period}'] = mean_or_blank(model_brier)
        row[f'baseline_{period}_brier'] = mean_or_blank(base_brier)
        row[f'brier_gain_{period}'] = mean_or_blank(base_brier-model_brier)
    return row


def selection_accuracy(pred_rows, target_rows):
    groups = defaultdict(lambda: {'candidates': set(), 'played': set(), 'games': set()})
    for r in pred_rows:
        key = (r['game_id'], r['player_id'], r['position_group'])
        g = groups[(parse_day(r['game_date']), r['position_group'])]
        g['candidates'].add(key)
        g['games'].add(r['game_id'])
    all_games = {r['game_id'] for r in pred_rows}
    for (gid, pid), r in target_rows.items():
        if gid not in all_games or r['participated'] != '1':
            continue
        k = (parse_day(r['game_date']), r['position_group'])
        if k in groups:
            groups[k]['played'].add((gid, pid, r['position_group']))
    out = []
    for (day, role), g in sorted(groups.items()):
        c, a = g['candidates'], g['played']
        tp = len(c & a)
        out.append({'game_date': day, 'position_group': role,
                    'games': len(g['games']), 'actual_participants': len(a),
                    'pregame_candidates': len(c), 'played_candidates': tp,
                    'unplayed_candidates': len(c-a), 'missing_played': len(a-c),
                    'precision': tp / len(c) if c else '',
                    'recall': tp / len(a) if a else ''})
    return out


def compare_external(graded, path: Path):
    """Optional existing predictions: game_id, player_id, market, predicted_mean,
    optionally over_probability. Only matched player-market rows are compared.
    """
    external = {}
    for r in iter_csv(path):
        key = (r['game_id'], r['player_id'], r['market'])
        if key in external:
            raise ValueError(f'Existing prediction duplicate: {key}')
        external[key] = r
    groups = defaultdict(list)
    for r in graded:
        key = (r['game_id'], r['player_id'], r['market'])
        if key not in external:
            continue
        e = external[key]
        mu = numeric(e.get('predicted_mean'), 'existing predicted_mean')
        if mu < 0:
            raise ValueError(f'Existing predicted mean negative: {key}')
        prob = numeric(e.get('over_probability'), 'existing probability', allow_blank=True)
        if math.isfinite(prob) and not 0 <= prob <= 1:
            raise ValueError(f'Existing probability out of bounds: {key}')
        groups[r['market']].append((r, mu, prob))
    out=[]
    for market in THRESHOLDS:
        rows=groups.get(market, ())
        row={'market':market,'n_overlap':len(rows)}
        if rows:
            model_err=np.array([abs(x['predicted_mean']-x['actual']) for x,_,_ in rows])
            other_err=np.array([abs(u-x['actual']) for x,u,_ in rows])
            row.update(model_mae=mean_or_blank(model_err),existing_mae=mean_or_blank(other_err),
                       mae_gain=mean_or_blank(other_err-model_err))
            valid=[(x,p) for x,_,p in rows if math.isfinite(p)]
            row['n_probability_overlap']=len(valid)
            if valid:
                mb=np.array([(x['over_probability']-x['actual_over'])**2 for x,p in valid])
                eb=np.array([(p-x['actual_over'])**2 for x,p in valid])
                row.update(model_brier=mean_or_blank(mb),existing_brier=mean_or_blank(eb),
                           brier_gain=mean_or_blank(eb-mb))
        out.append(row)
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, nargs='+', required=True,
                   help='One or more pregame prediction CSVs, or 01_predictions directory')
    p.add_argument('--targets', type=Path, required=True,
                   help='Completed NHL player-game targets, read only')
    p.add_argument('--output-dir', type=Path, default=OUT_DIR)
    p.add_argument('--official-ppp-cache', type=Path,
                   help='Optional cache location (verified official NHL stats)')
    p.add_argument('--existing-predictions', type=Path,
                   help='Optional separate existing model CSV with game_id/player_id/market/predicted_mean')
    args=p.parse_args(argv)
    if not args.targets.is_file():
        raise FileNotFoundError(args.targets)
    for path in args.predictions:
        if not path.exists():
            raise FileNotFoundError(path)
    pred, game_ids, days = load_predictions(args.predictions)
    outcomes = load_outcomes(args.targets, game_ids, days)
    # Fetch only played skaters from evaluated games; NEVER use outcomes to create predictions.
    skaters_by_season = defaultdict(dict)
    for key,r in outcomes.items():
        if r['position_group']=='SKATER' and r['participated']=='1':
            skaters_by_season[r['season']][key]=r
    official = {}
    for season,rows in skaters_by_season.items():
        cache = args.official_ppp_cache or (args.output_dir / f'official_ppp_eval_{season.replace("-", "_")}.json.gz')
        if args.official_ppp_cache and len(skaters_by_season)>1:
            raise ValueError('Explicit --official-ppp-cache is limited to one season')
        official.update(official_ppp_values(rows, season, cache))
    evaluated = []
    missed_count = 0
    for r in pred:
        key = (r['game_id'],r['player_id'])
        a = outcomes.get(key)
        if a is None or a['participated']!='1':
            continue
        if a['position_group'] != r['position_group']:
            raise ValueError(f'Prediction player position differs from outcome {key}')
        if r['market']=='power_play_points':
            value = official.get(key)
            if value is None:
                raise ValueError(f'Official PPP missing for played player {key}')
            actual = float(value)
        else:
            actual = numeric(a.get(r['market']),f'{r["market"]} outcome {key}')
        threshold = numeric(r['threshold'],'threshold')
        mu = numeric(r['predicted_mean'],'model mean')
        prob = numeric(r['over_probability'],'model probability')
        b5 = numeric(r['baseline_last5'],'baseline5',True)
        b10 = numeric(r['baseline_last10'],'baseline10',True)
        p5 = numeric(r['baseline_last5_over_probability'],'baseline5 probability',True)
        p10 = numeric(r['baseline_last10_over_probability'],'baseline10 probability',True)
        evaluated.append({'game_date':parse_day(r['game_date']),'game_id':r['game_id'],
                          'player_id':r['player_id'],'position_group':r['position_group'],
                          'market':r['market'],'actual':actual,'predicted_mean':mu,
                          'baseline_last5':b5,'baseline_last10':b10,
                          'absolute_error':abs(mu-actual),
                          'squared_error':(mu-actual)**2,
                          'actual_over':int(actual>threshold), 'over_probability':prob,
                          'baseline_last5_over_probability':p5,
                          'baseline_last10_over_probability':p10,
                          'was_played':1})
    if not evaluated:
        raise ValueError('No completed player-stat outcomes yet; rerun after games finish')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    grouped = defaultdict(list)
    daily = defaultdict(list)
    for r in evaluated:
        grouped[r['market']].append(r)
        daily[(r['game_date'],r['market'])].append(r)
    market_rows = [metrics(grouped[m],m) for m in THRESHOLDS if grouped[m]]
    daily_rows = [dict(game_date=d, **metrics(rows,m)) for (d,m),rows in sorted(daily.items())]
    selection_rows = selection_accuracy(pred,outcomes)
    atomic_csv(args.output_dir / 'accuracy_by_market.csv', market_rows, MARKET_FIELDS)
    atomic_csv(args.output_dir / 'accuracy_by_market_day.csv', daily_rows, DAY_MARKET_FIELDS)
    atomic_csv(args.output_dir / 'player_market_accuracy.csv', evaluated, SCORED_FIELDS)
    atomic_csv(args.output_dir / 'selection_accuracy.csv', selection_rows, SELECTION_FIELDS)
    if args.existing_predictions:
        if not args.existing_predictions.is_file():
            raise FileNotFoundError(args.existing_predictions)
        external = compare_external(evaluated, args.existing_predictions)
        atomic_csv(args.output_dir / 'existing_method_comparison.csv', external, EXTERNAL_FIELDS)
        print(f'EXISTING METHOD | matched rows={sum(x["n_overlap"] for x in external)}')
    print(f'STATUS | COMPLETE | games={len(game_ids)} | models={len(grouped)}/8 '
          f'| graded participant-markets={len(evaluated)}')
    print('COMPARISON | paired prior-5 and prior-10 player averages; positive gain means model better')
    print(f'OUTPUT | {args.output_dir}')
    print('NO ODDS, PROFITABILITY OR REPOSITORY WORKFLOW CHANGES')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FAILED | {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
