#!/usr/bin/env python3
"""Eight-market NHL player-stat predictions from VERIFIED pregame features.

Runs the existing build_features_asof.py as a subprocess or reads its sealed
slate CSV+manifest. No odds, no production workflow modifications, no grading.

Usage from the repository root (example):
 python docs/win/hockey/nhl/prop_engine/scripts/01_predict/predict_player_stats.py \
   --slate-date 2026-11-16 --targets /path/to/nhl_historical_player_game_targets_2009_2026.csv \
   --observed-at-utc 2026-11-16T16:00:00Z
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from model_scoring import load_models, over_probability

PROP_ROOT = Path(__file__).resolve().parents[2]
NHL_ROOT = PROP_ROOT.parent
DEFAULT_MODELS = PROP_ROOT / 'models' / 'nhl_8_market_portable_models.json.gz'
DEFAULT_OVERLAY = PROP_ROOT / 'models' / 'nhl_ppp_player_game_overlay_2009_2026_RESEARCH_ONLY_UPDATED.csv.gz'
BUILDER = PROP_ROOT / 'scripts' / '00_intake' / 'build_features_asof.py'
OUT_DIR = PROP_ROOT / '01_predictions'
EXPECTED_BUILDER = '2026-11-17-asof-v2-prop-engine-rolefix1'
API = 'https://api.nhle.com/stats/rest/en/skater/summary'
THRESHOLDS = {'shots_on_goal': ('SKATER', 2.5), 'points': ('SKATER', 0.5),
              'assists': ('SKATER', 0.5), 'goals': ('SKATER', 0.5),
              'blocked_shots': ('SKATER', 1.5), 'power_play_points': ('SKATER', 0.5),
              'saves': ('GOALIE', 25.5), 'goals_against': ('GOALIE', 2.5)}
OUTPUT_FIELDS = ('game_date', 'game_id', 'player_id', 'team', 'opponent_team',
                 'home_away', 'position_group', 'candidate_source',
                 'candidate_captured_utc', 'game_start_utc', 'prediction_as_of_utc',
                 'market', 'predicted_mean', 'baseline_last5', 'baseline_last10',
                 'threshold', 'over_probability', 'baseline_last5_over_probability',
                 'baseline_last10_over_probability', 'model_sha256',
                 'feature_sha256', 'builder_version')


def iter_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'Empty CSV: {path}')
        yield from reader


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for buf in iter(lambda: fh.read(2 * 1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def utc(value):
    raw = str(value or '').strip()
    if not raw:
        raise ValueError('Missing required UTC timestamp')
    x = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    if x.tzinfo is None or x.utcoffset() is None:
        raise ValueError(f'Timestamp lacks UTC offset: {raw}')
    return x.astimezone(timezone.utc)


def parse_day(value: str) -> str:
    raw = str(value).replace('_', '-')
    if not re.fullmatch(r'20\d{2}-\d{2}-\d{2}', raw):
        raise ValueError(f'Invalid game date: {raw!r}')
    return date.fromisoformat(raw).isoformat()


def atomic_csv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, suffix='.tmp',
                                     encoding='utf-8-sig', newline='',
                                     delete=False) as f:
        tmp = Path(f.name)
        try:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='raise')
            w.writeheader()
            w.writerows(rows)
        except Exception:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)



def choose_historical_pregame_asof(root: Path, day: str) -> str:
    """For an already-played date, freeze at the last PRE-first-puck snapshot.

    This cannot look at subsequent lineup updates or outcomes. A complete
    lineup snapshot must exist for every scheduled game.
    """
    sched = root / '00_intake' / 'nhl_schedule' / f'NHL_{day.replace("-", "_")}.csv'
    combos = root / '00_intake' / 'lineups' / f'{day}_line_combinations.csv'
    goalies = root / '00_intake' / 'lineups' / f'{day}_goalies.csv'
    for path in (sched, combos, goalies):
        if not path.is_file():
            raise FileNotFoundError(f'Missing dated pregame replay snapshot: {path}')
    schedule = {r['game_id']: utc(r['start_time_utc']) for r in iter_csv(sched)}
    if not schedule:
        raise ValueError(f'No games in dated schedule: {day}')
    first_start = min(schedule.values())
    captures = []
    coverage = set()
    for r in iter_csv(combos):
        gid = r['game_id']
        if gid not in schedule:
            continue
        cap = utc(r['CapturedUTC'])
        if cap >= schedule[gid] or r.get('lineup_category', '').lower() != 'ev':
            continue
        if r.get('lineup_updated_utc') and utc(r['lineup_updated_utc']) > cap:
            continue
        if cap < first_start:
            captures.append(cap)
            coverage.add(gid)
    if coverage != set(schedule) or not captures:
        raise ValueError(f'Incomplete pregame lineup coverage: {len(coverage)}/{len(schedule)}')
    for r in iter_csv(goalies):
        gid = r['game_id']
        if gid in schedule:
            cap = utc(r['CapturedUTC'])
            if cap < min(schedule[gid], first_start):
                captures.append(cap)
    return max(captures).isoformat(timespec='seconds').replace('+00:00', 'Z')

def run_existing_builder(targets: Path, slate_day: str, observed: str | None,
                         output_dir: Path):
    if not BUILDER.is_file():
        raise FileNotFoundError(f'Existing feature builder missing: {BUILDER}')
    if EXPECTED_BUILDER not in BUILDER.read_text('utf-8'):
        raise ValueError('Feature-builder version changed; inspect compatibility before scoring')
    command = [sys.executable, '-B', str(BUILDER), '--nhl-root', str(NHL_ROOT),
               '--targets', str(targets), '--slate-date', slate_day,
               '--output-dir', str(output_dir)]
    if observed:
        command += ['--observed-at-utc', observed]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError('Existing feature builder failed:\n' +
                           (result.stdout + '\n' + result.stderr)[-5000:])
    p = output_dir / f'nhl_player_features_asof_{slate_day.replace("-", "_")}.csv'
    return p, p.with_name(p.stem + '_manifest.json')


def load_slate_features(path: Path, manifest: Path, day: str):
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f'Missing generated feature file or manifest: {path} / {manifest}')
    meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if meta.get('mode') != 'slate' or meta.get('slate_date') != day:
        raise ValueError('Requires v2 SLATE features, never historical postgame rosters')
    if meta.get('builder_version') != EXPECTED_BUILDER:
        raise ValueError(f'Feature builder mismatch: {meta.get("builder_version")}')
    if meta.get('output_sha256') != checksum(path):
        raise ValueError('Feature manifest SHA-256 mismatch')
    if not meta.get('prediction_as_of_utc'):
        raise ValueError('Manifest missing pregame prediction as-of timestamp')
    freeze = utc(meta['prediction_as_of_utc'])
    df = pd.read_csv(path, dtype={'game_id': str, 'player_id': str, 'season': str,
                                  'game_type': str, 'game_date': str}, low_memory=False)
    if df.empty or df[['game_id', 'player_id']].duplicated().any():
        raise ValueError('Empty or duplicate player-game pregame features')
    required = {'game_date', 'game_id', 'player_id', 'season', 'game_start_utc',
                'candidate_source', 'candidate_captured_utc', 'team', 'opponent_team',
                'position_group', 'home_away', 'game_type', 'prior_player_last_game_date'}
    if required - set(df):
        raise ValueError(f'Missing mandatory slate features {sorted(required - set(df))}')
    if not df['game_date'].eq(day).all() or df['season'].nunique() != 1:
        raise ValueError('Mixed slate dates or seasons')
    allowed_sources = {'LINE_COMBINATIONS', 'DAILY_FACEOFF'}
    if not df['candidate_source'].isin(allowed_sources).all():
        raise ValueError('Feature file includes postgame or unidentified player candidates')
    if not df['position_group'].isin(('SKATER', 'GOALIE')).all():
        raise ValueError('Unexpected player position group')
    if not df['home_away'].isin(('HOME', 'AWAY')).all():
        raise ValueError('Unexpected home/away tag')
    for r in df.to_dict('records'):
        gid = str(r['game_id'])
        if (not re.fullmatch(r'\d{10}', gid) or gid[:4] != str(r['season'])[:4] or
            gid[4:6] != str(int(float(r['game_type']))).zfill(2)):
            raise ValueError(f'Game ID/season/type mismatch: {gid}')
        start = utc(r['game_start_utc'])
        captured = utc(r['candidate_captured_utc'])
        if freeze >= start:
            raise ValueError(f'Prediction decision time is not pregame for {r["game_id"]}')
        if captured >= start or captured > freeze:
            raise ValueError(f'Non-pregame player snapshot: {r["game_id"]}/{r["player_id"]}')
        if str(r.get('prior_player_last_game_date') or '').strip() not in ('', 'nan'):
            if parse_day(r['prior_player_last_game_date']) >= day:
                raise ValueError('Player feature includes same-day/future game result')
        for col in ('home_goalie_news_captured_utc', 'away_goalie_news_captured_utc',
                    'home_goalie_id_captured_utc', 'away_goalie_id_captured_utc',
                    'team_rating_captured_utc', 'opponent_rating_captured_utc'):
            stamp = r.get(col)
            if stamp is not None and str(stamp).strip() not in ('', 'nan'):
                if utc(stamp) >= start or utc(stamp) > freeze:
                    raise ValueError(f'Post-start {col} in game {r["game_id"]}')
    return df, meta


def read_target_metadata(targets: Path, current_season: str, cutoff: str):
    """Only completed earlier-day boxscore records are eligible PPP history."""
    info = {}
    played = {}
    for r in iter_csv(targets):
        gid = str(r['game_id']).strip()
        game_date = parse_day(r['game_date'])
        season = str(r['season']).strip()
        previous = info.setdefault(gid, (game_date, season))
        if previous != (game_date, season):
            raise ValueError(f'Conflicting target date/season game {gid}')
        if season != current_season or game_date >= cutoff:
            continue
        if r['position_group'] == 'SKATER' and str(r['participated']).strip() == '1':
            key = (gid, str(r['player_id']).strip())
            if key in played:
                raise ValueError(f'Duplicate current-season skater {key}')
            played[key] = r
    return info, played


def _official_stats_fetch(day: str, season_id: str, start: int):
    sort = json.dumps([{'property': 'gameId', 'direction': 'ASC'},
                       {'property': 'playerId', 'direction': 'ASC'}], separators=(',', ':'))
    query = (f'seasonId={season_id} and gameDate>="{day}" '
             f'and gameDate<="{day} 23:59:59"')
    url = API + '?' + urlencode({'isAggregate': 'false', 'isGame': 'true',
                                 'start': start, 'limit': 100, 'sort': sort,
                                 'cayenneExp': query})
    error = None
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={'Accept': 'application/json',
                                                'User-Agent': 'NHL-Pregame-Research/1.0'}),
                         timeout=25) as stream:
                obj = json.load(stream)
            if not isinstance(obj, dict) or not isinstance(obj.get('data'), list):
                raise ValueError(f'Invalid official NHL response {day}')
            return obj
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            error = exc
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                break
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'Official NHL Stats API {day} offset {start}: {error}')


def official_ppp_values(target_rows: dict, season: str, cache_path: Path) -> dict:
    """Officially verify EVERY requested played skater-game PPP; never fill blanks with zero.

    Date-scoped NHL queries avoid the 10k season cap and duplicate pagination.
    Cache is keyed by exact target game/player identities and boxscore statistics.
    """
    if not target_rows:
        return {}
    parts = re.fullmatch(r'(20\d{2})-(\d{2})', season)
    if not parts:
        raise ValueError(f'Invalid season: {season}')
    season_id = str(int(parts.group(1))) + str(int(parts.group(1)) + 1)
    date_keys = defaultdict(set)
    expected = {}
    for key, r in target_rows.items():
        d = parse_day(r['game_date'])
        date_keys[d].add(key)
        expected[key] = (d, str(r['goals']).strip(), str(r['assists']).strip(),
                         str(r['points']).strip(), str(r.get('power_play_goals') or '').strip())
    payload = '\n'.join(f'{gid}|{pid}|{expected[(gid,pid)]}' for gid,pid in sorted(expected))
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    days = sorted(date_keys)

    def verify(rows):
        result = {}
        for r in rows:
            key = (str(r.get('gameId', '')), str(r.get('playerId', '')))
            if key not in target_rows:
                continue
            if key in result:
                raise ValueError(f'Official NHL duplicate player-game {key}')
            ppp = r.get('ppPoints', r.get('powerPlayPoints'))
            ppg = r.get('ppGoals', r.get('powerPlayGoals'))
            if ppp is None or ppg is None:
                raise ValueError(f'Missing official PPP/PPG for {key}')
            val, pg = int(ppp), int(ppg)
            if val < pg or pg < 0:
                raise ValueError(f'Impossible official PP counts {key}')
            (_, goals, assists, points, ppg_box) = expected[key]
            for label, actual in (('goals', goals), ('assists', assists), ('points', points)):
                if not actual or r.get(label) is None or int(r[label]) != int(actual):
                    raise ValueError(f'Official NHL {label} mismatch {key}')
            if ppg_box and int(ppg_box) != pg:
                raise ValueError(f'Official NHL PPG/boxscore conflict {key}')
            if pg > int(goals) or val > int(points) or val-pg > int(assists):
                raise ValueError(f'Official NHL PPP exceeds individual totals {key}')
            result[key] = val
        if set(result) != set(target_rows):
            missing = sorted(set(target_rows) - set(result))
            raise ValueError(f'Official NHL PPP incomplete: {len(missing)} missing, first {missing[:4]}')
        return result

    if cache_path.is_file():
        try:
            with gzip.open(cache_path, 'rt', encoding='utf-8') as fh:
                obj = json.load(fh)
            if (obj.get('cache_version') == 7 and obj.get('season') == season
                and obj.get('target_fingerprint') == fingerprint
                and obj.get('dates') == days):
                return verify(obj['data'])
        except (OSError, EOFError, KeyError, TypeError, ValueError) as e:
            print(f'Official NHL PPP cache rejected: {e}', file=sys.stderr)
    def fetch_day(day):
        first = _official_stats_fetch(day, season_id, 0)
        count = int(first.get('total', -1))
        if count < 1 or count >= 2000 or len(first['data']) != min(count, 100):
            raise ValueError(f'NHL date filter did not return bounded rows {day}: {count}')
        rows = list(first['data'])
        for off in range(100, count, 100):
            page = _official_stats_fetch(day, season_id, off)
            if int(page.get('total', -1)) != count or len(page['data']) != min(100, count-off):
                raise ValueError(f'NHL pagination changed/overlapped {day} offset {off}')
            rows.extend(page['data'])
        if len(rows) != count:
            raise ValueError(f'Incomplete NHL date {day}')
        keys = [(str(r.get('gameId', '')), str(r.get('playerId', ''))) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f'Duplicate NHL date {day} player-game keys')
        if date_keys[day] - set(keys):
            raise ValueError(f'Official NHL date {day} missing played-skater keys')
        return rows

    all_rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_day, d): d for d in days}
        found = {}
        for future in as_completed(futures):
            d = futures[future]
            found[d] = future.result()
            if len(found) % 10 == 0 or len(found) == len(days):
                print(f'Official NHL PPP date batches: {len(found)}/{len(days)}', flush=True)
    for d in days:
        all_rows.extend(found[d])
    result = verify(all_rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + '.tmp')
    try:
        with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
            json.dump({'cache_version': 7, 'season': season,
                       'target_fingerprint': fingerprint, 'dates': days,
                       'verified_target_count': len(result),
                       'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                       'source': 'Official NHL date-scoped skater game statistics',
                       'data': all_rows}, fh, separators=(',', ':'))
        os.replace(tmp, cache_path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Official NHL PPP: VERIFIED {len(result)} played skater-games', flush=True)
    return result


def ppp_history_for_candidates(overlay: Path, game_info: dict, current: dict,
                               season: str, cutoff: str, candidate_ids: set):
    """Attach historical PPP by official game date; only candidate-player history."""
    history = defaultdict(list)
    op = gzip.open if str(overlay).endswith('.gz') else open
    seen = set()
    nrows = 0
    with op(overlay, 'rt', encoding='utf-8-sig', newline='') as stream:
        for r in csv.DictReader(stream):
            if r['position_group'] != 'SKATER' or str(r['participated']) != '1':
                continue
            nrows += 1
            gid, pid = str(r['game_id']), str(r['player_id'])
            if (gid, pid) in seen:
                raise ValueError(f'Duplicate historical PPP game/player {gid} {pid}')
            seen.add((gid, pid))
            info = game_info.get(gid)
            if not info or str(r['season']) != info[1]:
                raise ValueError(f'Historical PPP overlay game metadata mismatch {gid}')
            val = str(r['power_play_points']).strip()
            if not val or not val.isdecimal():
                raise ValueError(f'Invalid historical PPP for {gid} {pid}')
            if pid in candidate_ids and info[0] < cutoff:
                history[pid].append((info[0], info[1], int(val)))
    if nrows != 788503:
        raise ValueError(f'Historical PPP source unexpectedly changed: {nrows} rows (expected 788503)')
    for (gid, pid), val in current.items():
        if pid in candidate_ids:
            day, se = game_info[gid]
            if day >= cutoff or se != season:
                raise ValueError('Current PPP source violates prediction date cutoff')
            history[pid].append((day, se, val))
    for pid in history:
        history[pid].sort(key=lambda x: x[0])
    return history


def ppp_lags(records, player_id: str, day: str, season: str):
    previous = [x for x in records.get(player_id, ()) if x[0] < day]
    if not previous:
        return (float('nan'),) * 4
    career = sum(x[2] for x in previous) / len(previous)
    last5 = sum(x[2] for x in previous[-5:]) / min(5, len(previous))
    last10 = sum(x[2] for x in previous[-10:]) / min(10, len(previous))
    # Match the *PPP training code*, not build_features_asof's other-market
    # prior_season field: PPP uses the last previously played season.
    past_seasons = sorted({x[1] for x in previous if x[1] != season})
    if past_seasons:
        last_season = past_seasons[-1]
        vals = [x[2] for x in previous if x[1] == last_season]
        previous_season = sum(vals) / len(vals)
    else:
        previous_season = float('nan')
    return career, previous_season, last5, last10


def _nullable_num(v):
    if pd.isna(v):
        return float('nan')
    v = str(v).strip()
    if not v:
        return float('nan')
    f = float(v)
    return f if math.isfinite(f) else float('nan')


def predict(df: pd.DataFrame, models: dict, ppp_history: dict,
            day: str, meta: dict, model_sha: str, feature_sha: str):
    result = []
    for market, (role, threshold) in THRESHOLDS.items():
        f = df[df['position_group'] == role].copy()
        if f.empty:
            raise ValueError(f'No candidates for market {market} role {role}')
        model = models[market]
        if market == 'power_play_points':
            vals = [ppp_lags(ppp_history, str(pid), day, str(s))
                    for pid, s in zip(f['player_id'], f['season'])]
            for i, name in enumerate(model.features[-4:]):
                f[name] = [v[i] for v in vals]
        f['is_home'] = f['home_away'].eq('HOME').astype('float32')
        f['is_playoff'] = pd.to_numeric(f['game_type'], errors='raise').eq(3).astype('float32')
        absent = set(model.features) - set(f.columns)
        if absent:
            raise ValueError(f'{market} missing feature columns {sorted(absent)}')
        arr = f[list(model.features)].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
        forecast = model.predict(arr)
        if len(forecast) != len(f) or np.any(~np.isfinite(forecast)):
            raise ValueError(f'Invalid model predictions for {market}')
        p_model = over_probability(forecast, threshold, model.alpha)
        if market == 'power_play_points':
            base = 'player_power_play_points'
        else:
            base = 'player_' + market
        def baseline(suffix):
            name = base + '_mean_' + suffix
            if name not in f:
                raise ValueError(f'Missing baseline {name}')
            x = pd.to_numeric(f[name], errors='coerce').to_numpy(dtype=np.float64)
            return x
        baseline5 = baseline('last5')
        baseline10 = baseline('last10')
        def baseline_prob(values):
            pp = np.full(len(values), np.nan)
            valid = np.isfinite(values) & (values >= 0)
            if valid.any():
                pp[valid] = over_probability(values[valid], threshold, model.alpha)
            return pp
        p5 = baseline_prob(baseline5)
        p10 = baseline_prob(baseline10)
        for i, row in enumerate(f.to_dict('records')):
            def number(v):
                return '' if not math.isfinite(float(v)) else float(v)
            result.append(dict(game_date=day, game_id=row['game_id'], player_id=row['player_id'],
                               team=row['team'], opponent_team=row['opponent_team'],
                               home_away=row['home_away'], position_group=role,
                               candidate_source=row['candidate_source'],
                               candidate_captured_utc=row['candidate_captured_utc'],
                               game_start_utc=row['game_start_utc'],
                               prediction_as_of_utc=meta['prediction_as_of_utc'], market=market,
                               predicted_mean=number(forecast[i]), baseline_last5=number(baseline5[i]),
                               baseline_last10=number(baseline10[i]), threshold=threshold,
                               over_probability=number(p_model[i]),
                               baseline_last5_over_probability=number(p5[i]),
                               baseline_last10_over_probability=number(p10[i]),
                               model_sha256=model_sha, feature_sha256=feature_sha,
                               builder_version=meta['builder_version']))
    return sorted(result, key=lambda x: (x['game_id'], x['market'], str(x['player_id'])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slate-date', required=True, help='YYYY-MM-DD; must have pregame snapshots')
    p.add_argument('--targets', required=True, type=Path,
                   help='Existing complete historical player-game target CSV (read-only input)')
    p.add_argument('--observed-at-utc', help='Optional fixed pregame decision timestamp')
    p.add_argument('--features', type=Path, help='Use previously built and verified slate features')
    p.add_argument('--manifest', type=Path, help='Manifest for --features; otherwise inferred')
    p.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    p.add_argument('--ppp-overlay', type=Path, default=DEFAULT_OVERLAY)
    p.add_argument('--output-dir', type=Path, default=OUT_DIR)
    p.add_argument('--current-ppp-cache', type=Path,
                   help='Optional cache path for verified official current-season PPP stats')
    args = p.parse_args(argv)
    day = parse_day(args.slate_date)
    if not args.targets.is_file() or not args.models.is_file() or not args.ppp_overlay.is_file():
        raise FileNotFoundError('Missing --targets, models or packaged PPP overlay')
    if args.observed_at_utc:
        frozen = utc(args.observed_at_utc)
        if frozen > datetime.now(timezone.utc):
            raise ValueError('Prediction as-of timestamp cannot be in the future')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.features:
        if args.observed_at_utc:
            raise ValueError('Use the sealed --features manifest freeze; do not override it')
        feats = args.features
        manifest = args.manifest or feats.with_name(feats.stem + '_manifest.json')
    else:
        if args.manifest:
            raise ValueError('--manifest requires --features')
        observed = args.observed_at_utc
        if observed is None and day < datetime.now(ZoneInfo('America/New_York')).date().isoformat():
            observed = choose_historical_pregame_asof(NHL_ROOT, day)
            print(f'REPLAY PRESTART FREEZE | {observed}', flush=True)
        feats, manifest = run_existing_builder(args.targets, day, observed, args.output_dir)
    df, meta = load_slate_features(feats, manifest, day)
    season = str(df['season'].iat[0])
    model_set = load_models(args.models)
    info, current = read_target_metadata(args.targets, season, day)
    cache = args.current_ppp_cache or (args.output_dir / f'official_ppp_{season.replace("-", "_")}_pre_{day}.json.gz')
    official = official_ppp_values(current, season, cache)
    skaters = set(df.loc[df['position_group'].eq('SKATER'), 'player_id'].astype(str))
    history = ppp_history_for_candidates(args.ppp_overlay, info, official, season, day, skaters)
    predictions = predict(df, model_set, history, day, meta, checksum(args.models), checksum(feats))
    output = args.output_dir / f'{day.replace("-", "_")}_nhl_player_predictions.csv'
    atomic_csv(output, predictions, OUTPUT_FIELDS)
    print(f'STATUS | COMPLETE | date={day} | games={df.game_id.nunique()} '
          f'| candidates={len(df)} | markets=8/8 | predictions={len(predictions)}')
    print(f'OFFICIAL PPP | verified earlier current-season skater-games={len(official)}')
    print(f'OUTPUT | {output}')
    print('EXISTING REPO SCRIPTS UNCHANGED | NO ODDS/PROFITABILITY')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FAILED | {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
