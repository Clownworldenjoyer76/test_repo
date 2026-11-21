#!/usr/bin/env python3
"""NHL Item-5 season-to-date accuracy replay using strictly past-game roster proxy.

Inputs are local, EXISTING repository files. Never substitutes postgame game
participants as a pregame selection rule: candidates come exclusively from the
last N previously COMPLETED games for each club. Both candidate predictions and
graded participating predictions are saved. Actual, timestamped pregame slate
results are preserved SEPARATELY; this script does not claim proxy candidates
were available in lineup snapshots.

All 8 markets require VERIFIED official NHL PPP stats; if unavailable, fail
without publishing results. --offline-seven-market-check is for development
validation only and is explicitly labelled INCOMPLETE.

No odds/profitability. Does not edit any existing script or production output.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROP_ROOT = Path(__file__).resolve().parents[2]
PREDICT_DIR = PROP_ROOT / 'scripts' / '01_predict'
EVAL_DIR = PROP_ROOT / 'scripts' / '02_evaluate'
for dir_ in (PREDICT_DIR, EVAL_DIR):
    if str(dir_) not in sys.path:
        sys.path.insert(0, str(dir_))

from predict_player_stats import (THRESHOLDS, official_ppp_values, parse_day,
                                  checksum, atomic_csv)
from model_scoring import load_models, over_probability
from compare_prediction_accuracy import (metrics, MARKET_FIELDS, DAY_MARKET_FIELDS,
                                         SCORED_FIELDS, SELECTION_FIELDS)

STATS = ('shots_on_goal','points','assists','goals','blocked_shots',
         'power_play_goals','saves','goals_against')
MEASURES = ('goals_for','goals_against','shots_on_goal','blocked_shots',
            'goalie_saves','goalie_goals_against')
REQUIRED = {'season','game_id','game_date','game_type','player_id','team',
            'opponent_team','home_away','position_group','participated',*STATS}
MODEL_PATH = PROP_ROOT / 'models' / 'nhl_8_market_portable_models.json.gz'
OVERLAY_PATH = PROP_ROOT / 'models' / 'nhl_ppp_player_game_overlay_2009_2026_RESEARCH_ONLY_UPDATED.csv.gz'
TARGETS_PATH = PROP_ROOT / 'data' / 'nhl_historical_player_game_targets_2009_2026.csv'
OUT_ROOT = PROP_ROOT / '02_evaluation'
PROXY_RULE = 'PRIOR_3_TEAM_GAMES_PROXY'
EXPECTED_BUILDER = '2026-11-17-asof-v2-prop-engine-rolefix1'

class History:
    def __init__(self):
        self.n=0
        self.last_date=None
        self.sums=defaultdict(float);self.counts=defaultdict(int)
        self.last5=deque();self.sums5=defaultdict(float);self.counts5=defaultdict(int)
        self.last10=deque();self.sums10=defaultdict(float);self.counts10=defaultdict(int)
    def _window(self,queue,sums,counts,n,stats):
        if len(queue)==n:
            old=queue.popleft()
            for k,v in old.items():
                if v is not None:
                    sums[k]-=v;counts[k]-=1
        queue.append(stats)
        for k,v in stats.items():
            if v is not None:
                sums[k]+=v;counts[k]+=1
    def add(self,stats,day):
        self.n+=1
        for k,v in stats.items():
            if v is not None:
                self.sums[k]+=v;self.counts[k]+=1
        self._window(self.last5,self.sums5,self.counts5,5,stats)
        self._window(self.last10,self.sums10,self.counts10,10,stats)
        self.last_date=day
    def mean(self,key,scope):
        if scope in ('prior_career','prior_season'):
            sums,counts=self.sums,self.counts
        elif scope=='last5':
            sums,counts=self.sums5,self.counts5
        elif scope=='last10':
            sums,counts=self.sums10,self.counts10
        else:raise ValueError(scope)
        return round(sums[key]/counts[key],8) if counts.get(key,0) else np.nan

class FeatureState:
    """Reimplements only the read-only history state of build_features_asof.py.
    Uses its exactly dated update policy and eight target measures. Results
    compared against the user's actual-snapshot model forecasts before release.
    """
    def __init__(self):
        self.player_career=defaultdict(History)
        self.player_season=defaultdict(History)
        self.team_season=defaultdict(History)
        self.rosters=defaultdict(lambda:deque(maxlen=3))
    def features(self,rec,day,season):
        pid,role=rec['player_id'],rec['position_group']
        career=self.player_career.get((pid,role))
        present=self.player_season.get((season,pid,role))
        out={'prior_player_games_career':career.n if career else 0,
             'prior_player_games_season':present.n if present else 0,
             'prior_player_days_since_last_game':(day-career.last_date).days if career and career.last_date else np.nan}
        for s in STATS:
            for scope in ('prior_career','prior_season','last5','last10'):
                h=present if scope=='prior_season' else career
                out[f'player_{s}_mean_{scope}']=h.mean(s,scope) if h else np.nan
        for prefix,team in (('team',rec['team']),('opponent',rec['opponent_team'])):
            h=self.team_season.get((season,team))
            out[f'{prefix}_games_prior_season']=h.n if h else 0
            for s in MEASURES:
                for scope in ('prior_season','last5'):
                    out[f'{prefix}_{s}_mean_{scope}']=h.mean(s,scope) if h else np.nan
        out['is_home']=float(rec['home_away']=='HOME')
        out['is_playoff']=float(rec['game_type']=='3')
        return out
    def eligible(self,team,count):
        result={}
        # Prefer most recent role in the team's own previous games.
        for _,roster in list(self.rosters[team])[-count:][::-1]:
            for pid,role in roster.items():
                result.setdefault(pid,role)
        return result
    def update_day(self,rows,day):
        games=defaultdict(lambda:{'teams':defaultdict(lambda:defaultdict(int)),
                                  'opponents':{},'rosters':defaultdict(dict)})
        for r in rows:
            if r['participated']!=1:continue
            stats={s:r[s] for s in STATS}
            self.player_career[(r['player_id'],r['position_group'])].add(stats,day)
            self.player_season[(r['season'],r['player_id'],r['position_group'])].add(stats,day)
            gm=games[(r['season'],r['game_id'])]
            gm['opponents'][r['team']]=r['opponent_team']
            gm['rosters'][r['team']][r['player_id']]=r['position_group']
            team=gm['teams'][r['team']]
            if r['position_group']=='SKATER':
                team['goals_for']+=r['goals']
                team['shots_on_goal']+=r['shots_on_goal']
                team['blocked_shots']+=r['blocked_shots']
            else:
                team['goalie_saves']+=r['saves']
                team['goalie_goals_against']+=r['goals_against']
        for (season,gid),gm in games.items():
            teams=gm['teams']
            for team,opponent in gm['opponents'].items():
                if team not in teams or opponent not in teams:
                    raise ValueError(f'Incomplete game teams {gid}: {team}/{opponent}')
            for team,opponent in gm['opponents'].items():
                data={k:teams[team][k] for k in MEASURES if k!='goals_against'}
                data['goals_against']=teams[opponent]['goals_for']
                self.team_season[(season,team)].add(data,day)
                self.rosters[team].append((gid,gm['rosters'][team]))

class PPPHistory:
    def __init__(self):
        self.n=0;self.total=0;self.recent5=deque(maxlen=5);self.recent10=deque(maxlen=10)
        self.seasons=defaultdict(lambda:[0,0])
    def add(self,season,val):
        self.n+=1;self.total+=val
        self.recent5.append(val);self.recent10.append(val)
        self.seasons[season][0]+=val;self.seasons[season][1]+=1
    def means(self,season):
        if not self.n:return (np.nan,np.nan,np.nan,np.nan)
        previous=sorted(s for s in self.seasons if s!=season)
        prev_season=(self.seasons[previous[-1]][0]/self.seasons[previous[-1]][1]) if previous else np.nan
        return (self.total/self.n,prev_season,sum(self.recent5)/len(self.recent5),
                sum(self.recent10)/len(self.recent10))


def iter_target_days(path):
    """Externally sort source by actual game date, exactly as the feature builder.

    Game IDs are NOT guaranteed chronological. Do not use input row order.
    SQLite is a temporary scratch database, not a repository modification.
    """
    names=('season','game_id','game_date','game_type','player_id','team',
           'opponent_team','home_away','position_group','participated',*STATS)
    with tempfile.TemporaryDirectory(prefix='nhl_s2d_sort_') as td:
        db=sqlite3.connect(str(Path(td)/'targets.sqlite'))
        try:
            db.execute('PRAGMA journal_mode=OFF')
            db.execute('PRAGMA synchronous=OFF')
            columns=','.join('"'+name+'" TEXT' for name in names)
            db.execute(f'CREATE TABLE source ({columns}, PRIMARY KEY (game_id, player_id)) WITHOUT ROWID')
            sql='INSERT INTO source VALUES('+','.join('?' for _ in names)+')'
            batch=[];total=0
            with open(path,encoding='utf-8-sig',newline='') as fh:
                reader=csv.DictReader(fh)
                if not reader.fieldnames or REQUIRED-set(reader.fieldnames):
                    raise ValueError(f'Targets missing required columns: {sorted(REQUIRED-set(reader.fieldnames or []))}')
                for row in reader:
                    batch.append(tuple(row[n] for n in names));total+=1
                    if len(batch)>=20000:
                        db.executemany(sql,batch);batch=[]
                if batch:db.executemany(sql,batch)
            db.commit()
            db.execute('CREATE INDEX target_game_date_ix ON source(game_date,game_id)')
            db.commit()
            print(f'SOURCE SORT | {total} rows ordered by actual game_date/game_id',flush=True)
            current=None;batch=[]
            for tup in db.execute('SELECT '+','.join('"'+n+'"' for n in names)+
                                  ' FROM source ORDER BY game_date,game_id,player_id'):
                row=dict(zip(names,tup))
                d=parse_day(row['game_date'])
                if current is not None and d!=current:
                    yield current,batch
                    batch=[]
                current=d
                if row['participated'] not in ('0','1'):
                    raise ValueError(f'Invalid participation for {row["game_id"]}')
                row['participated']=int(row['participated'])
                row['position_group']=row['position_group'].upper()
                row['team']=row['team'].upper();row['opponent_team']=row['opponent_team'].upper()
                row['home_away']=row['home_away'].upper()
                for stat in STATS:
                    raw=str(row[stat]).strip()
                    row[stat]=(int(raw) if raw else None)
                    if row[stat] is not None and row[stat]<0:
                        raise ValueError(f'Negative {stat} in {row["game_id"]}')
                if row['participated'] and row['position_group']=='SKATER':
                    if any(row[k] is None for k in STATS[:5]):
                        raise ValueError(f'Skater missing target {row["game_id"]}')
                if row['participated'] and row['position_group']=='GOALIE':
                    if any(row[k] is None for k in ('saves','goals_against')):
                        raise ValueError(f'Goalie missing target {row["game_id"]}')
                if row['participated']==0 and any(row[k] is not None for k in STATS):
                    raise ValueError(f'Unplayed player has outcome {row["game_id"]}')
                batch.append(row)
            if current is not None:yield current,batch
        finally:db.close()

def get_current_rows(path,season,through):
    played={}; meta={'min_date':None,'max_date':None,'dates':set(),'games':set(),'source_rows':0}
    with open(path,encoding='utf-8-sig',newline='') as fh:
        reader=csv.DictReader(fh)
        for r in reader:
            if r['season']!=season:continue
            day=parse_day(r['game_date'])
            if day>through:continue
            meta['source_rows']+=1
            meta['dates'].add(day);meta['games'].add(r['game_id'])
            if meta['min_date'] is None or day<meta['min_date']:meta['min_date']=day
            if meta['max_date'] is None or day>meta['max_date']:meta['max_date']=day
            if r['position_group']=='SKATER' and r['participated']=='1':
                key=(r['game_id'],r['player_id'])
                if key in played:raise ValueError(f'Duplicate target skater {key}')
                played[key]=r
    if not played or not meta['dates']:
        raise ValueError(f'No completed games in {season} before {through}')
    return played,meta


def load_historical_ppp(path):
    result={};last=None
    with gzip.open(path,'rt',encoding='utf-8-sig',newline='') as fh:
        for row in csv.DictReader(fh):
            if row['position_group']!='SKATER' or row['participated']!='1':
                continue
            key=(row['game_id'],row['player_id'])
            if key in result:raise ValueError(f'Duplicate historical overlay key {key}')
            value=str(row['power_play_points']).strip()
            if not value.isdecimal():raise ValueError(f'Invalid historical PPP for {key}')
            result[key]=int(value)
    if len(result)!=788503:
        raise ValueError(f'Historical PPP row count changed: {len(result)} (expected 788503)')
    return result


def score_candidates(day,records,models,ppp_history,with_ppp):
    forecasts=[]
    for market,(role,threshold) in THRESHOLDS.items():
        if market=='power_play_points' and not with_ppp:continue
        selected=[r for r in records if r['position_group']==role]
        if not selected:continue
        model=models[market]
        X=[];b5=[];b10=[]
        for r in selected:
            f=r['features'].copy()
            if market=='power_play_points':
                hist=ppp_history.get(r['player_id'])
                vals=hist.means(r['season']) if hist else (np.nan,)*4
                for k,v in zip(model.features[-4:],vals):f[k]=v
                x5,x10=vals[2:]
            else:
                x5=f.get(f'player_{market}_mean_last5',np.nan)
                x10=f.get(f'player_{market}_mean_last10',np.nan)
            absent=set(model.features)-set(f)
            if absent:raise ValueError(f'Model feature mismatch {market}: {sorted(absent)}')
            X.append([f[k] for k in model.features]);b5.append(x5);b10.append(x10)
        feature_matrix=np.array(X,dtype=np.float32)
        mu=model.predict(feature_matrix)
        p=over_probability(mu,threshold,model.alpha)
        b5=np.array(b5,dtype=float);b10=np.array(b10,dtype=float)
        bp5=np.full(len(b5),np.nan);bp10=np.full(len(b10),np.nan)
        mask5=np.isfinite(b5)&(b5>=0);mask10=np.isfinite(b10)&(b10>=0)
        if mask5.any():bp5[mask5]=over_probability(b5[mask5],threshold,model.alpha)
        if mask10.any():bp10[mask10]=over_probability(b10[mask10],threshold,model.alpha)
        for i,r in enumerate(selected):
            forecasts.append({'game_date':day,'game_id':r['game_id'],
                              'player_id':r['player_id'],'team':r['team'],
                              'position_group':role,'market':market,
                              'candidate_source':PROXY_RULE,
                              'predicted_mean':float(mu[i]),'threshold':threshold,
                              'over_probability':float(p[i]),
                              'baseline_last5':float(b5[i]),'baseline_last10':float(b10[i]),
                              'baseline_last5_over_probability':float(bp5[i]),
                              'baseline_last10_over_probability':float(bp10[i])})
    return forecasts

CANDIDATE_FIELDS=('game_date','game_id','player_id','team','position_group','market',
                  'candidate_source','predicted_mean','threshold','over_probability',
                  'baseline_last5','baseline_last10','baseline_last5_over_probability',
                  'baseline_last10_over_probability')


def write_snapshot_copy(base,out_dir,through):
    m=base/'accuracy_by_market.csv';d=base/'accuracy_by_market_day.csv'
    if not m.is_file() or not d.is_file():
        return False
    with d.open(encoding='utf-8-sig',newline='') as fh:
        rows=list(csv.DictReader(fh))
    if not rows or any(parse_day(r['game_date'])>through for r in rows):
        return False
    shutil.copy2(m,out_dir/'actual_snapshot_accuracy_by_market.csv')
    shutil.copy2(d,out_dir/'actual_snapshot_accuracy_by_market_day.csv')
    if (base/'selection_accuracy.csv').is_file():
        shutil.copy2(base/'selection_accuracy.csv',out_dir/'actual_snapshot_selection_accuracy.csv')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',type=Path,default=TARGETS_PATH)
    p.add_argument('--models',type=Path,default=MODEL_PATH)
    p.add_argument('--ppp-overlay',type=Path,default=OVERLAY_PATH)
    p.add_argument('--season',default='2026-27')
    p.add_argument('--through',help='Inclusive YYYY-MM-DD cutoff; defaults to latest target date')
    p.add_argument('--proxy-games',type=int,choices=(1,2,3),default=3)
    p.add_argument('--output-dir',type=Path,default=OUT_ROOT/'season_to_date_2026_27')
    p.add_argument('--official-ppp-cache',type=Path,
                   help='Validated official NHL player-game PPP JSON gzip cache')
    p.add_argument('--offline-seven-market-check',action='store_true',
                   help='DEVELOPMENT ONLY: score seven markets without verifying official PPP')
    args=p.parse_args(argv)
    for path in (args.targets,args.models):
        if not path.is_file():raise FileNotFoundError(path)
    if not args.offline_seven_market_check and not args.ppp_overlay.is_file():
        raise FileNotFoundError(args.ppp_overlay)
    if args.through:
        through=parse_day(args.through)
    else:
        # Find max season target date (never use the calendar to invent completed games).
        with args.targets.open(encoding='utf-8-sig',newline='') as fh:
            through=max((parse_day(r['game_date']) for r in csv.DictReader(fh)
                         if r['season']==args.season),default='')
        if not through:raise ValueError('No matching target season')
    if through>datetime.now(timezone.utc).date().isoformat():
        raise ValueError('Cannot evaluate future game dates')
    builder=PROP_ROOT/'scripts'/'00_intake'/'build_features_asof.py'
    if not builder.is_file() or EXPECTED_BUILDER not in builder.read_text(encoding='utf-8'):
        raise ValueError('Existing feature builder missing or version changed; do not compare incompatible features')
    models=load_models(args.models)
    current,coverage=get_current_rows(args.targets,args.season,through)
    print(f'SEASON SOURCE | season={args.season} | through={coverage["max_date"]} '
          f'| games={len(coverage["games"])} | dates={len(coverage["dates"])} '
          f'| played skaters={len(current)}',flush=True)
    official={};overlay={}
    if not args.offline_seven_market_check:
        cache=args.official_ppp_cache or (args.output_dir/f'official_ppp_{args.season.replace("-","_")}_through_{through}.json.gz')
        official=official_ppp_values(current,args.season,cache)
        if len(official)!=len(current):raise ValueError('Official PPP verification failed')
        overlay=load_historical_ppp(args.ppp_overlay)
    all_scored=[];all_daily=defaultdict(list);metrics_by_market=defaultdict(list)
    selection_groups=defaultdict(lambda:{'candidates':set(),'actual':set(),'games':set()})
    output=args.output_dir
    # All outputs staged in a temporary directory, then published once complete.
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='nhl_season_proxy_',dir=output) as tempdir:
        staging=Path(tempdir)
        preds_file=staging/'proxy_candidate_predictions.csv'
        with preds_file.open('w',encoding='utf-8-sig',newline='') as pred_stream:
            writer=csv.DictWriter(pred_stream,fieldnames=CANDIDATE_FIELDS)
            writer.writeheader()
            state=FeatureState();ppp=defaultdict(PPPHistory)
            matched_historical=0
            seen_dates=set();seen_games=set();count_candidates=0
            for day,rows in iter_target_days(args.targets):
                if day>through:break
                current_day=any(r['season']==args.season for r in rows)
                if current_day:
                    game_rows=defaultdict(list)
                    for r in rows:
                        if r['season']==args.season:game_rows[r['game_id']].append(r)
                    eligible=[]
                    for gid,group in sorted(game_rows.items()):
                        seen_games.add(gid)
                        teams={r['team']:(r['opponent_team'],r['home_away']) for r in group}
                        if len(teams)!=2:raise ValueError(f'Unexpected team count for {gid}')
                        played={(r['player_id'],r['position_group']):r for r in group if r['participated']==1}
                        for team,(opponent,side) in teams.items():
                            proxy=state.eligible(team,args.proxy_games)
                            for pid,role in proxy.items():
                                r={'game_id':gid,'player_id':pid,'team':team,
                                   'opponent_team':opponent,'home_away':side,
                                   'position_group':role,'game_type':group[0]['game_type'],
                                   'season':args.season}
                                # All decisions here are based on earlier completed team games.
                                r['features']=state.features(r,date.fromisoformat(day),args.season)
                                eligible.append(r)
                                sel=selection_groups[(day,role)]
                                sel['candidates'].add((gid,pid,role,team));sel['games'].add(gid)
                        for (pid,role),actual in played.items():
                            sel=selection_groups[(day,role)]
                            sel['actual'].add((gid,pid,role,actual['team']));sel['games'].add(gid)
                    fcast=score_candidates(day,eligible,models,ppp,not args.offline_seven_market_check)
                    writer.writerows(fcast)
                    count_candidates+=len(eligible)
                    # Grade only actual participants who were SELECTED before the game.
                    lookup={(r['game_id'],r['player_id']):r
                            for r in rows if r['season']==args.season and r['participated']==1}
                    for f in fcast:
                        r=lookup.get((f['game_id'],f['player_id']))
                        if r is None or r['team']!=f['team']:continue
                        if r['position_group']!=f['position_group']:
                            raise ValueError(f'Actual role disagreement {f["game_id"]}/{f["player_id"]}')
                        if f['market']=='power_play_points':
                            key=(f['game_id'],f['player_id'])
                            if key not in official:raise ValueError(f'Missing verified PPP {key}')
                            actual=official[key]
                        else:actual=r[f['market']]
                        if actual is None:raise ValueError(f'Missing actual {f["market"]}')
                        scored={'game_date':day,'game_id':f['game_id'],'player_id':f['player_id'],
                                'position_group':r['position_group'],'market':f['market'],
                                'actual':float(actual),'predicted_mean':f['predicted_mean'],
                                'baseline_last5':f['baseline_last5'],
                                'baseline_last10':f['baseline_last10'],
                                'absolute_error':abs(f['predicted_mean']-actual),
                                'squared_error':(f['predicted_mean']-actual)**2,
                                'actual_over':int(actual>f['threshold']),
                                'over_probability':f['over_probability'],
                                'baseline_last5_over_probability':f['baseline_last5_over_probability'],
                                'baseline_last10_over_probability':f['baseline_last10_over_probability'],
                                'was_played':1}
                        metrics_by_market[f['market']].append(scored)
                        all_daily[(day,f['market'])].append(scored)
                        all_scored.append(scored)
                    seen_dates.add(day)
                    if len(seen_dates)%10==0 or day==coverage['max_date']:
                        print(f'PREGAME PROXY | dates={len(seen_dates)}/{len(coverage["dates"])} '
                              f'| games={len(seen_games)}/{len(coverage["games"])} '
                              f'| graded={len(all_scored)}',flush=True)
                # Update strictly AFTER scoring the whole day; no same-day target outcomes used.
                state.update_day(rows,date.fromisoformat(day))
                if not args.offline_seven_market_check:
                    for r in rows:
                        if r['position_group']!='SKATER' or r['participated']!=1:continue
                        key=(r['game_id'],r['player_id'])
                        if r['season']==args.season:
                            if key not in official:raise ValueError(f'Missing official PPP {key}')
                            ppp[r['player_id']].add(r['season'],official[key])
                        else:
                            if key not in overlay:raise ValueError(f'Historical PPP missing {key}')
                            ppp[r['player_id']].add(r['season'],overlay[key])
                            matched_historical+=1
        if not args.offline_seven_market_check and matched_historical!=len(overlay):
            raise ValueError(f'Historical PPP did not cover full target source: {matched_historical}/{len(overlay)}')
        if seen_dates!=coverage['dates'] or seen_games!=coverage['games']:
            raise ValueError('Season proxy did not process all target dates/games')
        if not all_scored:raise ValueError('No graded predictions')
        groups=sorted(metrics_by_market)
        if not args.offline_seven_market_check and set(groups)!=set(THRESHOLDS):
            raise ValueError('Incomplete market coverage: '+str(groups))
        summary=[metrics(metrics_by_market[m],m) for m in THRESHOLDS if m in metrics_by_market]
        daily=[dict(game_date=d,**metrics(items,m)) for (d,m),items in sorted(all_daily.items())]
        selection=[]
        for (day,role),vals in sorted(selection_groups.items()):
            c,a=vals['candidates'],vals['actual']
            tp=len(c&a)
            selection.append({'game_date':day,'position_group':role,'games':len(vals['games']),
                              'actual_participants':len(a),'pregame_candidates':len(c),
                              'played_candidates':tp,'unplayed_candidates':len(c-a),
                              'missing_played':len(a-c),
                              'precision':(tp/len(c) if c else ''),
                              'recall':(tp/len(a) if a else '')})
        atomic_csv(staging/'proxy_accuracy_by_market.csv',summary,MARKET_FIELDS)
        atomic_csv(staging/'proxy_accuracy_by_market_day.csv',daily,DAY_MARKET_FIELDS)
        atomic_csv(staging/'proxy_player_market_accuracy.csv',all_scored,SCORED_FIELDS)
        atomic_csv(staging/'proxy_selection_by_market_day.csv',selection,SELECTION_FIELDS)
        snapshot_preserved=write_snapshot_copy(OUT_ROOT,staging,through)
        manifest={'status':'INCOMPLETE_7_MARKET_OFFLINE_TEST' if args.offline_seven_market_check else 'COMPLETE_8_MARKETS',
                  'season':args.season,'through':coverage['max_date'],
                  'first_date':coverage['min_date'],'target_dates':len(seen_dates),
                  'games':len(seen_games),'target_rows':coverage['source_rows'],
                  'played_skaters_verified_ppp':len(official),'proxy_candidate_players':count_candidates,
                  'graded_player_markets':len(all_scored),'markets':groups,
                  'proxy_selection_policy':f'last {args.proxy_games} previous completed TEAM games only; date-strict',
                  'snapshot_results_preserved_separately':snapshot_preserved,
                  'historical_overlay_records':matched_historical,
                  'source_targets':str(args.targets),'source_targets_sha256':checksum(args.targets),
                  'source_models_sha256':checksum(args.models),
                  'expected_feature_builder_version':EXPECTED_BUILDER,
                  'scope_warning':'Proxy candidates are hypothetical and not archived pregame lineup snapshots; evaluated only where they played',
                  'existing_scripts_modified':False,
                  'generated_utc':datetime.now(timezone.utc).isoformat(timespec='seconds')}
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        for file in staging.iterdir():
            os.replace(file,output/file.name)
    print(f'STATUS | {manifest["status"]} | games={len(seen_games)} '
          f'| dates={len(seen_dates)} | markets={len(groups)}/8 '
          f'| graded={len(all_scored)} | pregame-proxy-candidates={count_candidates}',flush=True)
    print('IMPORTANT | Proxy-based player selection only; actual snapshots kept in separate files')
    print('NO ODDS, PROFITABILITY OR EXISTING SCRIPT MODIFICATIONS')
    print(f'OUTPUT | {output}',flush=True)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'FAILED | {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
