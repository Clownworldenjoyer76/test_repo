#!/usr/bin/env python3
"""NHL player WAR production calculation, 32-club scope.

Inputs: SDV's already date-filtered historical PBP and shifts, NHL official
postgame PBP, boxscores, skater TOI, GS, and official shift charts. No hard-coded
2026 player results, no Windows Downloads paths, no offline audit data dependency.

Model: independent weighted 5v5 xG RAPM, ridge lambda=100; empirically calibrated
2023-24 F/D depth-skater replacement rates, with 2025 held out; NHL official
PP/PK TOI and shot-on-ice attribution, official penalties and faceoffs.
"""
from __future__ import annotations
import bisect
import collections
from collections import defaultdict
import csv
import json
import math
import pathlib
from pathlib import Path
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

CLUBS = frozenset('ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split())
PARTS = ('ev_off','ev_def','pp','pk','pens','faceoffs')
GPW = 6.596
RIDGE_LAMBDA = 100.0
# Audited 2023-24 depth-skater calibration, weighted capped at 300 minutes.
# These are model parameters, not precomputed player or team WAR.
REPLACEMENT = {
 'F': {'off': 0.0029165010953079973, 'def': -0.2147217169431254},
 'D': {'off': 0.04540267306248921, 'def': -0.06318589143094956},
}
class DataFailure(RuntimeError):
    pass

def require(condition,message):
    if not condition: raise DataFailure(message)

def number(x):
    v=float(x)
    require(math.isfinite(v),f'Non-finite numerical input: {x!r}')
    return v

def asint(x):
    return int(x) if x is not None and str(x)!='' else None

def clock_seconds(x):
    if x is None:return None
    if isinstance(x,(int,float)):
        require(math.isfinite(x),f'Invalid time: {x!r}');return int(round(x))
    x=str(x).strip()
    if not x:return None
    if re.fullmatch(r'\d+(?:\.\d+)?',x):return int(round(float(x)))
    m=re.fullmatch(r'(\d{1,3}):(\d{2})',x)
    if m:require(int(m.group(2))<60,'Invalid seconds in time');return int(m.group(1))*60+int(m.group(2))
    raise DataFailure(f'Unrecognized time: {x!r}')


class TextHTML(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.nodes=[]
    def handle_data(self,data):
        if data.strip():self.nodes.append(data.strip())
    def handle_endtag(self,tag):
        if tag.lower() in ('tr','th','td','p','div','br','table'):self.nodes.append(' ')


def parse_gs_html(raw):
    txt=TextHTML();txt.feed(raw)
    s=' '.join(' '.join(txt.nodes).split())
    # NHL official away is listed before home in the GS report.
    pat=r'Power Plays\s*\(Goals\s*-\s*Opp\.?/\s*PPTime\)\s*(\d+)\s*-\s*(\d+)\s*/\s*(\d{1,2}:\d{2})'
    matches=re.findall(pat,s,flags=re.I)
    if len(matches)!=2:
        # Alternate separator/typography, but retain the exact official GS heading.
        pat=r'Power Plays\s*\(Goals-Opp\./PPTime\)\s*(\d+\s*-\s*\d+\s*/\s*\d{1,2}:\d{2})'
        mm=re.findall(pat,s,flags=re.I)
        matches=[]
        for text in mm:
            m=re.search(r'(\d+)\s*-\s*(\d+)\s*/\s*(\d{1,2}:\d{2})',text)
            if m:matches.append(m.groups())
    require(len(matches)==2,f'GS official PP time not parsed: count={len(matches)}; sample={s[s.find("PENALTY SUMMARY"):s.find("BY PERIOD")][:500]}')
    return [{'goals':int(a),'chances':int(b),'seconds':clock_seconds(c)} for a,b,c in matches]


def parse_situation(code):
    s=str(code or '')
    if not re.fullmatch(r'[01][3-6][3-6][01]',s):return None
    ag,away,home,hg=(int(c) for c in s)
    # Do not mislabel empty-net extra attackers as penalty advantage.
    if ag!=1 or hg!=1:return None
    if home>away:return 'home'
    if away>home:return 'away'
    return None


def game_seconds(period,elapsed):
    period=int(period);seconds=clock_seconds(elapsed)
    require(1<=period<=6 and seconds is not None and 0<=seconds<=1200,'Invalid period/time')
    return (period-1)*1200+seconds


def official_plays(payload):
    events=[]
    for v in payload.get('plays') or []:
        per=(v.get('periodDescriptor') or {}).get('number') or v.get('period')
        tm=v.get('timeInPeriod')
        if per is None or tm is None:continue
        if str((v.get('periodDescriptor') or {}).get('periodType') or '').upper()=='SO':continue
        ts=game_seconds(per,tm)
        events.append({'second':ts,'event_id':asint(v.get('eventId')),
            'order':asint(v.get('sortOrder')) or 0,'strength':parse_situation(v.get('situationCode')),
            'raw_code':v.get('situationCode'),'type':v.get('typeDescKey',''),
            'event':v})
    events.sort(key=lambda x:(x['second'],x['order']))
    return events


def official_shift_intervals(payload, gid, home, away, game_players=None):
    """Parse actual NHL shift starts/ends; never infer skaters from shift CHANGE deltas."""
    answer={'home':[], 'away':[]}
    included=set(game_players or ())
    for item in payload.get('data') or []:
        if asint(item.get('gameId')) not in (None,gid):continue
        pid=asint(item.get('playerId'))
        team=str(item.get('teamAbbrev') or '').strip().upper()
        if team not in (home,away) or pid is None:continue
        if included and pid not in included:continue  # exclude goalies/unknown identities
        period=asint(item.get('period'))
        start=item.get('startTime');end=item.get('endTime')
        if period is None or start is None or end is None:continue  # event markers are not shifts
        if period<1 or period>5:continue
        try:
            a,b=game_seconds(period,start),game_seconds(period,end)
        except (DataFailure,ValueError,TypeError):continue
        if b<=a or b-a>1200:continue
        side='home' if team==home else 'away'
        answer[side].append((a,b,pid))
    for side in answer:answer[side].sort()
    return answer


def chart_lineup(shifts,sec,home,away,valid_players,official_players,game_id,pp_side,expected=None):
    """Prefer independently sourced NHL per-player shifts at shot-second boundary."""
    for offset in (-1,0,-2,1,-3,2,3):
        t=max(0,int(sec)+offset)
        groups={side:{pid for a,b,pid in shifts[side] if a<=t<b} for side in ('home','away')}
        ok=True
        for side,team in (('home',home),('away',away)):
            ids=groups[side]
            needed=(expected or {}).get(side)
            if needed is not None and len(ids)!=needed:ok=False;break
            if len(ids)<3 or len(ids)>5:ok=False;break
            comp='pp' if side==pp_side else 'pk'
            if not all((pid,team) in valid_players and official_players.get((game_id,pid,team),{}).get(comp,0)>0 for pid in ids):
                ok=False;break
        if ok:return groups,'NHL_OFFICIAL_SHIFTS_'+str(offset)
    return None,None


def independent_lineup(shot,stint,shift_intervals,sec,home,away,valid_players,official_players,gid,pp_side,expected):
    """No guessed players: every member must be in an NHL shift at event time."""
    result,source=chart_lineup(shift_intervals,sec,home,away,valid_players,official_players,gid,pp_side,expected)
    if result:return result,source
    # The event might occur exactly as the officials record the shift change.
    # Require every candidate in the NHL shift chart within 3 seconds of the event.
    observed=onice_from_shot(shot,stint,valid_players,home,away)
    for side,team in (('home',home),('away',away)):
        ids=observed[side]
        if len(ids)!=(expected or {}).get(side,len(ids)):return None,None
        if not (3<=len(ids)<=5):return None,None
        comp='pp' if side==pp_side else 'pk'
        if not all(official_players.get((gid,pid,team),{}).get(comp,0)>0 for pid in ids):return None,None
        for pid in ids:
            if not any(a-3<=sec<b+3 and p==pid for a,b,p in shift_intervals[side]):return None,None
    return observed,'SDV_LINEUP_INDEPENDENTLY_CONFIRMED_NHL_SHIFTS'


class ShiftHTMLParser(HTMLParser):
    """Read NHL TV/TH cells by their actual published CSS classes."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells=[]; self.capturing=False; self.depth=0; self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='td':
            classes=dict(attrs).get('class','').lower()
            if not self.capturing and ('playerheading' in classes or ('lborder' in classes and 'bborder' in classes)):
                self.capturing=True; self.parts=[];self.depth=1
            elif self.capturing:self.depth+=1
    def handle_data(self,s):
        if self.capturing:self.parts.append(s)
    def handle_endtag(self,tag):
        if tag=='td' and self.capturing:
            self.depth-=1
            if self.depth==0:
                txt=' '.join(' '.join(self.parts).split())
                if txt:self.cells.append(txt)
                self.capturing=False;self.parts=[]


def official_roster_jersies(pbp,home,away):
    team_ids={'home':asint((pbp.get('homeTeam') or {}).get('id')),
              'away':asint((pbp.get('awayTeam') or {}).get('id'))}
    jersey={'home':{},'away':{}}
    for r in pbp.get('rosterSpots') or []:
        pid=asint(r.get('playerId'));num=asint(r.get('sweaterNumber'))
        tm=asint(r.get('teamId'))
        side=next((s for s,i in team_ids.items() if i==tm),None)
        if pid and num and side:
            if num in jersey[side] and jersey[side][num]!=pid:
                raise DataFailure(f'Roster duplicate jersey {side} #{num}')
            jersey[side][num]=pid
    return jersey


def html_shift_intervals(raw, gid, side, jersey, official_player_ids):
    parser=ShiftHTMLParser();parser.feed(raw)
    require(len(parser.cells)>=40,f'NHL {side} HTML shift report has too few cells for game {gid}: {len(parser.cells)}')
    intervals=[];pid=None;i=0;bad=[];roster_jerseys=set(jersey)
    while i<len(parser.cells):
        t=parser.cells[i]
        heading=re.match(r'^\s*#?\s*(\d{1,2})\s+(.+?),\s*(.+)$',t)
        if heading:
            num=int(heading.group(1));pid=jersey.get(num)
            # Goalie headings remain excluded by the verified skater roster.
            i+=1;continue
        if pid is not None and i+4<len(parser.cells):
            chunk=parser.cells[i:i+5]
            per=chunk[1].upper()
            if re.fullmatch(r'\d+',chunk[0]) and (per.isdigit() or per in ('OT','SO')):
                i+=5
                if pid not in official_player_ids or per=='SO':continue
                period=4 if per=='OT' else int(per)
                if not 1<=period<=5:continue
                # NHL report per-period summary rows have plain time cells,
                # not the elapsed/remaining pair used by genuine shift rows.
                # Reject the summaries: they create fictitious player shifts.
                if '/' not in chunk[2] or '/' not in chunk[3]:
                    continue
                start=chunk[2].split('/')[0].strip()
                end=chunk[3].split('/')[0].strip()
                try:
                    a=game_seconds(period,start)
                    b=game_seconds(period,end)
                    if b>a and b-a<=1200:intervals.append((a,b,pid))
                    else:bad.append((pid,period,start,end))
                except (ValueError,TypeError,DataFailure):bad.append((pid,period,start,end))
                continue
        i+=1
    # Absence of shift coverage is not converted to a guessed lineup.
    ids={pid for a,b,pid in intervals}
    require(len(ids)>=12 and len(intervals)>=100,
        f'Official NHL {side} HTML shifts for game {gid}: {len(ids)} skaters, {len(intervals)} intervals, rejected={len(bad)}')
    return sorted(set(intervals))


def get_official_shifts(gid, home,away, pbp,players,official_cache):
    shifturl='https://api.nhle.com/stats/rest/en/shiftcharts?'+urllib.parse.urlencode({'cayenneExp':f'gameId={gid}','limit':-1,'start':0})
    errors=[]
    try:
        shifts=json.loads(official_cache(shifturl,f'official_shiftcharts_{gid}.json'))
        if len(shifts.get('data') or [])>=60:
            result=official_shift_intervals(shifts,gid,home,away,players.keys())
            if all(len({pid for a,b,pid in result[s]})>=12 and len(result[s])>=100 for s in ('home','away')):
                return result,'NHL_JSON_SHIFTCHART'
        errors.append('JSON shift chart lacked skater intervals')
    except Exception as e:errors.append(f'JSON shift chart: {e}')
    # NHL publishes HTML TV/TH TOI on a separate official endpoint. The
    # jersey number is mapped through that game's official PBP rosterSpots.
    year=int(str(gid)[:4]);season=f'{year}{year+1}'
    jersies=official_roster_jersies(pbp,home,away)
    if any(len(jersies[s])<12 for s in ('home','away')):
        # PBP rosterSpots occasionally differs; cross-check NHL boxscore.
        box=json.loads(official_cache(f'https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore',f'boxscore_{gid}.json'))
        for side in ('home','away'):
            team=box.get(f'{side}Team') or {}
            for kind in ('forwards','defense'):
                for r in ((box.get('playerByGameStats') or {}).get(f'{side}Team') or {}).get(kind,[]):
                    pid=asint(r.get('playerId'));number=asint(r.get('sweaterNumber'))
                    if pid and number:jersies[side][number]=pid
    require(all(len(jersies[s])>=12 for s in ('home','away')),
        f'Official roster jersey lookup unavailable for game {gid}: {[len(jersies[s]) for s in ("home","away")]}')
    result={}
    for side,prefix in (('home','TH'),('away','TV')):
        url=f'https://www.nhl.com/scores/htmlreports/{season}/{prefix}{gid%1000000:06d}.HTM'
        try:
            h=official_cache(url,f'{prefix}{gid%1000000:06d}.html')
            result[side]=html_shift_intervals(h,gid,side,jersies[side],set(players))
        except Exception as e:
            errors.append(f'{side} HTML report: {e}')
    require(len(result)==2,
        f'NHL shift verification for game {gid} failed using JSON and HTML: {"; ".join(errors)}')
    return result,'NHL_OFFICIAL_TV_TH_HTML_SHIFTCHARTS'


def locate_shot_event(gid,shot,official_events,byid,bytime):
    """Only match an xG event to a time/type/participant-compatible official event."""
    ev=asint(shot.get('event_id'))
    sec=asint(shot.get('game_seconds'))
    scorer=asint(shot.get('event_player_1_id'))
    typ=re.sub(r'[^A-Z]+','_',str(shot.get('event_type') or '').upper()).strip('_')
    type_map={
        'SHOT':{'shot-on-goal'},'SHOT_ON_GOAL':{'shot-on-goal'},
        'GOAL':{'goal'},'MISS':{'missed-shot'},'MISSED_SHOT':{'missed-shot'},
        'BLOCKED_SHOT':{'blocked-shot'},'BLOCK':{'blocked-shot'}
    }
    expected=type_map.get(typ)
    def compatible(p):
        if sec is not None and abs(p['second']-sec)>1:return False
        if expected and p['type'] not in expected:return False
        details=p['event'].get('details') or {}
        official_actor=asint(details.get('shootingPlayerId') or details.get('scoringPlayerId'))
        if scorer is not None and official_actor is not None and official_actor!=scorer:return False
        return True
    if ev is not None and ev in byid and compatible(byid[ev]):return byid[ev],'event_id'
    candidates=[p for ss in ((sec-1,sec,sec+1) if sec is not None else ()) for p in bytime.get(ss,[]) if compatible(p)]
    if len(candidates)==1:return candidates[0],'second_type_player'
    return None,'UNMATCHED_EVENT'


def onice_from_shot(shot,stint,player_index,home,away):
    sides={}
    for side,team in (('home',home),('away',away)):
        ids={int(shot[k]) for i in range(1,8) for k in (f'{side}_on_{i}_id',)
             if k in shot and shot[k] is not None and str(shot[k]) not in ('','0')}
        # Exclude goaltenders and ghost/stale IDs using the audited game-team player index.
        ids={i for i in ids if (i,team) in player_index}
        if len(ids)<3 and stint:
            ss={int(x) for x in (stint.get(f'{side}_ids') or []) if (int(x),team) in player_index}
            if len(ss)>=3:ids=ss
        sides[side]=ids
    return sides


def select_stint(stints,starts,t):
    if not stints:return None
    i=bisect.bisect_right(starts,t)-1
    if i>=0 and int(stints[i]['start_s'])<=t<int(stints[i]['end_s']):return stints[i]
    if i>0 and int(stints[i-1]['start_s'])<t<=int(stints[i-1]['end_s']):return stints[i-1]
    return None


def create_model(stints, valid_games=None):
    """Build raw two-sided offense/defense design, without using prior EV player rates."""
    import numpy as np
    from scipy import sparse
    ids=set();time=collections.defaultdict(float);minutes_by_game=collections.defaultdict(float)
    work=[];bad_lineups=[];observed_games=set()
    for rec in stints:
        if str(rec['strength_state'])!='5v5':continue
        gid=int(rec['game_id']);sec=number(rec['duration'])
        require(sec>=0,f'Negative stint in game {gid}')
        if sec==0:continue
        if valid_games is not None:require(gid in valid_games,f'Unexpected stint game: {gid}')
        home=tuple(int(p) for p in (rec['home_ids'] or []));away=tuple(int(p) for p in (rec['away_ids'] or []))
        require(len(set(home))==len(home) and len(set(away))==len(away),'Repeated player in 5v5 stint')
        require(not (set(home)&set(away)),f'Player assigned both clubs in {gid}')
        require(len(home)==len(away)==5,
                f'Invalid 5v5 RAPM personnel in {gid}: home={len(home)} away={len(away)}')
        xh=number(rec['xgf_home']);xa=number(rec['xgf_away'])
        require(xh>=0 and xa>=0,f'Negative xG in game {gid}')
        observed_games.add(gid)
        for p in home:
            ids.add(p);time[p]+=sec;minutes_by_game[(gid,p,'home')]+=sec
        for p in away:
            ids.add(p);time[p]+=sec;minutes_by_game[(gid,p,'away')]+=sec
        work.append((gid,sec,home,away,xh,xa))
    require(len(work)>1000 and len(ids)>100,'Insufficient raw 5v5 game coverage')
    ids=sorted(ids);ix={p:i for i,p in enumerate(ids)};n=len(ids)
    row=[];col=[];data=[];target=[];weights=[];groups=[]
    for gid,sec,home,away,xh,xa in work:
        for attackers,defenders,xg,is_home in ((home,away,xh,1.),(away,home,xa,0.)):
            r=len(target)
            for p in attackers:row.append(r);col.append(ix[p]);data.append(1.)
            for p in defenders:row.append(r);col.append(ix[p]+n);data.append(1.)
            row.extend((r,r));col.extend((2*n,2*n+1));data.extend((is_home,1.))
            target.append(xg*3600./sec)
            weights.append(sec);groups.append(gid)
    X=sparse.coo_matrix((np.asarray(data,dtype=float),(row,col)),shape=(len(target),2*n+2)).tocsr()
    return dict(X=X,y=np.asarray(target),weights=np.asarray(weights),games=np.asarray(groups),
                players=ids,toi=time,game_toi=minutes_by_game,game_set=observed_games,
                bad_lineups=bad_lineups,stints=len(work))


def ridge(model,lam,keep=None):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    X=model['X']; y=model['y'];w=model['weights']
    if keep is not None:X=X[keep];y=y[keep];w=w[keep]
    A=(X.T@sparse.diags(w,format='csr')@X).tocsc()
    b=X.T@(w*y)
    m=sparse.eye(X.shape[1],format='csc')
    beta=np.asarray(spsolve(A+lam*m,b),dtype=float).ravel()
    require(len(beta)==X.shape[1] and np.isfinite(beta).all(),'Independent EV ridge solution is not finite')
    normal_err=float(np.max(np.abs(A@beta+lam*beta-b)))
    relative_err=normal_err/max(1.,float(np.max(np.abs(b))))
    require(relative_err<1e-7,f'RAPM linear system residual too high: {relative_err:.3g}')
    return beta,relative_err


def holdout_score(model,lam):
    import numpy as np
    games=sorted(int(g) for g in set(model['games']))
    test_games=set(g for i,g in enumerate(games) if i%5==0)
    is_test=np.isin(model['games'],list(test_games))
    require(int(is_test.sum())>=100 and int((~is_test).sum())>=500,'Insufficient independent game holdout')
    fit,fiterr=ridge(model,lam,~is_test)
    Y=model['y'][is_test];W=model['weights'][is_test];pred=model['X'][is_test]@fit
    # The baseline is learned solely on training games, not the game being predicted.
    trainmean=float(np.average(model['y'][~is_test],weights=model['weights'][~is_test]))
    modelrmse=math.sqrt(float(np.average((Y-pred)**2,weights=W)))
    nullrmse=math.sqrt(float(np.average((Y-trainmean)**2,weights=W)))
    return {'heldout_games':len(test_games),'train_games':len(games)-len(test_games),
            'heldout_observations':int(is_test.sum()),'ridge_heldout_weighted_rmse':modelrmse,
            'intercept_only_heldout_weighted_rmse':nullrmse,
            'ridge_vs_intercept_rmse_change':modelrmse-nullrmse,
            'training_normal_equations_relative_residual':fiterr}




def _retrieve(url: str, filename: str) -> str:
    # Cache official *completed-game* documents outside the repository.
    cache = Path.home() / '.cache' / 'nhl_war_official_v3'
    path = cache / filename
    if path.is_file() and path.stat().st_size >= 100:
        return path.read_text(encoding='utf-8-sig')
    exception = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent':'Mozilla/5.0 NHL WAR historical verification',
                'Accept':'application/json,text/html;q=0.9,*/*;q=0.1',
            })
            with urllib.request.urlopen(request, timeout=35) as response:
                body = response.read()
            require(len(body) >= 100, f'Empty NHL document: {url}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix+'.tmp')
            temp.write_bytes(body)
            temp.replace(path)
            return body.decode('utf-8-sig', errors='replace')
        except (OSError, TimeoutError, urllib.error.URLError, DataFailure) as exc:
            exception=exc
            if attempt < 3: time.sleep(min(6, 2**attempt))
    raise DataFailure(f'Official NHL record unavailable: {url}; {exception}')


def _json(url, filename):
    raw=_retrieve(url,filename)
    try: return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataFailure(f'Invalid official JSON {filename}: {exc}') from exc


def _fetch_game(gid: int):
    pbp = _json(f'https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play',
                f'play-by-play_{gid}.json')
    box = _json(f'https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore',
                f'boxscore_{gid}.json')
    home = str((pbp.get('homeTeam') or {}).get('abbrev') or '').upper()
    away = str((pbp.get('awayTeam') or {}).get('abbrev') or '').upper()
    require(home and away and home != away,
            f'Invalid official home/away teams in {gid}: {home}/{away}')
    year=int(str(gid)[:4]); season=f'{year}{year+1}'
    report = _retrieve(f'https://www.nhl.com/scores/htmlreports/{season}/GS{gid%1000000:06d}.HTM',
                       f'GS{gid}.html')
    gs=parse_gs_html(report)
    stats=_json('https://api.nhle.com/stats/rest/en/skater/timeonice?'+urllib.parse.urlencode({
        'isAggregate':'false','isGame':'true','start':0,'limit':-1,
        'cayenneExp':f'gameId={gid}'}), f'skater_time_{gid}.json')
    player_time={}
    for r in stats.get('data') or []:
        if int(r.get('gameId',gid))!=gid:continue
        pid=asint(r.get('playerId'))
        if pid is None:continue
        pp=clock_seconds(r.get('ppTimeOnIce',r.get('powerPlayTimeOnIce')))
        pk=clock_seconds(r.get('shTimeOnIce',r.get('shortHandedTimeOnIce')))
        require(pp is not None and pk is not None,
                f'Missing official PP/PK seconds {gid} player {pid}')
        team=r.get('teamAbbrev') or r.get('teamAbbreviation') or r.get('teamAbbrevs')
        if isinstance(team,dict):team=team.get('default')
        if isinstance(team,list):team=team[0] if len(team)==1 else None
        team=str(team or '').strip().upper()
        if team:require(team in (home,away),f'Wrong official TOI team: {gid}/{pid}/{team}')
        player_time[pid]={'pp':pp,'pk':pk,'team':team}
    require(len(player_time)>=30,f'NHL official player PP/PK time incomplete: {gid}: {len(player_time)}')
    roles={}; goalie_ids=set(); roster_teams={}
    for side,team in (('homeTeam',home),('awayTeam',away)):
        b=((box.get('playerByGameStats') or {}).get(side) or {})
        for category,role in (('forwards','F'),('defense','D'),('goalies','G')):
            for row in b.get(category) or []:
                pid=asint(row.get('playerId'))
                if pid is None:continue
                if pid in roster_teams:require(roster_teams[pid]==team,f'Conflicting player roster {gid}/{pid}')
                roster_teams[pid]=team
                roles[pid]=role
                if role=='G':goalie_ids.add(pid)
    require(len(set(player_time)-set(roster_teams))==0,
            f'NHL time/boxscore roster discrepancy game {gid}: {sorted(set(player_time)-set(roster_teams))[:12]}')
    shift_intervals,shift_source=get_official_shifts(
        gid,home,away,pbp,player_time,
        lambda url,name: _retrieve(url,f'{gid}_{name}'))
    return {'gid':gid,'home':home,'away':away,'gs':gs,'player_time':player_time,
            'roles':roles,'goalies':goalie_ids,'roster_teams':roster_teams,
            'shifts':shift_intervals,'shift_source':shift_source,
            'plays':official_plays(pbp),'pbp':pbp}


def _official_events(official):
    count=collections.defaultdict(collections.Counter)
    for event in official['pbp'].get('plays') or []:
        kind=event.get('typeDescKey')
        details=event.get('details') or {}
        if kind=='faceoff':
            for key,label in (('winningPlayerId','won'),('losingPlayerId','lost')):
                pid=asint(details.get(key))
                if pid is not None:count[pid][label]+=1
        if kind=='penalty':
            for key,label in (('committedByPlayerId','taken'),('drawnByPlayerId','drawn')):
                pid=asint(details.get(key))
                if pid is not None:count[pid][label]+=1
    return count



"""Conservative, production-score-preserving 5v5 RAPM source reconstruction.

Uses the already downloaded NHL official per-player shifts, official play-by-play
situation codes, and correctly period-ordered SDV source change events. Any
unsupported or contradictory exposure is discarded, never guessed.
"""

def _ev_ids(text):
    if not text:
        return set()
    if isinstance(text,(tuple,list,set)):
        return {int(p) for p in text if p is not None and str(p).strip().isdigit() and int(p)>0}
    return {int(p) for item in str(text).split(',') if (p:=item.strip()).isdigit() and int(p)>0}


def _ev_official_clock(period, stamp):
    if stamp is None or period is None:
        return None
    parts=str(stamp).split(':')
    if len(parts)!=2:
        return None
    try:
        minutes,seconds=int(parts[0]),int(parts[1])
        if not (0<=minutes<=20 and 0<=seconds<60):return None
        return (int(period)-1)*1200+minutes*60+seconds
    except (TypeError,ValueError):
        return None


def _ev_verified_timeline(shift_lists):
    # Official per-player shift intervals (not inferred SDV in/out changes).
    # Counters preserve overlapping/duplicated rows without falsely ending a
    # still-live player shift when one of two overlapping copies ends.
    edges=collections.defaultdict(lambda:[[],[]])
    for side in ('home','away'):
        for a,b,pid in shift_lists.get(side,[]):
            a,b,pid=int(a),int(b),int(pid)
            if 0<=a<b and b<=6000:
                edges[a][1].append((side,pid))
                edges[b][0].append((side,pid))
    if not edges:return [],[]
    active={'home':collections.Counter(),'away':collections.Counter()}
    boundaries=sorted(edges);segments=[]
    for a,b in zip(boundaries,boundaries[1:]):
        off,on=edges[a]
        for side,p in off:active[side][p]-=1
        for side,p in on:active[side][p]+=1
        for side in ('home','away'):
            for p,c in list(active[side].items()):
                if c<=0:del active[side][p]
        if a<b:
            segments.append((a,b,frozenset(active['home']),frozenset(active['away'])))
    return segments,[s[0] for s in segments]


def _ev_official_exact(segments,starts,start,end,home,away):
    if not segments or not start<end:return False
    idx=max(0,bisect.bisect_right(starts,start)-1)
    covered=0
    while idx<len(segments) and segments[idx][0]<end:
        a,b,h,v=segments[idx];length=max(0,min(b,end)-max(a,start))
        if length:
            if h!=home or v!=away:return False
            covered+=length
        idx+=1
    return covered==end-start


def _ev_name_abbrev(full_name):
    names={
    'Anaheim Ducks':'ANA','Arizona Coyotes':'ARI','Boston Bruins':'BOS','Buffalo Sabres':'BUF',
    'Calgary Flames':'CGY','Carolina Hurricanes':'CAR','Chicago Blackhawks':'CHI',
    'Colorado Avalanche':'COL','Columbus Blue Jackets':'CBJ','Dallas Stars':'DAL',
    'Detroit Red Wings':'DET','Edmonton Oilers':'EDM','Florida Panthers':'FLA',
    'Los Angeles Kings':'LAK','Minnesota Wild':'MIN','Montreal Canadiens':'MTL',
    'Montréal Canadiens':'MTL','MontrÃ©al Canadiens':'MTL','Nashville Predators':'NSH',
    'New Jersey Devils':'NJD','New York Islanders':'NYI','New York Rangers':'NYR',
    'Ottawa Senators':'OTT','Philadelphia Flyers':'PHI','Pittsburgh Penguins':'PIT',
    'San Jose Sharks':'SJS','Seattle Kraken':'SEA','St. Louis Blues':'STL',
    'Tampa Bay Lightning':'TBL','Toronto Maple Leafs':'TOR',
    'Utah Hockey Club':'UTA','Utah Mammoth':'UTA',
    'Vancouver Canucks':'VAN','Vegas Golden Knights':'VGK','Washington Capitals':'WSH',
    'Winnipeg Jets':'WPG'
    }
    name=str(full_name or '').strip()
    return names.get(name,name if name in CLUBS or name=='ARI' else None)


def _ev_shift_side(team, ids_on, ids_off, home, away, roster_teams):
    """Resolve a source CHANGE only from its team label or official player roster.

    Never assign a missing-label change if any involved player is not on the
    official NHL game roster, or if players belong to both teams.
    """
    if team==home:return 'home',False
    if team==away:return 'away',False
    participants=ids_on | ids_off
    if not participants:return None,False
    clubs={roster_teams.get(int(pid)) for pid in participants}
    if len(clubs)!=1:return None,False
    club=next(iter(clubs))
    if club==home:return 'home',True
    if club==away:return 'away',True
    return None,False


def _build_verified_5v5_ev(prior_shifts, scored_pbp, official_records):
    """Independently rebuild only NHL-verified 5v5 stints, using scored xG.

    Replaces SDV build_stints *for RAPM exposure only*. PP/PK attribution
    continues to use the existing separately verified NHL shot-on-ice logic.
    Accepts Polars inputs but performs source-to-official joins in Python so
    same-second period order, invalid NHL states and corrupt CHANGE events
    can be checked exactly. No shift feed is modified.
    """
    import collections,bisect,math
    events=collections.defaultdict(list)
    shifts=collections.defaultdict(list)
    ignored_unassigned_nonchanges=0
    for idx,row in enumerate(prior_shifts.iter_rows(named=True)):
        gid=int(row['game_id'])
        sec=row.get('game_seconds');per=row.get('period')
        if sec is None or per is None:continue
        team=_ev_name_abbrev(row.get('event_team'))
        add,remove=_ev_ids(row.get('ids_on')),_ev_ids(row.get('ids_off'))
        # SDV includes non-personnel records with no team, including period
        # markers. No player enters or leaves: such a row cannot alter a
        # lineup and must not introduce an artificial stint boundary.
        # Unknown-team records with actual personnel changes still fail closed.
        if team is None and not add and not remove:
            ignored_unassigned_nonchanges+=1
            continue
        shifts[gid].append((int(sec),int(per),idx,team,add,remove,row.get('event_team')))
    if ignored_unassigned_nonchanges:
        print(f'WAR INERT UNASSIGNED SHIFT ROWS SKIPPED | rows={ignored_unassigned_nonchanges}',flush=True)
    for row in scored_pbp.iter_rows(named=True):
        gid=int(row['game_id']);sec=row.get('game_seconds')
        if sec is not None:events[gid].append((int(sec),row))
    require(set(shifts)==set(events)==set(official_records),
        f'5v5 official game coverage mismatch: shifts={len(shifts)}, PBP={len(events)}, NHL={len(official_records)}')
    accepted=[];rejected=[];counters=collections.Counter();scores=collections.Counter()
    for game_idx,gid in enumerate(sorted(official_records),1):
        if game_idx%100==0:print('VERIFY EV',game_idx,'/',len(official_records),flush=True)
        r=official_records[gid]
        home,away=r['home'],r['away']
        goalie_ids=set(map(int,r['goalies']))
        roster={side:{int(pid) for pid,t in r['roster_teams'].items() if t==(home if side=='home' else away) and pid not in goalie_ids}
                for side in ('home','away')}
        official_segments,official_starts=_ev_verified_timeline(r['shifts'])
        official_events=[]
        for entry in r['pbp'].get('plays') or []:
            descriptor=entry.get('periodDescriptor') or {}
            if str(descriptor.get('periodType') or '').upper()=='SO':continue
            sec=_ev_official_clock(descriptor.get('number') or entry.get('period'),entry.get('timeInPeriod'))
            if sec is not None:official_events.append((sec,str(entry.get('situationCode') or '')))
        official_events.sort(key=lambda x:x[0]);official_secs=[x[0] for x in official_events]
        e=sorted(events[gid],key=lambda x:x[0]);evsecs=[row[0] for row in e]
        # Important: period order *inside tied game_seconds* is verified from
        # 2024020584/95-second cross-check; do not rely on original parquet row order.
        shifts_game=sorted(shifts[gid],key=lambda x:(x[0],x[1],x[2]))
        running={'home':set(),'away':set()};positions={};change_secs=[]
        for sec,per,source_order,team,add,remove,raw_team in shifts_game:
            side,inferred=_ev_shift_side(team,add,remove,home,away,r['roster_teams'])
            if inferred:counters['source_team_inferred_from_official_roster']+=1
            require(side is not None,
                f'Unmapped SDV shift team game={gid} second={sec} raw_team={raw_team!r} '
                f'normalized_team={team!r} ids_on={sorted(add)} ids_off={sorted(remove)}')
            if not change_secs or change_secs[-1]!=sec:change_secs.append(sec)
            running[side].difference_update(remove)
            running[side].update(add)
            positions[sec]=(frozenset(running['home']-goalie_ids),frozenset(running['away']-goalie_ids))
        # Exclude the synthetic end-of-regulation +1 second entirely. NHL
        # regular-season 5v5 does not extend beyond the regulation clock.
        end=min(3600,max(evsecs,default=0)+1)
        boundaries=[s for s in change_secs if s<end]
        if not boundaries:continue
        if boundaries[-1]!=end:boundaries.append(end)
        for a,b in zip(boundaries,boundaries[1:]):
            if b<=a:continue
            start=bisect.bisect_left(evsecs,a);stop=bisect.bisect_left(evsecs,b)
            window=[row for sec,row in e[start:stop]]
            if not window:continue
            states=[str(x.get('strength_state')) for x in window if x.get('strength_state') is not None]
            if not states:continue
            # Match SDV 0.0.75 mode semantics; at a tied maximum choose the
            # lexicographically first strength, not first encountered event.
            state_counts=collections.Counter(states)
            max_state_count=max(state_counts.values())
            modal_state=min(k for k,n in state_counts.items() if n==max_state_count)
            if modal_state!='5v5':continue
            counters['source_modal_5v5_intervals']+=1
            ids_h,ids_a=positions[a]
            why=None
            if len(ids_h)!=5 or len(ids_a)!=5:why='source_wrong_skater_count'
            elif not ids_h<=roster['home'] or not ids_a<=roster['away']:why='skater_not_official_game_roster'
            elif any(s!='5v5' for s in states):why='source_mixed_strength'
            elif not _ev_official_exact(official_segments,official_starts,a,b,ids_h,ids_a):why='official_shift_personnel_disagreement'
            else:
                lo=bisect.bisect_left(official_secs,a);hi=bisect.bisect_left(official_secs,b)
                codes=[code for _,code in official_events[lo:hi] if code]
                if not codes:why='missing_official_pbp_situation'
                elif any(code!='1551' for code in codes):why='official_pbp_non5v5_situation'
            xh=math.fsum(float(x.get('xg') or 0.) for x in window if x.get('event_team_abbr')==home)
            xa=math.fsum(float(x.get('xg') or 0.) for x in window if x.get('event_team_abbr')==away)
            require(math.isfinite(xh) and math.isfinite(xa) and xh>=0 and xa>=0,
                f'Invalid scored PBP xG game {gid} seconds {a}-{b}: {xh}/{xa}')
            scores['source_5v5_xg']+=xh+xa
            if why:
                counters['excluded_'+why]+=1;scores['excluded_scored_xg']+=xh+xa
                rejected.append({'game_id':gid,'start_s':a,'end_s':b,'home_count':len(ids_h),'away_count':len(ids_a),'cause':why,'scored_xg':xh+xa})
                continue
            accepted.append({'game_id':gid,'strength_state':'5v5','start_s':a,'end_s':b,'duration':b-a,
                'home_ids':sorted(ids_h),'away_ids':sorted(ids_a),'xgf_home':xh,'xgf_away':xa})
            counters['admitted_verified_5v5_intervals']+=1;scores['admitted_scored_xg']+=xh+xa
    require(accepted,'No NHL-verified 5v5 RAPM intervals')
    require(all(len(set(r['home_ids']))==len(set(r['away_ids']))==5 for r in accepted),
        'NHL-verified 5v5 reconstruction leaked invalid player counts')
    require({int(r['game_id']) for r in accepted}==set(official_records),
        'NHL-verified 5v5 reconstruction lost game coverage')
    print('WAR VERIFIED 5v5 SOURCE | '+str(dict(counters))+' | scores='+str(dict(scores)),flush=True)
    return accepted,dict(counters),dict(scores),rejected

def _valid_stints_with_goalies_removed(stints,goalies_by_game):
    """Return 5v5+all-state personnel stints; keep source xG unchanged."""
    result=[]
    for rec in stints.iter_rows(named=True):
        gid=int(rec['game_id'])
        require(gid in goalies_by_game, f'Missing official NHL goalie IDs game={gid}')
        g=goalies_by_game[gid]
        row=dict(rec)
        for side in ('home','away'):
            row[f'{side}_ids']=[int(p) for p in (rec.get(f'{side}_ids') or []) if int(p) not in g]
        result.append(row)
    return result




def _write_5v5_lineup_error_log(stints, model):
    """Report whether invalid positive-duration 5v5 lineups entered the RAPM design."""
    from collections import Counter

    log_path = Path(__file__).resolve().parents[2] / 'errors' / '00_intake' / 'nhl_war_production.txt'
    bad = []
    for rec in stints:
        if str(rec['strength_state']) != '5v5' or float(rec['duration']) <= 0:
            continue
        home = rec['home_ids'] or []
        away = rec['away_ids'] or []
        if len(home) != 5 or len(away) != 5:
            bad.append(rec)

    by_game = collections.defaultdict(list)
    patterns = Counter()
    for rec in bad:
        by_game[int(rec['game_id'])].append(rec)
        patterns[(len(rec['home_ids'] or []), len(rec['away_ids'] or []))] += 1
    bad_seconds = math.fsum(float(rec['duration']) for rec in bad)
    lines = [
        'NHL WAR | HISTORICAL 5v5 LINEUP AUDIT',
        f'Invalid 5v5 lineups entered RAPM design: {"YES" if bad else "NO"}',
        f'5v5 stints entering RAPM: {model["stints"]}',
        f'Invalid 5v5 stints entering RAPM: {len(model["bad_lineups"])}',
        f'Invalid RAPM observations: {2 * len(model["bad_lineups"])}',
        f'Affected games: {len(by_game)}',
        f'Invalid stint duration (seconds): {bad_seconds:.3f}',
        'Skater counts (home/away): ' + (
            ', '.join(f'{home}/{away}={count}' for (home, away), count in sorted(patterns.items()))
            if patterns else 'none'),
    ]
    if len(bad) != len(model['bad_lineups']):
        lines.append(f'WARNING: source records matched={len(bad)}; model flagged={len(model["bad_lineups"])}')
    ranked = sorted(by_game.items(), key=lambda item: (-math.fsum(float(r['duration']) for r in item[1]), item[0]))
    for game_id, rows in ranked[:20]:
        seconds = math.fsum(float(r['duration']) for r in rows)
        lines.append(f'GAME {game_id}: invalid_stints={len(rows)} duration_s={seconds:.3f}')
        for rec in sorted(rows, key=lambda r: (-float(r['duration']), float(r.get('start_s') or 0)))[:2]:
            home, away = rec['home_ids'] or [], rec['away_ids'] or []
            lines.append(
                f'  start_s={rec.get("start_s")} end_s={rec.get("end_s")}'
                f' duration_s={rec["duration"]} home_count={len(home)} away_count={len(away)}'
                f' home_ids={list(home)} away_ids={list(away)}'
                f' xgf_home={rec.get("xgf_home")} xgf_away={rec.get("xgf_away")}'
            )
    if len(ranked) > 20:
        lines.append(f'Additional affected games not detailed: {len(ranked) - 20}')
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    except OSError as exc:
        print(f'WAR 5v5 audit log could not be written: {log_path}: {exc}', file=sys.stderr)

def _model_and_player_times(records):
    """Rebuild two-sided RAPM independently, preserving per-game team attribution."""
    model=create_model(records)
    _write_5v5_lineup_error_log(records, model)
    coefficients,residual=ridge(model,RIDGE_LAMBDA)
    n=len(model['players'])
    rates={pid:(float(coefficients[i]),float(-coefficients[n+i]))
           for i,pid in enumerate(model['players'])}
    require(residual<1e-7,f'Independent ridge solver residual={residual}')
    return model,rates,residual



def team_rapm_from_verified_5v5(model, rates, game_teams):
    """Team TOI-weighted adjusted net xG/60 from *only* verified 5v5 stints.

    Rates are (offensive xGF/60 coefficient, defensive xGA suppression/60)
    from the independent WAR ridge model. Weight each player's net rate by
    exactly the verified seconds he played for the corresponding game/team;
    this prevents traded skaters from being attributed to the wrong club.
    """
    duration_by_team = collections.defaultdict(list)
    weighted_by_team = collections.defaultdict(list)
    for (game_id, player_id, side), raw_seconds in model['game_toi'].items():
        game_id, player_id = int(game_id), int(player_id)
        require(game_id in game_teams,
                f'RAPM game absent from verified source {game_id}')
        home, away = game_teams[game_id]
        require(side in ('home', 'away'), f'Invalid RAPM side {game_id}/{side}')
        club = home if side == 'home' else away
        require(player_id in rates, f'RAPM missing model player {game_id}/{player_id}')
        seconds = number(raw_seconds)
        require(seconds > 0, f'Invalid verified RAPM exposure {game_id}/{player_id}: {seconds}')
        offense, defense = map(number, rates[player_id])
        duration_by_team[club].append(seconds)
        weighted_by_team[club].append((offense + defense) * seconds)
    require(set(duration_by_team) == set(sum(([h, a] for h, a in game_teams.values()), [])),
            'RAPM validated 5v5 exposure does not cover every club')
    result = {}
    for club in sorted(duration_by_team):
        seconds = math.fsum(duration_by_team[club])
        require(seconds > 0, f'No verified 5v5 RAPM seconds for {club}')
        value = math.fsum(weighted_by_team[club]) / seconds
        require(math.isfinite(value), f'Non-finite verified 5v5 RAPM for {club}')
        result[club] = value
    print(f'WAR VERIFIED TEAM RAPM | clubs={len(result)} '
          f'verified_skater_seconds={math.fsum(math.fsum(v) for v in duration_by_team.values()):.0f}',
          flush=True)
    return result


def compute_all_32_war(prior_pbp, prior_shifts):
    """Produce WAR from prior games, independent of all Downloads audit artifacts.

    Returns (team WAR map, details). Missing official sources or player/team
    reconciliation causes a hard failure before any production CSV is published.
    """
    import polars as pl
    from sportsdataverse.nhl.nhl_xg import nhl_xg
    from sportsdataverse.nhl.nhl_rapm import build_stints
    require(isinstance(prior_pbp,pl.DataFrame) and isinstance(prior_shifts,pl.DataFrame),
            'WAR requires pregame-filtered Polars PBP and shifts')
    require({'game_id','game_date','home_abbr','away_abbr'}.issubset(prior_pbp.columns),
            'WAR PBP missing game/team/date columns')
    require({'game_id','game_date'}.issubset(prior_shifts.columns),
            'WAR shifts missing game/date columns')
    source_pair={}
    for rec in prior_pbp.select('game_id','home_abbr','away_abbr').unique().iter_rows(named=True):
        gid=int(rec['game_id'])
        home,away=str(rec['home_abbr']),str(rec['away_abbr'])
        if gid in source_pair:require(source_pair[gid]==(home,away),f'PBP game team mismatch {gid}')
        source_pair[gid]=(home,away)
    ids=set(int(x) for x in prior_shifts.get_column('game_id').drop_nulls().unique().to_list())
    require(set(source_pair)==ids,f'PBP/shift game coverage mismatch: PBP={len(source_pair)} shifts={len(ids)} gaps={len(set(source_pair)^ids)}')
    require(source_pair,'No prior games supplied to WAR')
    clubs=set(sum(([h,a] for h,a in source_pair.values()),[]))
    years={int(str(gid)[:4]) for gid in source_pair}
    require(len(years)==1,f'WAR mixed NHL seasons not allowed: {sorted(years)}')
    expected_clubs = CLUBS if next(iter(years)) >= 2024 else ((CLUBS - {'UTA'}) | {'ARI'})
    require(clubs==expected_clubs,
            f'WAR team coverage is not all 32: missing={sorted(expected_clubs-clubs)}, unexpected={sorted(clubs-expected_clubs)}')
    print(f'WAR OFFICIAL REBUILD | games={len(ids)} teams={len(clubs)}',flush=True)
    records={};problems=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(_fetch_game,gid):gid for gid in sorted(ids)}
        for count,future in enumerate(as_completed(futures),1):
            gid=futures[future]
            try:
                official=future.result()
                require((official['home'],official['away'])==source_pair[gid],
                        f'Official vs SDV team conflict game {gid}')
                records[gid]=official
            except Exception as exc:problems.append(f'{gid}: {exc}')
            if count%40==0:print(f'WAR OFFICIAL RECORDS | {count}/{len(ids)}',flush=True)
    require(not problems,f'Official NHL WAR sources failed for {len(problems)} games; sample={problems[:6]}')
    goalie_ids={g:r['goalies'] for g,r in records.items()}
    scored=nhl_xg(prior_pbp,league='nhl')
    require(scored.height==prior_pbp.height,'NHL xG source changed PBP event count')
    stints=build_stints(prior_shifts,scored)
    require({'game_id','strength_state','home_ids','away_ids','duration','xgf_home','xgf_away','start_s','end_s'}.issubset(stints.columns),
            'NHL shift-stint source schema differs from audited WAR model')
    original=_valid_stints_with_goalies_removed(stints,goalie_ids)
    # NHL official shifts are required to agree with every skater ID for the
    # full interval. No unverified or physically impossible 5v5 time enters RAPM.
    # Same-second SDV shift changes are period-ordered in this independent
    # reconstruction; the upstream SDV build_stints remains used for verified
    # PP/PK event fallback, not for 5v5 modeling.
    five,ev_reject_counts,ev_scored_totals,ev_examples = _build_verified_5v5_ev(prior_shifts,scored,records)
    model,rates,normal_residual=_model_and_player_times(five)
    require(model['game_set']==ids,'Five-on-five source does not cover every historical game')
    # Use the *same* NHL-confirmed 5v5 design and ridge coefficients for both
    # WAR and team skater_rapm; never rerun the unvalidated SDV RAPM path.
    verified_team_rapm=team_rapm_from_verified_5v5(model,rates,source_pair)
    require(set(verified_team_rapm)==expected_clubs,
            'WAR verified RAPM does not contain the expected 32 clubs')
    side_toi={}
    for (gid,pid,side),seconds in model['game_toi'].items():
        club=source_pair[gid][0 if side=='home' else 1]
        key=(gid,pid,club)
        require(key not in side_toi,f'Duplicate skater on both teams game {gid} player {pid}')
        side_toi[key]=seconds
    all_players={}
    for gid,record in records.items():
        h,a=record['home'],record['away']
        for pid,p in record['player_time'].items():
            team=record['roster_teams'][pid]
            require(not p['team'] or p['team']==team,
                    f'Wrong player/team PP time game={gid}, player={pid}')
            require(record['roles'].get(pid) in ('F','D'),
                    f'Skater has no verified forward/defense position {gid}/{pid}')
            all_players[(gid,pid,team)]={'pp':p['pp'],'pk':p['pk'],
                'role':record['roles'][pid]}
    require(all(k in all_players for k in side_toi),
            f'Five-on-five players absent from NHL official player games: {len(set(side_toi)-set(all_players))}')
    # Use official GS team PP clock, not the shift-code clock that differs in
    # 131 official team-games in the original 282-game audit.
    total_pp_seconds=sum(item['gs'][0]['seconds']+item['gs'][1]['seconds'] for item in records.values())
    require(total_pp_seconds>0,'Official league PP time is zero')
    xgf=collections.defaultdict(float);xga=collections.defaultdict(float)
    verified_shots=0;attacking_xg=0.0;unassigned=[]
    # Speed: filter PBP once by game, rather than scanning entire table for every game.
    pbp_by_game=collections.defaultdict(list)
    for shot in scored.iter_rows(named=True):pbp_by_game[int(shot['game_id'])].append(shot)
    stints_by_game=collections.defaultdict(list)
    for stint in original:stints_by_game[int(stint['game_id'])].append(stint)
    for gid,official in sorted(records.items()):
        h,a=official['home'],official['away']
        plays=official['plays']
        byid={r['event_id']:r for r in plays if r['event_id'] is not None}
        bytime=collections.defaultdict(list)
        for event in plays:bytime[event['second']].append(event)
        game_index={(p,t) for g,p,t in all_players if g==gid}
        raw=sorted(stints_by_game[gid],key=lambda r:(int(r['start_s']),int(r['end_s'])))
        starts=[int(s['start_s']) for s in raw]
        matched_ids=set()
        for shot in pbp_by_game[gid]:
            x=float(shot.get('xg') or 0.0)
            if x<=0:continue
            require(math.isfinite(x) and x<=1.01,f'Invalid NHL xG {gid}: {x}')
            event,match=locate_shot_event(gid,shot,plays,byid,bytime)
            sec=int(shot['game_seconds'])
            if event is not None and event['event_id'] in matched_ids:
                event=None
            if event is not None:
                pp_side=event['strength']
                code=str(event.get('raw_code') or '')
                expected={'away':int(code[1]),'home':int(code[2])} if re.fullmatch(r'[01][3-6][3-6][01]',code) else None
            else:
                hc=asint(shot.get('home_skaters'));ac=asint(shot.get('away_skaters'))
                hi=asint(shot.get('home_goalie_in'));ai=asint(shot.get('away_goalie_in'))
                expected={'home':hc,'away':ac} if hc in (3,4,5) and ac in (3,4,5) and hc!=ac and hi!=0 and ai!=0 else None
                pp_side=('home' if hc>ac else 'away') if expected else None
            if pp_side not in ('home','away'):
                if event is None and expected is None:
                    intervals=official['shifts']
                    for t in (sec-1,sec,sec+1):
                        hs={p for st,en,p in intervals['home'] if st<=t<en}
                        vs={p for st,en,p in intervals['away'] if st<=t<en}
                        if 3<=len(hs)<=5 and 3<=len(vs)<=5 and len(hs)!=len(vs):
                            pp_side='home' if len(hs)>len(vs) else 'away'
                            expected={'home':len(hs),'away':len(vs)}
                            break
                if pp_side not in ('home','away'):continue
            pp_team=h if pp_side=='home' else a
            pk_team=a if pp_side=='home' else h
            if str(shot.get('event_team_abbr') or '').upper()!=pp_team:continue
            stint=select_stint(raw,starts,sec)
            line,source=independent_lineup(shot,stint,official['shifts'],sec,h,a,
                                          game_index,all_players,gid,pp_side,expected)
            if line is None:
                unassigned.append((gid,sec,shot.get('event_id'),pp_team))
                continue
            ids_pp=line[pp_side]
            ids_pk=line['away' if pp_side=='home' else 'home']
            attacking_xg+=x;verified_shots+=1
            if event is not None and event['event_id'] is not None:
                matched_ids.add(event['event_id'])
            for pid in ids_pp:xgf[(gid,pid,pp_team)]+=x
            for pid in ids_pk:xga[(gid,pid,pk_team)]+=x
    require(not unassigned,
            f'Unverified official PP shot-on-ice assignments: {len(unassigned)}; first={unassigned[:8]}')
    require(verified_shots>0 and attacking_xg>0,'No verified NHL PP shots')
    rate=attacking_xg*3600.0/total_pp_seconds
    require(0.5<rate<20.,f'Invalid league PP xG/60 benchmark: {rate}')
    official_event_counts={gid:_official_events(rec) for gid,rec in records.items()}
    output=[]
    for key,rec in sorted(all_players.items()):
        gid,pid,team=key
        seconds=float(side_toi.get(key,0.0))
        role=rec['role']
        if seconds>0:require(pid in rates,f'Missing independent 5v5 RAPM for skater {gid}/{pid}')
        off,defense=rates.get(pid,(0.0,0.0))
        replacement=REPLACEMENT[role]
        ev_off=(off-replacement['off'])*seconds/3600.0
        ev_def=(defense-replacement['def'])*seconds/3600.0
        pp=xgf[key]-rate*rec['pp']/3600.0
        pk=rate*rec['pk']/3600.0-xga[key]
        events=official_event_counts[gid].get(pid,collections.Counter())
        pens=(events['drawn']-events['taken'])*.18
        faceoffs=(events['won']-events['lost'])*.01
        components=(ev_off,ev_def,pp,pk,pens,faceoffs)
        gar=math.fsum(components)
        output.append({'game_id':gid,'player_id':pid,'team':team,'position':role,
          'toi_5v5_minutes':seconds/60.0,'ev_off':ev_off,'ev_def':ev_def,
          'pp':pp,'pk':pk,'pens':pens,'faceoffs':faceoffs,
          'gar':gar,'war':gar/GPW})
    # Preserve player-game attribution (trades) instead of spreading season WAR
    # back to teams using one global player-to-team fraction.
    team_parts=collections.defaultdict(lambda:collections.defaultdict(list))
    player_parts=collections.defaultdict(lambda:collections.defaultdict(list))
    for rec in output:
        for part in PARTS:
            team_parts[rec['team']][part].append(rec[part])
            player_parts[rec['player_id']][part].append(rec[part])
    team_war={}
    for team,parts in team_parts.items():
        team_war[team]=math.fsum(math.fsum(parts[p]) for p in PARTS)/GPW
    require(set(team_war)==expected_clubs,
            f'WAR has not produced all 32 teams: {sorted(expected_clubs-set(team_war))}')
    for part in PARTS:
        delta=math.fsum(math.fsum(v[part]) for v in team_parts.values())-math.fsum(math.fsum(v[part]) for v in player_parts.values())
        require(abs(delta)<1e-7,f'All-32 WAR component reconciliation failed: {part}, {delta}')
    war_total=math.fsum(team_war.values())
    require(abs(war_total-math.fsum(x['war'] for x in output))<1e-7,'League WAR totals not conserved')
    print(f'WAR VERIFIED | games={len(records)} clubs={len(team_war)} skaters={len(player_parts)} '
          f'player_games={len(output)} verified_PP_shots={verified_shots} '
          f'EV_design_rows={model["X"].shape[0]} ridge_residual={normal_residual:.3g}',flush=True)
    for team in sorted(team_war):
        print(f'WAR TEAM | {team} | WAR={team_war[team]:+.8f}',flush=True)
    return team_war, {'teams':len(team_war),'games':len(records),'players':len(player_parts),
       'verified_team_rapm':verified_team_rapm,
       'player_games':len(output),'verified_special_teams_shots':verified_shots,
       'five_on_five_observations':int(model['X'].shape[0]),
       'ridge_normal_equation_residual':normal_residual,
       'total_war':war_total}


def reconstruct_official_shift_timeline(charts, plays):
    """Rebuild personnel-specific intervals from NHL raw shift start/end markers.

    Counts exclude goalies because the authoritative player universe is NHL
    /skater/timeonice, rather than an unverified SDV change set.
    """
    edges={0};movements=collections.defaultdict(list)
    for side in ('home','away'):
        for a,b,pid in charts[side]:
            edges.update((a,b))
            movements[a].append((side,pid,True))
            movements[b].append((side,pid,False))
    # Retain source-event boundaries as separate inspection intervals.
    for p in plays:edges.add(p['second'])
    active={'home':set(),'away':set()};out=[]
    times=sorted(edges)
    for a,b in zip(times,times[1:]):
        for side,pid,present in sorted(movements.get(a,[]),key=lambda v:v[2]):
            if present:active[side].add(pid)
            else:active[side].discard(pid)
        home,away=len(active['home']),len(active['away'])
        side='home' if 3<=away<home<=5 else 'away' if 3<=home<away<=5 else None
        if b>a:out.append({'start_s':a,'end_s':b,'duration':b-a,
            'home_skaters':home,'away_skaters':away,'pp_side':side or '',
            'source':'NHL_individual_shift_start_end'})
    return out

def selftest():
    # Test verified shift-time weighting (not raw SDV shift-time weighting),
    # including a player who changes teams between games.
    fake_model={'game_toi':{
        (1,10,'home'):60.,(1,11,'home'):120.,(1,12,'away'):90.,
        (2,10,'away'):180.,(2,13,'home'):60.,
    }}
    fake_rates={10:(0.6,0.2),11:(-0.1,-0.3),12:(0.1,0.0),13:(0.5,-0.1)}
    fake_teams={1:('AAA','BBB'),2:('BBB','AAA')}
    fake_result=team_rapm_from_verified_5v5(fake_model,fake_rates,fake_teams)
    require(set(fake_result)=={'AAA','BBB'},'Verified RAPM incomplete team mapping')
    require(abs(fake_result['AAA'] - ((0.8*60.-0.4*120.+0.8*180.)/360.))<1e-12,
            'Verified RAPM traded-skater and TOI weighting')
    require(abs(fake_result['BBB'] - ((0.1*90.+0.4*60.)/150.))<1e-12,
            'Verified RAPM team and defensive sign')
    require(_ev_name_abbrev('Utah Mammoth')=='UTA', '2026 Utah Mammoth source team mapping')
    require(_ev_name_abbrev('Utah Hockey Club')=='UTA', '2024 Utah historical source mapping')
    require(_ev_name_abbrev('UTA')=='UTA', 'Utah NHL abbreviation mapping')
    require(_ev_name_abbrev('Chicago Blackhawks')=='CHI', 'NHL opponent team mapping')
    require(_ev_name_abbrev('Unknown Team') is None, 'Unrecognized team must fail closed')
    roster={1:'UTA',2:'UTA',3:'CHI'}
    require(_ev_shift_side('UTA',{1},set(),'UTA','CHI',roster)==('home',False),
        'Known source team must not be inferred')
    require(_ev_shift_side(None,{1},{2},'UTA','CHI',roster)==('home',True),
        'Missing source team must use single-team official roster')
    require(_ev_shift_side(None,{1,3},set(),'UTA','CHI',roster)==(None,False),
        'Ambiguous two-team change must fail closed')
    require(_ev_shift_side(None,{999},set(),'UTA','CHI',roster)==(None,False),
        'Player outside NHL roster must not be inferred')
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    X=sparse.csr_matrix([[1.,0.,1.],[0.,1.,1.],[1.,1.,1.],[2.,-1.,1.],[1.,2.,1.]])
    w=np.asarray([5.,3.,7.,4.,2.]); y=np.asarray([1.5,2.5,4.,-1.,5.])
    fit,res=ridge({'X':X,'y':y,'weights':w},3.5)
    alternate=lsqr(sparse.vstack((sparse.diags(np.sqrt(w))@X,
                  math.sqrt(3.5)*sparse.eye(3))),
                  np.concatenate((np.sqrt(w)*y,np.zeros(3))),atol=1e-12,btol=1e-12)[0]
    require(max(abs(fit-alternate))<1e-8,'Independent EV solvers disagree')
    require(parse_situation('1451')=='home' and parse_situation('1541')=='away','NHL PP situation decoding')
    require(clock_seconds('05:32')==332 and clock_seconds(0)==0,'NHL official time parser')
    testcharts={'home':[(0,60,x) for x in range(1,6)]+[(60,120,x) for x in range(1,6)],
                'away':[(0,60,x) for x in range(11,16)]+[(60,120,x) for x in range(11,15)]}
    assert sum(r['duration'] for r in reconstruct_official_shift_timeline(testcharts,[]) if r['pp_side']=='home')==60
    assert len(REPLACEMENT)==2 and all(math.isfinite(v) for x in REPLACEMENT.values() for v in x.values())
    print('PASS | independent 5v5 ridge, NHL official clock, manpower situations and WAR calibration')

if __name__ == '__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--selftest':
        selftest()
    else:
        raise SystemExit('Usage: python nhl_war_production.py --selftest')
