"""Test revised event backgrounds; keep previous selections and metrics unchanged."""
from pathlib import Path
import base64
import gzip
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calculate_reviewed_distribution_20261013 import ROOT, cases, pairs, read_wave

OUT = ROOT / 'RAPID/outputs/revised_background_review_20261013'
OUT.mkdir(exist_ok=True)
DEFAULTS = {'02':'held_ramp','03':'model_minimum','05':'held_ramp','13':'held_ramp','37':'held_ramp','22':'lowered_end','45':'held_ramp'}
SELECTION_STATUS = {'02':'Accepted model crossing, horizontal start and ramp',
    '03':'Accepted model minimum','05':'Accepted horizontal start and ramp','13':'Accepted horizontal start and ramp',
    '15':'Accepted minimum start','37':'Accepted horizontal start and ramp','46':'Accepted lowered end',
    '22':'Slightly lower end with short horizontal finish: candidate for review',
    '45':'Accepted minimum start, ramp and horizontal finish'}
source = ROOT / 'RAPID/outputs/calibration_review/15_network_plateau_review.html'
original = source.read_text()
payload = json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>', original, re.S)[1])))


def iso(t):
    return pd.Timestamp(t, unit='s', tz='UTC').isoformat()


def check(t, q, knots):
    times, levels = np.array(knots).T
    tt = np.unique(np.r_[times, t[(t > times[0]) & (t < times[-1])]])
    qq = np.interp(tt, t, q)
    bg = np.interp(tt, times, levels)
    excess = qq - bg
    changed = np.flatnonzero(excess[:-1] * excess[1:] < 0)
    zeros = tt[changed] - excess[changed] * np.diff(tt)[changed] / np.diff(excess)[changed]
    zt = np.sort(np.r_[tt, zeros])
    ze = np.interp(zt, tt, excess)
    positive = float(np.trapz(np.maximum(ze, 0), zt))
    deficit = max(0., float(-np.trapz(np.minimum(ze, 0), zt)))
    minimum = int(excess.argmin())
    return dict(base_pass=bool(excess.min() >= -1e-8), minimum_excess_cms=float(excess.min()),
        minimum_excess_utc=iso(tt[minimum]), negative_excess_volume_m3=deficit,
        deficit_to_positive_volume_pct=100*deficit/positive,
        start_utc=iso(times[0]), end_utc=iso(times[-1]), start_cms=levels[0], end_cms=levels[-1],
        endpoint_excess_cms=float(excess[-1])), (tt, qq, bg)


def rising_crossing(t, q, level, after):
    ids = np.flatnonzero((t[1:] >= after) & (t[1:] <= t[q.argmax()]) & (q[:-1] < level) & (q[1:] >= level)) + 1
    if not len(ids):
        raise ValueError('No pre-peak rising crossing')
    j = ids[0]
    return t[j-1] + (t[j]-t[j-1]) * (level-q[j-1])/(q[j]-q[j-1])


def held_ramp(t, q, knots):
    """One extra knot, mathematically fitted to the base only; review, not adoption."""
    (a, q0), (b, qf) = knots
    use = (t > a) & (t < b)
    tt, qq = t[use], q[use]
    if qf > q0:
        eligible = qq < qf - 1e-9
        knee = max(a, float(np.max(((qf-q0)*tt[eligible]-(qq[eligible]-q0)*b)/(qf-qq[eligible]))))
        # A one-second buffer avoids floating-point overshoot at the tangency.
        knee = min(b-1, knee+1)
        return [(a,q0),(knee,q0),(b,qf)], 'Hold the starting level, then rise linearly'
    eligible = qq < q0 - 1e-9
    knee = min(b, float(np.min(a+(qf-q0)*(tt[eligible]-a)/(qq[eligible]-q0))))
    knee = max(a+1, knee-1)
    return [(a,q0),(knee,qf),(b,qf)], 'Reach the end level earlier, then hold it'


def main():
    rows, definitions, provenance = [], {}, {}
    fig, axes = plt.subplots(5,3,figsize=(15,17))
    for panel, ax in zip(payload['panels'], axes.flat):
        ex = panel['example']
        group = pairs[pairs.example_id.eq(ex)]
        record = group.iloc[0]
        t,q = read_wave(record.parent_outlet_parquet)
        provenance[ex] = dict(outlet=record.parent_outlet_parquet,
            sha256=hashlib.sha256(Path(record.parent_outlet_parquet).read_bytes()).hexdigest())
        c = cases[ex]
        reference = next(v for v in c['candidates'] if v['key']==c['default_candidate'])
        selected = next(v for v in reference['levels'] if v['raise_pct']==c['default_raise_pct'])
        a = pd.Timestamp(c['start']['start_utc']).timestamp()
        b = pd.Timestamp(selected['crossing_utc']).timestamp()
        q0,qf = c['start']['start_cms'], selected['plateau_cms']
        old = [(a,q0),(b,qf)]
        variants = []
        def add(key,label,knots,note):
            variants.append(dict(key=key,label=label,knots=knots,note=note))
        if ex == '02':
            lo,hi = [pd.Timestamp(s).timestamp() for s in ['2025-08-04T07:00Z','2025-08-04T20:00Z']]
            ix = np.flatnonzero((t>=lo)&(t<=hi));j=ix[q[ix].argmin()]
            b,qf = float(t[j]),float(q[j])
            forcing = pd.read_parquet(record.forcing_parquet)
            ft = pd.to_datetime(forcing.time_utc).astype('int64').to_numpy()/1e9
            q0 = float(np.interp(a,ft,forcing.forcing_cms))
            add('proposed','Input level at original July 31 02:00 start',[(a,q0),(b,qf)],
                'Literal same-time interpretation: input is 947.174 m³/s but model is still about 850.123 m³/s. Gauge and model boundaries are not interchangeable.')
            a = rising_crossing(t,q,q0,a)
            add('model_crossing','Same input level, at model rising crossing',[(a,q0),(b,qf)],
                'Moves the analysis start within July 31 to the model crossing. Small internal dip still needs checking.')
            adjusted,label = held_ramp(t,q,[(a,q0),(b,qf)])
            add('held_ramp',label,adjusted,'Additional review candidate; preserves both levels and uses a base-fitted horizontal start segment.')
        elif ex == '03':
            start1425 = rising_crossing(t,q,1425,a)
            add('proposed','1425 m³/s at the model rising crossing',[(start1425,1425),(b,qf)],
                'No routing rerun. Removes the initial dip but can change the connecting line enough to leave a late deficit.')
            ix=np.flatnonzero((t>=a)&(t<t[q.argmax()]));j=ix[q[ix].argmin()]
            add('model_minimum','Start at the saved pre-event model minimum',[(float(t[j]),float(q[j])),(b,qf)],
                'Recommended base candidate: analyze the existing run from its minimum; retain antecedent routing/storage. The model minimum is later than the input dip.')
        elif ex in ['05','13','15','22','37','45']:
            ix=np.flatnonzero((t>=a)&(t<t[q.argmax()]));j=ix[q[ix].argmin()]
            a,q0=float(t[j]),float(q[j])
            add('proposed','Pre-event model minimum to selected end',[(a,q0),(b,qf)],
                'Lowest saved model value between the previous start and dominant peak; moving the analysis boundary does not rerun routing.')
            result,_=check(t,q,variants[-1]['knots'])
            if not result['base_pass']:
                adjusted,label=held_ramp(t,q,variants[-1]['knots'])
                add('held_ramp',label,adjusted,'Additional base-fitted piecewise-linear candidate; not silently substituted for the requested straight line.')
        elif ex == '46':
            slopes=(q[1:]-q0)/(t[1:]-a)
            running=np.minimum.accumulate(slopes)
            passing=np.flatnonzero((t[1:]>b)&(q[1:]<qf)&(slopes<=running+1e-14))+1
            if not len(passing):
                raise ValueError('No admissible later lower endpoint in saved output')
            j=passing[0];b,qf=float(t[j]),float(q[j])
            add('proposed','Lower end at first admissible later model sample',[(a,q0),(b,qf)],
                'Searches the saved recession for the first later, lower endpoint whose straight background passes the base check, at one-minute resolution. This is a data-derived review boundary, not an independently observed baseflow.')
        elif ex == '54':
            knee=pd.Timestamp('2023-06-20T00:00Z').timestamp()
            add('proposed','Hold start level until June 20 00:00, then ramp',[(a,q0),(knee,q0),(b,qf)],
                'User-requested horizontal initial background, then a straight connection to the existing end reference.')
        else:
            add('proposed','Keep previous background',old,'Retained as requested (41 and 48 also unchanged). Tiny deficits remain reported; no substantive clipping.')
        if ex == '22':
            # Lower the background at the previous end time, then hold it until
            # the model crosses the lower level. Do not draw a new late chord.
            use=(t>a)&(t<b)
            upper=float(np.min(q0+(q[use]-q0)*(b-a)/(t[use]-a)))
            lower=float(np.floor(upper*10)/10)
            ids=np.flatnonzero((t[1:]>b)&(q[:-1]>lower)&(q[1:]<=lower))+1
            j=int(ids[0]);later=float(t[j-1]+60*(q[j-1]-lower)/(q[j-1]-q[j]))
            add('lowered_end','Slightly lower end, then short horizontal finish',[(a,q0),(b,lower),(later,lower)],
                '18415.000 m³/s: reach the lower level at the previous end time, then hold until its model recession crossing. A straight connection directly to the later crossing still has a tiny deficit.')
        add('previous','Previous straight-line background',old,'Previous review, retained for comparison.')
        panel['default_background']=DEFAULTS.get(ex,'proposed')
        panel['selection_status']=SELECTION_STATUS.get(ex,'Previous selection retained')
        origin=pd.Timestamp(panel['origin']).timestamp()
        for v in variants:
            result, (tt,qq,bg)=check(t,q,v['knots']);v.update(result)
            v['valid_children']=0;v['children_checked']=len(group)
            v['line_x']=[(p[0]-origin)/86400 for p in v['knots']]
            v['line_y']=[p[1] for p in v['knots']]
            v['event_x']=[(tt[0]-origin)/86400,(tt[-1]-origin)/86400]
            v['shade_x']=((tt-origin)/86400).tolist()
            bad=qq<bg-1e-8
            v['shade_q']=[float(x) if yes else None for x,yes in zip(qq,bad)]
            v['shade_b']=[float(x) if yes else None for x,yes in zip(bg,bad)]
            if v['key']==panel['default_background']:
                px=(tt-tt[0])/86400
                ax.plot(px,qq,color='#ce3d45',lw=1.2,label='Base Cunge')
                ax.plot(px,bg,color='#9b59b6',ls='--',lw=1.3,label='Current review background')
                ax.fill_between(px,qq,bg,where=bad,color='#e68c28',alpha=.5,interpolate=True)
                ax.set(title=f"{ex} · minimum Q−B: {result['minimum_excess_cms']:.4g} m³/s",xlabel='Days from proposed start',ylabel='Q (m³/s)')
                ax.grid(alpha=.2)
        # The same reference is applied to every child, without changing it per state.
        for child in group.itertuples():
            ct,cq=read_wave(child.outlet_parquet)
            for v in variants:
                checked,_=check(ct,cq,v['knots'])
                v['valid_children']+=int(checked['base_pass'])
        for v in variants:
            v['valid_pairs']=v['valid_children'] if v['base_pass'] else 0
            rows.append(dict(example=ex,**{k:val for k,val in v.items() if not k.startswith(('line_','shade_','event_')) and k!='knots'},
                knots_utc=json.dumps([[iso(x),y] for x,y in v['knots']])))
        panel['background_candidates']=variants
        panel['series']=[s for s in panel['series'] if s['legendgroup'] in ['curve_0','curve_3']]
        definitions[ex]=[{k:v for k,v in row.items() if not k.startswith(('line_','shade_','event_'))} for row in variants]
        print(ex,[(v['key'],round(v['minimum_excess_cms'],6),v['valid_pairs']) for v in variants],flush=True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Revised backgrounds: latest agreed choices and review candidates\nOrange: remaining negative excess; alternatives available in HTML',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'revised_backgrounds.png',dpi=140);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'candidate_checks.csv',index=False)
    (OUT/'candidate_definitions.json').write_text(json.dumps(dict(status='Review candidates; earlier approved selections and response outputs unchanged',
        times='UTC',negative_excess_tolerance_cms=1e-8,checks='Base and all 2172 unique children checked against the same per-network background',
        default_backgrounds={ex:DEFAULTS.get(ex,'proposed') for ex in definitions},selection_status=SELECTION_STATUS,cases=definitions),indent=2))
    script=Path(__file__).with_suffix('.js').read_text()
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())).decode()
    library=original.split('<script>',1)[1].split('</script>',1)[0]
    document='''<!doctype html><meta charset="utf-8"><title>Revised event backgrounds</title><style>body{font:14px Arial;color:#202a34;margin:24px}p{max-width:1300px;line-height:1.5}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px;z-index:20}select{padding:7px;margin-right:12px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.table{overflow:auto}</style><h1>Revised event backgrounds — 15-network review</h1><p>Blue: input. Red: saved base Cunge. Purple: selected background. Orange: background exceeds the base hydrograph. Proposed shows the latest requested changes literally, including 02 at the original clock time and 03 at the 1425 m³/s rising crossing. Select a network to inspect alternatives and their exact values. UTC throughout.</p><p>These are analysis-boundary changes, not model reruns. The latest agreed horizontal sections are selected by default; other alternatives remain available for review. A base pass does not guarantee a child pass: each child is checked against the same absolute-time background. Neither hydrographs nor earlier response results are overwritten.</p><div class="controls">Network <select id="network"><option value="all">All 15</option></select>View <select id="view"><option value="full">Full view</option><option value="event" selected>Proposed event</option><option value="peak">Peak</option></select>Background <select id="candidate"></select></div><p id="notes"></p><div id="plot"></div><h2>Candidate checks</h2><p>A deficit is retained whenever Q−B is below −1e−8 m³/s. Tiny and substantial deficits are shown numerically; no clipping or arbitrary acceptance percentage is applied. Valid pairs require both base and child to pass.</p><div class="table"><table><thead><tr><th>Network</th><th>Candidate</th><th>Start UTC / Q</th><th>End UTC / Q</th><th>Minimum Q−B</th><th>Deficit / positive volume (%)</th><th>Base passes</th><th>Valid pairs</th></tr></thead><tbody id="values"></tbody></table></div><script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
    document=document.replace('Proposed shows the latest requested changes literally, including 02 at the original clock time and 03 at the 1425 m³/s rising crossing.',
        'Defaults show the agreed choices: 02 at the model crossing with a horizontal start, 03 at its minimum, horizontal starts for 05/13/37, and the accepted 15/46 choices. For 22, the slightly lower end plus short horizontal finish is shown for review. 45 now uses the accepted horizontal finish.')
    document=document.replace('<h2>Candidate checks</h2>', '<p><a href="22_45_recession_detail.png">Zoomed explanation of the tiny 22 and 45 recession differences</a></p><h2>Candidate checks</h2>')
    (OUT/'15_network_revised_backgrounds.html').write_text(document.replace('LIBRARY',library).replace('PAYLOAD',packed).replace('SCRIPT',script))
    (OUT/'manifest.json').write_text(json.dumps(dict(provenance=provenance,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256((OUT/'15_network_revised_backgrounds.html').read_bytes()).hexdigest(),networks=15,unique_children=len(pairs),candidate_rows=len(rows)),indent=2))
    print(OUT)


if __name__=='__main__':
    main()
