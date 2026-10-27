"""Test single-diagonal endpoints moved along the saved model hydrograph.

Proposals only: no independently lowered endpoint values, clipping or rerouting.
The first passing saved sample is resolution-limited, not an exact optimum.
"""
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

from review_revised_backgrounds_20261013 import ROOT, OUT as PREVIOUS, DEFAULTS, pairs, read_wave, check, iso, rising_crossing

OUT=ROOT/'RAPID/outputs/straight_anchor_review_20261013'
OUT.mkdir(exist_ok=True)


def start_candidates(t,q,a,b,qf,direction):
    """A start passes iff its secant slope to the end is a suffix maximum."""
    ix=np.flatnonzero(t<b)
    tt,qq=t[ix],q[ix]
    slope=(qf-qq)/(b-tt)
    maximum=np.maximum.accumulate(slope[::-1])[::-1]
    good=(slope>=maximum-1e-14)&(tt<t[q.argmax()])
    good &= tt>a if direction=='later' else tt<a
    return ix[good]


def end_candidates(t,q,a,q0,b,direction):
    """An end passes iff its secant slope from the start is a prefix minimum."""
    ix=np.flatnonzero(t>a)
    tt,qq=t[ix],q[ix]
    slope=(qq-q0)/(tt-a)
    minimum=np.minimum.accumulate(slope)
    good=(slope<=minimum+1e-14)&(tt>t[q.argmax()])
    good &= tt>b if direction=='later' else tt<b
    return ix[good]


def main():
    previous=json.loads((PREVIOUS/'candidate_definitions.json').read_text())['cases']
    original=(PREVIOUS/'15_network_revised_backgrounds.html').read_text()
    payload=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',original,re.S)[1])))
    rows,definitions,provenance=[],{},{}
    fig,axes=plt.subplots(5,3,figsize=(15,17))
    for p,ax in zip(payload['panels'],axes.flat):
        ex=p['example'];group=pairs[pairs.example_id.eq(ex)]
        chosen=next(c for c in previous[ex] if c['key']==DEFAULTS.get(ex,'proposed'))
        (a,q0),(b,qf)=chosen['knots'][0],chosen['knots'][-1]
        t,q=read_wave(group.iloc[0].parent_outlet_parquet)
        provenance[ex]=dict(outlet=group.iloc[0].parent_outlet_parquet,
            sha256=hashlib.sha256(Path(group.iloc[0].parent_outlet_parquet).read_bytes()).hexdigest())
        variants=[];default='unchanged';search_note=''
        def add(key,label,knots,note):
            assert len(knots)==2
            v=dict(key=key,label=label,knots=knots,note=note)
            result,_=check(t,q,knots);v.update(result)
            v.update(start_shift_minutes=(knots[0][0]-a)/60,end_shift_minutes=(knots[-1][0]-b)/60,
                start_discharge_change_cms=knots[0][1]-q0,end_discharge_change_cms=knots[-1][1]-qf)
            j=int(np.searchsorted(t,knots[-1][0]));lo=max(0,j-1);hi=min(len(t)-1,j+1)
            v['end_local_change_cms_per_hour']=float((q[hi]-q[lo])/(t[hi]-t[lo])*3600)
            variants.append(v)
        add('unchanged','Straight line between current endpoints',[(a,q0),(b,qf)],
            'Remove any horizontal sections; keep the selected start and end points. The strict deficit is still reported.')
        if ex=='02':
            found=end_candidates(t,q,a,q0,b,'later')
            if len(found):
                j=found[0]
                add('later_end','First passing later end; on the next rise',[(a,q0),(float(t[j]),float(q[j]))],
                    'Start retained. The first passing later saved sample is after the selected minimum, on the next rise. Mathematical validity does not establish closure of the same recession event.')
                default='later_end'
                assert variants[-1]['base_pass']
            lo=pd.Timestamp('2025-07-31T14:00Z').timestamp();hi=pd.Timestamp('2025-07-31T20:00Z').timestamp()
            inside=np.flatnonzero((t>=lo)&(t<=hi));j=inside[q[inside].argmin()]
            na,nq=float(t[j]),float(q[j])
            add('afternoon_minimum','Requested afternoon minimum; previous end restored',[(na,nq),(b,qf)],
                'End restored to August 4 12:57:30. Start is the lowest saved model discharge between July 31 14:00 and 20:00 UTC; the resulting single diagonal is checked without clipping.')
            default='afternoon_minimum'
            found=start_candidates(t,q,na,b,qf,'later')
            if len(found):
                j=found[0]
                add('after_afternoon_minimum','Nearest passing start after the afternoon minimum',[(float(t[j]),float(q[j])),(b,qf)],
                    'Additional review candidate: same restored end, with start moved 81 minutes beyond the requested minimum and about 5 m³/s higher.')
                assert variants[-1]['base_pass']
        if ex in ['05','13','37','48','54']:
            found=start_candidates(t,q,a,b,qf,'later')
            if len(found):
                j=found[0]
                add('later_start','First passing later start on the model curve',[(float(t[j]),float(q[j])),(b,qf)],
                    'End retained. First passing one-minute sample before the peak; no independent discharge offset. This changes the selected event window and excludes its preceding portion.')
                assert variants[-1]['base_pass']
                if ex in ['05','13','37']:default='later_start'
        if ex=='54':
            na=pd.Timestamp('2023-06-18T12:01Z').timestamp();nq=float(np.interp(na,t,q))
            add('june18_noon','Requested June 18 at 12:01 UTC',[(na,nq),(b,qf)],
                'User-confirmed date. Model discharge interpolated to 12:01; endpoint levels are not moved independently of the model curve.')
            default='june18_noon'
            na=rising_crossing(t,q,337,a)
            add('q337','Requested start at model Q = 337 m³/s',[(na,337.),(b,qf)],
                'Use the model rising crossing of 337 m³/s; retain the current end. No routing rerun or independent vertical offset.')
            default='q337'
        if ex=='22':
            found=start_candidates(t,q,a,b,qf,'earlier')
            search_note=f'{len(found)} earlier saved start samples pass with the current end retained; search covers the complete earlier output.'
            for hours in [1,3,6]:
                na=a-hours*3600
                if na<t[0]:continue
                nq=float(np.interp(na,t,q))
                add(f'earlier_{hours}h',f'Start {hours} hours earlier',[(na,nq),(b,qf)],
                    f'Illustrative earlier-start trial along the model curve. {search_note}')
            default='earlier_3h'
        if ex in ['21','22','45','56']:
            found=end_candidates(t,q,a,q0,b,'earlier')
            if len(found):
                j=found[-1]
                add('earlier_end','Nearest passing earlier end on the recession',[(a,q0),(float(t[j]),float(q[j]))],
                    'Alternative for review: retains the start and moves the end earlier along the falling limb. Not silently adopted as the requested change.')
                assert variants[-1]['base_pass']
        origin=pd.Timestamp(p['origin']).timestamp()
        for v in variants:
            _,(tt,qq,bg)=check(t,q,v['knots'])
            v.update(valid_children=0,children_checked=len(group),line_x=[(x-origin)/86400 for x,y in v['knots']],
                line_y=[y for x,y in v['knots']],event_x=[(tt[0]-origin)/86400,(tt[-1]-origin)/86400],shade_x=((tt-origin)/86400).tolist())
            bad=qq<bg-1e-8
            v['shade_q']=[float(x) if yes else None for x,yes in zip(qq,bad)]
            v['shade_b']=[float(x) if yes else None for x,yes in zip(bg,bad)]
            if v['key']==default:
                x=(tt-tt[0])/86400
                ax.plot(x,qq,color='#ce3d45',lw=1.2,label='Base model')
                ax.plot(x,bg,color='#9b59b6',ls='--',lw=1.3,label='Single diagonal')
                ax.fill_between(x,qq,bg,where=bad,color='#e68c28',alpha=.5,interpolate=True)
                ax.set(title=f"{ex} · minimum Q−B: {v['minimum_excess_cms']:.4g}",xlabel='Days from candidate start',ylabel='Q (m³/s)')
                ax.grid(alpha=.2)
        for child in group.itertuples():
            ct,cq=read_wave(child.outlet_parquet)
            for v in variants:
                r,_=check(ct,cq,v['knots']);v['valid_children']+=int(r['base_pass'])
        for v in variants:
            v['valid_pairs']=v['valid_children'] if v['base_pass'] else 0
            row={k:value for k,value in v.items() if k not in ['knots'] and not k.startswith(('shade_','line_','event_'))}
            rows.append(dict(example=ex,display_default=v['key']==default,**row))
        definitions[ex]=[{k:value for k,value in v.items() if not k.startswith(('shade_','line_','event_'))} for v in variants]
        p.update(background_candidates=variants,default_background=default,selection_status='Straight-anchor proposal; previous approved choices preserved. '+search_note)
        p['series']=[s for s in p['series'] if s['legendgroup'] in ['curve_0','curve_3']]
        print(ex,[(v['key'],round(v['minimum_excess_cms'],6),v['valid_pairs']) for v in variants],flush=True)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Single straight backgrounds: endpoint-movement trials\nOrange: remaining deficit. Latest 02 minimum and 54 Q337 proposals are displayed.',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'straight_anchor_trials.png',dpi=140);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'candidate_checks.csv',index=False)
    (OUT/'candidate_definitions.json').write_text(json.dumps(dict(status='Proposals only; no approved definitions or response results overwritten',
        times='UTC',search_resolution_seconds=60,validation='Every reported passing proposal checked by full piecewise-linear Q−B evaluation; every child checked against the same line',cases=definitions),indent=2))
    script=(ROOT/'RAPID/investigations/review_revised_backgrounds_20261013.js').read_text()
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())).decode()
    library=original.split('<script>',1)[1].split('</script>',1)[0]
    doc='''<!doctype html><meta charset="utf-8"><title>Straight background endpoint trials</title><style>body{font:14px Arial;margin:24px;color:#202a34}p{max-width:1300px;line-height:1.5}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px;z-index:20}select{padding:7px}table{border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><h1>Single-diagonal backgrounds: endpoint trials</h1><p>Each candidate has exactly two endpoints, located on the saved model curve. There are no horizontal additions, independent discharge offsets or clipped deficits. Blue: input. Red: base model. Purple: diagonal. Orange: background exceeds the model. Times are UTC.</p><p>05, 13 and 37 show the nearest passing later start. 54 shows the requested June 18 12:01 start; its nearest passing alternative is selectable. 22 shows an illustrative three-hour-earlier start and provides other trials plus an earlier-end alternative. Other unchanged proposals retain their deficits. 02's passing later endpoint lies on the next rise: this is an event-definition issue even though its line passes. Previous accepted backgrounds and response calculations remain unchanged.</p><div class="controls">Network <select id="network"><option value="all">All 15</option></select> View <select id="view"><option value="full">Full view</option><option value="event" selected>Candidate event</option><option value="peak">Peak</option></select> Candidate <select id="candidate"></select></div><p id="notes"></p><div id="plot"></div><h2>All candidate values</h2><p>Base pass and valid parent–child pairs are distinct. Endpoints were selected on the base; no individual adjustment is made to children. Download <a href="candidate_checks.csv">times, discharge changes and checks</a>.</p><div style="overflow:auto"><table><thead><tr><th>Network</th><th>Candidate</th><th>Start UTC / Q</th><th>End UTC / Q</th><th>Minimum Q−B</th><th>Deficit / positive volume (%)</th><th>Base passes</th><th>Valid pairs</th></tr></thead><tbody id="values"></tbody></table></div><script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
    doc=doc.replace('54 shows the requested June 18 12:01 start; its nearest passing alternative is selectable.',
        '54 now shows the requested model crossing at Q337; previous start alternatives remain selectable.')
    doc=doc.replace("02's passing later endpoint lies on the next rise: this is an event-definition issue even though its line passes.",
        '02 now restores the August 4 12:57:30 end and starts at the minimum between July 31 14:00 and 20:00. The nearest passing later start is available as an alternative.')
    html=OUT/'15_network_straight_anchor_trials.html';html.write_text(doc.replace('LIBRARY',library).replace('PAYLOAD',packed).replace('SCRIPT',script))
    (OUT/'manifest.json').write_text(json.dumps(dict(provenance=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),networks=15,unique_children=len(pairs),candidate_rows=len(rows)),indent=2))
    print(OUT)


if __name__=='__main__':main()
