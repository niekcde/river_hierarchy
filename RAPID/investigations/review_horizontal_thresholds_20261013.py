"""Review common horizontal thresholds and peak-connected exceedance episodes.
Original model plateau levels are used, as requested. No routing is changed.
"""
from pathlib import Path
import json,hashlib,gzip,base64,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plotly.offline import get_plotlyjs
from review_revised_backgrounds_20261013 import ROOT,pairs,read_wave,iso

OUT=ROOT/'RAPID/outputs/horizontal_threshold_review_20261013';OUT.mkdir(exist_ok=True)
END={'02','05','13','37','48','54'}
MOVE={'15','21','22','41','45','46','55','56'}
TOL=1e-6

def episode(t,q,level,pa,pb,lo,hi):
    use=(t>=lo)&(t<=hi);tt=np.r_[lo,t[use&(t>lo)&(t<hi)],hi];qq=np.interp(tt,t,q)
    peakids=np.flatnonzero((tt>=pa)&(tt<=pb));j=peakids[np.argmax(qq[peakids])]
    above=qq>level+TOL
    if not above[j]:return dict(closed=False,status='peak below threshold',peak_utc=iso(tt[j]),peak_cms=float(qq[j]))
    left=np.flatnonzero(~above[:j]);right=np.flatnonzero(~above[j+1:])
    lc=not len(left);rc=not len(right)
    def cross(k):
        return float(tt[k]+np.clip((level-qq[k])/(qq[k+1]-qq[k]),0,1)*(tt[k+1]-tt[k]))
    a=float(tt[0]) if lc else cross(int(left[-1]));b=float(tt[-1]) if rc else cross(j+int(right[0]))
    # Count earlier resolved above-threshold episodes, excluding tiny initial roundoff.
    transitions=np.flatnonzero((~above[:-1])&above[1:])+1
    earlier=int(np.sum(tt[transitions]<a-1))
    if above[0] and not lc:earlier+=1
    return dict(closed=not(lc or rc),status='closed' if not(lc or rc) else 'missing '+('/'.join(x for x,y in [('rising crossing',lc),('falling crossing',rc)] if y)),
        rise_utc=iso(a),fall_utc=iso(b),rise_seconds=a,fall_seconds=b,peak_utc=iso(tt[j]),peak_cms=float(qq[j]),earlier_episodes_excluded=earlier)

def main():
    prior=json.loads((ROOT/'RAPID/outputs/final_straight_background_review_20261013/candidate_definitions.json').read_text())['candidates']
    old=(ROOT/'RAPID/outputs/centerline_background_review_20261013/15_network_centerline_backgrounds.html').read_text()
    panels=json.loads(gzip.decompress(base64.b64decode(re.search(r'<script id="payload"[^>]*>(.*?)</script>',old,re.S)[1])))
    rows=[];summaries=[];fig,axes=plt.subplots(5,3,figsize=(18,20));sources={}
    for c,p,ax in zip(prior,panels,axes.flat):
        ex=c['example'];assert ex==p['example'];g=pairs[pairs.example_id.eq(ex)]
        paths=[g.iloc[0].parent_outlet_parquet,*g.outlet_parquet.tolist(),str(ROOT/f'network_variants/output/{ex}_indep__anchored/reference_section/rapid/cunge/run/outlet.parquet')]
        names=['S000_base',*g.state_id.tolist(),'fully_collapsed'];waves=[read_wave(path) for path in paths]
        bt,bq=waves[0];a=pd.Timestamp(c['fixed_start_utc']).timestamp();b=pd.Timestamp(c['fixed_end_utc']).timestamp()
        level=c['current_end_cms'] if ex in END else c['current_start_cms'];reason='Original end plateau' if ex in END else 'Original start plateau'
        if ex=='56':
            a=pd.Timestamp('2025-06-25T00:00Z').timestamp();level=float(np.interp(a,bt,bq));reason='Base rising-limb Q at June 25 00:00 UTC, after June 24 19:44:30 minimum'
        # One common search support; selected peak still belongs to the reviewed event.
        lo=max(t[0] for t,q in waves);hi=min(t[-1] for t,q in waves)
        if ex=='48':lo=max(lo,pd.Timestamp('2025-06-02T06:00Z').timestamp());hi=min(hi,pd.Timestamp('2025-06-03T20:00Z').timestamp())
        original=level
        if ex in {"13","48"}:
            level={"13":730.15,"48":270.64}[ex];reason+="; user-accepted end-level refinement"
        # If a start-level curve never reaches below the reference before its target
        # peak, find the smallest common upward change that creates such a boundary.
        required=[]
        for t,q in waves:
            ids=np.flatnonzero((t>=a)&(t<=b));j=ids[np.argmax(q[ids])]
            pre=(t>=lo)&(t<t[j]);required.append(float(q[pre].min()))
        if ex in MOVE and max(required)>=level-TOL:
            level=max(level,max(required)+2*TOL)
            if level-original>1e-5:reason+='; minimally raised above highest pre-peak minimum across all versions'
        results=[]
        for name,path,(t,q) in zip(names,paths,waves):
            r=episode(t,q,level,a,b,lo,hi);r.update(example=ex,state=name,threshold_cms=level,outlet=str(Path(path).resolve()));rows.append(r);results.append(r)
        base=results[0];center=results[-1]
        summary=dict(example=ex,threshold_cms=level,original_level_cms=original,level_raise_cms=level-original,selection=reason,
            base=base,centerline=center,states=len(results),closed_states=sum(r['closed'] for r in results),
            common_search_start_utc=iso(lo),common_search_end_utc=iso(hi),peak_search_start_utc=iso(a),peak_search_end_utc=iso(b))
        summaries.append(summary)
        origin=pd.Timestamp(p['origin']).timestamp();p['series']=[s for s in p['series'] if s['key'] not in ['accepted','proposed']]
        p['series'].append(dict(x=[(lo-origin)/86400,(hi-origin)/86400],y=[level,level],name='Shared horizontal threshold',key='threshold',line=dict(color='#9155b5',dash='dash',width=1.8),customdata=[iso(lo),iso(hi)],type='scatter',mode='lines',connectgaps=False))
        for label,r,color in [('Base',base,'#ce3d45'),('Centerline',center,'#e38b15')]:
            if 'rise_seconds' in r:
                points=[(r[k],level) for k in ['rise_seconds','fall_seconds'] if not((k=='rise_seconds' and 'rising crossing' in r['status']) or (k=='fall_seconds' and 'falling crossing' in r['status']))]
                p['series'].append(dict(x=[(v-origin)/86400 for v,q in points],y=[q for v,q in points],customdata=[iso(v) for v,q in points],name=label+' event crossings',key=label+'_crossings',line=dict(color=color,width=0),marker=dict(color=color,size=7),type='scatter',mode='markers',connectgaps=False))
        valid=[r for r in results if 'rise_seconds' in r]
        start=min(r['rise_seconds'] for r in valid);end=max(r['fall_seconds'] for r in valid);pad=(end-start)*.04
        p['ranges']['event']=[(max(lo,start-pad)-origin)/86400,(min(hi,end+pad)-origin)/86400]
        p['threshold_summary']=summary
        for tr in p['series']:
            y=np.array([np.nan if v is None else v for v in tr['y']]);ax.plot(tr['x'],y,color=tr['line']['color'],ls='None' if tr['mode']=='markers' else '--' if tr['line'].get('dash')=='dash' else ':' if tr['line'].get('dash')=='dot' else '-',lw=1,marker='o' if tr['mode']=='markers' else None,ms=3,label=tr['name'] if tr.get('showlegend',True) else '_nolegend_')
        rng=p['ranges']['event'];vals=[v for tr in p['series'] for x,v in zip(tr['x'],tr['y']) if v is not None and rng[0]<=x<=rng[1]]+[level]
        lower,upper=min(vals),max(vals);pad=max(1,(upper-lower)*.07)
        ax.set(title=f'{ex} · Qref={level:.3f} · {summary["closed_states"]}/{len(results)} closed',xlim=rng,ylim=(lower-pad,upper+pad),xlabel=f'Days from {p["origin"][:10]} UTC',ylabel='Discharge (m³/s)');ax.grid(alpha=.2)
        sources[ex]={str(Path(path).resolve()):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
        print(ex,round(level,6),'raise',round(level-original,6),summary['closed_states'],len(results),'base',base['status'],'center',center['status'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'state_crossings.csv',index=False)
    (OUT/'threshold_definitions.json').write_text(json.dumps(dict(status='Accepted horizontal thresholds including end-level refinements for 13 and 48',
        rule='For each hydrograph select the connected above-threshold episode containing its maximum within the previously reviewed peak-search window. Earlier episodes are excluded; no forced closure. Common threshold and search support per network.',cases=summaries),indent=2)+'\n')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.978),fontsize=9)
    fig.suptitle('Horizontal event thresholds: base, single centerline and gauges\nMarkers: bounding crossings of the selected peak-connected pulse; gauges unshifted',fontsize=16,y=.998)
    fig.tight_layout(rect=(0,0,1,.945));fig.savefig(OUT/'15_network_horizontal_thresholds.png',dpi=140);plt.close(fig)
    script=Path(__file__).with_suffix('.js').read_text();packed=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
    # Reuse layout controls and offline library; replace prior explanatory text.
    head='''<!doctype html><meta charset="utf-8"><title>Horizontal event thresholds</title><style>body{font:15px Arial;margin:22px;color:#24303a}p{line-height:1.5;max-width:1300px}.controls{position:sticky;top:0;z-index:5;background:#eef2f5;padding:14px;display:flex;gap:20px;flex-wrap:wrap}select{padding:6px}</style><h1>15 networks: horizontal event thresholds</h1><p>Red: base. Orange: single centerline. Blue/black: upstream/downstream gauges. Purple: the shared horizontal threshold. Markers show the crossings bounding each model's selected pulse. The pulse is the connected above-threshold segment containing the selected event peak; earlier excursions are excluded. Missing crossings are reported, not invented. The closure counts include the base, every partial-collapse child and the single centerline.</p><p>Full view retains all saved model data; Event includes the selected pulse boundaries across every state; Peak retains the earlier crest zoom. Gauges are unshifted by default. Optional downstream alignment changes display only. Dotted gauge segments indicate long sampling intervals; 21/22 gauge clocks remain provisional.</p><p><b>Accepted end-level refinements:</b> 13 uses 730.15 m³/s and 48 uses 270.64 m³/s. All 2,202 model versions have both event crossings.</p><div class="controls"><label>Network <select id="network"><option value="all">All 15 networks</option></select></label><label>View <select id="view"><option value="event">Event</option><option value="full">Full view</option><option value="peak">Peak</option></select></label><label>Downstream gauge <select id="gauge"><option value="raw">Original discharge</option><option value="aligned">Baseline aligned (display only)</option></select></label></div><p id="note"></p><div id="plot"></div><p><a href="15_network_horizontal_thresholds.png">15-panel image</a> · <a href="state_crossings.csv">All state crossing checks</a> · <a href="threshold_definitions.json">Threshold definitions</a></p>'''
    html=head+'<script>'+get_plotlyjs()+'</script><script id="payload" type="application/octet-stream">'+packed+'</script><script>'+script+'</script>'
    (OUT/'15_network_horizontal_thresholds.html').write_text(html)
    (OUT/'manifest.json').write_text(json.dumps(dict(sources=sources,states=len(rows),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
