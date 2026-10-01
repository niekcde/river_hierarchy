"""Preliminary shape screen of active saved outlet series (not event attribution).

Remove constant-forcing spin-up using first detectable input change. Screen
whole remaining records, not the existing local-minimum event windows, which
can stop before a secondary peak. Peak prominence handles unequal initial and
final baselines without inventing an event-baseflow separation. Widths describe
the dominant peak ONLY. Multi-peak counts can reflect multiple forcing pulses.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/assets/event_shape_preliminary'
IDS='02 03 05 11 13 15 21 22 37 41 45 46 48 54 55 56'.split()

def local_path(value):
    s=str(value)
    if '/network_variants/' in s:
        return ROOT/'network_variants'/s.split('/network_variants/',1)[1]
    return Path(s)

def load(p):
    h=pd.read_csv(p)
    t=pd.to_datetime(h.time_utc,utc=True).astype('int64').to_numpy()/1e9
    return t,h.q_outlet_cms.to_numpy(),h.q_inflow_cms.to_numpy()

def shape(t,q,start,end,threshold=None):
    mask=(t>=start)&(t<=end);t=t[mask];q=q[mask]
    peaks,_=find_peaks(q)
    if not len(peaks):return None
    main=int(peaks[np.argmax(q[peaks])])
    prom=peak_prominences(q,[main])[0][0]
    if prom<=0:return None
    threshold=prom if threshold is None else threshold
    result=dict(peak=float(q[main]),prominence=float(prom))
    for fraction in (.01,.05,.1):
        ix,p=find_peaks(q,prominence=fraction*threshold)
        result[f'npeaks_{fraction:g}']=len(ix)
    for level in (.5,.95):
        _,_,left,right=peak_widths(q,[main],rel_height=level)
        result[f'width_{level:g}']=float(np.interp(right[0],np.arange(len(t)),t)-np.interp(left[0],np.arange(len(t)),t))
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];bases=[];issues=[]
    for e in IDS:
        p=ROOT/f'network_variants/output/{e}_indep__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv'
        if not p.exists():continue
        bt,bq,bi=load(p)
        changed=np.flatnonzero(np.abs(bi-bi[0])>max(np.ptp(bi)*1e-6,1e-8))
        if not len(changed):continue
        start=bt[max(0,changed[0]-1)]
        b=shape(bt,bq,start,bt[-1])
        if b is None:continue
        bases.append(dict(example=e,**b))
        for kind,filename in [('indep','rapid_run_registry.csv'),('compound_bubbles','rapid_run_registry.csv'),('island','island_rapid_run_registry.csv')]:
            p=ROOT/f'network_variants/output/{e}_{kind}__anchored'/filename
            if not p.exists():continue
            registry=pd.read_csv(p)
            if 'outlet_hydrograph_csv' not in registry:continue
            for _,r in registry.iterrows():
                sid=r.get('state_id',r.get('island_id',''))
                if sid=='S000_base' or pd.isna(r.outlet_hydrograph_csv):continue
                hp=local_path(r.outlet_hydrograph_csv)
                if not hp.exists():issues.append(dict(example=e,kind=kind,state=sid,issue='missing hydrograph'));continue
                t,q,qi=load(hp);shared_start=max(start,t[0])
                ref=shape(bt,bq,shared_start,bt[-1])
                if ref is None:continue
                v=shape(t,q,shared_start,t[-1],ref['prominence'])
                if v is None:continue
                row=dict(example=e,kind=kind,state=sid,hydrograph=str(hp),
                    peak_change_pct=100*(v['peak']/ref['peak']-1),
                    prominence_change_pct=100*(v['prominence']/ref['prominence']-1))
                for level in (.5,.95):
                    row[f'width_{level:g}_change_pct']=100*(v[f'width_{level:g}']/ref[f'width_{level:g}']-1)
                for f in (.01,.05,.1):
                    row[f'base_peaks_{f:g}']=ref[f'npeaks_{f:g}'];row[f'variant_peaks_{f:g}']=v[f'npeaks_{f:g}']
                rows.append(row)
        print(e,'done',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'comparisons.csv',index=False)
    pd.DataFrame(bases).to_csv(OUT/'baselines.csv',index=False)
    pd.DataFrame(issues).to_csv(OUT/'missing_files.csv',index=False)
    summary=[]
    for kind,g in d.groupby('kind'):
        r=dict(kind=kind,n=len(g))
        for col in ('peak_change_pct','prominence_change_pct','width_0.5_change_pct','width_0.95_change_pct'):
            r[col+'_abs_median']=g[col].abs().median();r[col+'_abs_p95']=g[col].abs().quantile(.95)
        for f in (.01,.05,.1):
            r[f'more_peaks_{f:g}']=int((g[f'variant_peaks_{f:g}']>g[f'base_peaks_{f:g}']).sum())
            r[f'fewer_peaks_{f:g}']=int((g[f'variant_peaks_{f:g}']<g[f'base_peaks_{f:g}']).sum())
        summary.append(r)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(d.loc[(d['variant_peaks_0.05']!=d['base_peaks_0.05']),['example','kind','state','base_peaks_0.05','variant_peaks_0.05']].to_string(index=False))
if __name__=='__main__':main()
