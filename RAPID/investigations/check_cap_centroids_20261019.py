"""Exploratory crest-cap centroids; production response definitions unchanged.

Run from repo root with PYTHONPATH=RAPID/src. Uses the finalized shared 0.5%
window, fixed baseline, every crest, and each state's own maximum excess.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rapid_tools.hydrograph_response import pulse, distribution, digest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'RAPID/outputs/hydrograph_response'
OUT = ROOT/'RAPID/outputs/hydrograph_cap_centroid_20261019'


def cap_centroid(t, excess, fraction):
    """Exact centroid of the piecewise-linear positive cap, including crossings."""
    if not 0 < fraction < 1:
        raise ValueError('Cap fraction must be between zero and one')
    t, excess = np.asarray(t, float), np.asarray(excess, float)
    if not np.isfinite(excess).all() or (excess < 0).any() or excess.max() <= 0:
        raise ValueError('Positive, finite excess required')
    level = fraction*excess.max()
    z = excess-level
    ix = np.flatnonzero(z[:-1]*z[1:] < 0)
    crossing = t[ix]+np.diff(t)[ix]*(-z[ix])/(z[ix+1]-z[ix])
    tt = np.sort(np.r_[t, crossing])
    cap = np.maximum(np.interp(tt, t, excess)-level, 0.)
    moments = distribution(tt, cap)
    return dict(cap_centroid_seconds=moments['centroid_seconds'],
                cap_volume_m3=moments['volume_m3'], cap_level_excess_cms=float(level))


def main():
    OUT.mkdir(exist_ok=True)
    states = pd.read_parquet(SOURCE/'state_metrics.parquet')
    assert set(states.window)=={'0.5%'}
    assert not states.duplicated(['example','state','window']).any()
    rows=[]
    for i, (_,r) in enumerate(states.iterrows()):
        t,q,_=pulse(r) # checks saved outlet fingerprint before using it
        for fraction in [.7,.8,.9]:
            rows.append(dict(example=r.example,state=r.state,aliases=r.aliases,
                             window=r.window,cap_fraction=fraction,**cap_centroid(t,q,fraction)))
        if (i+1)%200==0: print(f'{i+1}/{len(states)} states',flush=True)
    metrics=pd.DataFrame(rows)
    metrics.to_parquet(OUT/'state_metrics.parquet',index=False)
    base=metrics[metrics.state.eq('S000_base')].drop(columns=['state','aliases'])
    responses=metrics[metrics.state.ne('S000_base')].merge(base,on=['example','window','cap_fraction'],
                suffixes=('_child','_parent'),validate='many_to_one')
    responses['cap_centroid_shift_minutes']=(responses.cap_centroid_seconds_child-responses.cap_centroid_seconds_parent)/60
    original=pd.read_parquet(SOURCE/'responses.parquet')
    responses=responses.merge(original[['example','state','window','peak_shift_hours','centroid_shift_hours']],
                on=['example','state','window'],validate='many_to_one',indicator=True)
    assert responses._merge.eq('both').all() and len(responses)==3*len(original)
    responses=responses.drop(columns='_merge')
    responses['peak_shift_minutes']=60*responses.peak_shift_hours
    responses['full_centroid_shift_minutes']=60*responses.centroid_shift_hours
    responses.to_parquet(OUT/'responses.parquet',index=False)
    responses.to_csv(OUT/'responses.csv',index=False)
    pivot=responses.pivot(index=['example','state'],columns='cap_fraction',values='cap_centroid_shift_minutes')
    pivot.columns=['cap70_minutes','cap80_minutes','cap90_minutes']
    pivot['threshold_range_minutes']=pivot.max(axis=1)-pivot.min(axis=1)
    pivot['sign_change_outside_1min']=(pivot[['cap70_minutes','cap80_minutes','cap90_minutes']].min(axis=1)<-1)&(pivot[['cap70_minutes','cap80_minutes','cap90_minutes']].max(axis=1)>1)
    pivot.reset_index().to_csv(OUT/'threshold_sensitivity.csv',index=False)
    def summary(g):
        x=g.cap_centroid_shift_minutes
        return dict(n=len(g),exact_zero=int(x.eq(0).sum()),within_1second=int(x.abs().le(1/60).sum()),within_1minute=int(x.abs().le(1).sum()),median_abs_minutes=float(x.abs().median()),min_minutes=float(x.min()),max_minutes=float(x.max()))
    stats=[]
    for fraction,g in responses.groupby('cap_fraction'):
        for cohort,part in [('all',g),('unchanged_peak',g[g.peak_shift_minutes.eq(0)])]:
            stats.append(dict(cap_fraction=fraction,cohort=cohort,**summary(part)))
    pd.DataFrame(stats).to_csv(OUT/'summary.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(definition='w(t)=max(Q(t)-B-fraction*max(Q-B),0); centroid integral(t*w)/integral(w); child minus parent',
      fractions=[.7,.8,.9],window='Finalized shared 0.5% analysis window',
      baseline='Existing fixed example baseline',crests='All cap components retained; no selected-crest matching',
      status='Exploratory only; no production response replacements or predictor fits',
      unmatched_response_rows=0,state_count=len(states),comparison_count=len(original),
      sources={str(p):digest(p) for p in [SOURCE/'state_metrics.parquet',SOURCE/'responses.parquet',Path(__file__)]}),indent=2))
    print(pd.DataFrame(stats).to_string(index=False),flush=True)

if __name__=='__main__':main()
