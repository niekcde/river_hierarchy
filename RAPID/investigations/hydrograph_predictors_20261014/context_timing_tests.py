"""Separate predictor-context and parent-onset normalization experiments.

Prepare by default. --run starts fitting. No production response definitions change.
Context tests fit hours; normalization keeps baseline inputs and fits percent.
Every result is also evaluated in hours on identical whole-example held-out rows.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import focused_models as focused
import model_search as engine

ROOT=engine.ROOT
OUT=ROOT/'RAPID/outputs/hydrograph_context_timing_20261018'
TARGET='peak_shift_hours'
PCT='peak_shift_parent_rise_pct'


def parent_reference(t,q,input_start,initial,peak):
    """Parent-only tolerance: .005 * parent post-input max-minus-min range.

    Match the existing 30 consecutive one-minute departure samples convention;
    report the first sample, without adding the confirmation interval.
    """
    t,q=np.asarray(t,float),np.asarray(q,float)
    if not np.isfinite(t).all() or not np.isfinite(q).all() or not np.allclose(np.diff(t),60,rtol=0,atol=1e-6):
        raise ValueError('Expected finite one-minute parent outlet series')
    start=int(np.searchsorted(t,input_start))
    if start>=len(t) or t[start]!=input_start:raise ValueError('Input start absent from parent grid')
    amplitude=float(np.ptp(q[start:]))
    if amplitude<=0:raise ValueError('No positive parent discharge range')
    tolerance=.005*amplitude
    mask=np.abs(q[start:]-initial)>tolerance
    cs=np.r_[0,np.cumsum(mask)]
    found=np.flatnonzero(cs[30:]-cs[:-30]==30)
    if not len(found):raise ValueError('No sustained parent onset')
    onset=start+int(found[0]);t0=float(t[onset]);duration=(peak-t0)/3600
    if duration<=0:raise ValueError('Parent onset must precede global peak')
    return dict(t0_seconds=t0,t0_utc=pd.Timestamp(t0,unit='s',tz='UTC').isoformat(),
                parent_time_to_peak_hours=duration,tolerance_cms=tolerance,
                parent_range_cms=amplitude,onset_direction='rise' if q[onset]>initial else 'fall')


def references():
    state_path=ROOT/'RAPID/outputs/hydrograph_response/state_metrics.parquet'
    plan_path=ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'
    states=pd.read_parquet(state_path)
    parent=states[states.state.eq('S000_base')&states.window.eq('0.5%')]
    plan=json.loads(plan_path.read_text());cases={r['example']:r for r in plan['cases']}
    rows=[];hashes={str(p.relative_to(ROOT)):engine.sha(p) for p in [state_path,plan_path]}
    for r in parent.itertuples():
        outlet=Path(r.outlet);forcing_path=outlet.parent.parent/'forcing.parquet'
        assert engine.sha(outlet)==r.outlet_sha256
        hashes[str(outlet.relative_to(ROOT))]=r.outlet_sha256
        hashes[str(forcing_path.relative_to(ROOT))]=engine.sha(forcing_path)
        f=pd.read_parquet(outlet);forcing=pd.read_parquet(forcing_path)
        assert f.time_utc.equals(forcing.time_utc)
        input_start=pd.Timestamp(forcing.loc[forcing.phase.eq('input_event'),'time_utc'].iloc[0]).timestamp()
        t=f.time_utc.astype('int64').to_numpy()/1e9
        row=parent_reference(t,f.q_outlet_cms.to_numpy(),input_start,cases[r.example]['initial_cms'],r.peak_seconds)
        rows.append(dict(example=r.example,parent_peak_seconds=r.peak_seconds,**row))
    refs=pd.DataFrame(rows)
    assert len(refs)==15 and not refs.example.duplicated().any()
    return refs,hashes


def specifications():
    specs=[]
    for original in focused.specifications():
        for variant in ['baseline','relative_size','length_context','spread_context','normalized_timing']:
            # Replacing the SD/CV ratio by the same SD change makes these duplicates.
            if variant=='spread_context' and original['name'].startswith('cv__'):continue
            s=copy.deepcopy(original);s['name']=variant+'__'+s['name'];s['variant']=variant
            s['target']=PCT if variant=='normalized_timing' else TARGET
            if variant in ['relative_size','length_context']:s['columns'].append('B_mean_length_fraction')
            if variant=='length_context':
                s['columns']=['B_exposed_length_change' if c.endswith('B_ratio_route_length_mean') else c for c in s['columns']]
            if variant=='spread_context':
                s['columns']=['B_exposed_sd_change' if c.endswith('B_ratio_route_length_sd') else c for c in s['columns']]
            specs.append(s)
    return specs


def prepare(out):
    d,join=engine.load_data();refs,hashes=references();d=focused.transform(d)
    d=d.merge(refs,on='example',how='left',validate='many_to_one',suffixes=('','_reference'))
    np.testing.assert_allclose(d.parent_peak_seconds,d.parent_peak_seconds_reference)
    d[PCT]=100*d[TARGET]/d.parent_time_to_peak_hours
    cols=list(dict.fromkeys(c for s in specifications() for c in s['columns']))
    missing=~np.isfinite(d[cols+[TARGET,PCT]]).all(axis=1)
    excluded=d[~d.eligible|missing][['example','state','eligible']].copy()
    excluded['nonfinite_input_or_target']=missing.loc[excluded.index]
    d['eligible']=d.eligible&~missing
    # Explicit identities verify context uses the correct parent network scale.
    good=d[d.eligible]
    np.testing.assert_allclose(good.B_exposed_length_change,
        good.B_exposure*good.B_mean_length_fraction*(good.B_ratio_route_length_mean-1),atol=1e-12)
    paths=[Path(__file__),Path(engine.__file__),Path(focused.__file__),engine.SOURCE/'linked_responses.parquet',
           engine.SOURCE/'domain_graphs.csv',ROOT/'RAPID/outputs/hydrograph_response/manifest.json']
    hashes.update({str(p.relative_to(ROOT)):engine.sha(p) for p in paths})
    design=dict(specifications=specifications(),source_hashes=hashes,
        versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=engine.sklearn.__version__),
        onset='Parent only: first of 30 consecutive 1-minute samples after input start with abs(Qp-Qinitial) > .005*(max(Qp)-min(Qp)); range evaluated on parent outlet from input start through saved end. Confirmation adds no padding. No shared minimum-baseline level or child hydrographs used.',
        response='100*(child global peak time-parent global peak time)/(parent global peak time-parent t0). Original global peaks/window definitions unchanged. Requires the parent hydrograph at inference.',
        experiments={'baseline':'Agreed eight quantities; hours',
          'relative_size':'Baseline + mu_L,parent,D/mu_L,parent,N; hours',
          'length_context':'Baseline replace mean-length ratio with rin*(mu_child,D-mu_parent,D)/mu_parent,N, and add relative domain size; hours',
          'spread_context':'Baseline SD version replace spread ratio with rin*(SD_child,D-SD_parent,D)/mu_parent,N; hours; other length inputs unchanged',
          'normalized_timing':'Baseline inputs only; percentage target. No predictor-context changes.'},
        logarithms='Keep raw/log1p ratio alternatives. Newly added size fraction and signed context changes stay untransformed; never log a signed difference.',
        validation='Same whole-example outer holdout and 3 grouped inner folds as focused models. Equal-example MSE in fitted target units; normalization intentionally changes relative example weighting in physical units. Hyperparameters tuned separately per fixed model; no across-model selection.',
        output_units='All models assessed in hours and parent-rise percentage. Parent reference duration is never an input predictor. Whole-example scores and pair scores both reported.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists() and json.loads((out/'design.json').read_text())!=design and any((out/'checkpoints').glob('**/*.json')):
        raise ValueError('Design changed: use a fresh --output directory')
    engine.dump(out/'design.json',design)
    refs.to_csv(out/'parent_onset_audit.csv',index=False);excluded.to_csv(out/'exclusions.csv',index=False)
    join.to_csv(out/'join_audit.csv',index=False)
    rows=[]
    for scenario in ['all','without54']:
        source=engine.cohort(d,'partial',scenario)
        for variant in design['experiments']:
            ss=[s for s in specifications() if s['variant']==variant]
            rows.append(dict(scenario=scenario,experiment=variant,n=len(source),examples=source.example.nunique(),
                specifications=len(ss),fits=source.example.nunique()*sum(3*len(s['grid'])+1 for s in ss)))
    pd.DataFrame(rows).to_csv(out/'run_plan.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Prepared; use --run to fit. Parent times to peak [h]:',refs.parent_time_to_peak_hours.min(),refs.parent_time_to_peak_hours.max(),flush=True)
    return d,design


def summarize(out,design,data):
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    groups={}
    for path in (out/'checkpoints').glob('*/*/*.json'):
        r=json.loads(path.read_text());assert r['fingerprint']==fingerprint
        groups.setdefault((r['scenario'],r['spec']),[]).append(r)
    rows=[];settings=[];complete=set()
    for (scenario,spec),folds in groups.items():
        d=engine.cohort(data,'partial',scenario)
        if {r['held'] for r in folds}!=set(d.example):continue
        complete.add(scenario)
        for r in folds:
            for p in r['predictions']:rows.append(dict(scenario=scenario,pipeline=spec,variant=r['variant'],**p))
            settings.append(dict(scenario=scenario,pipeline=spec,held=r['held'],target=r['target'],params=json.dumps(r['params']),inner_mse=r['inner_mse']))
    for scenario in complete:
        d=engine.cohort(data,'partial',scenario)
        for held,test in d.groupby('example'):
            tr=d[d.example.ne(held)];w=engine.weights(tr)
            for label,target in [('zero',TARGET),('training_mean_hours',TARGET),('training_mean_pct',PCT)]:
                value=0. if label=='zero' else float(np.average(tr[target],weights=w))
                for r in test.itertuples():
                    h=value if target==TARGET else value*r.parent_time_to_peak_hours/100
                    rows.append(dict(scenario=scenario,pipeline=label,variant='reference',example=r.example,state=r.state,
                        observed_hours=r.peak_shift_hours,predicted_hours=h,observed_pct=getattr(r,PCT),predicted_pct=100*h/r.parent_time_to_peak_hours))
    if not rows:return
    pred=pd.DataFrame(rows);assert not pred.duplicated(['scenario','pipeline','example','state']).any()
    pred.to_parquet(out/'predictions.parquet',index=False);pd.DataFrame(settings).to_csv(out/'chosen_hyperparameters.csv',index=False)
    scores=[]
    for (scenario,pipeline,variant),g in pred.groupby(['scenario','pipeline','variant']):
        for unit in ['hours','pct']:
            for label,gg,w in [('equal_example',g,engine.weights(g)),('pair',g,np.ones(len(g))),
                *[('example_'+ex,z,np.ones(len(z))) for ex,z in g.groupby('example')]]:
                scores.append(dict(scenario=scenario,pipeline=pipeline,variant=variant,unit=unit,weighting=label,n=len(gg),
                    **engine.score(gg['observed_'+unit],gg['predicted_'+unit],w)))
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)


def run(data,design,out,args):
    fingerprint=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    variants={'baseline'}
    if 'context' in args.experiments:variants.update(['relative_size','length_context','spread_context'])
    if 'normalization' in args.experiments:variants.add('normalized_timing')
    specs=[s for s in design['specifications'] if s['variant'] in variants and s['model'] in args.models]
    for scenario in args.scenarios:
        d=engine.cohort(data,'partial',scenario)
        for held in sorted(d.example.unique()):
            tr,te=d[d.example.ne(held)],d[d.example.eq(held)]
            inner=[dict(train=sorted(tr.iloc[a].example.unique()),validation=sorted(tr.iloc[b].example.unique())) for a,b in engine.inner_splits(tr)]
            for s in specs:
                path=out/'checkpoints'/scenario/held/(s['name']+'.json')
                if path.exists():
                    assert json.loads(path.read_text())['fingerprint']==fingerprint;continue
                best,trials=engine.choose(tr,s['target'],s,args.jobs)
                p,audit=engine.fit_predict(tr,te,s['target'],s,best['params'],args.jobs)
                h=p if s['target']==TARGET else p*te.parent_time_to_peak_hours.to_numpy()/100
                rows=[dict(example=r.example,state=r.state,observed_hours=r.peak_shift_hours,predicted_hours=float(v),
                    observed_pct=getattr(r,PCT),predicted_pct=float(100*v/r.parent_time_to_peak_hours)) for r,v in zip(te.itertuples(),h)]
                engine.dump(path,dict(fingerprint=fingerprint,scenario=scenario,held=held,spec=s['name'],variant=s['variant'],
                    target=s['target'],params=best['params'],inner_mse=best['mse'],trials=trials,inner_groups=inner,audit=audit,predictions=rows))
            print(scenario,'held',held,'requested specifications complete',flush=True)
        summarize(out,design,data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--experiments',nargs='+',choices=['context','normalization'],default=['context','normalization'])
    p.add_argument('--models',nargs='+',choices=list(focused.GRIDS),default=list(focused.GRIDS))
    p.add_argument('--scenarios',nargs='+',choices=['all','without54'],default=['all','without54'])
    p.add_argument('--jobs',type=int,default=1)
    args=p.parse_args();d,design=prepare(args.output)
    if args.run:run(d,design,args.output,args)

if __name__=='__main__':main()
