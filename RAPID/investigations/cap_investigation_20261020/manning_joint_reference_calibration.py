"""Timing-first joint n/beta calibration of network 22's Manning reference.

Same reviewed targets and bounds as the existing BASED calibration. Child
states and RF artifacts are not used or changed.
"""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from manning_calibrated_network import ROOT, OUT, BUNDLE, parameters, digest
from delay_side_project import read, SOURCES
from manning_network_pilot import table
from rapid_tools.common_grid import route_interval_means

DEST=OUT/'manning_joint_reference22_20261027'


def main():
    DEST.mkdir(exist_ok=True)
    case=next(c for c in read(OUT/'routing_plan.json')['cases'] if c['example']=='22')
    job=next(j for j in case['jobs'] if j['state_id']=='S000_base')
    source=read(job['parameters']);dt=job['native_dt_seconds']
    previous=read(OUT/'manning_calibrated_network22_20261027/calibration.json')
    protocol=read(BUNDLE/'protocol.json');src=Path(protocol['source_directory'])
    data=read(src/'22/60s.parquet');meta=next(m for m in read(src/'22/60s.json') if m['method']=='cunge')
    timing=next(x for x in read(BUNDLE/'timing_targets.json')['records'] if x['example']=='22')
    height=next(x for x in read(BUNDLE/'height_targets.json')['records'] if x['example']=='22')
    archive=BUNDLE/'source_code/calibrate_common_grid_height.py';text=archive.read_text();SOURCES[str(archive)]=digest(archive)
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='feature')
    scope={'np':np,'pd':pd};exec(compile(ast.Module(body=[node],type_ignores=[]),str(archive),'exec'),scope);feature=scope['feature']
    valid=data.downstream_cms.notna().to_numpy();times=data.time_utc.astype('int64').to_numpy()/1e9
    goal=pd.Timestamp(timing['target_utc']).timestamp();lo,hi=height['low_cms'],height['high_cms']
    oldn=float(source.n_manning.iloc[0]);oldfit=previous['selected']
    design=dict(example='22',n_bounds=[.01,.1],beta_bounds=[.5,2.],
                selection='Archived timing-first protocol: fit n for each beta to reviewed crest time; within 1 minute choose lowest height distance, then beta nearest 1, then timing error.',
                depth='Manning reference depth recomputed for every trial n; fixed during routing',
                initial_beta_profile=[1.,.75,.5,1.5,2.],soft_height_range=[lo,hi],source_clock=meta.get('observation_clock_status'),
                recalibrate_children=False,production_modified=False,RF_fits=0)
    (DEST/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    cache={};trials=[];fits={};waves={}
    def run(n,beta,clock=1,subdivision=1):
        key=(float(n),float(beta),clock,subdivision)
        if key in cache:return cache[key]
        g=source.copy();g['diffusion_multiplier']=beta
        g['count']*=subdivision;g['rapid_effective_length_m']=g.link_length_m/g['count']
        f=parameters(g,n);step=dt/clock
        q,audit=route_interval_means(f,data.forcing_cms.to_numpy(),round(60/step),step,initial_flow=meta['initial_cms'])
        assert audit['minimum_native_flow_cms']>=-1e-9
        tm,h,kind,_=feature(q[valid],times[valid],'22',timing)
        rec=dict(n_manning=float(n),beta=float(beta),clock_divisor=clock,subdivision=subdivision,
                 timing_error_minutes=(tm-goal)/60,peak_cms=h,height_distance_cms=max(lo-h,0.,h-hi),
                 raw_rmse_cms=float(np.sqrt(np.mean((q[valid]-data.downstream_cms.to_numpy()[valid])**2))),feature_kind=kind,**audit)
        cache[key]=rec;waves[key]=q;trials.append(rec)
        pd.DataFrame(trials).to_csv(DEST/'trials.csv',index=False)
        return rec
    def fit_beta(beta):
        if beta in fits:return fits[beta]
        ns=sorted(set([.01,.0175,.025,.04,.07,.1,oldn,oldfit['n_manning']]+[x['n_manning'] for x in fits.values()]))
        candidates=[run(n,beta) for n in ns]
        for a,b in zip(ns[:-1],ns[1:]):
            ra,rb=run(a,beta),run(b,beta)
            if ra['timing_error_minutes']*rb['timing_error_minutes']>=0:continue
            for _ in range(18):
                ea,eb=ra['timing_error_minutes'],rb['timing_error_minutes']
                n=a+float(np.clip(-ea/(eb-ea),.15,.85))*(b-a);r=run(n,beta);candidates.append(r)
                if abs(r['timing_error_minutes'])<=.5:break
                if ea*r['timing_error_minutes']<=0:b=n;rb=r
                else:a=n;ra=r
        best=min(candidates,key=lambda r:(abs(r['timing_error_minutes']),abs(r['n_manning']-oldn)))
        fits[beta]=best
        pd.DataFrame(fits.values()).to_csv(DEST/'beta_profile.csv',index=False)
        print('Beta profile',beta,'n',best['n_manning'],'timing',best['timing_error_minutes'],'height gap',best['height_distance_cms'],flush=True)
        return best
    baseline=fit_beta(1.)
    for beta in [.75,.5,1.5,2.]:fit_beta(beta)
    # If both targets can be met, refine towards the least departure from beta=1.
    feasible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0]
    if feasible and baseline['height_distance_cms']>0:
        success=min(feasible,key=lambda r:abs(math.log(r['beta'])))['beta'];failed=1.
        for _ in range(3):
            beta=math.sqrt(success*failed);r=fit_beta(beta)
            if abs(r['timing_error_minutes'])<=1 and r['height_distance_cms']==0:success=beta
            else:failed=beta
    eligible=[r for r in fits.values() if abs(r['timing_error_minutes'])<=1]
    best=min(eligible,key=lambda r:(r['height_distance_cms'],abs(math.log(r['beta'])),abs(r['timing_error_minutes']))) if eligible else min(fits.values(),key=lambda r:(abs(r['timing_error_minutes']),r['height_distance_cms']))
    # Verify the inherited beta=.5/n pair before interpreting the new selection.
    inherited=run(oldfit['n_manning'],.5)
    np.testing.assert_allclose([inherited['peak_cms'],inherited['timing_error_minutes']],[oldfit['peak_cms'],oldfit['timing_error_minutes']],rtol=0,atol=1e-7)
    controls=[]
    for r in [best,baseline]:
        native=waves[r['n_manning'],r['beta'],1,1]
        for label,clock,sub in [('half_dt',2,1),('half_dx_half_dt',2,2)]:
            check=run(r['n_manning'],r['beta'],clock,sub)
            controls.append(dict(beta=r['beta'],control=label,timing_change_min=check['timing_error_minutes']-r['timing_error_minutes'],
                                 peak_change_cms=check['peak_cms']-r['peak_cms'],
                                 max_wave_change_cms=float(abs(waves[r['n_manning'],r['beta'],clock,sub][valid]-native[valid]).max())))
    pd.DataFrame(controls).to_csv(DEST/'numerical_checks.csv',index=False)
    wave=data.copy()
    for label,r in [('selected',best),('beta1',baseline),('inherited_beta05',inherited)]:
        wave[label+'_outlet_cms']=waves[r['n_manning'],r['beta'],1,1]
    wave.to_parquet(DEST/'waveforms.parquet',index=False)
    result=dict(selected=best,beta1=baseline,inherited=inherited,beta_at_boundary=best['beta'] in [.5,2.],
                n_at_boundary=best['n_manning'] in [.01,.1],height_range_met=best['height_distance_cms']==0,
                timing_tolerance_met=abs(best['timing_error_minutes'])<=1,trials=len(trials),
                numerical_checks=controls,design=design,independent_validation=False)
    (DEST/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    profile=pd.DataFrame(fits.values()).sort_values('beta')
    axes[0].plot(profile.beta,profile.peak_cms,'o-');axes[0].axhspan(lo,hi,color='#009E73',alpha=.15,label='Soft height target')
    axes[0].set(xlabel='Diffusion multiplier beta (n refitted at each point)',ylabel='Crest discharge (m³/s)',title='Timing-conditioned beta profile');axes[0].legend(fontsize=8)
    hours=(times-goal)/3600
    for label,color in [('selected','#0072B2'),('beta1','#D55E00')]:axes[1].plot(hours,wave[label+'_outlet_cms'],label=label,color=color)
    axes[1].axvline(0,color='grey',ls='--');axes[1].axhspan(lo,hi,color='#009E73',alpha=.15)
    axes[1].set(xlim=(-24,24),xlabel='Hours from reviewed crest target',ylabel='Discharge (m³/s)',title='Reference hydrograph near the crest');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(DEST/'calibration.png',dpi=160);plt.close(fig)
    report=f'''# Joint n/beta calibration: Manning reference, network 22

Both parameters were retried within the existing bounds: n=0.01–0.1 and
beta=0.5–2. The standard beta=1 case was evaluated explicitly, with its own
n fit. Depth, celerity, K and X were recalculated for every trial. Only the
full-complexity reference was used; no collapsed-state response entered fitting.

## Timing-conditioned calibration profile

{table(profile[['beta','n_manning','timing_error_minutes','peak_cms','height_distance_cms']])}

Selected n: **{best['n_manning']:.9f}**; beta: **{best['beta']:.6f}**.
Timing target met within one minute: **{result['timing_tolerance_met']}**.
Soft height interval 20,300–20,400 m³/s met: **{result['height_range_met']}**.
Selected beta is at a search boundary: **{result['beta_at_boundary']}**.

The selection follows the archived timing-first objective: first fit n to
crest timing for each beta, then prefer the smallest soft-height shortfall
among timing-feasible candidates, with beta closest to 1 as the tie-breaker.
It is not an unconstrained global fit or a fit of the entire hydrograph shape.
The saved provisional observation-clock status and soft target definitions
are unchanged. No independent validation is claimed.

![Calibration profile and crest hydrographs](calibration.png)

## Numerical checks

{table(pd.DataFrame(controls))}

The previously selected n/beta pair reproduces its saved calibration metrics
within 1e−7 absolute tolerance. All routed trials have nonnegative native flow.
The selected and beta=1 cases have half-timestep and half-cell-length/half-time
checks; all coefficients are regenerated for the refined grid.

## Interpretation

If beta=0.5 remains selected at its lower bound, that means reducing diffusion
continues to improve the soft height objective within the allowed range. It
does not establish that 0.5 is a physically required constant or identify an
interior optimum. Beta=1 can remain a valid unmodified Cunge comparison even
if its calibrated peak is lower. Choosing between them depends on how much
weight to give the existing soft height target; crest timing alone does not
establish the need for the extra diffusion adjustment.

Bounds were not widened to force a height match. Existing BASED results,
the completed Manning collapse experiment, and RF files remain unchanged.

## Reproduction

Run `RAPID/investigations/cap_investigation_20261020/manning_joint_reference_calibration.py`
with the river-hierarchy-rivgraph Python environment. trials.csv contains all
{len(trials)} routed trials, beta_profile.csv the per-beta selections,
result.json the final decision, numerical_checks.csv the convergence checks,
and waveforms.parquet the compared reference outputs. Source hashes and
output hashes are recorded in manifest.json.
'''
    (DEST/'REPORT.md').write_text(report)
    for path in [Path(__file__),Path(__file__).with_name('manning_calibrated_network.py'),Path(__file__).with_name('manning_network_pilot.py'),ROOT/'RAPID/src/rapid_tools/cunge.py',ROOT/'RAPID/src/rapid_tools/common_grid.py']:
        SOURCES[str(path)]=digest(path)
    for path,h in SOURCES.items():assert digest(path)==h
    (DEST/'manifest.json').write_text(json.dumps(dict(complete=True,sources=SOURCES,outputs={str(p):digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'},
                                                     child_runs=0,RF_fits=0,production_modified=False),indent=2)+'\n')
    print('SELECTED',json.dumps(best),flush=True)


if __name__=='__main__':main()
