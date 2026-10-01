"""Focused BASED/roughness routing sensitivity, separate from the completed grid.

One BASED kb and implied n per reference configuration, shared by all states.
No real-case rerouting or changes to production defaults. Resume exact cached jobs.
"""
from pathlib import Path
from dataclasses import asdict,replace
import sys,json,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
for p in [ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src']:sys.path.insert(0,str(p))
from synthetic_experiments.networks import make_states,canonical_network,digest
from synthetic_experiments.routing import Settings,schedule,route,buffer_seconds
from synthetic_experiments.pipeline import write_json,source_fingerprint
from synthetic_experiments.resistance import predictor
from rapid_tools.reference_section import predict_based_depth,DEFAULT_BASED_MODEL_PATH
from event_duration_verification import duration
OUT=ROOT/'synthetic_experiments/outputs/manning_sensitivity'
Y=['lag_shift_pct','event_duration_change_pct','peak_excess_change_pct']

def configurations():
    base=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.25,total_length_m=20000,internal_location='central')
    return [('baseline',base)]+[(label,dict(base,**change)) for label,change in [
        ('width_250',dict(total_width_m=250)),('width_1000',dict(total_width_m=1000)),
        ('slope_low',dict(reference_slope=5e-5)),('slope_high',dict(reference_slope=5e-4)),
        ('long_narrow_A',dict(outer_a_fraction=.05,length_multiplier=3.)),
        ('long_dominant_A',dict(outer_a_fraction=.95,length_multiplier=3.))]]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    default=Settings();source=source_fingerprint();rows=[];plans=[];refs=[];run_ids=set();infeasible=[]
    manifest=dict(reference='HYD-2026-09-29-MANNING-SENSITIVITY',status='running',source=source,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        based_model_sha256=hashlib.sha256(DEFAULT_BASED_MODEL_PATH.read_bytes()).hexdigest(),
        default_settings=asdict(default),configurations=[dict(case=k,parameters=p) for k,p in configurations()],
        variants=['legacy','based_n0350','based_n0025','based_n0035','based_n0050','based_derived'],
        n_reference='Wref Dref^(5/3) sqrt(Sref) / Qpeak; common to all states in each configuration',
        based_reference='single-centerline reference W=total width, S=reference slope, Q=saved peak 1000 m3/s',
        controls='200 m segmentation/floor, width allocation, X, forcing, celerity caps unchanged; clock recomputed within shared all-state stability bounds per configuration/variant',
        scope='7 configurations x 6 parameter variants x 7 comparisons; not a factorial grid or convergence study')
    write_json(OUT/'manifest.json',manifest)
    for case,params in configurations():
        states,pairs=make_states(params);nets={k:canonical_network(v) for k,v in states.items()}
        w=params['total_width_m'];s=params['reference_slope'];q=default.peak_cms
        with threadpool_limits(limits=1):h=predict_based_depth(discharge_cms=q,width_m=w,slope=s)
        kb=w/h;derived_n=w*h**(5/3)*np.sqrt(s)/q
        assert np.isclose(w*h**(5/3)*np.sqrt(s)/derived_n,q)
        refs.append(dict(case=case,width_m=w,slope=s,reference_q_cms=q,based_depth_m=h,kb=kb,derived_n=derived_n))
        variants=dict(legacy=default,based_n0350=replace(default,kb=kb),
            based_n0025=replace(default,kb=kb,n_manning=.025),based_n0035=replace(default,kb=kb,n_manning=.035),
            based_n0050=replace(default,kb=kb,n_manning=.05),based_derived=replace(default,kb=kb,n_manning=derived_n))
        for variant,settings in variants.items():
            try:
                clock=schedule(nets.values(),settings)
            except ValueError as exc:
                if 'No common stable integer timestep' not in str(exc):raise
                infeasible.append(dict(case=case,variant=variant,n_manning=settings.n_manning,kb=settings.kb,
                    affected_comparisons=len(pairs),reason=str(exc)))
                print(case,variant,'INFEASIBLE:',exc,flush=True)
                continue
            outputs={} 
            for state,net in nets.items():
                job=dict(network=net,settings=asdict(settings),clock=clock,source=source)
                rid=digest(job);run_ids.add(rid);folder=OUT/'runs'/rid
                if not (folder/'metrics.json').exists():
                    hydro,mt,seg=route(net,settings,clock)
                    mt.update(n_lower_celerity_clipped=int((seg.rapid_celerity_mps_raw<settings.min_celerity_mps).sum()),
                        n_upper_celerity_clipped=int((seg.rapid_celerity_mps_raw>settings.max_celerity_mps).sum()))
                    folder.mkdir(parents=True,exist_ok=True);write_json(folder/'job.json',job)
                    hydro.to_parquet(folder/'hydrograph.parquet',index=False)
                    write_json(folder/'metrics.json',mt)
                else:
                    mt=json.loads((folder/'metrics.json').read_text());hydro=pd.read_parquet(folder/'hydrograph.parquet')
                    assert json.loads((folder/'job.json').read_text())==job
                assert max(abs(mt['volume_error_fraction']),mt['pre_event_relative_error'],mt['final_relative_error'])<1e-4
                outputs[state]=(rid,hydro,mt)
                plans.append(dict(case=case,variant=variant,state=state,run_id=rid,kb=settings.kb,n_manning=settings.n_manning,dt_seconds=clock['dt_seconds']))
            for pair in pairs:
                pid,ph,pm=outputs[pair['parent_state']];cid,ch,cm=outputs[pair['child_state']]
                assert pm['peak_lag_seconds']>0
                amp=pm['peak_cms']-settings.baseflow_cms;spans={}
                for fraction in [.005,.01,.02]:
                    ds=[]
                    for hy in [ph,ch]:
                        t=hy.time_seconds.to_numpy()-buffer_seconds(clock)+clock['dt_seconds']/2
                        mask=t>=0
                        ds.append(duration(t[mask],hy.outlet_cms.to_numpy()[mask],settings.baseflow_cms,settings.baseflow_cms,fraction*amp,fraction*amp)['duration_seconds'])
                    spans[fraction]=ds
                row=dict(case=case,variant=variant,**params,**pair,kb=settings.kb,n_manning=settings.n_manning,derived_n=derived_n,
                    parent_run_id=pid,child_run_id=cid,dt_seconds=clock['dt_seconds'],
                    lag_shift_pct=100*(cm['peak_lag_seconds']/pm['peak_lag_seconds']-1),
                    event_duration_change_pct=100*(spans[.01][1]/spans[.01][0]-1),
                    peak_excess_change_pct=100*((cm['peak_cms']-settings.baseflow_cms)/amp-1),
                    parent_lag_seconds=pm['peak_lag_seconds'],child_lag_seconds=cm['peak_lag_seconds'],
                    parent_duration_seconds=spans[.01][0],child_duration_seconds=spans[.01][1],
                    parent_peak_cms=pm['peak_cms'],child_peak_cms=cm['peak_cms'],
                    within_one_dt=abs(cm['peak_lag_seconds']-pm['peak_lag_seconds'])<=clock['dt_seconds'],
                    max_volume_error=max(abs(pm['volume_error_fraction']),abs(cm['volume_error_fraction'])),
                    duration_change_005_pct=100*(spans[.005][1]/spans[.005][0]-1),
                    duration_change_020_pct=100*(spans[.02][1]/spans[.02][0]-1),
                    **predictor(states[pair['parent_state']],states[pair['child_state']],pair,settings))
                for side,mt in [('parent',pm),('child',cm)]:
                    for name in ['n_segments','n_celerity_capped','n_length_floored','n_lower_celerity_clipped','n_upper_celerity_clipped']:
                        row[side+'_'+name]=mt[name]
                rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/'results_partial.csv',index=False)
            print(case,variant,'complete; n=',round(settings.n_manning,5),'kb=',round(settings.kb,2),'dt=',clock['dt_seconds'],flush=True)
    d=pd.DataFrame(rows)
    assert len(d)+7*len(infeasible)==294 and not d.duplicated(['case','variant','child_state']).any()
    # Uniform kb changes scale dimensional resistance, not its matched ratio.
    for _,g in d.groupby(['case','child_state']):
        assert np.allclose(g.Lambda_matched,g.Lambda_matched.iloc[0],rtol=1e-7)
        assert np.allclose(g.resistance_predictor,g.resistance_predictor.iloc[0],rtol=1e-7,atol=1e-10)
    # Legacy controls must reproduce the available original grid at fixed inputs.
    original=pd.read_parquet(ROOT/'synthetic_experiments/outputs/full_grid_island/current_analysis/current_comparisons.parquet')
    keys=list(configurations()[0][1])+['child_state']
    matched=d[d.variant=='legacy'].merge(original,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(matched)==35
    for y in Y:assert np.allclose(matched[y+'_new'],matched[y+'_old'],rtol=1e-8,atol=1e-8),y
    d.to_csv(OUT/'results.csv',index=False);pd.DataFrame(plans).to_csv(OUT/'states.csv',index=False)
    pd.DataFrame(refs).to_csv(OUT/'references.csv',index=False)
    pd.DataFrame(infeasible,columns=['case','variant','n_manning','kb','affected_comparisons','reason']).to_csv(OUT/'infeasible.csv',index=False)
    manifest.update(status='complete_with_infeasible' if infeasible else 'complete',infeasible_comparisons=7*len(infeasible),attempted_comparisons=294,comparisons=len(d),unique_runs=len(run_ids),legacy_original_grid_matches=len(matched),
        max_volume_error=float(d.max_volume_error.max()),omega_invariance_verified=True)
    write_json(OUT/'manifest.json',manifest)
    print('COMPLETE',manifest['comparisons'],'comparisons;',len(run_ids),'unique runs',flush=True)
if __name__=='__main__':main()
