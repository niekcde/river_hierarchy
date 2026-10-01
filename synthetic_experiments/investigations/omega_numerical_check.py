"""Small matched audit of omega-invariant controls; separate from the grid.
Spatial/clock convergence, Muskingum-X dependence and forcing-duration dependence
are distinct questions. No fitted predictors use these diagnostic settings.
"""
from pathlib import Path
from dataclasses import replace
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
for p in [ROOT,ROOT/'synthetic_experiments/src',ROOT/'RAPID/src']:sys.path.insert(0,str(p))
from synthetic_experiments.networks import make_states,canonical_network
from synthetic_experiments.routing import Settings,schedule,route
from event_duration_verification import duration
from analysis_paths import RUN
OUT=RUN/'current_analysis'

def main():
    base=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.25,total_length_m=20000,internal_location='central')
    cases=[('baseline',base)]
    for f,levels in [('total_width_m',[250,1000]),('reference_slope',[5e-5,5e-4]),('total_length_m',[10000,40000])]:
        cases.extend((f'{f}={value}',dict(base,**{f:value})) for value in levels)
    selected=json.loads((RUN/'manifest.json').read_text()).get('config',{}).get('selected_hydraulic_reference')
    rows=[]
    for case,params in cases:
        default=Settings()
        local_schedule=schedule;local_route=route
        if selected:
            from selected_hydraulic_workflow import settings as reference_settings,clock_for
            from width_depth_scaling import adjusted,routing
            from segmentation_sensitivity import run_frame
            default=reference_settings(params)
            local_schedule=lambda nets,s:clock_for(nets,s,params['total_width_m'])
            local_route=lambda net,s,clock:run_frame(net,adjusted(routing.prepare(net,s),s,params['total_width_m'],selected['depth_alpha']),s,clock)
        variants={'baseline_200m':default,'refined_100m':replace(default,target_length_m=100,effective_length_floor_m=100),
                  'refined_50m':replace(default,target_length_m=50,effective_length_floor_m=50),
                  'larger_stable_dt':default,'muskingum_X_0.2':replace(default,x=.2),
                  'double_pulse_duration':replace(default,rise_hours=24,fall_hours=48)}
        states,_=make_states(params);nets=[canonical_network(states[k]) for k in ['nested/original','nested/full']]
        for variant,settings in variants.items():
            clock=local_schedule(nets,settings)
            if variant=='larger_stable_dt':clock['dt_seconds']=min(int(np.floor(clock['dt_max_seconds'])),2*clock['dt_seconds'])
            results=[local_route(n,settings,clock) for n in nets]
            amp=results[0][1]['peak_cms']-settings.baseflow_cms
            spans=[]
            for h,mt,_ in results:
                dt=mt['dt_seconds'];onset=round(mt['buffer_hours']*3600/dt)*dt;mask=h.time_seconds>=onset
                spans.append(duration(h.time_seconds[mask].to_numpy()-onset+dt/2,h.outlet_cms[mask].to_numpy(),settings.baseflow_cms,settings.baseflow_cms,.01*amp,.01*amp)['duration_seconds'])
            p,c=[r[1] for r in results]
            rows.append(dict(case=case,variant=variant,**params,
                lag_shift_pct=100*(c['peak_lag_seconds']/p['peak_lag_seconds']-1),
                event_duration_change_pct=100*(spans[1]/spans[0]-1),
                peak_excess_change_pct=100*((c['peak_cms']-settings.baseflow_cms)/(p['peak_cms']-settings.baseflow_cms)-1),
                dt_seconds=clock['dt_seconds'],n_capped=p['n_celerity_capped']+c['n_celerity_capped'],
                n_floored=p['n_length_floored']+c['n_length_floored'],
                max_volume_error=max(abs(p['volume_error_fraction']),abs(c['volume_error_fraction']))))
        print(case,'complete',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'omega_numerical_check.csv',index=False)
    (OUT/'omega_numerical_check.json').write_text(json.dumps(dict(family='nested/full',selected_hydraulic_reference=selected,n_geometries=len(cases),n_pair_reruns=len(rows),
        control='width, slope or extent varied one at a time; A=.8, inner=.5, multiplier1.25, central',
        spatial_check='segment target and associated minimum effective length reduced together; tests implemented refinement including its length floor',
        time_check='dt raised toward 2x while respecting shared parent-child stability bounds',
        interpretation='X changes model routing dispersion; pulse changes forcing; neither is a geometry predictor; no independent solver validation'),indent=2))
if __name__=='__main__':main()
