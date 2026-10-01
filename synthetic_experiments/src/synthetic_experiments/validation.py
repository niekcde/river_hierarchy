"""Baseline window, timestep, and segmentation sensitivity checks."""
from dataclasses import replace
import json
from pathlib import Path

from .networks import make_states, canonical_network
from .routing import Settings, schedule, route


def validate_baseline(config_path,output):
    from .pipeline import write_json
    config=json.loads(Path(config_path).read_text())
    params={k:v[0] for k,v in config['grid'].items()}
    if any(len(v)!=1 for v in config['grid'].values()):
        raise ValueError('Validation expects one baseline configuration')
    settings=Settings(**config['routing']); states,pairs=make_states(params)
    nets={name:canonical_network(g) for name,g in states.items()}
    baseline_clock=schedule(nets.values(),settings)
    modes=[('baseline',settings,baseline_clock),
           ('double_buffer',settings,dict(baseline_clock,buffer_hours=2*baseline_clock['buffer_hours'])),
           ('coarser_timestep',settings,dict(baseline_clock,dt_seconds=int((baseline_clock['dt_seconds']+baseline_clock['dt_max_seconds'])/2))),
           ('100m_segments',replace(settings,target_length_m=100,effective_length_floor_m=100),None)]
    values={}
    for label,s,clock in modes:
        clock=clock or schedule(nets.values(),s)
        cache={}; results={}
        for name,net in nets.items():
            key=json.dumps(net,sort_keys=True)
            if key not in cache:
                _,cache[key],_=route(net,s,clock)
            results[name]=cache[key]
        values[label]=dict(clock=clock,states=results)
    comparisons=[]
    for pair in pairs:
        record=dict(child_state=pair['child_state'])
        for label,mode in values.items():
            parent=mode['states'][pair['parent_state']]; child=mode['states'][pair['child_state']]
            record[label+'_lag_shift_seconds']=child['peak_lag_seconds']-parent['peak_lag_seconds']
            record[label+'_lag_shift_pct']=100*(child['peak_lag_seconds']/parent['peak_lag_seconds']-1)
        comparisons.append(record)
    # Longer initialization must not change reported peaks at the existing time resolution.
    for name in states:
        if abs(values['baseline']['states'][name]['peak_lag_seconds']-values['double_buffer']['states'][name]['peak_lag_seconds']) > baseline_clock['dt_seconds']:
            raise ValueError(f'Spin-up sensitivity exceeds one timestep: {name}')
    result=dict(modes=values,comparisons=comparisons,
                interpretation='Double-buffer peak lags must agree within one baseline step. Resolution sweeps are diagnostics, not assumed convergence; inspect their differences before interpreting small effects.')
    write_json(Path(output)/'numerical_validation.json',result)
    return result
