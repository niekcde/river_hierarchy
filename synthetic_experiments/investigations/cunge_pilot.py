"""Isolated current-method synthetic pilot. No legacy outputs or active selector edited."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for directory in [ROOT, ROOT/'RAPID/src', ROOT/'synthetic_experiments/src']:
    sys.path.insert(0, str(directory))
from synthetic_experiments.networks import make_states, canonical_network
from rapid_tools import cunge
from rapid_tools.common_grid_production import shared_clock
from rapid_tools.reference_section import predict_based_depth, DEFAULT_BASED_MODEL_PATH
from rapid_tools.hydrograph_response import distribution

CONFIG = ROOT/'synthetic_experiments/configs/cunge_pilot_20261026.json'
OUT = ROOT/'synthetic_experiments/outputs/cunge_pilot_20261026_refined'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(parameters, settings):
    graphs, pairs = make_states(parameters)
    depth = predict_based_depth(discharge_cms=settings['peak_cms'],
        width_m=parameters['total_width_m'], slope=parameters['reference_slope'])
    frames = {}
    for state, graph in graphs.items():
        net = canonical_network(graph)
        f = pd.DataFrame(net['edges'], columns=['u','v','link_length_m','width','slope'])
        f['id_link'] = np.arange(1, len(f)+1)
        f['depth_m'] = depth*(f.width/parameters['total_width_m'])**settings['depth_alpha']
        # Match the real uncapped preparation: a finite ceiling above all raw speeds.
        fastest = (5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/settings['grid_n_bounds'][0]
        ceiling = max(5.01, float(fastest.max())*1.01)
        grid = cunge.freeze_grid(f, settings['peak_cms'], n_bounds=settings['grid_n_bounds'],
            alpha=settings['grid_alpha'], min_celerity=settings['minimum_celerity_mps'], max_celerity=ceiling)
        grid['count'] *= settings['subdivision_multiplier']
        grid['rapid_effective_length_m'] = grid.link_length_m/grid['count']
        frames[state] = cunge.parameters(grid, settings['n_manning'], beta=settings['beta'])
        assert (frames[state].rapid_celerity_mps_raw < ceiling).all()
        np.testing.assert_allclose(grid['count']*grid.rapid_effective_length_m, f.link_length_m)
    return frames, pairs, float(depth)


def forcing(settings, tail):
    step = settings['report_seconds']
    onset = settings['spinup_hours']*3600
    peak = onset+settings['rise_hours']*3600
    end = peak+settings['fall_hours']*3600
    edges = np.arange(0, end+tail*3600+step/2, step)
    q = np.interp(edges, [0,onset,peak,end,edges[-1]],
        [settings['baseflow_cms'],settings['baseflow_cms'],settings['peak_cms'],settings['baseflow_cms'],settings['baseflow_cms']])
    # Exact interval means for the piecewise-linear pulse, with aligned corners.
    return (edges[:-1]+edges[1:])/2, (q[:-1]+q[1:])/2, end


def run_group(frames, settings, dt, tail):
    t, flow, end = forcing(settings, tail)
    waves, audits = {}, {}
    step = settings['report_seconds']; nsub = round(step/dt)
    assert np.isclose(nsub*dt, step)
    settled = True
    for state, frame in frames.items():
        wave, audit = cunge.route(frame, flow, nsub, dt)
        if audit['minimum_native_flow_cms'] < 0:
            raise ValueError(f'Negative native flow: {state}')
        outside = np.flatnonzero(np.abs(wave-settings['baseflow_cms']) > settings['settling_fraction']*(settings['peak_cms']-settings['baseflow_cms']))
        last = max(end, float(t[outside[-1]]) if len(outside) else end)
        ok = t[-1]-last >= settings['settling_guard_hours']*3600
        settled &= ok
        audit.update(settled=bool(ok), last_outside_seconds=last,
            excess_volume_error_fraction=float((wave-settings['baseflow_cms']).sum()/(flow-settings['baseflow_cms']).sum()-1))
        settled &= abs(audit['excess_volume_error_fraction']) < 1e-4
        waves[state], audits[state] = wave, audit
    return t, flow, waves, audits, settled


def responses(t, waves, pairs, settings):
    rows=[]; baseline=settings['baseflow_cms']
    for family in ['simple','nested','crossover']:
        family_pairs=[p for p in pairs if p['family']==family]
        parent=f'{family}/original'
        states=[parent]+[p['child_state'] for p in family_pairs]
        amplitude=float(waves[parent].max()-baseline)
        for fraction in settings['window_fractions']:
            # Synthetic counterpart of reviewed shared boundaries: the envelope of
            # threshold crossings across this original and all its children.
            crossings=[]
            for state in states:
                q=waves[state]-baseline; threshold=fraction*amplitude
                idx=np.flatnonzero(q>threshold)
                a,b=int(idx[0]),int(idx[-1])
                if a==0 or b==len(t)-1: raise ValueError('Censored analysis window')
                crossings.append((float(np.interp(threshold,q[a-1:a+1],t[a-1:a+1])),
                    float(np.interp(threshold,q[b:b+2][::-1],t[b:b+2][::-1]))))
            start=min(x[0] for x in crossings); end=max(x[1] for x in crossings)
            tt=np.r_[start,t[(t>start)&(t<end)],end]
            window_baseline=min(baseline, min(float(np.interp(tt,t,waves[state]).min()) for state in states))
            if baseline-window_baseline > 1e-8:
                raise ValueError('Substantive negative excess in shared window')
            stats={}
            for state in states:
                excess=np.interp(tt,t,waves[state])-window_baseline
                if excess.min()<0: raise ValueError(f'Negative excess in shared window: {state}, minimum={excess.min():.12g}, fraction={fraction}; no clipping allowed')
                stats[state]=distribution(tt,excess)
            for pair in family_pairs:
                p,c=stats[parent],stats[pair['child_state']]
                rows.append(dict(**pair,threshold_fraction=fraction,baseline_cms=window_baseline,
                    baseline_roundoff_adjustment_cms=baseline-window_baseline,
                    start_seconds=start,end_seconds=end,
                    centroid_shift_hours=(c['centroid_seconds']-p['centroid_seconds'])/3600,
                    duration80_change_pct=100*(c['duration80_hours']/p['duration80_hours']-1),
                    volume_change_pct=100*(c['volume_m3']/p['volume_m3']-1)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); out=args.output
    settings=json.loads(CONFIG.read_text())
    sources=[Path(__file__),CONFIG,Path(cunge.__file__),ROOT/'RAPID/src/rapid_tools/hydrograph_response.py',
        ROOT/'RAPID/src/rapid_tools/common_grid_production.py',DEFAULT_BASED_MODEL_PATH,
        ROOT/'synthetic_experiments/src/synthetic_experiments/networks.py',
        ROOT/'synthetic_experiments/src/synthetic_experiments/design.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['source_sha256'] != hashes:
            raise ValueError('Inputs/code changed: choose a new output directory')
    save(out/'manifest.json',dict(reference=settings['reference'],settings=settings,source_sha256=hashes,status='running'))
    all_rows=[]; checks=[]
    for name, changes in settings['cases'].items():
        parameters=dict(settings['base_geometry'],**changes)
        frames,pairs,depth=prepare(parameters,settings)
        dt=shared_clock(list(frames.values()),[settings['report_seconds']],settings['maximum_native_dt_seconds'])
        tail=settings['initial_tail_hours']
        while True:
            t,flow,waves,audits,settled=run_group(frames,settings,dt,tail)
            if settled:break
            tail*=2
            if tail>settings['maximum_tail_hours']:raise ValueError(f'Unsettled tail: {name}')
        rows=responses(t,waves,pairs,settings)
        for row in rows:row['configuration']=name
        all_rows.extend(rows)
        folder=out/name;folder.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(time_seconds=t,inflow_cms=flow,**waves)).to_parquet(folder/'hydrographs.parquet',index=False)
        save(folder/'audit.json',dict(parameters=parameters,reference_depth_m=depth,dt_seconds=dt,tail_hours=tail,states=audits))
        # Two independent resolution perturbations, same physical geometry/window.
        variants={'half_dt':(frames,dt/2),'double_sections':({k:v.assign(count=v['count']*2,
            rapid_effective_length_m=v.rapid_effective_length_m/2) for k,v in frames.items()},dt)}
        for variant,(fs,clock) in variants.items():
            if variant=='double_sections':
                fs={k:cunge.parameters(v,settings['n_manning'],beta=settings['beta']) for k,v in fs.items()}
                clock=shared_clock(list(fs.values()),[settings['report_seconds']],dt)
            _,_,ws,qa,ok=run_group(fs,settings,clock,tail)
            rr=responses(t,ws,pairs,settings)
            delta=max(abs(a['centroid_shift_hours']-b['centroid_shift_hours'])*60 for a,b in zip(rows,rr))
            checks.append(dict(configuration=name,check=variant,max_centroid_shift_difference_minutes=delta,
                max_wave_difference_pct_amplitude=100*max(float(np.max(np.abs(ws[k]-waves[k]))) for k in waves)/(settings['peak_cms']-settings['baseflow_cms']),settled=bool(ok)))
        max_volume=max(abs(a['excess_volume_error_fraction']) for a in audits.values())
        checks.append(dict(configuration=name,check='full_tail_volume',max_abs_error_fraction=max_volume,passed=bool(max_volume<1e-4)))
        print(name, 'states',len(frames),'dt',dt,'tail',tail,flush=True)
    pd.DataFrame(all_rows).to_csv(out/'responses.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'numerical_checks.csv',index=False)
    passed=all(c.get('passed',True) and c.get('settled',True) and c.get('max_centroid_shift_difference_minutes',0)<1 and c.get('max_wave_difference_pct_amplitude',0)<.1 for c in checks)
    save(out/'completion.json',dict(complete=True,numerical_screen_passed=bool(passed),
        comparisons=len(all_rows)//len(settings['window_fractions']),predictor_transfer_complete=False,
        note='Pilot only. <1 minute response difference, <0.1% amplitude waveform difference, <1e-4 full-tail volume error are screening tolerances, not observational uncertainty.'))
    save(out/'manifest.json',dict(reference=settings['reference'],settings=settings,source_sha256=hashes,status='complete' if passed else 'numerical_screen_failed'))
    if not passed: raise ValueError('Pilot numerical screen failed; inspect numerical_checks.csv')

if __name__=='__main__':main()
