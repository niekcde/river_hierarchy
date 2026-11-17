"""Freeze the inflow-based discharge convention and three-run design; no routing."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
DEST=OUT/'three_run_setup_20261028'
DRIVE=Path('/Volumes/PhD')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    sources={}
    def read(p):
        p=Path(p);sources[str(p)]=sha(p)
        return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text())
    prep=read(OUT/'manning_all_networks_preparation_20261027/run_specification.json')
    windows=read(OUT/'manning_all_networks_preparation_20261027/response_windows.parquet')
    slopepath=ROOT/'RAPID/outputs/compound_member_slopes_final_20261028/staged_slope_index.json'
    slopes=read(slopepath)
    for entry in slopes['cases']:
        for field in ['slope_file','contract_file','audit_file','accepted_geometry_candidate']:
            if field in entry:
                p=entry[field];sources[p]=sha(p)
        assert sources[entry['slope_file']]==entry['slope_sha256']
        if 'accepted_geometry_candidate' in entry:
            assert sources[entry['accepted_geometry_candidate']]==entry['accepted_geometry_sha256']
    rows=[];parameter_bytes=0;estimated_outlet_bytes=0
    for case in prep['cases']:
        ex=case['example'];force=read(case['forcing']);spin=force[force.phase.eq('spinup')];event=force[force.phase.eq('input_event')]
        q0=float(spin.forcing_cms.iloc[0]);peak=float(event.forcing_cms.max());mid=q0+.5*(peak-q0)
        assert np.allclose(spin.forcing_cms,q0,rtol=0,atol=1e-9)
        assert np.isclose(q0,event.forcing_cms.iloc[0]) and peak>q0
        parent=next(j for j in case['jobs'] if j['state']=='S000_base')
        f=read(parent['source_parameters']);roots=set(f.u)-set(f.v)
        assert len(roots)==1
        legacy=float(f.loc[f.u.isin(roots),'reference_discharge_cms'].sum())
        w=windows[(windows.example==ex)&(windows.window=='0.5%')].iloc[0]
        outlet=Path(w.outlet);sources[str(outlet)]=sha(outlet)
        estimated_outlet_bytes+=outlet.stat().st_size*len(case['jobs'])
        for j in case['jobs']:parameter_bytes+=Path(j['source_parameters']).stat().st_size
        rows.append(dict(example=ex,spinup_cms=q0,inflow_event_peak_cms=peak,midpoint_Qref_cms=mid,
            legacy_routing_Qref_cms=legacy,legacy_equals_selected_event_peak=bool(np.isclose(legacy,peak,rtol=1e-10,atol=1e-8)),
            peak_utc=str(event.loc[event.forcing_cms.idxmax(),'time_utc']),
            event_start_utc=str(event.interval_start_utc.iloc[0]),event_end_utc=str(event.interval_end_utc.iloc[-1]),
            forcing_path=case['forcing'],forcing_sha256=sources[case['forcing']],states=len(case['jobs'])))
    values=pd.DataFrame(rows)
    assert len(values)==15 and int(values.states.sum())==2215
    # Full outlet series + recalculated parameters + metrics/receipts; retain source inputs in place.
    estimated_one=(estimated_outlet_bytes+1.2*parameter_bytes+2215*20000+100*2**20)/2**30
    free=shutil.disk_usage(DRIVE).free/2**30
    budget=20.;reserve=20.
    space=dict(drive=str(DRIVE),free_GiB=free,estimated_one_run_GiB=estimated_one,
        estimated_three_runs_GiB=3*estimated_one,working_budget_GiB=budget,
        minimum_free_reserve_GiB=reserve,space_sufficient=free>=budget+reserve,
        estimate_basis='Saved parent outlet bytes per example times number of states, parameter bytes with 20% uplift, metrics/receipts and calibration allowance. Outputs are outlet-only, no internal-reach time series.',
        recheck_before_launch=True)
    assert space['space_sufficient'],space
    design=dict(status='frozen_design_not_started',networks=15,states_per_run=2215,accepted_collapses_per_run=2185,
        reference_discharge=dict(base='Prescribed constant upstream spinup, equal to first selected input-event discharge',
            peak='Maximum upstream forcing during input_event only, excluding padding and other events',
            midpoint='Qspinup + 0.5*(Qpeak_inflow - Qspinup)',
            freeze='Same event-level values for BASED and Manning and all representations; never recompute from model output',
            allocation='Allocate the frozen network reference discharge through each accepted topology; separately audit changes in retained-reach allocations'),
        runs=[dict(id='01_based_slope_geometry_correction',depth='Existing BASED reference anchor and width scaling',
                Qref='Exact legacy_routing_Qref_cms; do not silently substitute the newly frozen selected-event peak',
                calibration='Retain current n/beta only after verifying parent hydraulics and forcing unchanged; otherwise flag for review',
                purpose='Isolate staged slope and accepted geometry correction relative to saved BASED'),
              dict(id='02_based_midpoint',depth='Recompute BASED reference-depth anchor using frozen midpoint Qref, then apply existing width scaling',
                Qref='midpoint_Qref_cms',calibration='Refit n and beta on each full-complexity reference only; freeze across collapses',purpose='Measure reference-discharge effect against run01'),
              dict(id='03_manning_midpoint',depth='h=(n*Qref_reach/(B*sqrt(S)))^(3/5); recompute at each trial n, fixed over event',
                Qref='midpoint_Qref_cms',calibration='Refit n and beta on each full-complexity reference only; freeze across collapses',purpose='Compare depth formulations against run02')],
        output_root='/Volumes/PhD/river_hierarchy/three_run_comparison_20261028',
        shared=dict(staged_slope_index=str(slopepath),staged_slope_index_sha256=sha(slopepath),
            staged_scope='1467 compound members; other state types require an explicit input-policy check before launch',
            staged_fixed_cases=1413,staged_boundary_flex_cases=54,
            geometry='Use accepted geometry candidates where indexed; retain declared length/reversal and boundary-flex exceptions',
            calibration='Same reviewed timing/soft-height targets and bounds across runs02/03; n [0.01,0.1], beta [0.5,2]; no child fitting',
            forcing='Same selected events, initial steady flow and settling criteria',
            responses='Six existing metrics and original common baseline / 0.5%,1%,2% windows',
            numerics='Recompute K/X and check spatial discretization and stable timestep for every parameterization; shared spatial grid where feasible, representative timestep and grid convergence checks',
            outputs='Parameters, outlet hydrographs, metrics, calibration trials, receipts and comparison reports',
            checks='Hash validation, complete geometry/identity correspondence, declared exterior-change exceptions, flow conservation, positive depths/slopes, nonnegative discharge and settled tails'),
        launch_gates=['Resolve retained island/other input policy: compound-member index is not an all-state slope correction',
            'Validate indexed slope/geometry merges and all explicit boundary/length exceptions',
            'Verify run01 reference is unchanged; audit calibration target provenance',
            'Recheck drive free space; retain at least 20 GiB reserve'],
        routing_started=False,production_modified=False,RF_fits=0,space=space)
    if DEST.exists():raise FileExistsError('Frozen setup already exists; do not overwrite it silently')
    DEST.mkdir()
    values.to_csv(DEST/'frozen_reference_discharges.csv',index=False)
    (DEST/'run_setup.json').write_text(json.dumps(design,indent=2)+'\n')
    (DEST/'space_check.json').write_text(json.dumps(space,indent=2)+'\n')
    sources[str(Path(__file__).resolve())]=sha(__file__)
    for p,h in sources.items():assert sha(p)==h
    (DEST/'manifest.json').write_text(json.dumps(dict(frozen=True,source_hashes=sources,
        output_hashes={str(p):sha(p) for p in DEST.iterdir() if p.is_file()},routing_started=False),indent=2)+'\n')
    print(values[['example','spinup_cms','inflow_event_peak_cms','midpoint_Qref_cms','legacy_routing_Qref_cms','legacy_equals_selected_event_peak']].to_string(index=False))
    print(json.dumps(space,indent=2))

if __name__=='__main__':main()
