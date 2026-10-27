"""Prepare an isolated, hashed 15-network Manning run specification; never route.

Current child inputs are inventory sources, not approved local-intervention
inputs. The boundary-fix work must provide reviewed replacement inputs and a
complete retained-link correspondence before a routing launcher is enabled.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from manning_calibrated_network import ROOT, OUT, BUNDLE, digest, parameters

DEST=OUT/'manning_all_networks_preparation_20261027'
RUN_ROOT=Path('/Volumes/PhD/river_hierarchy/manning_reference_depth_runs')


def manning_parameters(source,n,beta):
    if not .01<=n<=.1 or not .5<=beta<=2:
        raise ValueError('Parameters outside the declared calibration bounds')
    f=source.copy();f['diffusion_multiplier']=beta
    return parameters(f,n)


def verify_exterior(parent,child,mapping):
    """Check an explicit reviewed exterior map, not inferred matching coverage.

    The caller must establish map completeness against the reviewed edit
    footprint. A successful field comparison alone does not certify coverage.
    """
    if not {'parent_link','child_link'}<=set(mapping):raise ValueError('Missing exterior map columns')
    if mapping.parent_link.duplicated().any() or mapping.child_link.duplicated().any():
        raise ValueError('Exterior map must be one-to-one')
    a=parent.set_index('id_link').loc[mapping.parent_link]
    b=child.set_index('id_link').loc[mapping.child_link]
    changes={}
    fields=['link_length_m','width','slope','reference_discharge_cms','reference_flow_fraction',
            'split_fraction','count','rapid_effective_length_m','n_manning','diffusion_multiplier',
            'depth_m','rapid_celerity_mps','rapid_k','rapid_x']
    for field in fields:
        mask=~np.isclose(a[field].to_numpy(),b[field].to_numpy(),rtol=1e-10,atol=1e-12)
        if mask.any():changes[field]=int(mask.sum())
    if changes:raise ValueError('Exterior hydraulic contract failed: '+json.dumps(changes))
    return dict(compared_links=len(mapping),fields=fields,
                scope='Field checks only; map completeness and boundary topology require separate approval')


def main():
    DEST.mkdir(exist_ok=True);hashes={}
    def read(p):
        p=Path(p);hashes[str(p)]=digest(p)
        if p.suffix=='.parquet':return pd.read_parquet(p)
        return json.loads(p.read_text())
    jobs={};cases={}
    for p in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',
              ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json',OUT/'routing_plan.json']:
        for c in read(p)['cases']:
            cases[c['example']]=c
            for j in c['jobs']:jobs[c['example'],j['state_id']]=j
    inventory=read(OUT/'validation_group_inventory_20261023/records.parquet')
    old=read(ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/state_metrics.parquet')
    new=read(OUT/'review/uncapped_state_metrics.parquet')
    metrics=pd.concat([old[~old.example.isin(new.example)],new],ignore_index=True)
    timing={r['example']:r for r in read(BUNDLE/'timing_targets.json')['records']}
    height={r['example']:r for r in read(BUNDLE/'height_targets.json')['records']}
    protocol=read(BUNDLE/'protocol.json');calroot=Path(protocol['source_directory'])
    planned=[];summary=[];all_windows=[]
    for ex,g in inventory.groupby('example'):
        ids=['S000_base',*g.state.tolist(),'single_centerline']
        assert len(ids)==len(set(ids))
        parent_windows=metrics[metrics.example.eq(ex)&metrics.state.eq('S000_base')]
        assert set(parent_windows.window)=={'0.5%','1%','2%'} and len(parent_windows)==3
        forcing=Path(parent_windows.iloc[0].outlet).parents[1]/'forcing.parquet'
        force=read(forcing)
        all_windows.append(parent_windows)
        waves=calroot/ex/'60s.parquet';meta=calroot/ex/'60s.json'
        read(waves);read(meta)
        parent=read(jobs[ex,'S000_base']['parameters'])
        trial=manning_parameters(parent,float(parent.n_manning.iloc[0]),float(parent.diffusion_multiplier.iloc[0]))
        assert np.isfinite(trial.depth_m).all() and trial.depth_m.gt(0).all()
        # Parent-depth sanity check only: inherited n is not a new calibration.
        entries=[]
        for state in ids:
            j=jobs[ex,state];p=Path(j['parameters']);hashes[str(p)]=digest(p)
            entries.append(dict(state=state,canonical_state='fully_collapsed' if state=='single_centerline' else state,
                source_parameters=str(p),source_sha256=hashes[str(p)],native_dt_seconds=j['native_dt_seconds'],
                aliases=j.get('aliases',[]),role='reference' if state=='S000_base' else 'control' if state=='single_centerline' else 'accepted_collapse',
                corrected_parameters=None,reviewed_exterior_map=None))
        planned.append(dict(example=ex,jobs=entries,forcing=str(forcing),forcing_sha256=hashes[str(forcing)],
            calibration_forcing=str(waves),calibration_metadata=str(meta),timing_target=timing[ex],height_target=height[ex],
            final_cms=cases[ex]['final_cms'],tolerance_cms=cases[ex]['tolerance_cms'],
            calibration_status='pending_reference_only_fit',boundary_contract_status='pending_other_chat_review'))
        summary.append(dict(example=ex,states=len(ids),accepted_collapses=len(g),
            parent_depth_min_at_old_n=trial.depth_m.min(),parent_depth_max_at_old_n=trial.depth_m.max(),
            forcing_rows=len(force),input_files_verified=True))
    assert len(planned)==15 and sum(len(c['jobs'])-2 for c in planned)==2185
    pd.concat(all_windows).to_parquet(DEST/'response_windows.parquet',index=False)
    pd.DataFrame(summary).to_csv(DEST/'preflight.csv',index=False)
    design=dict(status='prepared_not_routing_ready',run_root=str(RUN_ROOT),networks=15,
        states=sum(len(c['jobs']) for c in planned),accepted_collapses=2185,
        depth='Manning normal depth at allocated reference discharge; fixed through the event',
        calibration=dict(scope='Full-complexity reference only; recompute depth at each n trial',n_bounds=[.01,.1],
            beta_bounds=[.5,2.],beta_profile=[.5,.75,1.,1.5,2.],
            selection='Timing-first, soft height distance; preserve beta=1 candidate for optional sensitivity run',
            note='No network22 calibration is transferred to other examples; verify targets against current accepted calibration before fitting'),
        checks=['Manning discharge identity','reference-only calibration and held child n/beta',
                'reviewed complete exterior map and fixed exterior hydraulic fields',
                'boundary connectivity and conserved flow allocation',
                'stable native timestep for new K/X, plus representative half-timestep checks',
                'nonnegative flows and settled tails','same frozen baseline and 0.5/1/2% windows',
                'six existing response metrics against each model own parent','source/output hashes and safe resume'],
        blocked_until=['Corrected hydraulic inputs and explicit footprint/exterior correspondence reviewed in boundary-fix chat',
                       'Reference calibration and routing clock validation complete'],
        RF_fits=0,production_modified=False,routing_started=False,
        exclusions=['combination augmentation','internal-reach discharge time series'],cases=planned)
    (DEST/'run_specification.json').write_text(json.dumps(design,indent=2,default=str)+'\n')
    for p in [Path(__file__),Path(__file__).with_name('manning_calibrated_network.py'),Path(__file__).with_name('manning_network_pilot.py')]:
        hashes[str(p)]=digest(p)
    for p,h in hashes.items():assert digest(p)==h
    (DEST/'manifest.json').write_text(json.dumps(dict(preparation_complete=True,routing_ready=False,
        source_hashes=hashes,outputs={str(p):digest(p) for p in [DEST/'run_specification.json',DEST/'preflight.csv',DEST/'response_windows.parquet']}),indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False));print('Prepared',design['states'],'states; no routing started.')


if __name__=='__main__':main()
