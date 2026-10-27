"""Two closed-chain experiments: actual joint versus component routing and OOF RF."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/investigations/closed_unit_extension_20261019'),str(ROOT/'RAPID/investigations/closed_unit_pairing_20261019')]
from edited_features import pair
import pilot
from rapid_tools import event_window_batch as batch
from rapid_tools.common_grid_production import shared_clock
from responses import calculate
from analyze_pilot import response_window
from rapid_tools.hydrograph_response import response_rows
from tune_nine_inputs import fit,b,COLUMNS
OUT=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
DEST=OUT/'combination_information_pilot'
EXT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
sha,save=pilot.sha,pilot.save

def prepare():
    DEST.mkdir(exist_ok=True)
    path=DEST/'routing_plan.json'
    if path.exists():
        doc=json.loads(path.read_text());batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256']);return path
    selection=pilot.read(OUT/'physical_context_and_combinations/proposed_scale_extension.csv')
    selection=selection[selection.example.isin(['22','55'])&selection.target_fraction.eq(.5)]
    assert len(selection)==2
    blocks=pilot.read(pilot.OUT/'eligible_blocks.json')
    cases={c['example']:c for plan in [pilot.read(EXT/'routing_plan.json'),pilot.read(OUT/'routing_plan.json')] for c in plan['cases']}
    newcases=[];records=[];features=[];selections=[]
    for chosen in selection.itertuples():
        ex=chosen.example;case=copy.deepcopy(cases[ex]);jobs={j['state_id']:j for j in case['jobs']}
        parent=pilot.read(jobs['S000_base']['parameters']);seg=pilot.read(jobs['all_closed_units_collapsed']['parameters'])
        names=chosen.states.split(' | ');lookup={r['state']:dict(r) for r in blocks[ex]};sections={}
        for name in names:
            bb=lookup[name];sections[name]=seg[seg.block_state.eq(name)].copy();assert len(sections[name])==1
            # Segmented sections already use restored parent node identities.
            bb['child_source']=bb['source'];bb['child_sink']=bb['sink']
        n=len(names)//2
        variants=[('joint',names),('group_A',names[:n]),('group_B',names[n:])]+[('single_'+str(i).zfill(2),[name]) for i,name in enumerate(names)]
        case['jobs']=[copy.deepcopy(jobs['S000_base'])]
        done=pilot.read((OUT if ex=='22' else EXT)/f'runs/{ex}/completion.json')
        case['tail_hours']=done['final_tail_hours']
        case['simulation_end_utc']=(pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30,hours=case['tail_hours'])).isoformat()
        tracked={jobs['S000_base']['parameters']:sha(jobs['S000_base']['parameters']),case['forcing']:sha(case['forcing'])}
        for state,group in variants:
            bs=[lookup[name] for name in group];child,lineage=pilot.compose(parent,bs,sections)
            rev,_=pilot.compose(parent,list(reversed(bs)),sections);pd.testing.assert_frame_equal(child,rev)
            ids=set(i for bb in bs for i in bb['parent_ids']);cids=set(lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'])
            m=pair(parent,child,ids,cids);s=parent[parent.id_link.isin(ids)];w=s.reference_flow_fraction*s.link_length_m
            m.update(example=ex,state=state,parent_slope_slow=float(np.average(1/np.sqrt(s.slope),weights=w)),length_delta_km=(m['B_child_route_length_mean']-m['B_parent_route_length_mean'])/1000)
            assert np.isfinite([m[c] for c in COLUMNS]).all();features.append(m)
            folder=DEST/'parameters'/ex/state;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'parameters.parquet';child.to_parquet(target,index=False);lineage.to_csv(folder/'lineage.csv',index=False)
            dt=shared_clock([child],[30,60],min(jobs['S000_base']['native_dt_seconds'],jobs['all_closed_units_collapsed']['native_dt_seconds']))
            batch.equilibrium_check(child,dt)
            case['jobs'].append(dict(state_id=state,parameters=str(target),native_dt_seconds=dt,aliases=[state]))
            tracked[str(target)]=sha(target)
            records.append(dict(example=ex,state=state,components=group,blocks=len(group),edited_parent_km=m['domain_size']))
            if state=='joint':case['jobs'].append(dict(state_id='joint_half_dt',parameters=str(target),native_dt_seconds=dt/2,aliases=['joint_half_dt']))
        case['source_sha256']=tracked;newcases.append(case)
        selections.append(dict(example=ex,components=names,selection='Previously geometry-selected nearest 50% edited length; contrasting hydraulic contexts, exploratory not representative.',edited_parent_km=chosen.edited_parent_km))
    pd.DataFrame(features).to_parquet(DEST/'predictors.parquet',index=False);save(DEST/'variants.json',records);save(DEST/'selection.json',selections)
    code={str(Path(__file__)):sha(__file__),str(Path(pilot.__file__)):sha(pilot.__file__),**batch.code_sources()}
    save(path,dict(schema='reviewed-event-window-batch-v2',source_sha256=pilot.SOURCES,code_sha256=code,cases=newcases))
    return path

def analyze(path):
    doc=json.loads(path.read_text());batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256'])
    old=pd.concat([pd.read_parquet(EXT/'state_metrics.parquet').query("example=='55'"),pd.read_parquet(OUT/'review/uncapped_state_metrics.parquet').query("example=='22'")])
    rows=[];windows=[];replications=[];numerical=[]
    for case in doc['cases']:
        ex=case['example'];done=json.loads((DEST/f'runs/{ex}/completion.json').read_text());batch.verify(done['files'])
        folder=Path(done['final_directory']);wave=pd.read_parquet(folder/'forcing.parquet');t=wave.time_utc.astype('int64').to_numpy()/1e9;ii=np.flatnonzero(wave.phase.eq('input_event'))
        curves={j['state_id']:pd.read_parquet(folder/j['state_id']/'outlet.parquet') for j in case['jobs']}
        refs=old[old.example.eq(ex)];base=float(refs.baseline_cms.min())
        assert min(f.q_outlet_cms.iloc[ii[0]:].min() for f in curves.values())>=base-1e-9
        ref=refs[(refs.state=='S000_base')&(refs.window=='0.5%')].iloc[0]
        original=pd.read_parquet(ref.outlet);z=original.merge(curves['S000_base'],on='time_utc',suffixes=('_old','_new'))
        np.testing.assert_allclose(z.q_outlet_cms_old,z.q_outlet_cms_new,rtol=1e-12,atol=1e-8)
        replications.append(dict(example=ex,max_discharge_difference=float(abs(z.q_outlet_cms_old-z.q_outlet_cms_new).max())))
        for window in ['0.5%','1%','2%']:
            ref=refs[refs.window.eq(window)].iloc[0];start,end=float(ref.rise_seconds),float(ref.fall_seconds);tol=float(ref.tolerance_cms)
            for f in curves.values():
                assert f.time_utc.equals(wave.time_utc)
                a,z=response_window(f.q_outlet_cms.to_numpy(),ii[0],ii[-1],case['initial_cms'],case['final_cms'],tol)
                start=min(start,t[a]);end=max(end,t[z])
            windows.append(dict(example=ex,window=window,earlier_start_seconds=ref.rise_seconds-start,later_end_seconds=end-ref.fall_seconds,baseline_cms=base))
            for state,f in curves.items():
                target=folder/state/'outlet.parquet'
                rows.append(calculate(dict(example=ex,state=state,outlet=str(target),outlet_sha256=sha(target),tolerance_cms=tol),start,end,base,window))
    st=pd.DataFrame(rows);rr=response_rows(st);st.to_parquet(DEST/'state_metrics.parquet',index=False);rr.to_parquet(DEST/'responses.parquet',index=False)
    pd.DataFrame(windows).to_csv(DEST/'window_audit.csv',index=False);pd.DataFrame(replications).to_csv(DEST/'parent_reproduction.csv',index=False)
    metrics=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','volume_change_pct','asymmetry_change']
    interactions=[]
    for (ex,window),g in rr.groupby(['example','window']):
        g=g.set_index('state');singles=g[g.index.str.startswith('single_')]
        for metric in metrics:
            joint=float(g.loc['joint',metric]);a=float(g.loc['group_A',metric]);bb=float(g.loc['group_B',metric]);total=float(singles[metric].sum())
            interactions.append(dict(example=ex,window=window,response=metric,joint=joint,sum_individuals=total,sum_two_groups=a+bb,interaction_individuals=joint-total,interaction_groups=joint-a-bb,half_dt_difference=float(g.loc['joint_half_dt',metric]-joint)))
    pd.DataFrame(interactions).to_csv(DEST/'routing_additivity.csv',index=False)
    # Reconstruct current held-example models. No new observations enter training.
    data=pd.read_parquet(OUT/'augmented_tests/input.parquet');choices=json.loads((OUT/'nine_input_tuning/selections.json').read_text())
    x=pd.read_parquet(DEST/'predictors.parquet');predictions=[];model_checks=[]
    saved=pd.read_parquet(OUT/'nine_input_tuning/predictions.parquet').query("variant=='same_grid'")
    for ex,g in x.groupby('example'):
        choice=next(c for c in choices if c['held']==ex and c['variant']=='same_grid')
        model=fit(data[data.example.ne(ex)],choice['params'],4)
        check=data[data.example.eq(ex)];pr=b.forest.predict(model,check)
        z=check[['example','state']].merge(saved,on=['example','state'],validate='one_to_one')
        np.testing.assert_allclose(pr,z.predicted_hours,rtol=1e-12,atol=1e-12)
        model_checks.append(dict(example=ex,max_prediction_reproduction_error_h=float(abs(pr-z.predicted_hours.to_numpy()).max())))
        pred=b.forest.predict(model,g)
        for r,p in zip(g.itertuples(),pred):predictions.append(dict(example=ex,state=r.state,predicted_hours=float(p)))
        print(ex,'held-network predictions finished',flush=True)
    pred=pd.DataFrame(predictions);primary=rr[rr.window.eq('0.5%')];pred=pred.merge(primary,on=['example','state'],validate='one_to_one');pred.to_parquet(DEST/'predictions.parquet',index=False)
    summary=[]
    for ex,g in pred.groupby('example'):
        g=g.set_index('state');s=g[g.index.str.startswith('single_')];joint=g.loc['joint'];groups=g.loc[['group_A','group_B']]
        summary.append(dict(example=ex,actual_joint_h=joint.centroid_shift_hours,actual_sum_individuals_h=s.centroid_shift_hours.sum(),actual_sum_groups_h=groups.centroid_shift_hours.sum(),predicted_joint_h=joint.predicted_hours,predicted_sum_individuals_h=s.predicted_hours.sum(),predicted_sum_groups_h=groups.predicted_hours.sum(),joint_error_h=joint.predicted_hours-joint.centroid_shift_hours))
    summary=pd.DataFrame(summary);summary.to_csv(DEST/'summary.csv',index=False);pd.DataFrame(model_checks).to_csv(DEST/'model_reproduction.csv',index=False)
    batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256'])
    save(DEST/'completion.json',dict(complete=True,examples=['22','55'],training_augmented=False,production_modified=False,plan_sha256=sha(path),caveats=['Two deliberately contrasting examples, not a representative validation sample.','Summed percentage/shape changes are arithmetic diagnostics, not conserved quantities.','Only centroid predictions: no models trained for other response targets.']))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');args=p.parse_args();path=prepare()
    if args.run:batch.run(path,DEST/'runs');analyze(path)
