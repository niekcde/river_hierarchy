"""41 preselected chains; frozen held-example settings, 10/25% training mass."""
import argparse,copy,json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from combination_information_pilot import ROOT,OUT,EXT,pilot,pair,b,COLUMNS,fit,calculate,response_window,response_rows,batch,shared_clock,sha,save
DEST=OUT/'chain_augmentation'
METRICS=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','volume_change_pct','asymmetry_change']

def prepare():
    DEST.mkdir(exist_ok=True);path=DEST/'routing_plan.json'
    if path.exists():
        doc=json.loads(path.read_text());batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256']);return path
    selected=pilot.read(OUT/'physical_context_and_combinations/proposed_scale_extension.csv')
    assert len(selected)==41 and selected.example.nunique()==14
    assert not selected.duplicated(['example','states']).any() and not selected.duplicates_segmented.any()
    blocks=pilot.read(pilot.OUT/'eligible_blocks.json')
    cases={c['example']:c for plan in [pilot.read(EXT/'routing_plan.json'),pilot.read(OUT/'routing_plan.json')] for c in plan['cases']}
    newcases=[];features=[];records=[]
    for ex,selection in selected.groupby('example'):
        case=copy.deepcopy(cases[ex]);jobs={j['state_id']:j for j in case['jobs']}
        parent=pilot.read(jobs['S000_base']['parameters']);seg=pilot.read(jobs['all_closed_units_collapsed']['parameters'])
        lookup={r['state']:dict(r) for r in blocks[ex]};sections={}
        for name,bb in lookup.items():
            sections[name]=seg[seg.block_state.eq(name)].copy();assert len(sections[name])==1
            bb['child_source']=bb['source'];bb['child_sink']=bb['sink']
        case['jobs']=[copy.deepcopy(jobs['S000_base'])]
        current=OUT if ex in ['13','15','21','22','46','48','54'] else EXT
        done=pilot.read(current/f'runs/{ex}/completion.json');case['tail_hours']=done['final_tail_hours']
        case['simulation_end_utc']=(pd.Timestamp(case['window']['end_utc'])+pd.Timedelta(seconds=30,hours=case['tail_hours'])).isoformat()
        tracked={jobs['S000_base']['parameters']:sha(jobs['S000_base']['parameters']),case['forcing']:sha(case['forcing'])}
        for i,r in enumerate(selection.itertuples()):
            names=r.states.split(' | ');bs=[lookup[n] for n in names];state=f'chain_{i+1:02d}'
            child,lineage=pilot.compose(parent,bs,sections);reverse,_=pilot.compose(parent,list(reversed(bs)),sections);pd.testing.assert_frame_equal(child,reverse)
            ids=set(i for bb in bs for i in bb['parent_ids']);cids=set(lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'])
            m=pair(parent,child,ids,cids);s=parent[parent.id_link.isin(ids)];w=s.reference_flow_fraction*s.link_length_m
            m.update(example=ex,state=state,cohort='combined',parent_slope_slow=float(np.average(1/np.sqrt(s.slope),weights=w)),length_delta_km=(m['B_child_route_length_mean']-m['B_parent_route_length_mean'])/1000)
            assert np.isfinite([m[c] for c in COLUMNS]).all();np.testing.assert_allclose(m['domain_size'],r.edited_parent_km,rtol=1e-10)
            features.append(m);folder=DEST/'parameters'/ex/state;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'parameters.parquet';child.to_parquet(target,index=False);lineage.to_csv(folder/'lineage.csv',index=False)
            dt=shared_clock([child],[30,60],min(jobs['S000_base']['native_dt_seconds'],jobs['all_closed_units_collapsed']['native_dt_seconds']))
            batch.equilibrium_check(child,dt)
            case['jobs'].append(dict(state_id=state,parameters=str(target),native_dt_seconds=dt,aliases=[state]));tracked[str(target)]=sha(target)
            records.append(dict(example=ex,state=state,components=names,blocks=len(names),edited_parent_km=m['domain_size'],target_fraction=r.target_fraction,actual_fraction=r.actual_fraction))
        case['source_sha256']=tracked;newcases.append(case);print(ex,'prepared',len(selection),'chains',flush=True)
    pd.DataFrame(features).to_parquet(DEST/'predictors.parquet',index=False);save(DEST/'variants.json',records)
    save(path,dict(schema='reviewed-event-window-batch-v2',source_sha256=pilot.SOURCES,code_sha256={str(Path(pilot.__file__)):sha(pilot.__file__),**batch.code_sources()},preparation_script_sha256=sha(__file__),cases=newcases))
    return path

def responses(path):
    doc=json.loads(path.read_text());batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256'])
    uc=pd.read_parquet(OUT/'review/uncapped_state_metrics.parquet');old=pd.read_parquet(EXT/'state_metrics.parquet')
    old=pd.concat([old[~old.example.isin(uc.example.unique())],uc],ignore_index=True)
    rows=[];windows=[];reproduction=[];changed=set()
    for case in doc['cases']:
        ex=case['example'];done=json.loads((DEST/f'runs/{ex}/completion.json').read_text());batch.verify(done['files'])
        folder=Path(done['final_directory']);wave=pd.read_parquet(folder/'forcing.parquet');t=wave.time_utc.astype('int64').to_numpy()/1e9;ii=np.flatnonzero(wave.phase.eq('input_event'))
        curves={j['state_id']:pd.read_parquet(folder/j['state_id']/'outlet.parquet') for j in case['jobs']}
        refs=old[old.example.eq(ex)];base=min(float(refs.baseline_cms.min()),min(float(f.q_outlet_cms.iloc[ii[0]:].min()) for f in curves.values()))
        ref=refs[(refs.state=='S000_base')&(refs.window=='0.5%')].iloc[0];original=pd.read_parquet(ref.outlet)
        z=original.merge(curves['S000_base'],on='time_utc',suffixes=('_old','_new'));np.testing.assert_allclose(z.q_outlet_cms_old,z.q_outlet_cms_new,rtol=1e-12,atol=1e-8)
        reproduction.append(dict(example=ex,max_discharge_difference=float(abs(z.q_outlet_cms_old-z.q_outlet_cms_new).max())))
        for window in ['0.5%','1%','2%']:
            subset=refs[refs.window.eq(window)];ref=subset.iloc[0];start,end=float(ref.rise_seconds),float(ref.fall_seconds);tol=float(ref.tolerance_cms)
            for f in curves.values():
                assert f.time_utc.equals(wave.time_utc)
                a,z=response_window(f.q_outlet_cms.to_numpy(),ii[0],ii[-1],case['initial_cms'],case['final_cms'],tol);start=min(start,t[a]);end=max(end,t[z])
            change=start!=ref.rise_seconds or end!=ref.fall_seconds or base!=ref.baseline_cms
            windows.append(dict(example=ex,window=window,earlier_start_seconds=ref.rise_seconds-start,later_end_seconds=end-ref.fall_seconds,baseline_change_cms=base-ref.baseline_cms,changed=change))
            if change:
                changed.add((ex,window))
                for record in subset.to_dict('records'):rows.append(calculate(record,start,end,base,window))
            else:rows.extend(subset.to_dict('records'))
            for state,f in curves.items():
                if state=='S000_base':continue
                target=folder/state/'outlet.parquet';rows.append(calculate(dict(example=ex,state=state,outlet=str(target),outlet_sha256=sha(target),tolerance_cms=tol),start,end,base,window))
        print(ex,'responses calculated',flush=True)
    rows.extend(old[~old.example.isin([c['example'] for c in doc['cases']])].to_dict('records'))
    st=pd.DataFrame(rows);assert not st.duplicated(['example','state','window']).any();rr=response_rows(st)
    st.to_parquet(DEST/'state_metrics.parquet',index=False);rr.to_parquet(DEST/'responses.parquet',index=False)
    pd.DataFrame(windows).to_csv(DEST/'window_audit.csv',index=False);pd.DataFrame(reproduction).to_csv(DEST/'parent_reproduction.csv',index=False)
    primary=rr[rr.window.eq('0.5%')];new=pd.read_parquet(DEST/'predictors.parquet').merge(primary[['example','state',*METRICS]],on=['example','state'],validate='one_to_one',how='left',indicator=True)
    assert len(new)==41 and new._merge.eq('both').all();new=new.drop(columns='_merge');new.to_parquet(DEST/'combined_input.parquet',index=False)
    base=pd.read_parquet(OUT/'augmented_tests/input.parquet');revised=base.drop(columns=METRICS,errors='ignore').merge(primary[['example','state',*METRICS]],on=['example','state'],how='left',validate='one_to_one',indicator=True)
    assert len(revised)==2185 and revised._merge.eq('both').all();revised=revised.drop(columns='_merge')
    np.testing.assert_allclose(base[COLUMNS],revised[COLUMNS],rtol=0,atol=0)
    revisions=base[['example','state',*METRICS]].merge(revised[['example','state',*METRICS]],on=['example','state'],suffixes=('_old','_new'),validate='one_to_one')
    for m in METRICS:revisions[m+'_difference']=revisions[m+'_new']-revisions[m+'_old']
    revisions.to_parquet(DEST/'existing_target_changes.parquet',index=False);revised.to_parquet(DEST/'evaluation_input.parquet',index=False)
    assert np.isfinite(new[COLUMNS+METRICS].to_numpy()).all() and np.isfinite(revised[COLUMNS+METRICS].to_numpy()).all()
    save(DEST/'response_check.json',dict(combined_matched=41,existing_matched=2185,unmatched=0,changed_windows=sorted(changed),max_centroid_target_change_h=float(abs(revisions.centroid_shift_hours_difference).max())))

def weighted_fit(old,new,params,fraction):
    if fraction==0:return fit(old,params,4),[]
    train=pd.concat([old,new],ignore_index=True);y=train.centroid_shift_hours.to_numpy();w=np.zeros(len(train));audit=[]
    for ex in sorted(train.example.unique()):
        orig=train.example.eq(ex)&train.cohort.ne('combined');aug=train.example.eq(ex)&train.cohort.eq('combined')
        mass=fraction if aug.any() else 0
        w[orig]=(1-mass)/orig.sum()
        if aug.any():w[aug]=mass/aug.sum()
        assert np.isclose(w[train.example.eq(ex)].sum(),1)
        audit.append(dict(training_example=ex,old_rows=int(orig.sum()),combined_rows=int(aug.sum()),combined_mass=mass,total_mass=float(w[train.example.eq(ex)].sum())))
    w/=w.mean();pre=b.engine.Preprocess(COLUMNS,None,None).fit(train,y)
    mean=float(np.average(y,weights=w));sd=float(np.sqrt(np.average((y-mean)**2,weights=w)))
    model=RandomForestRegressor(**params,random_state=b.engine.SEED,n_jobs=4)
    model.fit(pre.transform(train),(y-mean)/sd,sample_weight=w)
    return dict(preprocessor=pre,estimator=model,target_mean=mean,target_sd=sd,columns=COLUMNS),audit

def models():
    old=pd.read_parquet(DEST/'evaluation_input.parquet');new=pd.read_parquet(DEST/'combined_input.parquet');choices=json.loads((OUT/'nine_input_tuning/selections.json').read_text())
    files=[DEST/'evaluation_input.parquet',DEST/'combined_input.parquet',OUT/'nine_input_tuning/selections.json',Path(__file__),Path(b.__file__),Path(b.engine.__file__),Path(b.forest.__file__)]
    design=dict(sources={str(f):sha(f) for f in files},target='centroid_shift_hours',columns=COLUMNS,fractions=[0,.1,.25],seed=b.engine.SEED,validation='15 whole-example holdouts. Reuse each fold same_grid selected settings, no tuning. All original and segmented test rows identical across treatments. Combined tests reported separately.',weighting='Equal total mass per training network; combinations share 10/25 percent, previous rows share remainder. Network37 has no eligible long-chain candidate and keeps original mass1.',limitation='Exploratory reused networks; correlated augmentation is not independent new physical evidence.',training_counts=dict(existing=len(old),combined=len(new)))
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();save(DEST/'design.json',design)
    rows=[];extra=[];weights=[];checks=[]
    reference=pd.read_parquet(OUT/'nine_input_tuning/predictions.parquet').query("variant=='same_grid'")
    unchanged=json.loads((DEST/'response_check.json').read_text())['max_centroid_target_change_h']==0
    for ex in sorted(old.example.unique()):
        train,test=old[old.example.ne(ex)],old[old.example.eq(ex)];aug=new[new.example.ne(ex)];extest=new[new.example.eq(ex)]
        assert ex not in set(train.example)|set(aug.example)
        params=next(c['params'] for c in choices if c['held']==ex and c['variant']=='same_grid')
        for fraction,label in [(0,'baseline'),(.1,'aug10'),(.25,'aug25')]:
            checkpoint=DEST/'folds'/ex/f'{label}.json';result=b.forest.checked(checkpoint,fp)
            if result is None:
                model,audit=weighted_fit(train,aug,params,fraction);pred=b.forest.predict(model,test)
                if fraction==0 and unchanged:
                    z=test[['example','state']].merge(reference,on=['example','state'],validate='one_to_one');np.testing.assert_allclose(pred,z.predicted_hours,atol=1e-12,rtol=1e-12)
                pred_extra=b.forest.predict(model,extest) if len(extest) else []
                result=dict(fingerprint=fp,held=ex,label=label,params=params,weights=audit,
                    rows=[dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(p)) for r,p in zip(test.itertuples(),pred)],
                    extra=[dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(p)) for r,p in zip(extest.itertuples(),pred_extra)])
                save(checkpoint,result)
            rows.extend(dict(variant=label,**r) for r in result['rows']);extra.extend(dict(variant=label,**r) for r in result['extra']);weights.extend(dict(held=ex,variant=label,**r) for r in result['weights'])
            print(ex,label,'saved',flush=True)
    p=pd.DataFrame(rows);p.to_parquet(DEST/'predictions.parquet',index=False);pd.DataFrame(extra).to_parquet(DEST/'combined_predictions.parquet',index=False);pd.DataFrame(weights).to_csv(DEST/'weight_audit.csv',index=False)
    assert len(p)==2185*3 and not p.duplicated(['variant','example','state']).any()
    scores=[];networks=[]
    for (variant,cohort),g in p.groupby(['variant','cohort']):
        for target in ['centroid_shift_hours','peak_shift_hours']:
            for subset,z in [('all',g),('without46',g[g.example.ne('46')]),('without54',g[g.example.ne('54')])]:
                y,pr,w=z[target].to_numpy(),z.predicted_hours.to_numpy(),b.engine.weights(z)
                scores.append(dict(variant=variant,cohort=cohort,target=target,subset=subset,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
        for ex,z in g.groupby('example'):
            y,pr=z.centroid_shift_hours.to_numpy(),z.predicted_hours.to_numpy()
            networks.append(dict(variant=variant,cohort=cohort,example=ex,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,np.ones(len(z)),1),mae_minutes=float(np.mean(abs(pr-y))*60),mse_minutes2=float(np.mean((pr-y)**2)*3600)))
    s=pd.DataFrame(scores);s.to_csv(DEST/'scores.csv',index=False);pd.DataFrame(networks).to_csv(DEST/'network_scores.csv',index=False);p[p.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
    for file,h in design['sources'].items():assert sha(file)==h
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,combined_states=41,combined_networks=14,held_networks=15,fits=45,production_modified=False,retuned=False,baseline_reproduced=unchanged))
    print(s.query("subset=='all' and target=='centroid_shift_hours'").to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--models-only',action='store_true');args=parser.parse_args()
    path=prepare()
    if args.run:
        if not args.models_only:batch.run(path,DEST/'runs');responses(path)
        models()
