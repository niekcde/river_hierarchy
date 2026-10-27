"""41 preselected chains; frozen held-example settings, 10/25% training mass."""
import argparse,copy,json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from combination_information_pilot import ROOT,OUT,EXT,pilot,pair,b,COLUMNS,fit,calculate,response_window,response_rows,batch,shared_clock,sha,save
DEST=OUT/'chain80_tuning'
METRICS=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','volume_change_pct','asymmetry_change']

def prepare():
    DEST.mkdir(exist_ok=True);path=DEST/'routing_plan.json'
    if path.exists():
        doc=json.loads(path.read_text());batch.verify(doc['source_sha256']);batch.verify(doc['code_sha256']);return path
    selected=select80()
    assert len(selected)==80 and selected.example.nunique()==15
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
    assert len(new)==80 and new._merge.eq('both').all();new=new.drop(columns='_merge');new.to_parquet(DEST/'combined_input.parquet',index=False)
    base=pd.read_parquet(OUT/'augmented_tests/input.parquet');revised=base.drop(columns=METRICS,errors='ignore').merge(primary[['example','state',*METRICS]],on=['example','state'],how='left',validate='one_to_one',indicator=True)
    assert len(revised)==2185 and revised._merge.eq('both').all();revised=revised.drop(columns='_merge')
    np.testing.assert_allclose(base[COLUMNS],revised[COLUMNS],rtol=0,atol=0)
    revisions=base[['example','state',*METRICS]].merge(revised[['example','state',*METRICS]],on=['example','state'],suffixes=('_old','_new'),validate='one_to_one')
    for m in METRICS:revisions[m+'_difference']=revisions[m+'_new']-revisions[m+'_old']
    revisions.to_parquet(DEST/'existing_target_changes.parquet',index=False);revised.to_parquet(DEST/'evaluation_input.parquet',index=False)
    assert np.isfinite(new[COLUMNS+METRICS].to_numpy()).all() and np.isfinite(revised[COLUMNS+METRICS].to_numpy()).all()
    save(DEST/'response_check.json',dict(combined_matched=80,existing_matched=2185,unmatched=0,changed_windows=sorted(changed),max_centroid_target_change_h=float(abs(revisions.centroid_shift_hours_difference).max())))


def select80():
    inventory=pilot.read(OUT/'physical_context_and_combinations/combination_inventory.csv')
    inventory=inventory[~inventory.duplicates_segmented].copy()
    physical=pilot.read(OUT/'physical_context_and_combinations/segmented_physical_context.csv').set_index('example')
    inventory['actual_fraction']=[r.edited_parent_km/physical.loc[r.example,'edited_parent_km'] for r in inventory.itertuples()]
    initial=pilot.read(OUT/'physical_context_and_combinations/proposed_scale_extension.csv')
    chosen=initial.to_dict('records');used={(r['example'],r['states']) for r in chosen}
    while len(chosen)<80:
        eligible=[]
        for ex,g in inventory.groupby('example'):
            candidates=g[[ (ex,x) not in used for x in g.states ]]
            if len(candidates):eligible.append((sum(r['example']==ex for r in chosen),ex,candidates))
        _,ex,g=min(eligible,key=lambda x:(x[0],x[1]))
        previous=[r for r in chosen if r['example']==ex]
        def novelty(row):
            if not previous:return row.actual_fraction
            members=set(row.states.split(' | '))
            return min(abs(row.actual_fraction-r['actual_fraction'])+.25*(1-len(members&set(r['states'].split(' | ')))/len(members|set(r['states'].split(' | ')))) for r in previous)
        g=g.copy();g['novelty']=g.apply(novelty,axis=1)
        r=g.sort_values(['novelty','states'],ascending=[False,True]).iloc[0].to_dict()
        r['target_fraction']=r['actual_fraction'];r['selection_reason']='Largest minimum geometric fraction/constituent-set distance; balance counts across networks.'
        chosen.append(r);used.add((ex,r['states']))
    result=pd.DataFrame(chosen);result.to_csv(DEST/'selection80.csv',index=False)
    assert len(used)==80 and set(zip(initial.example,initial.states))<=used
    return result

def tuning():
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.model_selection import GroupKFold
    import joblib
    old=pd.read_parquet(DEST/'evaluation_input.parquet');new=pd.read_parquet(DEST/'combined_input.parquet')
    data=pd.concat([old,new],ignore_index=True)
    prior=json.loads((OUT/'nine_input_tuning/design.json').read_text());candidates=prior['candidates'];assert len(candidates)==6
    original_choices=json.loads((OUT/'nine_input_tuning/selections.json').read_text())
    sourcefiles=[DEST/'evaluation_input.parquet',DEST/'combined_input.parquet',DEST/'selection80.csv',OUT/'nine_input_tuning/design.json',OUT/'nine_input_tuning/selections.json',OUT/'nine_input_tuning/predictions.parquet',Path(__file__),Path(b.__file__),Path(b.engine.__file__),Path(b.forest.__file__)]
    design=dict(sources={str(f):sha(f) for f in sourcefiles},candidates=candidates,columns=COLUMNS,seed=b.engine.SEED,target='centroid_shift_hours',validation='15 outer whole-network holdouts; 3 inner GroupKFold folds; equal weight per case within network and equal total network weight.',objective='Mean original and segmented cohort network-balanced relative errors with one-minute floor. Combined cohort reported but not added to tuning objective, preserving previous objective.',selection='All previous41 plus39 geometry-only distinct contiguous chains. No observed response/error selection.',report='All six inner candidates per outer fold; same-grid first-four control and expanded-six winner; frozen-fold control. Existing and combined held-out cohorts separate; pooled is secondary.',limitations='Exploratory reused networks. Correlated added states are not independent rivers. Inner candidate scores are tuning results, not unbiased test scores.',production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();save(DEST/'design.json',design)
    b.COLUMNS=list(COLUMNS)
    def direct_fit(tr,params):return b.fit(tr,params,jobs=1)
    def objective(frame,pred):
        by={}
        for cohort,g in frame.assign(prediction=pred).groupby('cohort'):
            by[cohort]=b.forest.sweep.relative_score(g.centroid_shift_hours.to_numpy(),g.prediction.to_numpy(),b.engine.weights(g),1)
        return (by['original']+by['segmented'])/2,by
    def tune(tr,label):
        folds=list(GroupKFold(3).split(tr,groups=tr.example))
        def trial(item):
            i,c=item;path=DEST/'inner'/label/f'{i:02d}.json';saved=b.forest.checked(path,fp)
            if saved is None:
                predicted=np.full(len(tr),np.nan);groups=[]
                for ti,vi in folds:
                    train,val=tr.iloc[ti],tr.iloc[vi];assert set(train.example).isdisjoint(val.example)
                    model=direct_fit(train,c['params']);predicted[vi]=b.forest.predict(model,val)
                    groups.append(dict(train=sorted(train.example.unique()),validation=sorted(val.example.unique())))
                assert np.isfinite(predicted).all();score,by=objective(tr,predicted)
                saved=dict(fingerprint=fp,index=i,name=c['name'],params=c['params'],score=score,cohort_scores=by,groups=groups)
                save(path,saved)
            print(label,c['name'],round(saved['score'],3),flush=True);return saved
        with ThreadPoolExecutor(max_workers=3) as pool:ranked=list(pool.map(trial,enumerate(candidates)))
        return sorted(ranked,key=lambda r:(r['score'],r['index']))
    def records(frame,pred):
        return [dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(p)) for r,p in zip(frame.itertuples(),pred)]
    rows=[];sweep=[];selections=[];folds=[]
    reference=pd.read_parquet(OUT/'nine_input_tuning/predictions.parquet')
    assert json.loads((DEST/'response_check.json').read_text())['max_centroid_target_change_h']==0,'Historical controls need retuning on changed targets'
    for ex in sorted(old.example.unique()):
        tr=data[data.example.ne(ex)].reset_index(drop=True);te=data[data.example.eq(ex)].copy();ranked=tune(tr,ex)
        sweep.extend(dict(held=ex,**r) for r in ranked)
        # Report the complete predeclared grid on held-out examples as well.
        # These scores do not select the nested winner or the final artifact.
        def outer_candidate(item):
            i,c=item;path=DEST/'outer'/ex/f'aug_{i:02d}.json'
            result=b.forest.checked(path,fp)
            if result is None:
                model=direct_fit(tr,c['params'])
                result=dict(fingerprint=fp,rows=records(te,b.forest.predict(model,te)))
                save(path,result)
            return i,result
        with ThreadPoolExecutor(max_workers=3) as pool:
            outer=list(pool.map(outer_candidate,enumerate(candidates)))
        for i,result in outer:
            rows.extend(dict(variant=f'candidate_{i:02d}',**r) for r in result['rows'])
        choices={'retuned4':next(r for r in ranked if r['index']<4),'retuned6':ranked[0]}
        fixed=next(c for c in original_choices if c['held']==ex and c['variant']=='same_grid')
        choices['fixed']=dict(index=next(i for i,c in enumerate(candidates) if c['params']==fixed['params']),params=fixed['params'],name=fixed['name'])
        for label,c in choices.items():
            path=DEST/'outer'/ex/f'aug_{c["index"]:02d}.json';saved=b.forest.checked(path,fp)
            if saved is None:
                model=direct_fit(tr,c['params']);saved=dict(fingerprint=fp,rows=records(te,b.forest.predict(model,te)));save(path,saved)
            rows.extend(dict(variant='aug80_'+label,**r) for r in saved['rows']);selections.append(dict(held=ex,variant=label,**c))
        for label,previous in [('baseline4','same_grid'),('baseline6','expanded_grid')]:
            choice=next(c for c in original_choices if c['held']==ex and c['variant']==previous)
            path=DEST/'outer'/ex/f'baseline_{choice["index"]:02d}.json';saved=b.forest.checked(path,fp)
            if saved is None:
                model=direct_fit(old[old.example.ne(ex)],choice['params']);pred=b.forest.predict(model,te)
                z=te[['example','state']].assign(pred=pred).merge(reference[reference.variant.eq(previous)],on=['example','state'],validate='one_to_one')
                assert len(z)==len(old[old.example.eq(ex)]);np.testing.assert_allclose(z.pred,z.predicted_hours,rtol=1e-12,atol=1e-12)
                saved=dict(fingerprint=fp,rows=records(te,pred));save(path,saved)
            rows.extend(dict(variant=label,**r) for r in saved['rows'])
        folds.append(dict(held=ex,training_rows=len(tr),training_networks=tr.example.nunique(),test_original=int(te.cohort.eq('original').sum()),test_segmented=int(te.cohort.eq('segmented').sum()),test_combined=int(te.cohort.eq('combined').sum())))
        save(DEST/'selections.json',selections);print('HELD',ex,'complete',flush=True)
    p=pd.DataFrame(rows);assert len(p)==2265*11 and not p.duplicated(['variant','example','state']).any();p.to_parquet(DEST/'predictions.parquet',index=False)
    pd.DataFrame(folds).to_csv(DEST/'fold_audit.csv',index=False)
    sweep_rows=[]
    for r in sweep:sweep_rows.append(dict(held=r['held'],index=r['index'],name=r['name'],score=r['score'],**r['cohort_scores']))
    pd.DataFrame(sweep_rows).to_csv(DEST/'full_inner_sweep.csv',index=False)
    pd.DataFrame(sweep_rows).groupby(['index','name'])[['score','original','segmented','combined']].mean().to_csv(DEST/'sweep_mean_inner_scores.csv')
    scores=[];networks=[]
    for variant,g in p.groupby('variant'):
        groups=list(g.groupby('cohort'))+[('existing_pooled',g[g.cohort.ne('combined')]),('all_pooled',g)]
        for cohort,h in groups:
            for target in ['centroid_shift_hours','peak_shift_hours']:
                for subset,z in [('all',h),('without46',h[h.example.ne('46')]),('without54',h[h.example.ne('54')])]:
                    y,pr=z[target].to_numpy(),z.predicted_hours.to_numpy();w=b.engine.weights(z)
                    scores.append(dict(variant=variant,cohort=cohort,target=target,subset=subset,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
            if cohort in ['original','segmented','combined']:
                for ex,z in h.groupby('example'):
                    y,pr=z.centroid_shift_hours.to_numpy(),z.predicted_hours.to_numpy()
                    networks.append(dict(variant=variant,cohort=cohort,example=ex,mae_minutes=float(np.mean(abs(pr-y))*60),relative_error_pct=b.forest.sweep.relative_score(y,pr,np.ones(len(z)),1)))
    scores=pd.DataFrame(scores);scores.to_csv(DEST/'scores.csv',index=False);pd.DataFrame(networks).to_csv(DEST/'network_scores.csv',index=False);p[p.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
    selected=tune(data,'final_all_data');save(DEST/'final_selection.json',selected)
    for name,choice in [('same_grid',next(r for r in selected if r['index']<4)),('expanded_grid',selected[0])]:
        artifact=direct_fit(data,choice['params']);joblib.dump(artifact,DEST/f'research_model_{name}.joblib',compress=3)
        restored=joblib.load(DEST/f'research_model_{name}.joblib');np.testing.assert_allclose(b.forest.predict(artifact,data),b.forest.predict(restored,data),rtol=1e-12,atol=1e-12)
        save(DEST/f'research_model_{name}.json',dict(choice=choice,columns=COLUMNS,training_rows=len(data),warning='Research artifact only; final selection score is not independent test performance.',sha256=sha(DEST/f'research_model_{name}.joblib')))
    for f,h in design['sources'].items():assert sha(f)==h
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,combined_cases=80,regular_cases=2170,segmented_cases=15,networks=15,production_modified=False,inner_candidate_evaluations=96,inner_fits=288,warning='Nested evaluation; six-setting exploratory sweep, not exhaustive RF optimization.'))
    print(scores.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--tune-only',action='store_true');args=parser.parse_args()
    path=prepare()
    if args.run:
        if not args.tune_only:batch.run(path,DEST/'runs');responses(path)
        tuning()
