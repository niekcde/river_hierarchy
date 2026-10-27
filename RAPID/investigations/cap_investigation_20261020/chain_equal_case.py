"""Append existing chains using the baseline equal-case-within-network weights."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from chain_augmentation import DEST as SOURCE,OUT,b,COLUMNS,fit,sha,save
DEST=OUT/'chain_equal_case'

def main():
    DEST.mkdir(exist_ok=True)
    assert json.loads((SOURCE/'completion.json').read_text())['complete']
    files=[SOURCE/'evaluation_input.parquet',SOURCE/'combined_input.parquet',SOURCE/'predictions.parquet',SOURCE/'design.json',OUT/'nine_input_tuning/selections.json',Path(__file__),Path(b.__file__),Path(b.engine.__file__),Path(b.forest.__file__)]
    sources={str(f):sha(f) for f in files}
    old=pd.read_parquet(files[0]);new=pd.read_parquet(files[1]);previous=pd.read_parquet(files[2]);choices=json.loads(files[4].read_text())
    assert len(old)==2185 and len(new)==41 and new.example.nunique()==14
    design=dict(sources=sources,columns=COLUMNS,seed=b.engine.SEED,target='centroid_shift_hours',weighting='Baseline fit function on concatenated old/new rows: equal total weight per network and equal weight per case within network. No reserved augmentation share.',validation='15 whole-example holdouts; frozen same_grid fold settings; existing 2185 test cases unchanged. Historical baseline/10/25 predictions reused after target/key verification.',limitations='Exploratory reused networks; one seed, no retuning, no new independent networks.',production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();save(DEST/'design.json',design)
    rows=[];extras=[];weights=[]
    for ex in sorted(old.example.unique()):
        tr=pd.concat([old[old.example.ne(ex)],new[new.example.ne(ex)]],ignore_index=True);te=old[old.example.eq(ex)];extra=new[new.example.eq(ex)]
        assert ex not in set(tr.example) and tr.example.nunique()==14
        w=b.engine.weights(tr);tr=tr.copy();tr['audit_weight']=w
        totals=tr.groupby('example').audit_weight.sum();np.testing.assert_allclose(totals,totals.iloc[0])
        for net,g in tr.groupby('example'):
            np.testing.assert_allclose(g.audit_weight,g.audit_weight.iloc[0]);m=g.cohort.eq('combined')
            weights.append(dict(held=ex,training_example=net,old_rows=int((~m).sum()),new_rows=int(m.sum()),combined_mass=float(g.loc[m,'audit_weight'].sum()/g.audit_weight.sum()),network_weight=float(g.audit_weight.sum())))
        params=next(c['params'] for c in choices if c['held']==ex and c['variant']=='same_grid')
        path=DEST/'folds'/f'{ex}.json';result=b.forest.checked(path,fp)
        if result is None:
            model=fit(tr,params,4);pred=b.forest.predict(model,te);px=b.forest.predict(model,extra) if len(extra) else []
            def records(frame,pred):
                return [dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(p)) for r,p in zip(frame.itertuples(),pred)]
            result=dict(fingerprint=fp,held=ex,params=params,rows=records(te,pred),extra=records(extra,px));save(path,result)
        rows.extend(dict(variant='equal_case',**r) for r in result['rows']);extras.extend(dict(variant='equal_case',**r) for r in result['extra']);print(ex,'saved',flush=True)
    p=pd.concat([previous,pd.DataFrame(rows)],ignore_index=True)
    assert len(p)==2185*4 and not p.duplicated(['variant','example','state']).any()
    for variant,g in p.groupby('variant'):
        z=old[['example','state','centroid_shift_hours','peak_shift_hours']].merge(g,on=['example','state'],validate='one_to_one',suffixes=('_old','_new'))
        assert len(z)==2185
        for c in ['centroid_shift_hours','peak_shift_hours']:np.testing.assert_allclose(z[c+'_old'],z[c+'_new'],atol=0,rtol=0)
    p.to_parquet(DEST/'predictions.parquet',index=False);pd.DataFrame(extras).to_parquet(DEST/'combined_predictions.parquet',index=False);pd.DataFrame(weights).to_csv(DEST/'weight_audit.csv',index=False)
    scores=[];nets=[]
    for (variant,cohort),g in p.groupby(['variant','cohort']):
        for target in ['centroid_shift_hours','peak_shift_hours']:
            for subset,z in [('all',g),('without46',g[g.example.ne('46')]),('without54',g[g.example.ne('54')])]:
                y,pr,w=z[target].to_numpy(),z.predicted_hours.to_numpy(),b.engine.weights(z)
                scores.append(dict(variant=variant,cohort=cohort,target=target,subset=subset,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
        for ex,z in g.groupby('example'):
            y,pr=z.centroid_shift_hours.to_numpy(),z.predicted_hours.to_numpy()
            nets.append(dict(variant=variant,cohort=cohort,example=ex,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,np.ones(len(z)),1),mae_minutes=float(np.mean(abs(pr-y))*60)))
    s=pd.DataFrame(scores);s.to_csv(DEST/'scores.csv',index=False);n=pd.DataFrame(nets);n.to_csv(DEST/'network_scores.csv',index=False)
    delta=n[n.variant.eq('equal_case')].merge(n[n.variant.eq('baseline')],on=['example','cohort'],suffixes=('','_baseline'),validate='one_to_one')
    for c in ['relative_error_pct','mae_minutes']:delta[c+'_difference']=delta[c]-delta[c+'_baseline']
    delta.to_csv(DEST/'network_changes.csv',index=False);p[p.cohort.eq('segmented')].to_csv(DEST/'segmented_predictions.csv',index=False)
    s.query("target=='centroid_shift_hours' and subset=='all'").groupby('variant').relative_error_pct.mean().rename('equal_cohort_objective_pct').to_csv(DEST/'objective.csv')
    for file,h in sources.items():assert sha(file)==h
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,fits=15,existing_test_cases=2185,combined_training_candidates=41,production_modified=False,retuned=False))
    print(s.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False))
if __name__=='__main__':main()
