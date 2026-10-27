"""Segmented-only augmentation: fixed controls and nested grouped forest tuning."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
import augmentation as aug
b=aug.baseline
OUT=aug.OUT/'segmented_search'
COLUMNS=list(b.COLUMNS)
TARGET='centroid_shift_hours'

def objective(frame,pred):
    # Equal evaluation weight to cohorts and, within each cohort, networks.
    by={}
    for cohort in ['original','segmented']:
        mask=frame.cohort.eq(cohort).to_numpy();z=frame.loc[mask]
        if z.empty:raise ValueError('Both cohorts required')
        by[cohort]=b.forest.sweep.relative_score(z[TARGET].to_numpy(),np.asarray(pred)[mask],b.engine.weights(z),1.)
    return float(np.mean(list(by.values()))),by

def candidates(params):
    settings=[dict(params)]
    for criterion,leaf,split,depth,features in [
        ('absolute_error',1,2,None,1.),('absolute_error',2,2,16,1.),
        ('absolute_error',1,2,None,.75),('squared_error',1,2,None,1.),
        ('squared_error',2,2,16,1.),('squared_error',1,2,None,.75),
        ('squared_error',5,10,10,.75)]:
        settings.append(dict(params,n_estimators=200,criterion=criterion,min_samples_leaf=leaf,min_samples_split=split,max_depth=depth,max_features=features,ccp_alpha=0.))
    return [dict(size=size,params=p) for size in ['current','none','edited'] for p in settings]

def transformed(frame,size):
    frame=frame.copy()
    if size=='edited':frame['domain_size']=frame.domain_size_edited
    return frame

def fit(train,candidate,jobs):
    b.COLUMNS=[c for c in COLUMNS if candidate['size']!='none' or c!='domain_size']
    artifact=b.fit(transformed(train,candidate['size']),candidate['params'],jobs)
    artifact['size_definition']=candidate['size']
    return artifact

def predict(artifact,frame):return b.forest.predict(artifact,transformed(frame,artifact['size_definition']))

def records(frame,pred):
    return [dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,peak_shift_hours=r.peak_shift_hours,predicted_hours=float(v)) for r,v in zip(frame.itertuples(),pred)]

def tune(train,grids,folder,fp,jobs):
    folds=list(GroupKFold(3).split(train,groups=train.example));trials=[]
    for i,c in enumerate(grids):
        path=folder/f'{i:02d}.json';saved=b.forest.checked(path,fp)
        if saved is None:
            predictions=np.full(len(train),np.nan)
            for ti,vi in folds:
                tr,va=train.iloc[ti],train.iloc[vi];assert set(tr.example).isdisjoint(va.example)
                predictions[vi]=predict(fit(tr,c,jobs),va)
            assert np.isfinite(predictions).all()
            score,by=objective(train,predictions)
            saved=dict(fingerprint=fp,index=i,candidate=c,selection_score=score,cohort_scores=by)
            aug.pilot.save(path,saved)
        trials.append(saved)
        if (i+1)%8==0:print(folder.name,i+1,'/24 candidates',flush=True)
    return sorted(trials,key=lambda r:(r['selection_score'],r['index']))

def summarize(rows):
    p=pd.DataFrame(rows);p.to_parquet(OUT/'predictions.parquet',index=False);scores=[];net=[]
    for (variant,cohort),g in p.groupby(['variant','cohort']):
        for target in [TARGET,'peak_shift_hours']:
            for subset,z in [('all',g),('excluding22',g[g.example!='22']),('excluding54',g[g.example!='54'])]:
                w=b.engine.weights(z);y=z[target].to_numpy();pr=z.predicted_hours.to_numpy()
                scores.append(dict(variant=variant,cohort=cohort,target=target,subset=subset,n=len(z),relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
            for ex,z in g.groupby('example'):
                net.append(dict(variant=variant,cohort=cohort,target=target,example=ex,mae_minutes=float(abs(z.predicted_hours-z[target]).mean()*60)))
    pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False);pd.DataFrame(net).to_csv(OUT/'network_scores.csv',index=False)
    print(pd.DataFrame(scores).query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    source=aug.OUT/'size_comparison/input.parquet';d=pd.read_parquet(source);d=d[d.cohort.isin(['original','segmented'])].reset_index(drop=True)
    assert d.cohort.value_counts().to_dict()=={'original':2170,'segmented':15};assert d.example.nunique()==15 and not d.duplicated(['example','state']).any()
    assert d[d.cohort.eq('segmented')].groupby('example').size().eq(1).all()
    params=json.loads((b.OUT/'final_model.json').read_text())['params'];grids=candidates(params)
    design=dict(target=TARGET,rows=len(d),cohort_counts=d.cohort.value_counts().to_dict(),columns=COLUMNS,candidates=grids,
        selection='Mean of original and segmented cohort floored relative errors (1 minute); each cohort gives networks equal weight',
        training='Unchanged inverse-network-count sample weights and training-only target standardization. Cohort balancing applies to selection, not training weights.',
        validation='15 outer held-network folds, each with three grouped inner folds; jointly select size and forest settings using inner predictions only. Per-size inner winners also evaluated separately.',
        final='Select one configuration using 3-fold grouped CV on all data; fit separate research artifact. Its CV selection score is not a test score.',
        limitations=['Exploratory design informed by previous errors, not a new external test','Only 15 segmented outcomes, one per independent network','Eight forest configurations per size, targeted sweep not exhaustive'],
        sources={str(p):b.engine.sha(p) for p in [Path(__file__),Path(b.__file__),Path(b.forest.__file__),Path(b.engine.__file__),source,b.OUT/'final_model.json',aug.OUT/'augmentation/predictions.parquet']})
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (OUT/'design.json').exists() and json.loads((OUT/'design.json').read_text())!=design:raise ValueError('Search design changed; use new output')
    aug.pilot.save(OUT/'design.json',design);d.to_parquet(OUT/'input.parquet',index=False)
    print('2185 comparisons, no pairs; 24 candidates; 1152 inner fits + up to 45 outer fits + 30 fixed controls + 1 final fit.',flush=True)
    if not args.run:return
    rows=[];old=pd.read_parquet(aug.OUT/'augmentation/predictions.parquet')
    for oldvariant,name in [('original_only','original_training_reference'),('plus_segmented','fixed_current')]:
        z=old[old.variant.eq(oldvariant)&old.cohort.isin(['original','segmented'])].copy()
        keys=['example','state','cohort',TARGET,'peak_shift_hours'];pd.testing.assert_frame_equal(z[keys].sort_values(['example','state']).reset_index(drop=True),d[keys].sort_values(['example','state']).reset_index(drop=True),check_dtype=False)
        z['variant']=name;rows.extend(z.to_dict('records'))
    for size in ['none','edited']:
        for ex in sorted(d.example.unique()):
            path=OUT/'fixed'/size/f'{ex}.json';saved=b.forest.checked(path,fp)
            if saved is None:
                tr,te=d[d.example.ne(ex)],d[d.example.eq(ex)];artifact=fit(tr,dict(size=size,params=params),args.jobs)
                saved=dict(fingerprint=fp,rows=records(te,predict(artifact,te)));aug.pilot.save(path,saved)
            rows.extend(dict(variant='fixed_'+size,**r) for r in saved['rows'])
        print('Fixed',size,'done',flush=True)
    summarize(rows)
    selections=[]
    for ex in sorted(d.example.unique()):
        tr,te=d[d.example.ne(ex)],d[d.example.eq(ex)];ranked=tune(tr,grids,OUT/'inner'/ex,fp,args.jobs)
        chosen={size:next(r for r in ranked if r['candidate']['size']==size) for size in ['current','none','edited']};chosen['joint']=ranked[0]
        held={}
        for label,choice in chosen.items():
            i=choice['index'];path=OUT/'outer'/ex/f'{i:02d}.json';saved=b.forest.checked(path,fp)
            if saved is None:
                artifact=fit(tr,choice['candidate'],args.jobs);saved=dict(fingerprint=fp,rows=records(te,predict(artifact,te)));aug.pilot.save(path,saved)
            rows.extend(dict(variant='tuned_'+label,**r) for r in saved['rows']);selections.append(dict(held_example=ex,selection=label,**choice))
        pd.DataFrame(selections).to_json(OUT/'outer_selections.json',orient='records',indent=2);print('Outer',ex,'done, joint size',ranked[0]['candidate']['size'],flush=True)
    summarize(rows)
    ranked=tune(d,grids,OUT/'inner'/'all_data',fp,args.jobs);best=ranked[0];artifact=fit(d,best['candidate'],args.jobs);artifact['fingerprint']=fp
    path=OUT/'research_model.joblib';joblib.dump(artifact,path,compress=3);restored=joblib.load(path);np.testing.assert_allclose(predict(artifact,d),predict(restored,d),rtol=1e-12,atol=1e-12)
    aug.pilot.save(OUT/'research_model.json',dict(fingerprint=fp,selected=best,training_rows=len(d),warning='Research artifact; frozen production model unchanged; use nested scores for evaluation',sha256=b.engine.sha(path)))
    print('Completed; separate research artifact saved.',flush=True)
if __name__=='__main__':main()
