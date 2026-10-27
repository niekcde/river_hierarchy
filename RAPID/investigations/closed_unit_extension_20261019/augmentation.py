"""Fixed-setting grouped augmentation: no held-network rows enter training."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from geometry_audit import ROOT,OUT,pilot
import drop_exposure_check as baseline
engine=baseline.engine;forest=baseline.forest
VARIANTS={'original_only':[], 'plus_pairs':['pair'], 'plus_segmented':['segmented'], 'plus_both':['pair','segmented']}

def main():
    dest=OUT/'augmentation';dest.mkdir(exist_ok=True)
    original=pd.read_parquet(baseline.SOURCE/'input.parquet');responses=pd.read_parquet(OUT/'responses.parquet');responses=responses[responses.window=='0.5%']
    targets=['centroid_shift_hours','peak_shift_hours'];columns=['example','state',*baseline.COLUMNS,*targets]
    d=original.drop(columns=targets).merge(responses[['example','state',*targets]],on=['example','state'],how='left',validate='one_to_one')
    assert len(d)==2170 and d[targets].notna().all().all();d=d[columns].copy();d['cohort']='original'
    new=pd.read_parquet(OUT/'new_comparisons.parquet');new=new[new.window.eq('0.5%')][columns+['cohort']]
    assert len(new)==21 and new.cohort.value_counts().to_dict()=={'segmented':15,'pair':6}
    all_rows=pd.concat([d,new],ignore_index=True);assert not all_rows.duplicated(['example','state']).any();assert np.isfinite(all_rows[baseline.COLUMNS+targets].to_numpy(float)).all()
    all_rows.to_parquet(dest/'input.parquet',index=False)
    info=json.loads((baseline.OUT/'final_model.json').read_text());params=info['params']
    files=[Path(__file__),Path(baseline.__file__),Path(engine.__file__),Path(forest.__file__),baseline.SOURCE/'input.parquet',baseline.OUT/'final_model.json',OUT/'responses.parquet',OUT/'edited_predictors.parquet',OUT/'new_comparisons.parquet']
    design=dict(variants=VARIANTS,columns=baseline.COLUMNS,params=params,seed=engine.SEED,target=forest.TARGET,weights='Each network equal total weight; each comparison within a network equal weight. Augmentation changes per-case weight within its network.',preprocessing='Fit on training fold only, including target standardization; absolute-error criterion with ccp_alpha retains the existing standardized-target convention.',validation='Hold out all comparisons from one network; evaluate original, pair and segmented separately. Shared updated response window for every comparison within each network.',qualifications=['Fixed settings previously selected on original data, no new nested tuning','Only six pairs from three deliberately chosen networks; pair results exploratory','Added cases do not add independent networks','Original seven predictor scopes retained; new combined cases exclude inter-block connectors and tails, not unchanged branches within existing outer compound scopes'],sources={str(f):engine.sha(f) for f in files})
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();pilot.save(dest/'design.json',design)
    predictions=[];folds=[]
    for variant,include in VARIANTS.items():
        for ex in sorted(d.example.unique()):
            tr=all_rows[all_rows.example.ne(ex)&all_rows.cohort.isin(['original',*include])];te=all_rows[all_rows.example.eq(ex)]
            assert ex not in set(tr.example) and set(tr.example).isdisjoint(set(te.example))
            checkpoint=dest/'checkpoints'/variant/f'{ex}.json';saved=forest.checked(checkpoint,fp)
            if saved is None:
                artifact=baseline.fit(tr,params);estimate=forest.predict(artifact,te)
                saved=dict(fingerprint=fp,variant=variant,held_example=ex,training_rows=len(tr),training_networks=int(tr.example.nunique()),training_target_sd_hours=float(artifact['target_sd']),predictions=[dict(example=r.example,state=r.state,cohort=r.cohort,predicted_hours=float(v),centroid_shift_hours=float(r.centroid_shift_hours),peak_shift_hours=float(r.peak_shift_hours)) for r,v in zip(te.itertuples(),estimate)])
                pilot.save(checkpoint,saved);print(variant,ex,'saved',flush=True)
            predictions.extend(dict(variant=variant,**r) for r in saved['predictions']);folds.append({k:v for k,v in saved.items() if k!='predictions'})
    p=pd.DataFrame(predictions);p.to_parquet(dest/'predictions.parquet',index=False);pd.DataFrame(folds).to_csv(dest/'fold_audit.csv',index=False)
    scores=[];networks=[]
    for variant,g in p.groupby('variant'):
        for response in targets:
            for cohort in ['original','pair','segmented']:
                group=g[g.cohort==cohort]
                for subset,z in [('all',group),('excluding_extreme37',group[~((group.example=='37')&(group.state=='CB001_bubble_3'))]),('excluding54',group[group.example!='54'])]:
                    if z.empty:continue
                    y=z[response].to_numpy();prediction=z.predicted_hours.to_numpy();w=engine.weights(z)
                    scores.append(dict(variant=variant,response=response,cohort=cohort,subset=subset,n=len(z),networks=int(z.example.nunique()),relative_error_pct=forest.sweep.relative_score(y,prediction,w,1),**engine.score(y*60,prediction*60,w)))
                for ex,z in group.groupby('example'):
                    y=z[response].to_numpy();prediction=z.predicted_hours.to_numpy();networks.append(dict(variant=variant,response=response,cohort=cohort,example=ex,n=len(z),relative_error_pct=forest.sweep.relative_score(y,prediction,np.ones(len(z)),1),mae_minutes=float(np.mean(abs(y-prediction))*60)))
    pd.DataFrame(scores).to_csv(dest/'scores.csv',index=False);pd.DataFrame(networks).to_csv(dest/'network_scores.csv',index=False)
    # Isolate how much response-window harmonization changes historical model scores.
    old=pd.read_parquet(baseline.OUT/'predictions.parquet');old=old[old.variant=='seven_inputs'];old=old[['example','state','predicted_hours']].merge(d[['example','state',*targets]],on=['example','state'],validate='one_to_one')
    historic=[]
    for target in targets:
        w=engine.weights(old);y=old[target].to_numpy();pred=old.predicted_hours.to_numpy();historic.append(dict(response=target,relative_error_pct=forest.sweep.relative_score(y,pred,w,1),**engine.score(y*60,pred*60,w)))
    pd.DataFrame(historic).to_csv(dest/'historical_predictions_new_targets.csv',index=False)
    print(pd.DataFrame(scores).query("response=='centroid_shift_hours' and subset=='all'").to_string(index=False))
if __name__=='__main__':main()
