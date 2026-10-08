"""Lag-only synthetic model selection, frozen holdout/extension/full-real tests."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error
sys.path.insert(0,str(Path(__file__).parent))
from full_population_lag_search import OUT,ROOT,FEATURES,CANDIDATES,RESPONSES,model,enrich
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.refreshed_analysis import interpolated_peak
SOURCE=ROOT/'RAPID/validation/deduplicated_response_benchmark_20261005'
EXT=ROOT/'synthetic_experiments/outputs/hydraulic_alpha065_20260930/embedded_width_length_extension'

def extension():
    s=pd.read_parquet(EXT/'predictor_validation/full_features.parquet');rows=[]
    for rid in sorted(set(s.parent_run_id)|set(s.child_run_id)):
        folder=EXT/'runs'/rid;h=pd.read_parquet(folder/'hydrograph.parquet');m=json.loads((folder/'metrics.json').read_text());onset=round(m['buffer_hours']*3600/m['dt_seconds'])*m['dt_seconds'];peak,status=interpolated_peak(h.time_seconds,h.outlet_cms)
        assert status=='ok';rows.append(dict(run_id=rid,peak=peak-onset,input=float(h.time_seconds.iloc[h.inlet_cms.argmax()])-onset))
    peaks=pd.DataFrame(rows).set_index('run_id');p=peaks.reindex(s.parent_run_id).reset_index(drop=True);c=peaks.reindex(s.child_run_id).reset_index(drop=True);np.testing.assert_allclose(p.input,c.input,rtol=0,atol=1e-8)
    s['interpolated_peak_shift_hours']=(c.peak-p.peak)/3600;s['interpolated_lag_shift_pct']=100*(c.peak-p.peak)/(p.peak-p.input)
    s=s[~s.is_original_grid_anchor].drop_duplicates(['parent_run_id','child_run_id','domain_mode','domain_start_m','domain_end_m']).copy();s.to_parquet(OUT/'extension_features.parquet',index=False)
    return enrich(s)

def main():
    s=enrich(pd.read_parquet(SOURCE/'synthetic_matched_features.parquet'));train=s[s.split.eq('development')];test=s[s.split.eq('test')];ext=extension();real=enrich(pd.read_parquet(OUT/'features.parquet'));audit=[];choices=[];scores=[];predictions=[]
    assert not set(train.geometry_group)&set(test.geometry_group)
    splits=list(GroupKFold(4).split(train,groups=train.geometry_group))
    for response in RESPONSES:
        ranks=[]
        for name,kind in CANDIDATES:
            cols=FEATURES[name];errors=[]
            for a,b in splits:
                t=train.iloc[a];v=train.iloc[b];assert not set(t.geometry_group)&set(v.geometry_group)
                fit=model(kind).fit(t[cols],t[response]);pred=fit.predict(v[cols]);errors.extend(pd.DataFrame({'group':v.geometry_group.to_numpy(),'error':abs(pred-v[response].to_numpy())}).groupby('group').error.mean().tolist())
            loss=float(np.mean(errors));ranks.append((loss,len(cols),name,kind));audit.append(dict(response=response,feature_set=name,model=kind,grouped_development_mae=loss))
        loss,_,name,kind=min(ranks);choices.append(dict(response=response,feature_set=name,model=kind,grouped_development_mae=loss))
        # Freeze the development-selected model and retain readable fixed baselines.
        for label,fn,mk in [('selected',name,kind),('omega','omega','boosted_7'),('components','components','boosted_7'),('selected_six','selected_six','boosted_7')]:
            fit=model(mk).fit(train[FEATURES[fn]],train[response])
            for dataset,frame in [('synthetic_holdout',test),('novel_embedding_extension',ext),('full_real_transfer',real)]:
                pred=fit.predict(frame[FEATURES[fn]]);obs=frame[response].to_numpy();mae=mean_absolute_error(obs,pred);zero=abs(obs).mean()
                scores.append(dict(response=response,model_label=label,feature_set=fn,model=mk,dataset=dataset,n=len(frame),r2=r2_score(obs,pred),mae=mae,zero_mae=zero,skill_vs_zero=1-mae/zero))
                detail=pd.DataFrame(dict(response=response,model_label=label,dataset=dataset,example_id=frame.example_id.to_numpy(),plot_category=frame.plot_category.to_numpy(),observed=obs,predicted=pred));predictions.append(detail)
        print(response,name,kind,'complete',flush=True)
    pd.DataFrame(audit).to_csv(OUT/'synthetic_inner_scores.csv',index=False);pd.DataFrame(choices).to_csv(OUT/'synthetic_selections.csv',index=False);pd.DataFrame(scores).to_csv(OUT/'synthetic_transfer_scores.csv',index=False);pd.concat(predictions).to_parquet(OUT/'synthetic_transfer_predictions.parquet',index=False)
    (OUT/'synthetic_provenance.json').write_text(json.dumps(dict(development_rows=len(train),holdout_rows=len(test),novel_unique_extension_rows=len(ext),extension_raw_rows=980,original_anchor_rows_excluded=63,full_real_rows=len(real),candidates=len(CANDIDATES),selection='4 grouped geometry folds inside original development only; equal geometry-group MAE. No extension, original test or real responses enter selection.',peaks='Extension re-extracted with same guarded outlet interpolation and common-input-clock checks; spinup removed per run',limits='Extension and real tests are exploratory previously examined populations, not untouched confirmatory validation. Wider feature/model search only; no new routing.'),indent=2))
    print(pd.DataFrame(scores).query("model_label=='selected'").to_string(index=False),flush=True)

if __name__=='__main__':main()
