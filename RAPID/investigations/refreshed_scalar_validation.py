"""Full-cohort scalar/component benchmark without path-enumeration exclusions."""
from refreshed_predictor_validation import *

def main():
 d=pd.read_parquet(OUT/'analysis.parquet');d=d[~d.technical_alias].drop_duplicates(['example_id','fingerprint','matched_base_link_ids','child_link_ids']).copy()
 d['omega']=d.len_fraction_width_weighted_log_lambda;d['resistance_log']=d.neg_log10_lambda;d['lag_shift_pct']=d.lag_to_inflow_peak_pct_shift_vs_base
 rows=[]
 for label,cols in {'resistance_only':[R],'length_fraction_only':[LF],'exposure_only':[E],'omega':['omega'],'components':[R,LF,E]}.items():
  for y in Y:
   valid=d[np.isfinite(d[cols+[y]]).all(axis=1)].reset_index(drop=True);obs=valid[y].to_numpy()
   for kind in ['linear','boosted']:
    pred=np.zeros(len(valid));null=pred.copy()
    for ex in valid.example_id.unique():
     test=valid.example_id.eq(ex);train=~test;fit=estimator(kind).fit(valid.loc[train,cols],valid.loc[train,y]);pred[test]=fit.predict(valid.loc[test,cols]);null[test]=valid.loc[train,y].mean()
    rows.extend(score_rows(obs,pred,valid,validation='full_cohort_leave_river_out',feature_set=label,model=kind,response=y))
   if label=='omega':rows.extend(score_rows(obs,null,valid,validation='full_cohort_leave_river_out',feature_set='training_mean',model='baseline',response=y))
 pd.DataFrame(rows).to_csv(OUT/'full_cohort_scores.csv',index=False)
 s=pd.read_csv(OUT/'scores.csv',keep_default_na=False);s.loc[s.model.eq('null'),'model']='baseline';s.to_csv(OUT/'scores.csv',index=False)
 print('Full-cohort unique comparisons',len(d),flush=True)
if __name__=='__main__':main()
