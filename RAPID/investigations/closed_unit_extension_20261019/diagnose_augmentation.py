"""Held-22 diagnostic; not a new validation or model-selection result."""
import json
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
import augmentation as aug
b=aug.baseline

def main():
 d=pd.read_parquet(OUT/'augmentation/input.parquet');params=json.loads((b.OUT/'final_model.json').read_text())['params'];dest=OUT/'augmentation/diagnostic22';dest.mkdir(exist_ok=True)
 tr=d[d.example.ne('22')];te=d[d.example.eq('22')&d.cohort.eq('original')];original=tr[tr.cohort.eq('original')];expanded=tr[tr.cohort.isin(['original','segmented'])]
 def sd(z):
  y=z[b.forest.TARGET].to_numpy();w=b.engine.weights(z);return np.sqrt(np.average((y-np.average(y,weights=w))**2,weights=w))
 results=[];pred=[];importance=[]
 for name,z,alpha in [('original',original,params['ccp_alpha']),('segmented',expanded,params['ccp_alpha']),('segmented_same_physical_alpha',expanded,params['ccp_alpha']*sd(original)/sd(expanded)),('segmented_no_pruning',expanded,0.)]:
  a=b.fit(z,dict(params,ccp_alpha=alpha));p=b.forest.predict(a,te);err=abs(p-te.centroid_shift_hours.to_numpy())*60
  results.append(dict(variant=name,alpha=alpha,target_sd=sd(z),mae_minutes=float(err.mean()),median_error_minutes=float(np.median(err)),max_error_minutes=float(err.max())))
  pred.extend(dict(variant=name,state=r.state,actual_hours=r.centroid_shift_hours,predicted_hours=float(v)) for r,v in zip(te.itertuples(),p))
  importance.extend(dict(variant=name,column=c,importance=float(v)) for c,v in zip(a['preprocessor'].kept,a['estimator'].feature_importances_))
  if name=='segmented':
   w=b.engine.weights(z);q=z[['example','state','cohort','centroid_shift_hours','domain_size']].copy();q['weight']=w;q[q.cohort.eq('segmented')].to_csv(dest/'added_training_weights.csv',index=False)
  print(results[-1],flush=True)
 pd.DataFrame(results).to_csv(dest/'controls.csv',index=False);pd.DataFrame(pred).to_parquet(dest/'predictions.parquet',index=False);pd.DataFrame(importance).to_csv(dest/'importance.csv',index=False)
 p=pd.read_parquet(OUT/'augmentation/predictions.parquet');p=p[p.cohort.eq('original')]
 rows=[]
 for variant,g in p.groupby('variant'):
  for subset,z in [('all',g),('excluding22',g[g.example!='22'])]:
   w=b.engine.weights(z);y=z.centroid_shift_hours.to_numpy();pr=z.predicted_hours.to_numpy();s=b.engine.score(y*60,pr*60,w)
   rows.append(dict(variant=variant,subset=subset,**s))
 pd.DataFrame(rows).to_csv(dest/'excluding22.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
