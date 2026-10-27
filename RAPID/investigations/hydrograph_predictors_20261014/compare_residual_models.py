"""Compare already saved held-out errors on exactly the same inputs and rows."""
from pathlib import Path
from itertools import combinations
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'RAPID/outputs/hydrograph_context_timing_20261018'
OUT=ROOT/'RAPID/outputs/hydrograph_residual_review_20261018'
MODELS=['ridge','elastic_net','forest','boosting']
PREFIX='length_context__sd__log1p__'

def corr(x,y,w):
 x=np.asarray(x);y=np.asarray(y);x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
 den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
 return float(np.sum(w*x*y)/den) if den>0 else np.nan

def main():
 pred=pd.read_parquet(SOURCE/'predictions.parquet');d=pred[pred.pipeline.isin([PREFIX+m for m in MODELS])].copy()
 d['model']=d.pipeline.str[len(PREFIX):];d['error_min']=60*(d.predicted_hours-d.observed_hours)
 labels=pd.read_csv(OUT/'classification_audit.csv',dtype={'example':str})[['example','state','edit_type']]
 d=d.merge(labels,on=['example','state'],validate='many_to_one');net=[];pairs=[];cases=[];types=[]
 for scenario,g in d.groupby('scenario'):
  assert not g.duplicated(['example','state','model']).any()
  assert g.groupby(['example','state']).model.nunique().eq(4).all()
  assert g.groupby(['example','state']).observed_hours.nunique().eq(1).all()
  e=g.pivot(index=['example','state'],columns='model',values='error_min')[MODELS]
  counts=pd.Series(e.index.get_level_values('example')).value_counts()
  w=np.array([1/counts[ex] for ex in e.index.get_level_values('example')])
  for a,b in combinations(MODELS,2):
   within=[corr(z[a],z[b],np.ones(len(z))) for _,z in e.groupby(level='example')]
   pairs.append(dict(scenario=scenario,model_a=a,model_b=b,equal_example_error_correlation=corr(e[a],e[b],w),
     median_within_example_correlation=float(np.nanmedian(within))))
  for (ex,model),z in g.groupby(['example','model']):
   net.append(dict(scenario=scenario,example=ex,model=model,n=len(z),bias_min=z.error_min.mean(),mae_min=z.error_min.abs().mean(),rmse_min=np.sqrt(np.mean(z.error_min**2))))
  for (kind,model),z in g.groupby(['edit_type','model']):
   ww=1/z.groupby('example').example.transform('size').to_numpy()
   types.append(dict(scenario=scenario,edit_type=kind,model=model,n=len(z),bias_min=np.average(z.error_min,weights=ww),mae_min=np.average(abs(z.error_min),weights=ww)))
  c=e.copy();c['scenario']=scenario
  c['all_positive']=e.gt(0).all(axis=1);c['all_negative']=e.lt(0).all(axis=1)
  c['all_errors_over_5min_same_direction']=e.gt(5).all(axis=1)|e.lt(-5).all(axis=1)
  c['error_range_min']=e.max(axis=1)-e.min(axis=1);c['median_abs_error_min']=e.abs().median(axis=1)
  cases.append(c.reset_index())
 pd.DataFrame(net).to_csv(OUT/'model_network_errors.csv',index=False)
 pd.DataFrame(pairs).to_csv(OUT/'model_error_correlations.csv',index=False)
 pd.DataFrame(types).to_csv(OUT/'model_edit_type_errors.csv',index=False)
 c=pd.concat(cases,ignore_index=True);c.to_csv(OUT/'model_case_agreement.csv',index=False)
 summary=[]
 for scenario,g in c.groupby('scenario'):
  w=1/g.groupby('example').example.transform('size').to_numpy()
  summary.append(dict(scenario=scenario,n=len(g),equal_example_same_sign_fraction=np.average(g.all_positive|g.all_negative,weights=w),
    equal_example_common_error_over_5min_fraction=np.average(g.all_errors_over_5min_same_direction,weights=w)))
 pd.DataFrame(summary).to_csv(OUT/'model_agreement_summary.csv',index=False)
 manifest=dict(scope='Same SD/log1p length-context inputs; each family uses its own inner-selected hyperparameters. Saved out-of-example predictions only.',
  caveat='Errors share the observed target, so correlated residuals alone do not prove missing information. Agreement can also reflect common training support limitations and fitting constraints.',
  new_fits=0,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'predictions.parquet',Path(__file__)]})
 (OUT/'model_comparison_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(pd.DataFrame(pairs).to_string(index=False));print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
