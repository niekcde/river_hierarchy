"""Interpret the frozen seven-input LONO forests; never modify the baseline.

Run with the original sklearn/numpy environment and --shap-path if needed.
All exclusions are summary-only; no sensitivity refits or tuning are performed.
"""
from pathlib import Path
import argparse, hashlib, json, sys, warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/'RAPID/investigations/hydrograph_predictors_20261014'
sys.path.insert(0,str(LEGACY))
import drop_exposure_check as frozen
import sklearn
E=frozen.engine
OUT=ROOT/'RAPID/outputs/hydrograph_frozen_interpretation_20261019'
SOURCE=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019/input.parquet'
BASE=ROOT/'RAPID/outputs/hydrograph_no_exposure_20261019'
C=frozen.COLUMNS
LABELS=['Mean length ratio','Parent size (km)','Length SD ratio','Mean width ratio','Width CV ratio','Entropy ratio','Parent dominance']
LABEL=dict(zip(C,LABELS))
GROUPS={**{c:[c] for c in C},'length_geometry':[C[0],C[1],C[2]],'width_geometry':[C[3],C[4]],'branching':[C[5],C[6]],'width_entropy':[C[3],C[4],C[5]],'length_width':[C[0],C[1],C[2],C[3],C[4]],'all_inputs':C}
VIEWS=['all','without54_summary','without_extreme_summary','without_both_summary']
def mask(d,view):
 m=np.ones(len(d),bool)
 if view in ['without54_summary','without_both_summary']:m &= d.example.to_numpy()!='54'
 if view in ['without_extreme_summary','without_both_summary']:m &= ~((d.example.to_numpy()=='37')&(d.state.to_numpy()=='CB001_bubble_3'))
 return m

def rho(x,y):
 if len(x)<4 or np.ptp(x)<1e-10 or np.ptp(y)<1e-10:return np.nan
 return pd.Series(np.asarray(x)).corr(pd.Series(np.asarray(y)),method='spearman')

def raw(c,x):return np.expm1(x) if c.startswith('log1p__') else np.asarray(x)
def quantiles(x,w,q):
 o=np.argsort(x);x=np.asarray(x)[o];w=np.asarray(w)[o]
 return np.interp(q,(np.cumsum(w)-.5*w)/w.sum(),x)
def pred(a,d):return frozen.forest.predict(a,d)*60

def run(args):
 if args.shap_path:sys.path.insert(0,args.shap_path)
 import shap
 OUT.mkdir(parents=True,exist_ok=True)
 info=json.loads((BASE/'final_model.json').read_text())
 assert C==info['columns']
 assert sklearn.__version__==info['versions']['sklearn']
 assert np.__version__==info['versions']['numpy']
 d=pd.read_parquet(SOURCE).reset_index(drop=True)
 saved=pd.read_parquet(BASE/'predictions.parquet').query("variant=='seven_inputs'")
 assert len(d)==2170 and d.example.nunique()==15 and not d.duplicated(['example','state']).any()
 assert not (d.state=='fully_collapsed').any()
 assert set(zip(d.example,d.state))==set(zip(saved.example,saved.state))
 assert np.isfinite(d[C+['centroid_shift_hours']]).all().all()
 files=[Path(__file__),Path(frozen.__file__),Path(E.__file__),Path(frozen.forest.__file__),SOURCE,BASE/'final_model.json',BASE/'predictions.parquet']
 design=dict(sources={str(f):E.sha(f) for f in files},columns=C,params=info['params'],seed=E.SEED,repeats=args.repeats,groups=GROUPS,versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,shap=shap.__version__),sensitivity='Summary exclusions only; original folds retained. Permutations use full held-network donor pools even for exclusions.',ale='First-order ALE with equal-network-weighted pooled quantile bin edges; local differences computed at held-network rows by their held-out forest. Equal-network mean local differences per supported bin; center with equal-network bin frequencies. Per-network curves use own quantile bins. No inference outside observed ranges. 8 bins, plus 4/12-bin sensitivity.',interpretation='Fixed settings previously selected on these networks; exploratory, not fresh nested validation. No baseline or historical artifacts overwritten.')
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
 if (OUT/'design.json').exists():assert json.loads((OUT/'design.json').read_text())==design,'Design changed; use a new destination'
 E.dump(OUT/'design.json',design)
 edges={(c,b):np.unique(quantiles(d[c],E.weights(d),np.linspace(0,1,b+1))) for c in C for b in [4,8,12]}
 sh=[];per=[];inter=[];ale=[];checks=[];support=[];predictions=[]
 for ex in sorted(d.example.unique()):
  dest=OUT/'folds'/ex;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'complete.json').exists():
   assert json.loads((dest/'complete.json').read_text())['fingerprint']==fp
  else:
   tr=d[d.example!=ex];te=d[d.example==ex].copy();print('Fitting',ex,flush=True)
   a=frozen.fit(tr,info['params'],jobs=args.jobs);p=pred(a,te)
   ref=te[['example','state']].merge(saved,on=['example','state'],validate='one_to_one')
   np.testing.assert_allclose(p,ref.predicted_hours*60,rtol=1e-10,atol=1e-8)
   model=a['estimator'];model.n_jobs=1
   xt=a['preprocessor'].transform(te);kept=a['preprocessor'].kept
   explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent',model_output='raw')
   sv=np.asarray(explainer.shap_values(xt,check_additivity=True))*a['target_sd']*60
   base=float(np.asarray(explainer.expected_value).item())*a['target_sd']*60+a['target_mean']*60
   np.testing.assert_allclose(base+sv.sum(1),p,atol=1e-6,rtol=1e-9)
   si=np.asarray(explainer.shap_interaction_values(xt))*a['target_sd']*60
   np.testing.assert_allclose(si.sum(2),sv,atol=1e-6,rtol=1e-9)
   sr=[];sp=[];ar=[];pr=[]
   for j,c in enumerate(C):
    assert c in kept
    v=sv[:,kept.index(c)]
    sr.extend(dict(example=ex,state=r.state,feature=c,value=float(x),shap_minutes=float(s),base_minutes=base,predicted_minutes=float(pp)) for r,x,s,pp in zip(te.itertuples(),raw(c,te[c]),v,p))
    sp.append(dict(example=ex,feature=c,train_min=tr[c].min(),train_max=tr[c].max(),held_min=te[c].min(),held_max=te[c].max(),held_unique=te[c].nunique(),outside_training_fraction=float(((te[c]<tr[c].min())|(te[c]>tr[c].max())).mean())))
    for bins in [4,8,12]:
     for grid in ['common','network']:
      ed=edges[c,bins] if grid=='common' else np.unique(te[c].quantile(np.linspace(0,1,bins+1)).to_numpy())
      ids=np.clip(np.searchsorted(ed,te[c],side='right')-1,0,len(ed)-2)
      lo=te.copy();hi=te.copy();lo[c]=ed[ids];hi[c]=ed[ids+1]
      diff=pred(a,hi)-pred(a,lo)
      for view in VIEWS:
       use=mask(te,view)
       for k in range(len(ed)-1):
        m=(ids==k)&use;n=int(m.sum())
        ar.append(dict(example=ex,view=view,feature=c,bins=bins,grid=grid,bin=k,lower=ed[k],upper=ed[k+1],n=n,total=int(use.sum()),local_difference_minutes=float(diff[m].mean()) if n else np.nan))
   y=te.centroid_shift_hours.to_numpy()*60
   loss0=100*abs(p-y)/np.maximum(abs(y),1);se0=(p-y)**2
   for name,cols in GROUPS.items():
    inds=[kept.index(c) for c in cols]
    for rep in range(args.repeats):
     order=np.random.default_rng(E.SEED+int(ex)*1000+rep).permutation(len(te))
     xp=xt.copy();xp[:,inds]=xt[order][:,inds]
     pp=model.predict(xp)*a['target_sd']*60+a['target_mean']*60
     dl=100*abs(pp-y)/np.maximum(abs(y),1)-loss0;ds=(pp-y)**2-se0
     for view in VIEWS:
      m=mask(te,view)
      if m.any():pr.append(dict(example=ex,view=view,group=name,repeat=rep,delta_relative_pp=dl[m].mean(),delta_mse_minutes2=ds[m].mean()))
   pd.DataFrame(sr).to_parquet(dest/'shap.parquet',index=False)
   ii=te[['example','state',C[3],C[1]]].copy();ii['interaction_minutes']=2*si[:,kept.index(C[3]),kept.index(C[1])];ii.to_parquet(dest/'interaction.parquet',index=False)
   pd.DataFrame(ar).to_parquet(dest/'ale.parquet',index=False)
   pd.DataFrame(pr).to_parquet(dest/'permutation.parquet',index=False)
   pd.DataFrame(sp).to_csv(dest/'support.csv',index=False)
   pp=te[['example','state','centroid_shift_hours']].copy();pp['predicted_minutes']=p;pp.to_parquet(dest/'predictions.parquet',index=False)
   E.dump(dest/'complete.json',dict(fingerprint=fp,example=ex,prediction_max_error_minutes=float(np.max(abs(p-ref.predicted_hours*60))),shap_additivity_max_error_minutes=float(np.max(abs(base+sv.sum(1)-p))),interaction_additivity_max_error_minutes=float(np.max(abs(si.sum(2)-sv)))))
   print('Verified and interpreted',ex,flush=True)
  sh.append(pd.read_parquet(dest/'shap.parquet'));per.append(pd.read_parquet(dest/'permutation.parquet'));inter.append(pd.read_parquet(dest/'interaction.parquet'));ale.append(pd.read_parquet(dest/'ale.parquet'));support.append(pd.read_csv(dest/'support.csv',dtype={'example':str}));predictions.append(pd.read_parquet(dest/'predictions.parquet'));checks.append(json.loads((dest/'complete.json').read_text()))
 for name,parts in [('shap_values',sh),('permutation_repeats',per),('shap_width_size_interaction',inter),('ale_local_differences',ale),('reconstructed_predictions',predictions)]:pd.concat(parts,ignore_index=True).to_parquet(OUT/f'{name}.parquet',index=False)
 pd.concat(support).to_csv(OUT/'support.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False)
 summarize(d)


def summarize(d):
 s=pd.read_parquet(OUT/'shap_values.parquet');p=pd.read_parquet(OUT/'reconstructed_predictions.parquet');pm=pd.read_parquet(OUT/'permutation_repeats.parquet');al=pd.read_parquet(OUT/'ale_local_differences.parquet')
 sns=[];cs=[];pairs=[];scores=[];ps=[];als=[];isummary=[]
 inter=pd.read_parquet(OUT/'shap_width_size_interaction.parquet')
 for view in VIEWS:
  a=d[mask(d,view)];z=p[mask(p,view)];w=E.weights(z);y=z.centroid_shift_hours.to_numpy()*60
  variance=np.average((y-np.average(y,weights=w))**2,weights=w)
  scores.append(dict(view=view,n=len(z),networks=z.example.nunique(),relative_error_pct=np.average(100*abs(z.predicted_minutes-y)/np.maximum(abs(y),1),weights=w),**E.score(y,z.predicted_minutes,w)))
  b=pm[pm.view==view].groupby(['example','group'])[['delta_relative_pp','delta_mse_minutes2']].mean().reset_index();b['view']=view;b['r2_decrease']=b.delta_mse_minutes2/variance
  for group,g in b.groupby('group'):
   row=dict(view=view,group=group,networks=len(g))
   for metric in ['delta_relative_pp','delta_mse_minutes2','r2_decrease']:
    row.update({metric:g[metric].mean(),metric+'_sd':g[metric].std(),metric+'_min':g[metric].min(),metric+'_max':g[metric].max(),metric+'_positive_networks':int((g[metric]>0).sum())})
   # Monte Carlo variability of the equal-network estimate, NOT sampling uncertainty.
   mc=pm[(pm.view==view)&(pm.group==group)].groupby('repeat').delta_relative_pp.mean()
   row['shuffle_mc_sd_relative_pp']=mc.std();ps.append(row)
  b.to_csv(OUT/f'permutation_by_network_{view}.csv',index=False)
  ss=s[mask(s,view)]
  for (ex,c),g in ss.groupby(['example','feature']):
   sns.append(dict(view=view,example=ex,feature=c,mean_abs_shap_minutes=g.shap_minutes.abs().mean(),signed_shap_value_rho=rho(g.value,g.shap_minutes)))
  for c in C:
   cs.append(dict(view=view,example='pooled_rows',feature=c,rho=rho(a[c],a.centroid_shift_hours)))
   for ex,g in a.groupby('example'):cs.append(dict(view=view,example=ex,feature=c,rho=rho(g[c],g.centroid_shift_hours)))
  for ex,g in [('pooled_rows',a),*list(a.groupby('example'))]:
   for j,c in enumerate(C):
    for cc in C[j+1:]:pairs.append(dict(view=view,example=ex,feature1=c,feature2=cc,rho=rho(g[c],g[cc])))
  ii=inter[mask(inter,view)].groupby('example').interaction_minutes.agg(lambda x:x.abs().mean())
  isummary.append(dict(view=view,mean_absolute_pair_minutes=ii.mean(),sd=ii.std(),min=ii.min(),max=ii.max()))
  # Each local difference is averaged equally across networks with data in that bin.
  aa=al[(al.view==view)&(al.grid=='common')]
  for (c,bins),g in aa.groupby(['feature','bins']):
   rows=[]
   for k,h in g.groupby('bin'):
    h=h[h.total>0];supported=h[h.n>0]
    rows.append(dict(view=view,feature=c,bins=bins,bin=k,lower=h.lower.iloc[0],upper=h.upper.iloc[0],networks=len(supported),rows=int(h.n.sum()),local_difference_minutes=supported.local_difference_minutes.mean(),local_difference_sd=supported.local_difference_minutes.std(),mass=(h.n/h.total).mean()))
   q=pd.DataFrame(rows);q['ale_minutes']=q.local_difference_minutes.cumsum()-.5*q.local_difference_minutes
   q.ale_minutes-=np.average(q.ale_minutes,weights=q.mass)
   als.extend(q.to_dict('records'))
 pd.DataFrame(scores).to_csv(OUT/'scores_summary_exclusions.csv',index=False)
 pd.DataFrame(ps).to_csv(OUT/'permutation_summary.csv',index=False)
 sn=pd.DataFrame(sns);sn['rank']=sn.groupby(['view','example']).mean_abs_shap_minutes.rank(ascending=False,method='min');sn.to_csv(OUT/'shap_by_network.csv',index=False)
 sn.groupby(['view','feature']).agg(mean_abs_shap_minutes=('mean_abs_shap_minutes','mean'),network_sd=('mean_abs_shap_minutes','std'),network_min=('mean_abs_shap_minutes','min'),network_max=('mean_abs_shap_minutes','max'),median_rank=('rank','median'),rank_min=('rank','min'),rank_max=('rank','max'),direction_rho_mean=('signed_shap_value_rho','mean'),direction_positive=('signed_shap_value_rho',lambda x:int((x>0).sum())),direction_negative=('signed_shap_value_rho',lambda x:int((x<0).sum()))).to_csv(OUT/'shap_summary.csv')
 cr=pd.DataFrame(cs);cr.to_csv(OUT/'response_correlations.csv',index=False)
 cr[cr.example!='pooled_rows'].groupby(['view','feature']).agg(mean_within_rho=('rho','mean'),median_within_rho=('rho','median'),min_rho=('rho','min'),max_rho=('rho','max'),valid_networks=('rho','count'),positive=('rho',lambda x:int((x>0).sum())),negative=('rho',lambda x:int((x<0).sum()))).join(cr[cr.example=='pooled_rows'].set_index(['view','feature']).rho.rename('pooled_row_rho')).to_csv(OUT/'response_correlation_summary.csv')
 pc=pd.DataFrame(pairs);pc.to_csv(OUT/'predictor_correlations.csv',index=False)
 pc[pc.example!='pooled_rows'].groupby(['view','feature1','feature2']).agg(mean_within_rho=('rho','mean'),min_rho=('rho','min'),max_rho=('rho','max'),positive=('rho',lambda x:int((x>0).sum())),negative=('rho',lambda x:int((x<0).sum()))).to_csv(OUT/'predictor_correlation_summary.csv')
 pd.DataFrame(als).to_csv(OUT/'ale_summary.csv',index=False);pd.DataFrame(isummary).to_csv(OUT/'interaction_summary.csv',index=False)
 print('Completed',OUT,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--shap-path');ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--repeats',type=int,default=20);args=ap.parse_args();run(args)
