"""Descriptive candidate reduction; no RF selection, fitting, routing or p-values."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning')
OUT=Path('/Volumes/SyntheticResults20261027/analysis/candidate_reduction_20261030')
FACTORS=['total_width_m','reference_slope','outer_a_fraction','inner_fraction','length_multiplier','total_length_m','internal_location','branch_width_factor']
RESPONSES=['centroid_shift_minutes','peak_shift_minutes']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    x=np.asarray(x,dtype=float).copy();finite=np.isfinite(x)
    if not finite.any():return x
    tol=1e-10*max(np.max(abs(x[finite])),1.)
    x[finite]=np.round(x[finite]/tol)*tol
    if np.ptp(x[finite])<=tol:x[finite]=np.mean(x[finite])
    return x

def corr(x,y,rank=False):
    ok=np.isfinite(x)&np.isfinite(y);a=np.asarray(x)[ok];b=np.asarray(y)[ok]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    if rank:a,b=rankdata(a),rankdata(b)
    a=a-a.mean();b=b-b.mean();return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))

def matrix(frame,cols,rank=False):
    x=np.column_stack([clean(frame[c]) for c in cols]);assert np.isfinite(x).all()
    if rank:x=rankdata(x,axis=0)
    valid=np.ptp(x,axis=0)>0
    x-=x.mean(axis=0);den=np.sqrt((x*x).sum(axis=0));valid &= den>1e-10
    x[:,valid]/=den[valid];x[:,~valid]=0
    c=np.clip(x.T@x,-1.,1.);c[~valid,:]=np.nan;c[:,~valid]=np.nan
    return c,valid

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 meta=json.loads((SOURCE/'provenance.json').read_text())
 for p,h in meta['source_sha256'].items():assert sha(p)==h,p
 cat=pd.read_csv(SOURCE/'candidate_catalog.csv');cols=cat.variable.tolist();assert len(cols)==205
 f=pd.read_parquet(SOURCE/'features.parquet');assert len(f)==137781
 # First use a full-column scale to suppress subtraction roundoff in null strata.
 # This changes audit copies only, never the source features or response values.
 for c in cols:f[c]=clean(f[c])
 quality=[]
 for c in cols:
  raw=c.removeprefix('log1p__')
  if raw.startswith('B_ratio_'):
   metric=raw[8:];a=f['B_parent_'+metric];b=f['B_child_'+metric]
   invalid=~np.isfinite(a)|~np.isfinite(b)|(a<=0)|(b<0)
   reason='derived ratio with zero/nonfinite denominator or invalid numerator after roundoff cleaning'
  elif c.startswith('B_') and 'route_length_skew' in c:
   sides=['parent','child'] if '_delta_' in c else ['child' if '_child_' in c else 'parent']
   invalid=np.logical_or.reduce([f['B_'+side+'_route_length_sd'].eq(0) for side in sides]);reason='skewness of deterministic route distribution'
  else:continue
  affected=np.isfinite(f[c])&invalid
  for (method,state),z in f[affected].groupby(['method','child_state']):quality.append(dict(variable=c,method=method,child_state=state,n=len(z),reason=reason))
  f.loc[invalid,c]=np.nan
 pd.DataFrame(quality).to_csv(OUT/'finite_but_undefined_audit.csv',index=False)
 keys=['case','child_state'];g=f[f.method.eq('variable_manning')].sort_values(keys).reset_index(drop=True)
 assert len(g)==45927 and not g.duplicated(keys).any()
 for method,h in f.groupby('method'):
  h=h.sort_values(keys).reset_index(drop=True)
  pd.testing.assert_frame_equal(g[keys],h[keys]);np.testing.assert_allclose(g[cols],h[cols],rtol=1e-9,atol=1e-8,equal_nan=True)
 # Domain validity and exact implementation-level reasons, no imputation.
 coverage=[];undefined=[]
 for state,z in g.groupby('child_state'):
  for col in cols:
   x=clean(z[col]);ok=np.isfinite(x);a=x[ok]
   coverage.append(dict(variable=col,child_state=state,n=len(z),finite=int(ok.sum()),finite_pct=100*ok.mean(),varying=bool(len(a)>1 and np.ptp(a)>0),minimum=a.min() if len(a) else np.nan,maximum=a.max() if len(a) else np.nan))
   if ok.all():continue
   raw=col.removeprefix('log1p__')
   if raw.startswith('B_ratio_'):
    metric=raw[8:];p=z['B_parent_'+metric];c=z['B_child_'+metric]
    flags={'nonfinite_parent_or_child':~np.isfinite(p)|~np.isfinite(c),'nonpositive_parent_denominator':np.isfinite(p)&(p<=0),'negative_child_excluded_by_ratio_convention':np.isfinite(c)&(c<0)}
   elif 'route_length_skew' in col:
    sides=['parent','child'] if '_delta_' in col else ['child' if '_child_' in col else 'parent']
    flags={'zero_route_spread':np.logical_or.reduce([np.isclose(z['B_'+s+'_route_length_sd'],0,rtol=0,atol=1e-8) for s in sides])}
   else:
    sides=['parent','child'] if '_delta_' in col else ['child' if '_child_' in col else 'parent']
    flags={'no_bifurcation_in_relevant_state':np.logical_or.reduce([z['B_'+s+'_bifurcations'].eq(0) for s in sides])}
   assigned=np.zeros(len(z),bool)
   for reason,mask in flags.items():
    mask=np.asarray(mask)&~ok&~assigned;assigned|=mask
    if mask.any():undefined.append(dict(variable=col,child_state=state,reason=reason,n=int(mask.sum())))
   if (~ok&~assigned).any():undefined.append(dict(variable=col,child_state=state,reason='unresolved_requires_review',n=int((~ok&~assigned).sum())))
 cov=pd.DataFrame(coverage);cov.to_csv(OUT/'coverage_by_type.csv',index=False)
 pd.DataFrame(undefined).to_csv(OUT/'undefined_reasons.csv',index=False)
 status=cov.groupby('variable').agg(finite_pct=('finite_pct','mean'),minimum_type_coverage=('finite_pct','min'),varying_types=('varying','sum')).reset_index().merge(cat,on='variable')
 status['global_varying']=[np.ptp(clean(g[c])[np.isfinite(g[c])])>0 if np.isfinite(g[c]).any() else False for c in status.variable]
 status['status']=np.select([status.minimum_type_coverage.lt(100),~status.global_varying,status.varying_types.eq(0)],['partly_defined','globally_constant','between_type_only'],default='within_type_varying')
 status.to_csv(OUT/'inventory.csv',index=False)
 complete=status[status.minimum_type_coverage.eq(100)&status.global_varying].variable.sort_values().tolist()
 print('Inventory',status.status.value_counts().to_dict(),len(complete),'complete nonconstant',flush=True)
 # Complete-data correlations: no shifting pairwise-complete populations for redundancy.
 pear,valid=matrix(g,complete);spear,_=matrix(g,complete,True)
 pear_type={};spear_type={};active=[]
 for state,z in g.groupby('child_state'):
  pear_type[state],_=matrix(z,complete);spear_type[state],act=matrix(z,complete,True);active.append(act)
 act=np.array(active);stack=np.array(list(spear_type.values()));pstack=np.array(list(pear_type.values()))
 records=[];aliases=[];n=len(complete)
 for i in range(n):
  for j in range(i+1,n):
   a,b=complete[i],complete[j];r=stack[:,i,j];p=pstack[:,i,j];usable=np.isfinite(r);same=np.array_equal(act[:,i],act[:,j])
   # A conservative similarity: strong pooled and within every informative type;
   # differing active-type sets block an interchangeability claim.
   similarity=min(abs(spear[i,j]),float(np.min(abs(r[usable])))) if usable.any() and same else (abs(spear[i,j]) if same and not act[:,i].any() else 0.)
   consistent=bool(np.all(np.sign(r[usable])==np.sign(spear[i,j])))
   if not consistent:similarity=0.
   residual=np.nan;slope=np.nan;intercept=np.nan
   if abs(pear[i,j])>1-1e-11:
    x=g[a].to_numpy();y=g[b].to_numpy();slope=np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean());intercept=y.mean()-slope*x.mean();residual=float(np.max(abs(y-(intercept+slope*x)))/max(np.ptp(y),1e-30))
    if residual<=1e-8:aliases.append(dict(a=a,b=b,slope=slope,intercept=intercept,max_residual_over_range=residual,evidence='empirical full-grid affine equality; not automatically a universal identity'))
   records.append(dict(a=a,b=b,spearman_pooled=spear[i,j],pearson_pooled=pear[i,j],mean_abs_spearman=float(np.mean(abs(r[usable]))) if usable.any() else np.nan,min_abs_spearman=float(np.min(abs(r[usable]))) if usable.any() else np.nan,min_abs_pearson=float(np.min(abs(p[usable]))) if usable.any() else np.nan,n_informative_types=int(usable.sum()),same_active_types=same,consistent_sign=consistent,conservative_similarity=similarity,affine_max_residual_over_range=residual))
 pairs=pd.DataFrame(records);pairs.to_csv(OUT/'predictor_pairs.csv',index=False);pd.DataFrame(aliases).to_csv(OUT/'affine_equalities.csv',index=False)
 for stat,c in [('spearman_pooled',spear),('pearson_pooled',pear)]:pd.DataFrame(c,index=complete,columns=complete).to_csv(OUT/(stat+'.csv'))
 rows=[]
 for state in spear_type:
  for i,j in itertools.combinations(range(n),2):rows.append(dict(child_state=state,a=complete[i],b=complete[j],spearman=spear_type[state][i,j],pearson=pear_type[state][i,j]))
 pd.DataFrame(rows).to_parquet(OUT/'predictor_pairs_by_type.parquet',index=False)
 distance=np.ones((n,n));np.fill_diagonal(distance,0)
 for r in pairs.itertuples():i=complete.index(r.a);j=complete.index(r.b);distance[i,j]=distance[j,i]=1-r.conservative_similarity
 tree=linkage(squareform(distance,checks=True),method='complete',optimal_ordering=True)
 clusters=[]
 for threshold in [.85,.90,.95,.99]:
  labels=fcluster(tree,t=1-threshold+1e-12,criterion='distance')
  clusters.extend(dict(variable=c,threshold=threshold,cluster=int(k)) for c,k in zip(complete,labels))
 pd.DataFrame(clusters).to_csv(OUT/'clusters.csv',index=False);np.save(OUT/'linkage.npy',tree)
 print('Correlations complete',len(pairs),'pairs;',len(aliases),'affine edges',flush=True)
 # Recompute signed Spearman and Pearson for all three response methods, per type.
 assoc=[]
 for (method,state),z in f.groupby(['method','child_state']):
  for response in RESPONSES:
   y=z[response].to_numpy();span=np.ptp(y);resolved=span>(4/60 if response.startswith('centroid') else .5)
   for c in cols:
    x=clean(z[c]);assoc.append(dict(method=method,child_state=state,response=response,variable=c,n_finite=int(np.isfinite(x).sum()),resolved=bool(resolved),response_span_minutes=span,spearman=corr(x,y,True) if resolved else np.nan,pearson=corr(x,y) if resolved else np.nan))
  print('Response associations',method,state,flush=True)
 assoc=pd.DataFrame(assoc);assoc.to_csv(OUT/'response_associations.csv',index=False)
 summary=assoc.groupby(['method','response','variable']).agg(mean_abs_spearman=('spearman',lambda x:x.abs().mean()),min_abs_spearman=('spearman',lambda x:x.abs().min()),max_abs_spearman=('spearman',lambda x:x.abs().max()),mean_abs_pearson=('pearson',lambda x:x.abs().mean()),n_types=('spearman','count'),positive_types=('spearman',lambda x:int((x>0).sum())),negative_types=('spearman',lambda x:int((x<0).sum()))).reset_index()
 summary.to_csv(OUT/'response_summary.csv',index=False)
 # Descriptor factorial main/pair signatures are inherited and indexed, not refitted.
 sources={str(SOURCE/n):sha(SOURCE/n) for n in ['features.parquet','candidate_catalog.csv','factorial_signatures.csv','provenance.json']};sources[str(Path(__file__))]=sha(__file__)
 metadata=dict(reference='SYN-CANDIDATE-REDUCTION-2026-10-30',sources=sources,n_candidates=len(cols),n_geometry_pairs=len(g),n_response_rows=len(f),geometry_invariant_across_methods=True,roundoff='Full-column then per-stratum tolerance 1e-10 max(max absolute predictor,1); null-stratum subtraction roundoff suppressed; response values not rounded',cluster_rule='Complete linkage, 1 minus minimum absolute Spearman across pooled and informative within-type correlations; require matching variable-active type sets and consistent sign. Thresholds .85,.90,.95,.99 are sensitivity settings, not significance cutoffs.',affine_tolerance='Maximum full-grid affine residual / target range <=1e-8; empirical grid identities only',inference='Descriptive finite factorial population. No p-values, no iid confidence intervals, no causal identification beyond imposed grid controls, no RF selection.',derived_validity='Ratios with zero/nonfinite cleaned parent denominators and skewness at zero route SD are NaN in audit copies; finite source exceptions exported separately.',routing_runs=0,rf_fits=0,versions={'numpy':np.__version__,'pandas':pd.__version__})
 (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
