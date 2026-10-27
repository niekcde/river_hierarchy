"""Slope-ratio associations with frozen nine-input held-network residuals; no fits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.current_domains import allocation_tables,validate_domain
OUT=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
EXT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
SC=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
DEST=OUT/'slope_ratio_audit_20261023'
SOURCES={}
def read(p):
 p=Path(p);SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.suffix=='.json':return json.loads(p.read_text())
 if p.suffix=='.parquet':return pd.read_parquet(p)
 return pd.read_csv(p,dtype={'example':str})
def corr(x,y,w):
 x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float);w=w/w.sum()
 x=x-np.sum(w*x);y=y-np.sum(w*y)
 den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
 return float(np.sum(w*x*y)/den) if den>1e-15 else np.nan
def main():
 DEST.mkdir(exist_ok=True)
 d=read(OUT/'augmented_tests/input.parquet');p=read(OUT/'nine_input_tuning/predictions.parquet');p=p[p.variant.eq('same_grid')]
 d=d.merge(p[['example','state','cohort','centroid_shift_hours','predicted_hours']],on=['example','state','cohort','centroid_shift_hours'],how='outer',indicator=True,validate='one_to_one')
 assert d._merge.eq('both').all();d=d.drop(columns='_merge');assert len(d)==2185
 meta=read(SC/'geometry_responses.parquet').set_index(['example','state'])
 graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope'])
 jobs={}
 for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',EXT/'routing_plan.json',OUT/'routing_plan.json']:
  for case in read(path)['cases']:
   lookup=jobs.setdefault(case['example'],{})
   for j in case['jobs']:
    for name in set([j['state_id'],*j.get('aliases',[])]):lookup[name]=j
 missing=[(r.example,r.state) for r in d.itertuples() if r.state not in jobs[r.example]]
 assert not missing,missing
 parents={ex:read(j['S000_base']['parameters']) for ex,j in jobs.items() if Path(j['S000_base']['parameters']).exists()};cache={};features=[];skipped=[]
 historical=read(EXT/'slope_attribution/slope_descriptors.csv').set_index(['example','side'])
 history_manifest=read(EXT/'slope_attribution/manifest.json')
 for ex,group in d.groupby('example'):
  for r in group.itertuples():
   if ex not in parents or not Path(jobs[ex][r.state]['parameters']).exists():
    if r.cohort!='segmented':
     skipped.append(dict(example=ex,state=r.state,reason='active parent or child parameters unavailable on external drive'));continue
    # Reuse immutable geometric summaries only when their recorded inputs match
    # the current unchanged-regime plan; never substitute historical responses.
    hp=jobs[ex]['S000_base']['parameters'];hc=jobs[ex][r.state]['parameters']
    extcase=next(c for c in read(EXT/'routing_plan.json')['cases'] if c['example']==ex)
    for path in [hp,hc]:assert history_manifest['sources'][path]==extcase['source_sha256'][path]
    record=dict(example=ex,state=r.state,source='saved geometric summary; active-input hashes match')
    for side in ['parent','child']:
     h=historical.loc[(ex,side)]
     record[side+'_slope_mean']=float(h.slope_mean);record[side+'_slope_slow']=float(h.slope_effective**-.5);record[side+'_length_km']=float(h.length_km)
    child=read(hc);assert SOURCES[hc]==history_manifest['sources'][hc];lineage=read(EXT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv');ff=child[child.id_link.isin(lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'])]
    ww=ff.reference_flow_fraction*ff.link_length_m
    np.testing.assert_allclose(record['child_slope_slow'],np.average(1/np.sqrt(ff.slope),weights=ww),rtol=1e-10)
    np.testing.assert_allclose(record['child_slope_mean'],np.average(ff.slope,weights=ww),rtol=1e-10)
    np.testing.assert_allclose(record['parent_slope_slow'],r.parent_slope_slow,rtol=1e-10)
    for side in ['parent','child']:np.testing.assert_allclose(record[side+'_length_km'],getattr(r,'B_'+side+'_route_length_mean')/1000,rtol=1e-10)
    record['slope_mean_ratio']=record['child_slope_mean']/record['parent_slope_mean'];record['slope_slow_ratio']=record['child_slope_slow']/record['parent_slope_slow'];record['slope_effective_ratio']=record['slope_slow_ratio']**-2
    features.append(record);continue
   child=read(jobs[ex][r.state]['parameters']);fulls={'parent':parents[ex],'child':child}
   record=dict(example=ex,state=r.state,source='active parameters')
   if r.cohort=='segmented':
    lineage=read(EXT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv');selected=lineage[lineage.role.eq('collapsed_block')]
   else:
    scope=meta.loc[(ex,r.state),'B_scope'];g=graphs.loc[(ex,r.state,scope)]
   for side,full in fulls.items():
    if r.cohort=='segmented':
     f=full[full.id_link.isin(selected[side+'_link_id'])].copy();w=(f.reference_flow_fraction*f.link_length_m).to_numpy()
    else:
     key=g[side+'_graph']
     if key not in cache:
      geom=read(SC/'graphs'/f'{key}.parquet');cache[key]=(geom,allocation_tables(geom)[1].set_index('id_link'))
     geom,alloc=cache[key];f,_=validate_domain(full,geom);f=f.set_index('id_link').loc[alloc.index]
     np.testing.assert_allclose(f[['link_length_m','width']],geom.set_index('id_link').loc[alloc.index,['link_length_m','width']],rtol=1e-10,atol=1e-7)
     w=alloc.mean_length_contribution_m.to_numpy()
    assert (f.slope>0).all() and np.isfinite(f.slope).all()
    record[side+'_slope_mean']=float(np.average(f.slope,weights=w))
    record[side+'_slope_slow']=float(np.average(1/np.sqrt(f.slope),weights=w))
    record[side+'_length_km']=float(w.sum()/1000)
   np.testing.assert_allclose(record['parent_slope_slow'],r.parent_slope_slow,rtol=1e-10)
   for side in fulls:np.testing.assert_allclose(record[side+'_length_km'],getattr(r,'B_'+side+'_route_length_mean')/1000,rtol=1e-10,atol=1e-8)
   record['slope_mean_ratio']=record['child_slope_mean']/record['parent_slope_mean']
   record['slope_slow_ratio']=record['child_slope_slow']/record['parent_slope_slow']
   record['slope_effective_ratio']=record['slope_slow_ratio']**-2
   features.append(record)
  print(ex,'available descriptors',sum(v['example']==ex for v in features),'unavailable',sum(v['example']==ex for v in skipped),flush=True)
 pd.DataFrame(skipped).to_csv(DEST/'unavailable_sources.csv',index=False)
 f=pd.DataFrame(features);assert len(f)+len(skipped)==2185
 d=d.merge(f.drop(columns='parent_slope_slow'),on=['example','state'],validate='one_to_one')
 d['signed_error_h']=d.predicted_hours-d.centroid_shift_hours;d['absolute_error_h']=abs(d.signed_error_h)
 d['mean_length_ratio']=np.expm1(d.log1p__B_ratio_route_length_mean)
 d['slope_distribution_ratio']=d.slope_slow_ratio/np.sqrt(d.mean_length_ratio)
 identity_error=float(abs(d.slope_mean_ratio*d.mean_length_ratio-1).max())
 assert identity_error<1e-9
 for cohort,g in d.groupby('cohort'):
  w=1/g.groupby('example').example.transform('size')
  d.loc[g.index,'sse_share_pct']=100*w*g.signed_error_h**2/(w*g.signed_error_h**2).sum()
 d.to_parquet(DEST/'input_and_residuals.parquet',index=False)
 features=['slope_mean_ratio','slope_slow_ratio','slope_effective_ratio','slope_distribution_ratio']
 targets=['signed_error_h','absolute_error_h','centroid_shift_hours','mean_length_ratio','parent_slope_slow']
 rows=[];within=[]
 for cohort,g in d.groupby('cohort'):
  for subset,z in [('all',g),*[(f'without{ex}',g[g.example.ne(ex)]) for ex in ['22','37','46','54','55']]]:
   w=1/z.groupby('example').example.transform('size')
   for feature in features:
    for target in targets:
     rows.append(dict(cohort=cohort,subset=subset,feature=feature,target=target,n=len(z),networks=z.example.nunique(),pearson=corr(z[feature],z[target],w),spearman=corr(rankdata(z[feature]),rankdata(z[target]),w)))
  if cohort=='original':
   for ex,z in g.groupby('example'):
    for feature in features:
     within.append(dict(example=ex,feature=feature,n=len(z),spearman_signed=corr(rankdata(z[feature]),rankdata(z.signed_error_h),np.ones(len(z))),spearman_absolute=corr(rankdata(z[feature]),rankdata(z.absolute_error_h),np.ones(len(z)))))
 pd.DataFrame(rows).to_csv(DEST/'associations.csv',index=False);pd.DataFrame(within).to_csv(DEST/'within_network.csv',index=False)
 # Remove network and existing length-ratio rank trends as descriptive controls.
 # This is not cross-validated residual correction or an added predictor fit.
 rows=[]
 z=d[d.cohort.eq('original')].copy();w=1/z.groupby('example').example.transform('size');sw=np.sqrt(w.to_numpy())
 for control in ['network_only','network_and_length_ratio']:
  X=pd.get_dummies(z.example,dtype=float).to_numpy()
  if control.endswith('length_ratio'):X=np.column_stack([X,rankdata(z.mean_length_ratio)])
  for feature in features:
   for target in ['signed_error_h','absolute_error_h']:
    residuals=[]
    for col in [feature,target]:
     y=rankdata(z[col]);beta=np.linalg.lstsq(X*sw[:,None],y*sw,rcond=None)[0];residuals.append(y-X@beta)
    rows.append(dict(control=control,feature=feature,target=target,partial_rank_correlation=corr(*residuals,w)))
 pd.DataFrame(rows).to_csv(DEST/'controlled_rank_associations.csv',index=False)
 columns=['example','state','cohort','centroid_shift_hours','predicted_hours','signed_error_h','sse_share_pct',*features,'parent_slope_slow','mean_length_ratio']
 pd.concat([g.nlargest(10,'sse_share_pct') for _,g in d.groupby('cohort')])[columns].to_csv(DEST/'largest_errors.csv',index=False)
 d[d.cohort.eq('segmented')][columns].to_csv(DEST/'segmented.csv',index=False)
 for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
 (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),mean_slope_length_reciprocity_max_error=identity_error,unmatched_response_joins=0,unavailable_source_rows=len(skipped),rows=len(d),training_target='full excess-volume centroid shift',prediction='frozen nine-input same_grid nested held-network predictions, no augmentations',scope='B domain for regular cases; edited footprint excluding connectors for segmented; independently recomputed r_e L_e weights on each side',error_sign='predicted minus observed: positive means underpredicted advance for negative responses',correlations='Equal total network weight within each cohort; rank correlations weighted after ranking. No row-independent significance tests. Top-error table is selected by weighted squared error; correlations use full cohorts.',no_fits=True,production_modified=False),indent=2)+'\n')
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
