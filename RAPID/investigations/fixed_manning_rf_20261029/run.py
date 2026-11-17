"""Fresh fixed-Manning descriptors and network-held-out refits of existing RFs."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/cap_investigation_20261020'),str(ROOT/'RAPID/investigations/closed_unit_extension_20261019'),str(ROOT/'RAPID/investigations/hydrograph_predictors_20261014')]
import tune_nine_inputs as tuning
from edited_features import describe_edited
from flow_resistance import solve_unit_flow
from rapid_tools.current_domains import allocation_tables,validate_domain
from network_variants.equivalent_transition_domain import extract_complete_directed_domain
BASE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
DEST=BASE/'fixed_manning_rf_20261029'
SOURCE=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028/03_manning_midpoint')
EXT=ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
HASHES={};CACHE={};SOLVES=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);HASHES[str(p)]=sha(p)
 if p.suffix=='.json':return json.loads(p.read_text())
 return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,dtype={'example':str})
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str)+'\n');tmp.replace(p)
def active(ex,state):
 folder=SOURCE/'states'/ex/state;p=folder/'parameters.parquet';receipt=read(folder/'completion.json')
 f=read(p);assert HASHES[str(p)]==receipt['files'][str(p)]
 # The routed u/v identities are authoritative after staged geometry restoration.
 f['id_us_node']=f.u;f['id_ds_node']=f.v
 return f
def terminal(f):
 g=nx.MultiDiGraph();g.add_edges_from(zip(f.u,f.v));s=[x for x in g if g.in_degree(x)==0];t=[x for x in g if g.out_degree(x)==0]
 assert len(s)==len(t)==1 and nx.is_directed_acyclic_graph(g) and nx.is_weakly_connected(g)
 return s[0],t[0]
def resistance(f):
 key=hashlib.sha256(f[['id_link','u','v','link_length_m','width','depth_m']].to_numpy().tobytes()).hexdigest()
 if key in CACHE:return CACHE[key]
 s,t=terminal(f);edges=f[['id_link','u','v']].rename(columns={'id_link':'edge_id'}).copy();edges['K_e']=f.link_length_m/(f.width**2*f.depth_m**3)
 result=solve_unit_flow(edges,s,t,tolerance=1e-8,max_iterations=500)
 SOLVES.append({k:v for k,v in result.items() if k!='flows'}|dict(graph=key,links=len(f)))
 if not result['success']:raise RuntimeError(f'Resistance solve failed: {SOLVES[-1]}')
 CACHE[key]=result['K_eq'];return CACHE[key]
def describe(f,selected=None):
 ids=set(f.id_link) if selected is None else set(selected)
 key=('geometry',hashlib.sha256(f[['id_link','u','v','link_length_m','width']].to_numpy().tobytes()).hexdigest(),tuple(sorted(ids)))
 if key not in CACHE:CACHE[key]=describe_edited(f,ids)
 return CACHE[key]

def prepare():
 DEST.mkdir(exist_ok=True)
 if (DEST/'preparation.json').exists():
  prep=json.loads((DEST/'preparation.json').read_text());assert prep['script_sha256']==sha(__file__)
  for p,h in prep['sources'].items():assert sha(p)==h,p
  assert sha(DEST/'input.parquet')==prep['input_sha256'];return pd.read_parquet(DEST/'input.parquet')
 records=read(BASE/'validation_group_inventory_20261023/records.parquet')
 old=read(BASE/'resistance_inverse_depth_panels_20261023/input.parquet').set_index(['example','state'])
 response=read(SOURCE/'responses.csv');receipt=read(SOURCE/'completion.json');assert HASHES[str(SOURCE/'responses.csv')]==receipt['files'][str(SOURCE/'responses.csv')]
 response=response[(response.window=='0.5%')&(response.state!='fully_collapsed')].set_index(['example','state'])
 settings=read(BASE/'current_model_comparison_20261023/settings.json');columns=sorted(set(sum([a['columns'] for a in settings.values()],[])))
 rows=[];audit=[]
 for ex,group in records.groupby('example'):
  parent=active(ex,'S000_base');pi=parent.set_index('id_link')
  for record in group.itertuples():
   child=active(ex,record.state);seg=record.cohort=='segmented';oldrow=old.loc[(ex,record.state)]
   footprint=set(map(int,record.parent_footprint_links))
   if seg:
    lineage=read(EXT/f'variants/{ex}/{record.state}/lineage.csv');lineage=lineage[lineage.role=='collapsed_block']
    p,c=parent,child;pmask=set(map(int,lineage.parent_link_id));cmask=set(map(int,lineage.child_link_id))
    assert pmask==footprint and cmask<=set(c.id_link)
   else:
    ids=set(map(int,record.descriptor_links));assert ids<=set(pi.index)
    p=parent[parent.id_link.isin(ids)].copy();s,t=terminal(p);p,_=validate_domain(parent,p)
    c=extract_complete_directed_domain(child,int(s),int(t));c,_=validate_domain(child,c)
    pmask=set(p.id_link);cmask=set(c.id_link)
   a=describe(p,pmask);b=describe(c,cmask)
   pa=allocation_tables(p)[1].set_index('id_link');ca=allocation_tables(c)[1].set_index('id_link')
   psel=p.set_index('id_link').loc[sorted(pmask)];csel=c.set_index('id_link').loc[sorted(cmask)]
   wp=pa.loc[psel.index,'mean_length_contribution_m'];wc=ca.loc[csel.index,'mean_length_contribution_m']
   np.testing.assert_allclose([wp.sum(),wc.sum()],[a['route_length_mean'],b['route_length_mean']],rtol=1e-10)
   row=dict(example=ex,state=record.state,cohort=record.cohort,edit_type=record.edit_type,
    centroid_shift_hours=float(response.loc[(ex,record.state),'centroid_shift_hours']),
    peak_shift_hours=float(response.loc[(ex,record.state),'peak_shift_hours']))
   for k in ['route_length_mean','route_length_sd','width_mean','width_cv','route_entropy']:
    ratio=b[k]/a[k] if a[k]>0 else np.nan
    row['log1p__B_ratio_'+k]=float(np.log1p(ratio))
   row['domain_size']=float(pa.loc[sorted(footprint),'mean_length_contribution_m'].sum()/1000)
   row['B_parent_dominance_mean']=a['dominance_mean']
   row['parent_slope_slow']=float(np.average(1/np.sqrt(psel.slope),weights=wp))
   row['length_delta_km']=(b['route_length_mean']-a['route_length_mean'])/1000
   row['delta_G_inverse_depth']=float((wc/(csel.depth_m*np.sqrt(csel.slope))).sum()-(wp/(psel.depth_m*np.sqrt(psel.slope))).sum())
   row['resistance_explicit_depth']=-np.log10(resistance(c)/resistance(p))
   assert np.isfinite([row[col] for col in columns]).all(),(ex,record.state,row)
   # Parent context and edited footprint are unchanged by collapse corrections.
   for col in ['domain_size','B_parent_dominance_mean','parent_slope_slow']:
    np.testing.assert_allclose(row[col],oldrow[col],rtol=1e-9,atol=1e-9,err_msg=str((ex,record.state,col)))
   changed=[col for col in columns if not np.isclose(row[col],oldrow[col],rtol=1e-9,atol=1e-9)]
   audit.append(dict(example=ex,state=record.state,parent_domain_links=len(p),child_domain_links=len(c),changed_features='|'.join(changed)))
   rows.append(row)
  print('Prepared',ex,len(group),'cases',flush=True)
  pd.DataFrame(rows).to_parquet(DEST/'input.partial.parquet',index=False)
  pd.DataFrame(audit).to_csv(DEST/'descriptor_audit.csv',index=False)
  pd.DataFrame(SOLVES).to_csv(DEST/'resistance_solves.csv',index=False)
 d=pd.DataFrame(rows);assert len(d)==2185 and d.example.nunique()==15 and not d.duplicated(['example','state']).any()
 d.to_parquet(DEST/'input.parquet',index=False)
 save(DEST/'preparation.json',dict(sources=HASHES,script_sha256=sha(__file__),input_sha256=sha(DEST/'input.parquet'),rows=len(d),
  definitions='Original descriptor definitions. Current routed u/v topology. Regular B domains use saved parent membership and complete child domain between same parent identities; segmented edited geometry/G exclude connectors, resistance includes full network.',
  depth='Current fitted fixed-Manning depths; no response-derived descriptors. Depth-informed predictions assume reference calibration is available for held-out network.'))
 return d

def score(g):
 y=g.centroid_shift_hours.to_numpy();p=g.predicted_hours.to_numpy();w=1/g.groupby('example').example.transform('size').to_numpy();e=p-y
 denominator=float(np.sum(w*(y-np.average(y,weights=w))**2))
 return dict(n=len(g),examples=g.example.nunique(),mae_minutes=float(60*np.average(abs(e),weights=w)),
  r2=float(1-np.sum(w*e*e)/denominator) if denominator else None,
  relative_error_pct=float(100*np.average(abs(e)/np.maximum(abs(y),1/60),weights=w)))
def train(d):
 settings=read(BASE/'current_model_comparison_20261023/settings.json')
 choices={'geometry_9':[x for x in read(BASE/'nine_input_tuning/selections.json') if x['variant']=='same_grid'],
          'depth_informed_10':read(BASE/'ten_input_tuning_20261023/selections.json')}
 design=dict(input_sha256=sha(DEST/'input.parquet'),models=settings,choices=choices,seed=tuning.b.engine.SEED,
  validation='Fresh refits, 15 whole-network holdouts per model. Reuse each model\'s previous training-only fold-selected hyperparameters; no new parameter search. Equal-network training weights; training-only preprocessing/target standardization.',
  caveats=['Exploratory reused networks and previously explored feature sets, not external validation.',
  'Depth-informed model uses reference-calibrated Manning depths; not a prediction for an entirely uncalibrated river.',
  'Two models have different historical hyperparameters; difference is not a pure depth-feature ablation.'],
  script_sha256=sha(__file__),jobs=2)
 fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();save(DEST/'design.json',design)
 rows=[]
 for model,cols in [(m,s['columns']) for m,s in settings.items()]:
  tuning.b.COLUMNS=list(cols)
  for choice in choices[model]:
   ex=choice['held'];tr=d[d.example!=ex];te=d[d.example==ex]
   assert len(tr.example.unique())==14 and not set(tr.example)&set(te.example)
   for fold in choice['inner_groups']:assert ex not in fold['train']+fold['validation']
   path=DEST/'folds'/model/(ex+'.json')
   if path.exists():
    saved=json.loads(path.read_text());assert saved['fingerprint']==fp
   else:
    start=time.monotonic();artifact=tuning.b.fit(tr,choice['params'],jobs=2);pred=tuning.b.forest.predict(artifact,te)
    assert np.isfinite(pred).all()
    saved=dict(fingerprint=fp,held_network=ex,training_networks=sorted(tr.example.unique()),params=choice['params'],seconds=time.monotonic()-start,
     rows=[dict(example=r.example,state=r.state,cohort=r.cohort,edit_type=r.edit_type,centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),model=model) for r,v in zip(te.itertuples(),pred)])
    save(path,saved)
   rows.extend(saved['rows']);print('Fitted',model,ex,'seconds',round(saved['seconds'],1),flush=True)
 p=pd.DataFrame(rows);assert len(p)==4370 and not p.duplicated(['model','example','state']).any();p.to_parquet(DEST/'predictions.parquet',index=False)
 scores=[];delay=[]
 for model,g in p.groupby('model'):
  for name,z in [('all',g),*[(c,z) for c,z in g.groupby('cohort')],*[(c,z) for c,z in g.groupby('edit_type')]]:
   scores.append(dict(model=model,group=name,**score(z)))
   y=z.centroid_shift_hours*60;pred=z.predicted_hours*60;actual=y>1+1e-6;called=pred>1+1e-6;tp=int((actual&called).sum())
   delay.append(dict(model=model,group=name,n=len(z),actual_delays=int(actual.sum()),predicted_delays=int(called.sum()),true_delays=tp,
    missed_delays=int((actual&~called).sum()),false_delays=int((~actual&called).sum()),
    recall=tp/actual.sum() if actual.sum() else None,precision=tp/called.sum() if called.sum() else None))
 pd.DataFrame(scores).to_csv(DEST/'scores.csv',index=False);pd.DataFrame(delay).to_csv(DEST/'delay_detection.csv',index=False)
 flags=pd.read_csv(BASE/'three_run_review_20261029/preliminary_depth_sensitivity/comparison_reach_flags.csv',dtype={'example':str})
 errors=p.merge(flags,on=['example','state'],validate='many_to_one');errors['error_minutes']=60*(errors.predicted_hours-errors.centroid_shift_hours)
 errors.to_parquet(DEST/'errors_with_depth_flags.parquet',index=False)
 errors.loc[errors.groupby('model').error_minutes.transform(lambda x:abs(x)).sort_values(ascending=False).index].groupby('model').head(15).to_csv(DEST/'largest_errors.csv',index=False)
 diagnostics=[]
 for model,g in errors.groupby('model'):
  for col in ['depth5_any','depth10_any','depth20_any','radius10_any','radius20_any']:
   for flag in [False,True]:
    z=g[g[col]==flag]
    if len(z):diagnostics.append(dict(model=model,flag=col,flagged=flag,**score(z)))
 pd.DataFrame(diagnostics).to_csv(DEST/'scores_by_depth_flags.csv',index=False)
 save(DEST/'completion.json',dict(complete=True,fingerprint=fp,fits=30,rows=4370,production_modified=False,
  files={str(DEST/x):sha(DEST/x) for x in ['input.parquet','predictions.parquet','scores.csv','delay_detection.csv']}))
 print(pd.DataFrame(scores).to_string(index=False),flush=True);print(pd.DataFrame(delay).query("group=='all'").to_string(index=False),flush=True)

if __name__=='__main__':train(prepare())
