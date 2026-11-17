"""Read-only paired BASED/Manning comparison and reach-flag exclusion scan."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from review import ROOT,SOURCE,DEST,RUNS,digest

OUT=DEST/'preliminary_depth_sensitivity'
CRITERIA={'depth5':lambda f:f.depth_m>5,'depth10':lambda f:f.depth_m>10,'depth20':lambda f:f.depth_m>20,
 'radius10':lambda f:(1+2*f.depth_m/f.width)**(2/3)-1>.10,
 'radius20':lambda f:(1+2*f.depth_m/f.width)**(2/3)-1>.20}

def indexed(f):
 f=f.copy();f.index=pd.MultiIndex.from_arrays([f.u,f.v,f.link_length_m.round(6),f.width.round(6)])
 assert f.index.is_unique
 return f

def summarize(g,name,window):
 a=g.centroid_min_based;b=g.centroid_min_manning;d=b-a
 return dict(group=name,window=window,comparisons=len(g),examples=g.example.nunique(),
 median_abs_centroid_difference_min=d.abs().median(),p95_abs_centroid_difference_min=d.abs().quantile(.95),
 max_abs_centroid_difference_min=d.abs().max(),centroid_difference_gt1min=int((d.abs()>1+1e-8).sum()),
 centroid_sign_flips=int((a*b<0).sum()),material_sign_flips=int(((a*b<0)&(a.abs()>1+1e-8)&(b.abs()>1+1e-8)).sum()),
 based_delays_gt1min=int((a>1+1e-8).sum()),manning_delays_gt1min=int((b>1+1e-8).sum()),
 median_abs_peak_difference_min=g.peak_difference_min.abs().median(),p95_abs_peak_difference_min=g.peak_difference_min.abs().quantile(.95),
 max_abs_peak_difference_min=g.peak_difference_min.abs().max(),
 p95_abs_attenuation_difference_pp=g.attenuation_difference_pp.abs().quantile(.95))

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 frames=[]
 for run in RUNS[1:]:
  receipt=json.loads((SOURCE/run/'completion.json').read_text())
  p=SOURCE/run/'responses.csv';assert digest(p)==receipt['files'][str(p)]
  f=pd.read_csv(p,dtype={'example':str});f=f[f.state!='fully_collapsed'].copy()
  f['centroid_min']=60*f.centroid_shift_hours
  f['peak_min']=(f.child_peak_seconds-f.parent_peak_seconds)/60
  frames.append(f)
 m=frames[0].merge(frames[1],on=['example','state','window'],suffixes=('_based','_manning'),validate='one_to_one')
 m['centroid_difference_min']=m.centroid_min_manning-m.centroid_min_based
 m['peak_difference_min']=m.peak_min_manning-m.peak_min_based
 m['attenuation_difference_pp']=m.peak_excess_change_pct_manning-m.peak_excess_change_pct_based
 rows=[];distinct={c:set() for c in CRITERIA}
 for ex,group in m.groupby('example'):
  parent=indexed(pd.read_parquet(SOURCE/RUNS[2]/'states'/ex/'S000_base'/'parameters.parquet'))
  pf={c:fn(parent).to_numpy() for c,fn in CRITERIA.items()}
  for state in group.state.unique():
   folder=SOURCE/RUNS[2]/'states'/ex/state;p=folder/'parameters.parquet'
   receipt=json.loads((folder/'completion.json').read_text());assert digest(p)==receipt['files'][str(p)]
   child=indexed(pd.read_parquet(p));shared=parent.index.intersection(child.index)
   changed=np.zeros(len(shared),bool)
   for col in ['slope','reference_discharge_cms','n_manning','diffusion_multiplier']:
    changed|=~np.isclose(parent.loc[shared,col],child.loc[shared,col],rtol=1e-12,atol=1e-12)
   changedkeys=shared[changed]
   pa=~parent.index.isin(shared)|parent.index.isin(changedkeys)
   ca=~child.index.isin(shared)|child.index.isin(changedkeys)
   row=dict(example=ex,state=state,shared_reaches=len(shared),shared_changed_inputs=int(changed.sum()),
      removed_reaches=int((~parent.index.isin(shared)).sum()),new_reaches=int((~child.index.isin(shared)).sum()))
   for c,fn in CRITERIA.items():
    cf=fn(child).to_numpy();pflag=pf[c]
    row[c+'_any']=bool(pflag.any() or cf.any())
    row[c+'_affected']=bool((pflag&pa).any() or (cf&ca).any())
    row[c+'_unchanged_shared']=bool((pflag&~pa).any())
    for f,flag in [(parent,pflag),(child,cf)]:
     distinct[c].update((ex,*key) for key in f.index[flag])
   rows.append(row)
  print('Scanned',ex,len(group.state.unique()),'comparisons',flush=True)
 flags=pd.DataFrame(rows);flags.to_csv(OUT/'comparison_reach_flags.csv',index=False)
 m=m.merge(flags,on=['example','state'],validate='many_to_one')
 m['case_type']=np.select([m.state.str.startswith('island'),m.state.str.startswith('CB'),m.state.str.startswith('S')],['island','compound_bubble','unit'],default='combined')
 m.to_csv(OUT/'paired_comparisons.csv',index=False)
 stats=[]
 for window,g in m.groupby('window'):
  stats.append(summarize(g,'all',window))
  for c in CRITERIA:
   for scope in ['any','affected']:
    for flag in [True,False]:
     sub=g[g[c+'_'+scope]==flag];stats.append(summarize(sub,f'{c}_{scope}_'+('flagged' if flag else 'unflagged'),window))
  for ex,sub in g.groupby('example'):stats.append(summarize(sub,'example_'+ex,window))
  for kind,sub in g.groupby('case_type'):stats.append(summarize(sub,'type_'+kind,window))
 pd.DataFrame(stats).to_csv(OUT/'group_summary.csv',index=False)
 g=m[m.window=='0.5%'].copy()
 g.loc[g.centroid_difference_min.abs().nlargest(25).index].to_csv(OUT/'largest_centroid_disagreements.csv',index=False)
 flips=g[(g.centroid_min_based*g.centroid_min_manning<0)&(g.centroid_min_based.abs()>1+1e-8)&(g.centroid_min_manning.abs()>1+1e-8)]
 flips.to_csv(OUT/'material_sign_reversals.csv',index=False)
 (OUT/'method.json').write_text(json.dumps(dict(script_sha256=digest(__file__),comparison_count=len(flags),
 distinct_flagged_geometry_counts={c:len(s) for c,s in distinct.items()},
 source_response_hashes={run:digest(SOURCE/run/'responses.csv') for run in RUNS[1:]},
 definitions={'any':'Either member of the pair contains a flagged reach; includes unchanged inherited reaches.',
 'affected':'Flagged reach is removed/new geometry or a shared geometry with changed slope, allocated Qref, n or beta.',
 'identity':'Example, endpoints, length and width rounded to 1 micrometre; not reach ID.',
 'radius':'Wide-section discharge overestimate at fixed depth,n,slope compared with full rectangular equation; not routed discharge error.',
 'material_sign_flip':'Opposite signs with magnitude >1 minute under both formulations; descriptive threshold, not uncertainty bound.'},
 limitations=['Flags are from fixed-Manning depths only, not physical validity tests.','Comparison includes independently fitted n/beta, not a depth-only intervention.',
 'Comparisons share reaches and parents; counts are not independent observations.','No new routing or RF fitting.','Exclusions can change example composition; inspect per-example results.','Shared inherited slope changes are retained as existing inputs.']),indent=2)+'\n')
 print(pd.DataFrame(stats).query("window=='0.5%' and not group.str.startswith('example_') and not group.str.startswith('type_')",engine='python').to_string(index=False),flush=True)

if __name__=='__main__':main()
