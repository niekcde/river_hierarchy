"""Read-only comparison and fitted-depth screening of completed three-run batch."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028')
DEST=ROOT/'RAPID/outputs/uncapped_event_window_20261020/three_run_review_20261029'
RUNS=['01_based_slope_geometry_correction','02_based_midpoint','03_manning_midpoint']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(DEST/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')
def main():
 DEST.mkdir(exist_ok=True)
 for run in RUNS:
  rec=json.loads((SOURCE/run/'completion.json').read_text())
  for path,h in rec['files'].items():assert digest(path)==h,path
 print('Stage aggregates verified',flush=True)
 frames=[pd.read_csv(SOURCE/r/'responses.csv',dtype={'example':str}) for r in RUNS]
 a,b=frames[:2];keys=['example','state','window'];m=a.merge(b,on=keys,suffixes=('_old','_mid'),validate='one_to_one');m=m[m.state.ne('fully_collapsed')].copy()
 for name,factor in [('centroid_shift_hours',60),('peak_shift_hours',60),('peak_excess_change_pct',1),('duration80_change_pct',1)]:
  m[name+'_difference']=factor*(m[name+'_mid']-m[name+'_old'])
 m.to_csv(DEST/'based_paired_responses.csv',index=False)
 sums=[]
 for window,d in m.groupby('window'):
  for col in ['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct']:
   x=d[col+'_difference'];sums.append(dict(window=window,metric=col,comparisons=len(d),median_signed=x.median(),median_absolute=x.abs().median(),p95_absolute=x.abs().quantile(.95),maximum_absolute=x.abs().max(),above_one=int(x.abs().gt(1+1e-8).sum())))
 pd.DataFrame(sums).to_csv(DEST/'based_difference_summary.csv',index=False)
 # Counts describe collapse-minus-own-parent responses, not absolute event times.
 signs=[]
 for run,d in zip(RUNS,frames):
  d=d[d.state.ne('fully_collapsed')]
  for window,g in d.groupby('window'):
   for col in ['centroid_shift_hours','peak_shift_hours']:
    z=(g.child_peak_seconds-g.parent_peak_seconds)/60 if col=='peak_shift_hours' else g[col]*60
    signs.append(dict(run=run,window=window,metric=col,total=len(g),delays=int(z.gt(0).sum()),subminute_delays=int((z.gt(0)&z.lt(1-1e-8)).sum()),one_minute_delays=int(np.isclose(z,1,rtol=0,atol=1e-8).sum()),delays_over_1min=int(z.gt(1+1e-8).sum())))
 pd.DataFrame(signs).to_csv(DEST/'delay_counts.csv',index=False)
 d=m[m.window.eq('0.5%')].copy()
 d['centroid_sign_flip']=d.centroid_shift_hours_old*d.centroid_shift_hours_mid<0
 d['centroid_material_sign_flip']=(d.centroid_shift_hours_old*60>1)&(d.centroid_shift_hours_mid*60< -1)|(d.centroid_shift_hours_old*60< -1)&(d.centroid_shift_hours_mid*60>1)
 d.nlargest(15,'centroid_shift_hours_difference').to_csv(DEST/'largest_centroid_increases.csv',index=False)
 d.loc[d.centroid_shift_hours_difference.abs().nlargest(15).index].to_csv(DEST/'largest_absolute_centroid_changes.csv',index=False)
 per=[]
 for ex,g in d.groupby('example'):
  per.append(dict(example=ex,comparisons=len(g),median_abs_centroid_difference_min=g.centroid_shift_hours_difference.abs().median(),max_abs_centroid_difference_min=g.centroid_shift_hours_difference.abs().max(),centroid_sign_flips=int(g.centroid_sign_flip.sum()),material_sign_flips=int(g.centroid_material_sign_flip.sum())))
 pd.DataFrame(per).to_csv(DEST/'based_by_example.csv',index=False)
 parents=[]
 for run in RUNS:
  for p in sorted((SOURCE/run/'calibration').glob('*/result.json')):
   ex=p.parent.name;r=json.loads(p.read_text());v=r['selected']
   parents.append(dict(run=run,example=ex,n=v['n_manning'],beta=v['beta'],timing_error_minutes=v.get('timing_error_minutes'),height_distance_cms=v.get('height_distance_cms')))
 pd.DataFrame(parents).to_csv(DEST/'calibration_comparison.csv',index=False)
 # Parent hydrograph changes are reported separately from collapse responses.
 pa=a[(a.window=='0.5%')].drop_duplicates('example');pb=b[(b.window=='0.5%')].drop_duplicates('example')
 pm=pa.merge(pb,on='example',suffixes=('_old','_mid'))
 parentrows=[]
 for x in pm.itertuples():parentrows.append(dict(example=x.example,centroid_change_min=(x.parent_centroid_seconds_mid-x.parent_centroid_seconds_old)/60,peak_time_change_min=(x.parent_peak_seconds_mid-x.parent_peak_seconds_old)/60,peak_height_change_pct=100*(x.parent_peak_cms_mid/x.parent_peak_cms_old-1)))
 pd.DataFrame(parentrows).to_csv(DEST/'based_parent_changes.csv',index=False)
 # Inspect every fitted Manning parameter table, retaining state/reach context.
 states=[];extremes=[];parentlinks=[]
 for exdir in sorted((SOURCE/RUNS[2]/'states').iterdir()):
  for folder in sorted(exdir.iterdir()):
   if folder.name=='single_centerline':continue
   path=folder/'parameters.parquet';receipt=json.loads((folder/'completion.json').read_text());assert digest(path)==receipt['files'][str(path)]
   f=pd.read_parquet(path);h=f.depth_m;ratio=h/f.width
   row=dict(example=exdir.name,state=folder.name,links=len(f),n=float(f.n_manning.iloc[0]),beta=float(f.diffusion_multiplier.iloc[0]),depth_max_m=float(h.max()),depth_median_m=float(h.median()),depth_width_ratio_max=float(ratio.max()))
   for lim in [10,20,30,50]:row['links_depth_gt_'+str(lim)]=int((h>lim).sum())
   for lim in [.1,.2,.5,1]:row['links_ratio_gt_'+str(lim)]=int((ratio>lim).sum())
   states.append(row)
   flag=(h>20)|(ratio>.2)
   top=f.loc[flag,['id_link','width','slope','depth_m','n_manning','reference_discharge_cms']].copy();top['depth_width_ratio']=ratio[flag];top['example']=exdir.name;top['state']=folder.name;extremes.append(top)
   if folder.name=='S000_base':parentlinks.append(f.assign(example=exdir.name))
  print('Depth audit',exdir.name,flush=True)
 ds=pd.DataFrame(states);ds.to_csv(DEST/'manning_depth_by_state.csv',index=False)
 flags=pd.concat(extremes,ignore_index=True);flags.to_csv(DEST/'manning_flagged_reach_occurrences.csv',index=False)
 pd.concat(parentlinks,ignore_index=True).to_parquet(DEST/'manning_parent_parameters.parquet',index=False)
 counts=[]
 for group,g in [('all_including_parents',ds),('parents',ds[ds.state.eq('S000_base')]),('collapses',ds[ds.state.ne('S000_base')])]:
  for col in [c for c in ds if c.startswith('links_')]:counts.append(dict(group=group,criterion=col,states=int(g[col].gt(0).sum()),reach_occurrences=int(g[col].sum()),total_states=len(g),total_reach_occurrences=int(g.links.sum())))
 pd.DataFrame(counts).to_csv(DEST/'manning_depth_counts.csv',index=False)
 save('summary.json',dict(based_comparisons=len(d),centroid_sign_flips=int(d.centroid_sign_flip.sum()),centroid_material_sign_flips=int(d.centroid_material_sign_flip.sum()),
  max_fitted_depth_m=float(ds.depth_max_m.max()),max_fitted_depth_width_ratio=float(ds.depth_width_ratio_max.max()),depth_states=len(ds),
  qualifications=['Nonsegmented single-centerline controls excluded. Depth thresholds are screening flags, not physical admissibility limits.','Repeated reaches across states are counted as reach occurrences, not independent physical reaches.','BASED comparison is run1 slope/geometry-corrected original Qref versus run2 midpoint Qref plus depth/calibration/grid changes.'],
  source_aggregate_hashes={str(SOURCE/r/'completion.json'):digest(SOURCE/r/'completion.json') for r in RUNS},script_sha256=digest(__file__)))
 print(pd.DataFrame(sums).to_string(index=False),flush=True)
 print(pd.DataFrame(counts).to_string(index=False),flush=True)
if __name__=='__main__':main()
