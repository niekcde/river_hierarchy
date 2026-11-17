"""Descriptive first pass on completed fixed-Manning real-network responses."""
import json
import numpy as np
import pandas as pd
from review import SOURCE,DEST,RUNS,ROOT,digest

OUT=DEST/'fixed_manning_real_cases'
BASE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
EPS=1e-6 # minutes; avoids floating representation of exact minute peak bins.

def counts(g,metric,group):
 z=g[metric];positive=z>EPS;exact=np.isclose(z,1,atol=EPS,rtol=0)
 return dict(group=group,metric=metric,comparisons=len(g),delays=int(positive.sum()),
  subminute_delays=int((positive&(z<1-EPS)).sum()),one_minute_delays=int(exact.sum()),
  delays_gt1min=int((z>1+EPS).sum()),advances_gt1min=int((z< -1-EPS).sum()),
  within_plusminus1min=int((abs(z)<=1+EPS).sum()),median_min=z.median(),
  minimum_min=z.min(),maximum_min=z.max())

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 p=SOURCE/RUNS[2]/'responses.csv';receipt=json.loads((SOURCE/RUNS[2]/'completion.json').read_text())
 assert digest(p)==receipt['files'][str(p)]
 f=pd.read_csv(p,dtype={'example':str});f=f[f.state!='fully_collapsed'].copy()
 inventory=BASE/'validation_group_inventory_20261023/records.parquet'
 labels=pd.read_parquet(inventory)[['example','state','edit_type','selected_unit_ids','compound_ids','nested']]
 f=f.merge(labels,on=['example','state'],how='left',validate='many_to_one');assert f.edit_type.notna().all()
 f['centroid_min']=60*f.centroid_shift_hours;f['peak_min']=(f.child_peak_seconds-f.parent_peak_seconds)/60
 flags=pd.read_csv(DEST/'preliminary_depth_sensitivity/comparison_reach_flags.csv',dtype={'example':str})
 f=f.merge(flags,on=['example','state'],validate='many_to_one')
 based=pd.read_csv(SOURCE/RUNS[1]/'responses.csv',dtype={'example':str})
 based=based[['example','state','window','centroid_shift_hours']].rename(columns={'centroid_shift_hours':'based_centroid_hours'})
 f=f.merge(based,on=['example','state','window'],validate='one_to_one')
 f['based_centroid_min']=60*f.based_centroid_hours
 f.to_parquet(OUT/'case_inventory.parquet',index=False)
 rows=[]
 for window,g in f.groupby('window'):
  for metric in ['centroid_min','peak_min']:
   rows.append(dict(window=window,**counts(g,metric,'All')))
   for kind,sub in g.groupby('edit_type'):rows.append(dict(window=window,**counts(sub,metric,kind)))
 pd.DataFrame(rows).to_csv(OUT/'response_counts_by_type.csv',index=False)
 main=f[f.window=='0.5%'].copy();assert len(main)==2185
 per=[]
 for ex,g in main.groupby('example'):
  per.append(dict(example=ex,**counts(g,'centroid_min','All')))
 pd.DataFrame(per).to_csv(OUT/'centroid_counts_by_example.csv',index=False)
 delayed=main[main.centroid_min>1+EPS].copy().sort_values('centroid_min',ascending=False)
 delayed['based_also_delayed_gt1min']=delayed.based_centroid_min>1+EPS
 across=f.pivot(index=['example','state'],columns='window',values='centroid_min')
 robust=(across>1+EPS).all(axis=1).rename('delayed_gt1min_all_windows')
 delayed=delayed.merge(robust,on=['example','state'],validate='one_to_one')
 cols=['example','state','edit_type','centroid_min','peak_min','based_centroid_min','based_also_delayed_gt1min',
  'delayed_gt1min_all_windows','peak_excess_change_pct','depth5_any','depth10_any','depth20_any','radius10_any','radius20_any','shared_changed_inputs']
 delayed[cols].to_csv(OUT/'centroid_delays_gt1min.csv',index=False)
 main[(main.edit_type=='Simple')&((main.centroid_min>EPS)|(main.peak_min>EPS))].to_csv(OUT/'simple_unit_delays.csv',index=False)
 main[(main.centroid_min>1+EPS)&(main.peak_min<=EPS)].to_csv(OUT/'centroid_delay_without_peak_delay.csv',index=False)
 summary=dict(comparisons=len(main),examples=main.example.nunique(),centroid_delays_gt1min=len(delayed),
  centroid_delays_gt1min_all_windows=int(delayed.delayed_gt1min_all_windows.sum()),
  centroid_delays_also_based_gt1min=int(delayed.based_also_delayed_gt1min.sum()),
  centroid_delays_with_based_advance_gt1min=int((delayed.based_centroid_min< -1-EPS).sum()),
  centroid_delay_gt1min_but_peak_not_delayed=int(((main.centroid_min>1+EPS)&(main.peak_min<=EPS)).sum()),
  source_response_sha256=digest(p),inventory_sha256=digest(inventory),script_sha256=digest(__file__),
  qualifications=['Fixed depth and constant fitted n/beta, not variable-depth results.',
   'Case labels reused from existing inventory; old responses and RF predictions are not reused.',
   'All counts are collapse-reference comparisons, not unique reaches or independent observations.',
   'Depth flags inherited from fixed-Manning reach audit; not proofs of physical invalidity.',
   'Existing slope and flow-allocation differences between configurations are retained.',
   'Timing direction alone does not identify a diffusion mechanism.'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(pd.DataFrame(rows).query("window=='0.5%'").to_string(index=False));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
