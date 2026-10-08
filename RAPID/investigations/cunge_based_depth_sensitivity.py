"""Bounded BASED-depth scaling: identifiability audit and case-11 recalibration.

Depth multipliers are sensitivity bounds, not measured confidence intervals.
Q, widths, slopes, forcing and depth exponent remain fixed. No production edits.
"""
from pathlib import Path
import sys,json
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import cunge_calibration as cal
from rapid_tools import cunge
OUT=ROOT/'RAPID/outputs/cunge_based_depth_sensitivity'
FACTORS=[.75,1.,1.25,1.5,2.]

def archived(ex):
 rows=pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str});row=rows[rows.example.eq(ex)&rows.method.eq('cunge')].iloc[0]
 p=Path(row.output_dir);m=json.loads((p/'case.json').read_text());w=cal.original.read_schedule().loc[ex]
 wave=pd.read_parquet(p/'cunge_best_wave.parquet');event=wave.loc[pd.to_datetime(wave.time_utc,utc=True)<=w.last_sample,'q_inflow_cms'].to_numpy()
 target=m['target'];target['series']={name:pd.read_parquet(p/f'observed_{name}.parquet') for name in ['upstream','downstream']}
 return dict(ex=ex,window=w,old=row,target=target,grid=pd.read_parquet(p/'frozen_grid.parquet'),event=event,step=m['forcing_step_s'],dt=m['routing_dt_s'],nsub=m['nsub'],dest=p,manifest=m)

def adjust(case,factor):
 g=case['grid'].copy();g['depth_m']*=factor;g['celerity_times_n']*=factor**(2/3)
 # Common grid across factors, selected at fastest depth/n combination.
 fast=cunge.freeze_grid(case['grid'].assign(depth_m=case['grid'].depth_m*max(FACTORS)),case['manifest']['reference_discharge_cms'],alpha=2.)
 g['count']=fast['count'];g['rapid_effective_length_m']=g.link_length_m/g['count']
 result=dict(case);result['grid']=g;result['dt'],result['nsub']=cunge.common_clock(fast,case['step'])
 result['dest']=OUT/case['ex']/f'depth_{factor:g}';result['dest'].mkdir(parents=True,exist_ok=True)
 return result

def audit():
 rows=[]
 for ex in pd.read_csv(cal.OUT/'comparison.csv',dtype={'example':str}).example.unique():
  case=archived(ex);n=float(case['old'].n_manning)
  for fac in FACTORS:
   mapped=n*fac**(2/3)
   # Extend validation bounds only for this identity proof, not the fit.
   g=case['grid'].copy();g['depth_m']*=fac;g['celerity_times_n']*=fac**(2/3)
   g['grid_n_max']=max(.1,mapped);g['grid_n_min']=min(.01,mapped)
   a=cunge.parameters(g,mapped)
   bgrid=g.copy();bgrid['depth_m']/=fac;bgrid['celerity_times_n']/=fac**(2/3)
   b=cunge.parameters(bgrid,n)
   errs={k:float(np.max(abs(a[k]-b[k])/np.maximum(abs(b[k]),1e-12))) for k in ['rapid_celerity_mps','rapid_k','rapid_x']}
   assert max(errs.values())<1e-10
   rows.append(dict(example=ex,depth_factor=fac,kb_factor=1/fac,original_n=n,equivalent_n=mapped,equivalent_n_within_search=.01<=mapped<=.1,**errs))
 pd.DataFrame(rows).to_csv(OUT/'all_network_parameter_equivalence.csv',index=False)
 print('Verified parameter equivalence: 16 networks x 5 depth factors.',flush=True)

def run(factor):
 case=adjust(archived('11'),factor);cache={}
 def evaluate(n):
  key=round(float(n),12)
  if key not in cache:
   cache[key]=cal.trial(case,'cunge',key)[0];pd.DataFrame(cache.values()).to_csv(case['dest']/'trials.csv',index=False)
  return cache[key]
 # Dense scan, plus exact equivalent of existing best, if inside bounds.
 samples=np.unique(np.r_[np.linspace(.01,.1,31),.035,.07953125*factor**(2/3)])
 for n in samples:
  if .01<=n<=.1:evaluate(n)
 valid=lambda:sorted([r for r in cache.values() if r['valid']],key=lambda r:r['n_manning'])
 vals=valid()
 if not vals:raise ValueError('No settled, positive valid trial')
 best=lambda:min(valid(),key=lambda r:(r['abs_error_h'],abs(r['n_manning']-.035)))
 vals=valid();i=vals.index(best());lo=vals[max(0,i-1)]['n_manning'];hi=vals[min(len(vals)-1,i+1)]['n_manning']
 for n in np.linspace(lo,hi,9):evaluate(n)
 for _ in range(9):
  vals=valid()
  if best()['abs_error_h']<=case['step']/7200:break
  cross=[(a,b) for a,b in zip(vals[:-1],vals[1:]) if a['error_h']*b['error_h']<0]
  if not cross:break
  for a,b in cross:evaluate((a['n_manning']+b['n_manning'])/2)
 chosen=best();r,w,f=cal.trial(case,'cunge',chosen['n_manning'],save_wave=True)
 w.to_parquet(case['dest']/'best_wave.parquet',index=False)
 # Ensure the candidate is insensitive to half timestep and doubled grid.
 half=cal.trial(case,'cunge',r['n_manning'],clock_factor=2)[0]
 fine=cal.trial(case,'cunge',r['n_manning'],grid_factor=2)[0]
 r.update(depth_factor=factor,kb_factor=1/factor,effective_original_depth_n=r['n_manning']/factor**(2/3),
          reference_depth_m=4.680543775233037*factor,link54_depth_m=float(f.loc[f.id_link.eq(54),'depth_m'].iloc[0]),
          half_clock_lag_change_h=half['lag_h']-r['lag_h'],double_grid_lag_change_h=fine['lag_h']-r['lag_h'],
          trials=len(cache),valid_trials=len(valid()),best_unfiltered_error_h=min(v['abs_error_h'] for v in cache.values()),
          output_dir=str(case['dest']))
 (case['dest']/'best.json').write_text(json.dumps(r,indent=2))
 print(json.dumps({k:r[k] for k in ['depth_factor','n_manning','lag_h','abs_error_h','within_sampling','normalized_shape_rmse','valid_trials']}),flush=True)
 return r

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);audit()
 with ProcessPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,FACTORS))
 pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False)
 print('DONE',flush=True)
