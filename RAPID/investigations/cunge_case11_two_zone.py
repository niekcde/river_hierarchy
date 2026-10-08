"""Two-zone Manning calibration: link 54 versus the remaining case-11 network.

BASED depths, reference Q, slopes, widths, counts and clock remain fixed.
No independent validation or production parameter changes are implied.
"""
from pathlib import Path
import sys,json,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(Path(__file__).parent)]
import cunge_based_depth_sensitivity as depth
from rapid_tools import cunge
from rapid_tools.engine import run_prepared_state
cal=depth.cal
OUT=ROOT/'RAPID/outputs/cunge_case11_two_zone'
CASE=None

def case():
 global CASE
 if CASE is None: CASE=depth.archived('11')
 return CASE

def evaluate(pair,clock=1,grid=1,save=False):
 nf,nr=map(float,pair);source=case();cc=dict(source);g=source['grid'].copy()
 # Algebraic adapter to the scalar-n evaluator; depths remain untouched.
 mask=g.id_link.eq(54);assert mask.sum()==1
 g.loc[mask,'celerity_times_n']*=nr/nf
 cc['grid']=g;cc['dest']=OUT/f'flat_{nf:.10f}_rest_{nr:.10f}';cc['dest'].mkdir(parents=True,exist_ok=True)
 r,w,f=cal.trial(cc,'cunge',nr,clock_factor=clock,grid_factor=grid,save_wave=save)
 r.update(n_flat=nf,n_rest=nr,n_ratio=nf/nr,output_dir=str(cc['dest']))
 if f is not None:
  # Independent check of the adapter against physical n values per link.
  local_n=np.where(f.id_link.eq(54),nf,nr)
  expected=(5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/local_n
  np.testing.assert_allclose(f.rapid_celerity_mps_raw,expected,rtol=1e-12)
  f['n_manning']=local_n
  f['celerity_times_n']=(5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)
 if save:return r,w,f
 return r

def main():
 OUT.mkdir(parents=True,exist_ok=True);source=case()
 metadata=dict(example='11',flat_link_id=54,flat_length_m=float(source['grid'].loc[source['grid'].id_link.eq(54),'link_length_m'].iloc[0]),
               bounds=[.01,.1],interpretation='Diagnostic search bounds, not established physically plausible roughness limits.',
               fixed=['BASED depth and exponent','reference discharge','width','slope','grid','forcing','routing timestep'],
               source_case=json.loads((source['dest']/'case.json').read_text()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'experiment.json').write_text(json.dumps(metadata,indent=2))
 cache={}
 with ProcessPoolExecutor(max_workers=2) as pool:
  def batch(pairs):
   keys=sorted(set((round(float(a),10),round(float(b),10)) for a,b in pairs)-set(cache))
   for pair,r in zip(keys,pool.map(evaluate,keys)):
    cache[pair]=r
    pd.DataFrame(cache.values()).to_csv(OUT/'trials.csv',index=False)
   good=[v for v in cache.values() if v['valid']]
   print(json.dumps(dict(trials=len(cache),valid=len(good),best_error_h=min(v['abs_error_h'] for v in good),max_valid_lag_h=max(v['lag_h'] for v in good))),flush=True)
  ns=np.array([.01,.015,.02,.025,.03,.035,.04,.05,.06,.07,.08,.09,.1])
  batch([(a,b) for a in ns for b in ns]+[(.07953125,.07953125)])
  def good():return [v for v in cache.values() if v['valid']]
  def timing():return min(good(),key=lambda r:(r['abs_error_h'],r['normalized_shape_rmse'],abs(np.log(r['n_ratio']))))
  def shape():return min(good(),key=lambda r:r['normalized_shape_rmse'])
  # Refine the timing and shape neighborhoods, without assuming monotonic lag.
  for radius in [.01,.003]:
   pairs=[]
   for b in [timing(),shape()]:
    xs=np.linspace(max(.01,b['n_flat']-radius),min(.1,b['n_flat']+radius),7)
    ys=np.linspace(max(.01,b['n_rest']-radius),min(.1,b['n_rest']+radius),7)
    pairs.extend((a,c) for a in xs for c in ys)
   batch(pairs)
  # Bracket every sampled timing-target crossing in each coordinate direction.
  for _ in range(6):
   pairs=[]
   for fixed,vary in [('n_flat','n_rest'),('n_rest','n_flat')]:
    df=pd.DataFrame(good())
    for val,group in df.groupby(fixed):
     rows=group.sort_values(vary).to_dict('records')
     for a,b in zip(rows[:-1],rows[1:]):
      if a['error_h']*b['error_h']<0 and b[vary]-a[vary]>1e-7:
       mid=(a[vary]+b[vary])/2;pairs.append((val,mid) if fixed=='n_flat' else (mid,val))
   if not pairs or timing()['abs_error_h']<=source['step']/7200:break
   batch(pairs)
 candidates=[]
 for label,b in [('best_timing',timing()),('best_shape',shape())]:
  r,w,f=evaluate((b['n_flat'],b['n_rest']),save=True)
  half=evaluate((b['n_flat'],b['n_rest']),clock=2)
  fine=evaluate((b['n_flat'],b['n_rest']),grid=2)
  r.update(candidate=label,half_clock_lag_change_h=half['lag_h']-r['lag_h'],double_grid_lag_change_h=fine['lag_h']-r['lag_h'],
           double_grid_peak_change_pct=100*(fine['peak_cms']/r['peak_cms']-1),n_flat_at_boundary=r['n_flat'] in [.01,.1],n_rest_at_boundary=r['n_rest'] in [.01,.1])
  dest=OUT/label;dest.mkdir(exist_ok=True);w.to_parquet(dest/'wave.parquet',index=False);f.to_parquet(dest/'hydraulic_links.parquet',index=False)
  # Cross-check changing forcing against the actual sparse RAPID engine.
  changing=np.flatnonzero(abs(np.diff(source['event']))>0);stop=min(len(source['event']),int(changing[0])+32)
  probe=source['event'][:stop];pm=cunge.write_prepared_state(f,probe,source['step'],source['dt'],dest/'engine_probe',metadata=metadata)
  _,q=run_prepared_state(dest/'engine_probe',return_qout=True);actual=q[:,np.array(pm['outlet_reach_ids'])-1].sum(axis=1)
  expected,_=cunge.route(f,probe,source['nsub'],source['dt']);r['engine_probe_max_error_cms']=float(np.max(abs(actual-expected)))
  assert r['engine_probe_max_error_cms']<1e-8*max(1,float(np.ptp(probe)))
  (dest/'result.json').write_text(json.dumps(r,indent=2));candidates.append(r)
 pd.DataFrame(candidates).to_csv(OUT/'candidates.csv',index=False)
 # Equal zone values must reproduce the previous one-n result exactly.
 diag=cache[(.07953125,.07953125)]
 assert abs(diag['lag_h']-source['old'].lag_h)<1e-9
 assert abs(diag['peak_cms']-source['old'].peak_cms)<1e-6
 print('DONE',flush=True)

if __name__=='__main__':main()
