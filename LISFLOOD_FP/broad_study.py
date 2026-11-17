"""15-network baseline/calibration and selected collapse transfer, 100m working grid."""
import os,sys,json,time,hashlib,traceback,argparse,zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import verify_inertial56 as v
ROOT=v.ROOT
OUT=ROOT/'LISFLOOD_FP/results/broad_15_100m';OUT.mkdir(parents=True,exist_ok=True)
v.OUT=OUT
sys.path.insert(0,str(ROOT/'RAPID/investigations'))
from calibrate_common_grid_height import feature
PLAN=json.loads((ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
CASES={c['example']:c for c in PLAN['cases']}
CALDATA=ROOT/'RAPID/outputs/master_workflow_verification_20261009/calibration/bounded_beta_20261009'
TARGETS={r['example']:r for r in json.loads((CALDATA/'timing_targets.json').read_text())['records']}
HEIGHTS={r['example']:r for r in json.loads((CALDATA/'height_targets.json').read_text())['records']}
SUBSET=['02','15','22','37','46','48','56']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
 def clean(x):
  if isinstance(x,np.generic):return clean(x.item())
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [clean(v) for v in x]
  if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
  return x
 p=Path(p);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(clean(r),indent=2,allow_nan=False));tmp.replace(p)
def frame(ex,state):return pd.read_parquet(OUT/'inputs'/ex/(state+'.parquet'))
def forcing(ex):return pd.read_parquet(OUT/'inputs'/ex/'forcing.parquet')
def build(cells,n):
 f=cells.reset_index(drop=True);N=len(f);a=[];b=[];by_u={u:g.index.to_numpy() for u,g in f.groupby('u')}
 for i,r in enumerate(f.itertuples()):
  for j in by_u.get(r.v,[N]):a.append(i);b.append(j)
 a=np.array(a,dtype=np.int64);b=np.array(b,dtype=np.int64);terminal=b==N
 assert terminal.sum()==1
 w=f.width.to_numpy(float);dx=f.link_length_m.to_numpy(float);z=((f.head_us+f.head_ds)/2).to_numpy(float)
 r=f.iloc[a[terminal][0]];slope=float((r.head_us-r.head_ds)/r.link_length_m)
 assert slope>0 and np.all(dx>0) and np.all(w>0)
 source=np.flatnonzero(~f.u.isin(f.v));assert len(source)==1
 wf=np.minimum(w[a],w[np.minimum(b,N-1)]);wf[terminal]=w[a[terminal]]
 dxf=.5*(dx[a]+dx[np.minimum(b,N-1)]);dxf[terminal]=dx[a[terminal]]/2
 return dict(z=np.r_[z,float(r.head_ds)],w=w,dx=dx,a=a,b=b,wf=wf,dxf=dxf,source=int(source[0]),n=float(n),outlet_slope=slope,stage=0.,normal=True)

def prepare():
 rows=[];selection={};sources={}
 for ex,c in CASES.items():
  dest=OUT/'inputs'/ex;dest.mkdir(exist_ok=True,parents=True)
  j=next(j for j in c['jobs'] if j['state_id']=='S000_base');f=pd.read_parquet(j['parameters'])
  assert sha(j['parameters'])==c['source_sha256'][j['parameters']]
  f.to_parquet(dest/'S000_base.parquet',index=False)
  receipt=json.loads((ROOT/f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json').read_text());saved=Path(receipt['final_directory'])
  assert sha(saved/'forcing.parquet')==receipt['files'][str(saved/'forcing.parquet')]
  q=pd.read_parquet(saved/'forcing.parquet');q.to_parquet(dest/'forcing.parquet',index=False)
  obs=pd.read_parquet(CALDATA/ex/'observed_downstream.parquet');obs.to_parquet(dest/'observed_downstream.parquet',index=False)
  pd.read_parquet(saved/'S000_base/outlet.parquet').to_parquet(dest/'mc_base.parquet',index=False)
  candidates=[]
  for j in c['jobs']:
   if j['state_id'] in ['S000_base','single_centerline']:continue
   ff=pd.read_parquet(j['parameters']);reduction=1-len(ff)/len(f)
   if 0<reduction<.95:candidates.append((j,ff,reduction))
  chosen=[]
  for target in [.05,.30]:
   available=[x for x in candidates if x[0]['state_id'] not in chosen]
   if not available:break
   j,ff,reduction=min(available,key=lambda x:abs(x[2]-target))
   state=j['state_id'];chosen.append(state)
   assert sha(j['parameters'])==c['source_sha256'][j['parameters']]
   ff.to_parquet(dest/(state+'.parquet'),index=False)
   pd.read_parquet(saved/state/'outlet.parquet').to_parquet(dest/(state+'_mc.parquet'),index=False)
   sources[f'{ex}/{state}']=dict(path=j['parameters'],sha256=sha(j['parameters']),reduction=reduction,aliases=j.get('aliases'))
  selection[ex]=chosen
  sources[f'{ex}/S000_base']=dict(path=next(j['parameters'] for j in c['jobs'] if j['state_id']=='S000_base'),forcing_path=str(saved/'forcing.parquet'),forcing_sha256=sha(saved/'forcing.parquet'))
  rows.append(dict(example=ex,physical_reaches=len(f),cells100=int(np.ceil(f.link_length_m/100).sum()),total_length_km=float(f.link_length_m.sum()/1000),minimum_length_m=float(f.link_length_m.min()),length_ratio=float(f.link_length_m.max()/f.link_length_m.min()),width_ratio=float(f.width.max()/f.width.min()),minimum_slope=float(f.slope.min()),maximum_branches=int(f.groupby('u').size().max()),days=len(q)/1440,mc_n=float(f.n_manning.iloc[0]),full_experiment_states=len(c['jobs']),selected_children=','.join(chosen)))
 pd.DataFrame(rows).to_csv(OUT/'inventory.csv',index=False)
 save(OUT/'selection.json',dict(refinement_examples=SUBSET,children=selection,sources=sources,rationale='All 15 originals and two collapses per example (~5% and ~30% physical-reach reduction). Refinement subset covers high roughness (02), three-way splits (15,22,37), largest network/width contrast (22), shortest reaches (37), flat long network (46), steep network (48), previous control (56). Selection uses geometry, not new model outcomes.'))
 save(OUT/'protocol.json',dict(working_max_length_m=100,cfl=.35,cfl_retries=[.175,.0875],normal_flow=True,roughness_bounds=[.01,.1],calibration='Original network only; inherited reviewed crest timing feature, target tolerance one minute; height secondary diagnostic, no geometry or outlet fitting. Full-event final validation, frozen n for selected collapses.',target_source=str(CALDATA/'timing_targets.json'),target_sha256=sha(CALDATA/'timing_targets.json'),kernel_sha256=sha(v.KERNEL),refinement=[200,100,50],scope='Steps 1-4 only. Full all-collapse production deferred.'))
 print(pd.DataFrame(rows).to_string(index=False),flush=True)

def initial_guess(f,cells,m,Q):
 alloc=f.set_index('id_link').reference_flow_fraction
 shares=np.array([float(alloc.loc[i]) for i in cells.id_link]);slope=((cells.head_us-cells.head_ds)/cells.link_length_m).to_numpy()
 h=(Q*shares*m['n']/(m['w']*np.sqrt(slope)))**.6
 q=np.zeros(len(m['a']))
 for i in np.unique(m['a']):
  js=np.flatnonzero(m['a']==i);dest=m['b'][js]
  widths=np.array([m['w'][j] if j<len(cells) else m['w'][i] for j in dest]);q[js]=Q*shares[i]*widths/widths.sum()/m['wf'][js]
 return h,q

def trial(ex,state='S000_base',n=None,length=100,phase='full',cfl=.35):
 f=frame(ex,state);n=float(f.n_manning.iloc[0]) if n is None else float(n)
 tag=f'{state}_dx{length:g}_n{n:.10f}_{phase}_cfl{cfl:g}';folder=OUT/ex/tag;folder.mkdir(parents=True,exist_ok=True);record=folder/'result.json'
 if record.exists():return refresh_feature(json.loads(record.read_text()))
 started=time.monotonic();cells=v.grid(f,f'max{length}');m=build(cells,n)
 q=forcing(ex);origin=q.interval_start_utc.iloc[0]
 if phase=='search':
  end=max(q.loc[q.phase.eq('input_event'),'interval_end_utc'].iloc[-1],pd.Timestamp(TARGETS[ex]['target_utc'])+pd.Timedelta(days=3))
  q=q[q.interval_start_utc<end].copy()
 times=np.r_[(q.interval_start_utc-origin).dt.total_seconds(),(q.interval_end_utc.iloc[-1]-origin).total_seconds()];vals=np.r_[q.forcing_cms,q.forcing_cms.iloc[-1]]
 initial=initial_guess(f,cells,m,vals[0]);warm=[]
 # An interrupted event may already have a completed steady initializer.
 # It is evolved and convergence-checked again before any event resumes.
 if (folder/'initial.npz').exists():warm.append((0.,folder/'initial.npz'))
 # Search/full horizons share the same initial inflow and geometry. Reuse
 # either steady initializer, then recheck convergence at the requested n.
 for path in (OUT/ex).glob(f'{state}_dx{length:g}_n*_*_cfl*/initial.npz'):
  try:
   rr=json.loads((path.parent/'result.json').read_text())
   if rr.get('valid'):warm.append((abs(np.log(rr['n']/n)),path))
  except (OSError,ValueError):pass
 for _,path in sorted(warm,key=lambda x:x[0]):
  try:
   with np.load(path) as data:hh,qq=data['h'].copy(),data['q'].copy()
   if hh.shape==(len(cells),) and qq.shape==(len(m['a']),) and np.isfinite(hh).all() and np.isfinite(qq).all():
    initial=(hh,qq);break
  except (OSError,ValueError,EOFError,zipfile.BadZipFile):continue
 try:
  for day in range(1,61):
   spun=v.solve(m,[0,86400],[vals[0],vals[0]],initial,cfl=cfl,max_dt=150);initial=(spun['h'],spun['q'])
   if not np.isfinite(initial[0]).all():raise ValueError('Nonfinite spinup')
   if spun['audit']['final_node_imbalance_cms']/max(vals[0],1)<1e-8:break
  else:raise ValueError('Spinup did not converge in 60 days')
  np.savez(folder/'initial.npz',h=initial[0],q=initial[1]);m['abort_on_invalid']=True
  result=v.solve(m,times,vals,initial,cfl=cfl,max_dt=150,samples=np.array([0,len(times)-1]))
  means=np.diff(result['wave'][:,4])/60;t=(q.time_utc-origin).dt.total_seconds().to_numpy()
  selected=q.phase.eq('input_event').to_numpy()
  obs=pd.read_parquet(OUT/'inputs'/ex/'observed_downstream.parquet').sort_values('time')
  feature_mask=(q.time_utc>=obs.time.iloc[0])&(q.time_utc<=obs.time.iloc[-1])&(q.time_utc>=q.loc[selected,'interval_start_utc'].iloc[0])
  tm,height,kind,lobes=feature(means[feature_mask],t[feature_mask],ex,TARGETS[ex])
  target=(pd.Timestamp(TARGETS[ex]['target_utc'])-origin).total_seconds()
  event=q[selected];baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[vals[0],q.loc[selected,'forcing_cms'].iloc[-1]])
  met=v.metrics(t,means,baseline);audit=result['audit'];relative=audit['maximum_mass_residual_m3']/max(result['wave'][-1,3],1)
  valid=bool(np.isfinite(means).all() and audit['clipping_volume_m3']<1e-6 and audit['limited_face_steps']==0 and relative<1e-9)
  wave=q[['time_utc','phase']].copy();wave['q_outlet_cms']=means;wave.to_parquet(folder/'wave.parquet',index=False)
  obs=pd.read_parquet(OUT/'inputs'/ex/'observed_downstream.parquet').sort_values('time');ot=(obs.time-origin).dt.total_seconds().to_numpy();oq=obs.discharge_cms.to_numpy();ix=np.searchsorted(ot,t);left=np.clip(ix-1,0,len(ot)-1);right=np.clip(ix,0,len(ot)-1)
  cadence=float(np.median(np.diff(ot)));supported=(t>=ot[0])&(t<=ot[-1])&((ot[right]-ot[left])<=1.5*cadence)&feature_mask
  observed=np.interp(t,ot,oq);err=means[supported]-observed[supported]
  rmse=float(np.sqrt(np.mean(err**2))) if len(err) else None
  nse=float(1-np.sum(err**2)/np.sum((observed[supported]-observed[supported].mean())**2)) if len(err)>1 and np.ptp(observed[supported]) else None
  low=HEIGHTS[ex].get('low_cms');high=HEIGHTS[ex].get('high_cms')
  if low is None:low=high=HEIGHTS[ex].get('height_target_nominal_cms',height)
  r=dict(example=ex,state=state,n=n,length=length,phase=phase,cfl=cfl,cells=len(cells),valid=valid,seconds=time.monotonic()-started,spinup_days=day,spinup_audit=spun['audit'],timing_error_minutes=(tm-target)/60,feature_utc=str(origin+pd.to_timedelta(tm,unit='s')),feature_kind=kind,feature_height_cms=height,height_distance_cms=max(low-height,height-high,0),raw_downstream_rmse_cms=rmse,raw_downstream_nse=nse,relative_mass_residual=relative,final_outlet_error_cms=float(means[-1]-vals[-1]),maximum_last_48h_error_cms=float(abs(means[-min(2880,len(means)):]-vals[-1]).max()),folder=str(folder),**met,**audit)
  r['peak_utc']=str(origin+pd.to_timedelta(r['peak_seconds'],unit='s'))
 except Exception as exc:
  r=dict(example=ex,state=state,n=n,length=length,phase=phase,cfl=cfl,cells=len(cells),valid=False,error=str(exc),spinup_audit=spun['audit'] if 'spun' in locals() else None,seconds=time.monotonic()-started,folder=str(folder))
 save(record,r);print({k:r[k] for k in ['example','state','n','length','phase','cfl','valid','seconds']},flush=True);return r

def refresh_feature(r):
 if not r.get('valid'):return r
 ex=r['example'];q=pd.read_parquet(Path(r['folder'])/'wave.parquet');f=forcing(ex);origin=f.interval_start_utc.iloc[0]
 obs=pd.read_parquet(OUT/'inputs'/ex/'observed_downstream.parquet').sort_values('time')
 mask=(q.time_utc>=obs.time.iloc[0])&(q.time_utc<=obs.time.iloc[-1])&(q.time_utc>=f.loc[f.phase.eq('input_event'),'interval_start_utc'].iloc[0])
 t=(q.time_utc-origin).dt.total_seconds().to_numpy();tm,height,kind,lobes=feature(q.q_outlet_cms.to_numpy()[mask],t[mask],ex,TARGETS[ex])
 r.update(timing_error_minutes=(tm-(pd.Timestamp(TARGETS[ex]['target_utc'])-origin).total_seconds())/60,feature_utc=str(origin+pd.to_timedelta(tm,unit='s')),feature_kind=kind,feature_height_cms=height)
 low=HEIGHTS[ex].get('low_cms');high=HEIGHTS[ex].get('high_cms')
 if low is None:low=high=HEIGHTS[ex].get('height_target_nominal_cms',height)
 r['height_distance_cms']=max(low-height,height-high,0)
 ot=(obs.time-origin).dt.total_seconds().to_numpy();oq=obs.discharge_cms.to_numpy();ix=np.searchsorted(ot,t);left=np.clip(ix-1,0,len(ot)-1);right=np.clip(ix,0,len(ot)-1)
 supported=mask&((ot[right]-ot[left])<=1.5*np.median(np.diff(ot)))
 observed=np.interp(t,ot,oq);errors=q.q_outlet_cms.to_numpy()[supported]-observed[supported]
 r['raw_downstream_rmse_cms']=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
 r['raw_downstream_nse']=float(1-np.sum(errors**2)/np.sum((observed[supported]-observed[supported].mean())**2)) if len(errors)>1 and np.ptp(observed[supported]) else None
 return r

def robust(ex,state='S000_base',n=None,length=100,phase='full'):
 for cfl in [.35,.175,.0875]:
  r=trial(ex,state,n,length,phase,cfl)
  if r['valid']:return r
  spin=r.get('spinup_audit') or {}
  if 'Critical-flow limiter active' in r.get('error','') and spin.get('limited_face_steps',0)>0 and spin.get('clipping_volume_m3',1)<1e-6:
   r['rejection_class']='limiter_active_in_steady_initial_state';return r
 return r

def baseline(ex):
 r=robust(ex);save(OUT/ex/'baseline.json',r);return r

def calibrate(ex):
 dest=OUT/ex/'calibration.json'
 if dest.exists():return json.loads(dest.read_text())
 f=frame(ex,'S000_base');n0=float(f.n_manning.iloc[0]);records={}
 def evaluate(n):
  n=float(np.clip(n,.01,.1));key=round(n,10)
  if key not in records:records[key]=robust(ex,n=key,phase='search')
  return records[key]
 # Fixed coarse scan detects peak switches and invalid low-roughness regions.
 for n in sorted(set([.01,.0175,.025,.04,.07,.1,n0])):evaluate(n)
 ns=sorted(records)
 for a,b in zip(ns[:-1],ns[1:]):
  ra=records[a];rb=records[b]
  if not(ra['valid'] and rb['valid']):continue
  if ra['timing_error_minutes']*rb['timing_error_minutes']>=0:continue
  for iteration in range(16):
   ea=ra['timing_error_minutes'];eb=rb['timing_error_minutes'];frac=float(np.clip(-ea/(eb-ea),.15,.85));n=a+frac*(b-a);r=evaluate(n)
   if not r['valid']:break
   if abs(r['timing_error_minutes'])<=.5:break
   if ea*r['timing_error_minutes']<=0:b=n;rb=r
   else:a=n;ra=r
 good=[r for r in records.values() if r['valid']]
 if not good:
  result=dict(example=ex,status='no_valid_trial',trials=list(records.values()));save(dest,result);return result
 feasible=[r for r in good if abs(r['timing_error_minutes'])<=1]
 best=min(feasible,key=lambda r:(r['height_distance_cms'],abs(np.log(r['n']/n0)))) if feasible else min(good,key=lambda r:abs(r['timing_error_minutes']))
 final=robust(ex,n=best['n'],phase='full')
 result=dict(example=ex,status='timing_calibrated' if final['valid'] and abs(final['timing_error_minutes'])<=1 else 'timing_target_unresolved',selected_n=best['n'],mc_n=n0,final=final,trials=list(records.values()))
 save(dest,result);return result

def transfer(ex):
 cal=json.loads((OUT/ex/'calibration.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
 if 'selected_n' not in cal:return dict(example=ex,status='no_n_available')
 rows=[robust(ex,state,n=cal['selected_n']) for state in selection['children'][ex]]
 result=dict(example=ex,parent_calibration_status=cal['status'],n=cal['selected_n'],runs=rows);save(OUT/ex/'transfer.json',result);return result

def refinement(job):
 ex,state,length=job;return robust(ex,state,length=length)

def pool(fn,jobs,workers=3):
 with ProcessPoolExecutor(max_workers=workers) as executor:
  futures={executor.submit(fn,j):j for j in jobs}
  for fut in as_completed(futures):
   try:fut.result()
   except Exception:print('WORKER FAILED',futures[fut],traceback.format_exc(),flush=True)

def topology_check():
 f=pd.DataFrame(dict(id_link=range(5),u=[0,1,1,1,2],v=[1,2,2,2,3],width=[300.,100.,100.,100.,300.],link_length_m=[200.]*5,head_us=[3.,2.8,2.8,2.8,2.6],head_ds=[2.8,2.6,2.6,2.6,2.4]))
 m=build(f,.025);h=np.full(5,2.);unit=2**(5/3)*np.sqrt(.001)/.025;q=np.full(len(m['a']),unit)
 r=v.solve(m,[0,3600],[unit*300]*2,(h,q),cfl=.35)
 assert abs(r['h']-2).max()<1e-9 and r['audit']['maximum_mass_residual_m3']<1e-5
 save(OUT/'three_way_junction_check.json',dict(passed=True,maximum_depth_error_m=float(abs(r['h']-2).max()),audit=r['audit']))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','baseline','refine','calibrate','transfer']);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
 if args.stage=='prepare':prepare();topology_check()
 elif args.stage=='baseline':pool(baseline,list(CASES),args.workers)
 elif args.stage=='calibrate':pool(calibrate,list(CASES),args.workers)
 elif args.stage=='transfer':pool(transfer,list(CASES),args.workers)
 else:
  selection=json.loads((OUT/'selection.json').read_text());jobs=[(ex,state,length) for ex in SUBSET for state in ['S000_base',*selection['children'][ex]] for length in [200,100,50] if not(state=='S000_base' and length==100)]
  pool(refinement,jobs,args.workers)
